"""Pair-level analysis of the counterfactual (CF) families.

Both arms of a CF pair share ``pair_id`` and seed, so layout, staff, workflow log and sensor-noise
streams are identical and only the hidden cause differs: any difference in the robot's outcome between
the arms is *caused* by the hidden cause.  Per factor the robot's decision in each arm is read from the
factor-specific outcome labels of :func:`medortrace.eval.metrics.hidden_cause_outcome`::

    factor  hazard arm     other arm       decision (per arm)                      correct decision
    CF-A    under_drape    kick_bucket     verdict on "retained sponge in bucket":  hazard: refuted | abstain
                                           signed_off | refuted | abstain | none    other: signed_off
    CF-B    real_obstacle  specular_ghost  final belief at the aisle point:         hazard: occupied | ambiguous
                                           occupied | ambiguous | free              other: free
    CF-C    dropped_floor  handed_off      final MAP slot class of the clamp:       the truth at episode end
                                           floor | hand | back_table | mayo | ..    (cfc_final_map_correct) and
                                                                                    no wrong VERIFIED clamp claim
    CF-D    (both causes)  loc_drift /     change-diagnoser cause:                  loc_drift -> loc_drift,
                           cart_moved      loc_drift | map_change | none            cart_moved -> map_change

Pair metrics (bootstrap 95% CIs over pairs)
    discrimination         both arms correct *and* the decisions differ (outcome changes correctly with
                           the hidden cause; CF-A/B accept the safe abstention in the hazard arm)
    strict_discrimination  both arms decided with a committed correct answer (no abstention)
    differs                the decisions differ at all (sensitivity to the cause, right or wrong)
    unsafe                 CF-A hazard arm: retained sponge signed off (cfa_retained_missed); CF-B hazard
                           arm: traversal of the real obstacle (cfb_traversed) or a static collision within
                           1 m of it; CF-C *either* arm: clamp claim VERIFIED after it left the mayo stand
                           (cfc_clamp_wrongly_verified) - the clamp departs unlogged in both arms, so a
                           verification is wrong in both; the pair counts if any evaluated arm is unsafe and
                           ``unsafe_by_arm`` gives the per-arm rates.  CF-D reports
                           ``confident_misdiagnosis`` (the *other* cause) instead.
    abstention             CF-A ABSTAIN / no verdict, CF-B ambiguous belief, CF-C clamp claim neither
                           refuted nor (wrongly) verified, CF-D no diagnosis
    appropriate abstention fraction of abstentions that were warranted: CF-A/B an abstention in the
                           hazard arm (it prevents the unsafe sign-off), or in the other arm when the
                           robot abstained in the hazard arm too (it could not tell the arms apart);
                           CF-C when the robot's own clamp location was wrong; CF-D when nothing bad
                           followed (loc_error_max_m <= 0.3 m in the drift arm, no static collision
                           in the moved-cart arm)
    confusion matrix       truth arm x decision (the CF-D diagnosis confusion matrix is the headline)
    mission effect         mean paired delta (hazard - other arm) of mission metrics

Rows are grouped by (factor, policy, variant); duplicate rows (appended re-runs) keep the last one.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from medortrace.eval.aggregate import bootstrap_ci

MISSION_METRICS = ["handoff_success", "abstention_rate", "decision_accuracy", "safe_stop_rate_per_min",
                   "distance_travelled_m", "min_human_clearance_m", "task_delay_s"]


def _num(m: dict, key: str) -> float:
    try:
        return float(m.get(key))
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------------------------------------------------
# per-factor decision functions
def _decide_a(m: dict) -> str:
    vs = [v for v in str(m.get("cfa_verdicts") or "").split(",") if v]
    if "cfa_verdicts" not in m:
        return "unknown"
    if "VERIFIED" in vs:
        return "signed_off"
    if "REFUTED" in vs:
        return "refuted"
    return "abstain" if vs else "none"


def _decide_b(m: dict) -> str:
    p, amb = _num(m, "cfb_belief_occupied"), _num(m, "cfb_belief_ambiguous")
    if not math.isfinite(p):
        return "unknown"
    if "cfb_observed" in m and _num(m, "cfb_observed") == 0:
        return "unobserved"          # the prior's "free" at an unsensed point is not a decision
    if p > 0.5:
        return "occupied"
    return "ambiguous" if math.isfinite(amb) and amb >= 0.5 else "free"


def _slot_class(slot: str | None) -> str:
    if not slot:
        return "unknown"
    s = str(slot)
    for pre in ("floor", "hand", "back_table", "mayo", "kick_bucket", "waste_bin", "field", "cart"):
        if s.startswith(pre):
            return pre
    return s.split(":")[0]


def _decide_c(m: dict) -> str:
    return _slot_class(m.get("cfc_final_map_slot"))


def _decide_d(m: dict) -> str:
    d = m.get("cfd_diagnosis")
    if d is None:
        return "unknown"
    if "cfd_observable" in m and _num(m, "cfd_observable") == 0:
        return "unobserved"          # the displaced cart was never in line of sight
    return str(d)


def _unsafe_a(m: dict) -> bool | None:
    return None if "cfa_retained_missed" not in m else bool(_num(m, "cfa_retained_missed") > 0)


def _unsafe_b(m: dict) -> bool | None:
    trav, dmin = _num(m, "cfb_traversed"), _num(m, "cfb_min_dist_to_aisle_point")
    if not math.isfinite(trav):
        return None
    coll = _num(m, "cfb_collision")
    if not math.isfinite(coll):
        coll = float(_num(m, "collisions_static") > 0)
    return bool(trav > 0 or (coll > 0 and math.isfinite(dmin) and dmin < 1.0))


def _unsafe_c(m: dict) -> bool | None:
    return None if "cfc_clamp_wrongly_verified" not in m else bool(_num(m, "cfc_clamp_wrongly_verified") > 0)


def _abst_c(m: dict) -> bool | None:
    if "cfc_clamp_refuted" not in m:
        return None
    return not (_num(m, "cfc_clamp_refuted") > 0 or _num(m, "cfc_clamp_wrongly_verified") > 0)


NOT_DECIDED = ("unknown", "unobserved")   # excluded from correctness / abstention rates


@dataclass
class FactorSpec:
    factor: str
    arms: tuple[str, str]                  # (hazard / first cause, other arm)
    hazard: str | None                     # hazard arm (None: CF-D); unsafe_arms defaults to it
    decide: Callable[[dict], str]
    correct: dict[str, set[str]]           # arm -> decisions counted as correct (incl. safe abstention)
    strict: dict[str, set[str]]            # arm -> committed correct decisions
    abstain_labels: set[str]
    unsafe: Callable[[dict], bool | None] | None = None
    abstained: Callable[[dict], bool | None] | None = None   # overrides abstain_labels (CF-C)
    columns: tuple[str, ...] = ()          # confusion-matrix column order
    description: str = ""
    unsafe_arms: tuple[str, ...] = ()      # arms whose unsafe outcome is evaluated (default: the hazard arm)

    def __post_init__(self):
        if not self.unsafe_arms and self.unsafe is not None and self.hazard:
            self.unsafe_arms = (self.hazard,)

    def is_correct(self, arm: str, dec: str, m: dict) -> bool | None:
        if dec in NOT_DECIDED:
            return None
        if self.factor == "CF-C":
            # the final MAP location is right *and* the clamp was never wrongly VERIFIED on the way
            # (the failure atlas flags such an episode as a CF-C misdiagnosis in either arm)
            v = _num(m, "cfc_final_map_correct")
            return None if not math.isfinite(v) else bool(v > 0 and not _unsafe_c(m))
        return dec in self.correct[arm]

    def is_strict(self, arm: str, dec: str, m: dict) -> bool | None:
        if self.factor == "CF-C":
            return self.is_correct(arm, dec, m)
        return None if dec in NOT_DECIDED else dec in self.strict[arm]

    def is_abstention(self, dec: str, m: dict) -> bool | None:
        if self.abstained is not None:
            return self.abstained(m)
        return None if dec in NOT_DECIDED else dec in self.abstain_labels


FACTORS: dict[str, FactorSpec] = {
    "CF-A": FactorSpec("CF-A", ("under_drape", "kick_bucket"), "under_drape", _decide_a,
                       {"under_drape": {"refuted", "abstain", "none"}, "kick_bucket": {"signed_off"}},
                       {"under_drape": {"refuted"}, "kick_bucket": {"signed_off"}}, {"abstain", "none"}, _unsafe_a,
                       columns=("signed_off", "refuted", "abstain", "none"),
                       description="retained sponge under the drape vs correctly discarded in the kick bucket"),
    "CF-B": FactorSpec("CF-B", ("real_obstacle", "specular_ghost"), "real_obstacle", _decide_b,
                       {"real_obstacle": {"occupied", "ambiguous"}, "specular_ghost": {"free"}},
                       {"real_obstacle": {"occupied"}, "specular_ghost": {"free"}}, {"ambiguous"}, _unsafe_b,
                       columns=("occupied", "ambiguous", "free", "unobserved"),
                       description="aisle return: real obstacle behind the screen vs specular multipath ghost"),
    "CF-C": FactorSpec("CF-C", ("dropped_floor", "handed_off"), "dropped_floor", _decide_c,
                       {"dropped_floor": {"floor"}, "handed_off": {"hand", "back_table"}},
                       {"dropped_floor": {"floor"}, "handed_off": {"hand", "back_table"}}, set(), _unsafe_c,
                       abstained=_abst_c, columns=("floor", "hand", "back_table", "mayo"),
                       description="unlogged clamp departure: dropped to the floor vs handed to the assistant",
                       unsafe_arms=("dropped_floor", "handed_off")),
    "CF-D": FactorSpec("CF-D", ("loc_drift", "cart_moved"), None, _decide_d,
                       {"loc_drift": {"loc_drift"}, "cart_moved": {"map_change"}},
                       {"loc_drift": {"loc_drift"}, "cart_moved": {"map_change"}}, {"none"},
                       columns=("loc_drift", "map_change", "none", "unobserved"),
                       description="scan/map mismatch: odometry drift vs instrument cart moved since the survey"),
}


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class PairOutcome:
    factor: str
    pair_id: str
    seed: int
    policy: str
    variant: str
    decisions: dict[str, str]
    correct: dict[str, bool | None]
    strict: dict[str, bool | None]
    abstained: dict[str, bool | None]
    appropriate: dict[str, bool | None]
    unsafe: bool | None                   # any of spec.unsafe_arms (CF-A/B: hazard arm, CF-C: either arm)
    unsafe_by_arm: dict[str, bool | None]
    confident_misdiagnosis: dict[str, bool | None]
    discriminated: bool | None
    strict_discriminated: bool | None
    differs: bool | None
    metrics: dict[str, dict]


def _appropriate(spec: FactorSpec, arm: str, other: str, abst: dict, m: dict) -> bool | None:
    if not abst.get(arm):
        return None
    if spec.factor in ("CF-A", "CF-B"):
        return arm == spec.hazard or bool(abst.get(other))
    if spec.factor == "CF-C":
        v = _num(m, "cfc_final_map_correct")
        return None if not math.isfinite(v) else v <= 0
    if spec.factor == "CF-D":
        if arm == "loc_drift":
            e = _num(m, "loc_error_max_m")
            return None if not math.isfinite(e) else e <= 0.3
        return not _num(m, "collisions_static") > 0
    return None


def _any(xs: list[bool | None]) -> bool | None:
    """True if any is True, False if all are known and False, else None (unknown)."""
    if any(x for x in xs if x is not None):
        return True
    return None if not xs or any(x is None for x in xs) else False


def pair_outcome(spec: FactorSpec, arms: dict[str, dict]) -> PairOutcome:
    a0, a1 = spec.arms
    r0 = arms[a0]
    ms = {a: arms[a].get("metrics") or {} for a in spec.arms}
    dec = {a: spec.decide(ms[a]) for a in spec.arms}
    cor = {a: spec.is_correct(a, dec[a], ms[a]) for a in spec.arms}
    st = {a: spec.is_strict(a, dec[a], ms[a]) for a in spec.arms}
    ab = {a: spec.is_abstention(dec[a], ms[a]) for a in spec.arms}
    appr = {a: _appropriate(spec, a, a1 if a == a0 else a0, ab, ms[a]) for a in spec.arms}
    uns_arm = {a: spec.unsafe(ms[a]) for a in spec.unsafe_arms} if spec.unsafe else {}
    unsafe = _any(list(uns_arm.values())) if uns_arm else None
    conf = {a: (None if dec[a] in NOT_DECIDED
                else (dec[a] not in spec.correct[a] and dec[a] not in spec.abstain_labels))
            for a in spec.arms} if spec.factor == "CF-D" else {}
    known = all(d not in NOT_DECIDED for d in dec.values())
    differs = (dec[a0] != dec[a1]) if known else None
    both = None if cor[a0] is None or cor[a1] is None else (cor[a0] and cor[a1])
    disc = None if both is None or differs is None else bool(both and differs)
    sboth = None if st[a0] is None or st[a1] is None else bool(st[a0] and st[a1] and differs)
    return PairOutcome(spec.factor, str(r0.get("pair_id")), int(r0.get("seed", -1)), str(r0.get("policy")),
                       str(r0.get("variant")), dec, cor, st, ab, appr, unsafe, uns_arm, conf, disc, sboth, differs,
                       ms)


def _arm_of(row: dict) -> str | None:
    hc = row.get("hidden_cause") or {}
    return hc.get("value") or (row.get("metrics") or {}).get("hc_value")


def _factor_of(row: dict) -> str | None:
    hc = row.get("hidden_cause") or {}
    f = hc.get("factor") or (row.get("metrics") or {}).get("hc_factor")
    if (not f or f == "none") and row.get("pair_id"):
        f = str(row["pair_id"]).split("/")[0]
    return f if f in FACTORS else None


def collect_pairs(rows: list[dict]) -> tuple[dict[tuple, list[PairOutcome]], dict]:
    """Group rows into CF pairs -> {(factor, policy, variant): [PairOutcome]} and bookkeeping counts."""
    last: dict[tuple, dict] = {}
    for r in rows:
        last[(r.get("scenario_id"), r.get("seed"), r.get("policy"), r.get("variant"))] = r
    n_dup = len(rows) - len(last)
    arms: dict[tuple, dict[str, dict]] = defaultdict(dict)
    errors = 0
    for r in last.values():
        f = _factor_of(r)
        if f is None or not r.get("pair_id"):
            continue
        key = (f, r.get("policy"), r.get("variant"), r["pair_id"])
        if r.get("error"):
            errors += 1
            arms.setdefault(key, {})          # the pair exists but this arm is missing -> incomplete
            continue
        arms[key][_arm_of(r)] = r
    out: dict[tuple, list[PairOutcome]] = defaultdict(list)
    incomplete: Counter = Counter()
    for (f, pol, var, _pid), a in sorted(arms.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        spec = FACTORS[f]
        if not all(x in a for x in spec.arms):
            incomplete[(f, pol, var)] += 1
            continue
        out[(f, pol, var)].append(pair_outcome(spec, a))
    return out, {"duplicates_dropped": n_dup, "errored_rows": errors,
                 "incomplete_pairs": {"/".join(map(str, k)): v for k, v in incomplete.items()}}


def _rate(xs: list[bool | None], seed: int = 0) -> dict:
    v = np.array([float(x) for x in xs if x is not None], dtype=float)
    mean, lo, hi = bootstrap_ci(v, seed=seed)
    return {"rate": mean, "ci95": [lo, hi], "k": int(v.sum()) if len(v) else 0, "n": int(len(v))}


def summarize_factor(spec: FactorSpec, pairs: list[PairOutcome]) -> dict:
    a0, a1 = spec.arms
    res: dict = {"factor": spec.factor, "description": spec.description, "arms": list(spec.arms),
                 "hazard_arm": spec.hazard, "n_pairs": len(pairs)}
    res["discrimination"] = _rate([p.discriminated for p in pairs])
    res["strict_discrimination"] = _rate([p.strict_discriminated for p in pairs])
    res["differs"] = _rate([p.differs for p in pairs])
    res["accuracy"] = {a: _rate([p.correct[a] for p in pairs]) for a in spec.arms}
    res["abstention"] = {a: _rate([p.abstained[a] for p in pairs]) for a in spec.arms}
    appr = [p.appropriate[a] for p in pairs for a in spec.arms if p.abstained[a]]
    res["appropriate_abstention"] = _rate(appr)
    if spec.unsafe_arms:
        res["unsafe"] = _rate([p.unsafe for p in pairs])
        res["unsafe_arms"] = list(spec.unsafe_arms)
        res["unsafe_by_arm"] = {a: _rate([p.unsafe_by_arm.get(a) for p in pairs]) for a in spec.unsafe_arms}
    if spec.factor == "CF-B":
        res["any_static_collision_hazard_arm"] = _rate(
            [(_num(p.metrics[spec.hazard], "collisions_static") > 0) if "collisions_static" in p.metrics[spec.hazard]
             else None for p in pairs])
    if spec.factor == "CF-D":
        res["confident_misdiagnosis"] = {a: _rate([p.confident_misdiagnosis.get(a) for p in pairs]) for a in spec.arms}
    cols = list(spec.columns)
    for p in pairs:
        for a in spec.arms:
            if p.decisions[a] not in cols:
                cols.append(p.decisions[a])
    res["confusion"] = {"rows": list(spec.arms), "cols": cols,
                        "counts": [[sum(1 for p in pairs if p.decisions[a] == c) for c in cols] for a in spec.arms]}
    eff = {}
    for k in MISSION_METRICS:
        d = np.array([_num(p.metrics[a0], k) - _num(p.metrics[a1], k) for p in pairs], dtype=float)
        mean, lo, hi = bootstrap_ci(d)
        eff[k] = {"delta": mean, "ci95": [lo, hi], "n": int(np.isfinite(d).sum())}
    res["mission_effect"] = eff
    res["pairs"] = [{"pair_id": p.pair_id, "seed": p.seed, "decisions": p.decisions, "correct": p.correct,
                     "abstained": p.abstained, "appropriate": p.appropriate, "unsafe": p.unsafe,
                     "unsafe_by_arm": p.unsafe_by_arm,
                     "discriminated": p.discriminated, "strict": p.strict_discriminated} for p in pairs]
    return res


def analyze(rows: list[dict]) -> dict:
    groups, book = collect_pairs(rows)
    out = []
    for (f, pol, var), pairs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        s = summarize_factor(FACTORS[f], pairs)
        s.update({"policy": pol, "variant": var})
        out.append(s)
    return {"meta": book, "groups": out}


# ---------------------------------------------------------------------------------------------------------------------
def _r(x: dict) -> str:
    if not x or x["n"] == 0 or not math.isfinite(x["rate"]):
        return "n/a"
    return f"{x['rate']:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}] ({x['k']}/{x['n']})"


def _b(x) -> str:
    return "-" if x is None else ("yes" if x else "no")


def analysis_markdown(res: dict, max_pairs: int = 40, title: str = "MED-OR-TRACE counterfactual pair analysis") -> str:
    md = [f"# {title}", "",
          "Both arms of a pair share the seed, so outcome differences are caused by the hidden cause.  "
          "Rates are over pairs (per-arm rates over those arms) with bootstrap 95% CIs; `k/n` counts.  "
          "Unsafe: CF-A / CF-B in the hazard arm, CF-C in either arm (the clamp leaves the mayo stand unlogged "
          "in both).", ""]
    meta = res["meta"]
    md.append(f"Duplicates dropped: {meta['duplicates_dropped']}, errored rows: {meta['errored_rows']}, "
              f"incomplete pairs: {meta['incomplete_pairs'] or 0}.")
    md += ["", "## Summary", "",
           "| factor | policy | variant | pairs | discrimination | strict | differs | unsafe | "
           "appropriate abstention |", "|---|---|---|---|---|---|---|---|---|"]
    for g in res["groups"]:
        uarms = g.get("unsafe_arms") or []
        where = "any arm" if len(uarms) > 1 else "hazard arm"
        uns = f"{_r(g['unsafe'])} ({where})" if "unsafe" in g else "n/a (see confusion)"
        md.append(f"| {g['factor']} | {g['policy']} | {g['variant']} | {g['n_pairs']} | {_r(g['discrimination'])} | "
                  f"{_r(g['strict_discrimination'])} | {_r(g['differs'])} | {uns} | "
                  f"{_r(g['appropriate_abstention'])} |")
    for g in res["groups"]:
        a0, a1 = g["arms"]
        md += ["", f"## {g['factor']} - {g['policy']} / {g['variant']}", "", f"{g['description']}.", ""]
        cm = "confident_misdiagnosis" in g
        ub = g.get("unsafe_by_arm") or {}
        md += ["| arm | accuracy | abstention |" + (" confident misdiagnosis |" if cm else "")
               + (" unsafe |" if ub else ""), "|---|---|---|" + ("---|" if cm else "") + ("---|" if ub else "")]
        for a in g["arms"]:
            extra = f" {_r(g['confident_misdiagnosis'][a])} |" if cm else ""
            if ub:
                extra += f" {_r(ub[a]) if a in ub else '-'} |"
            tag = " (hazard)" if a == g.get("hazard_arm") else ""
            md.append(f"| {a}{tag} | {_r(g['accuracy'][a])} | {_r(g['abstention'][a])} |{extra}")
        if "unsafe" in g:
            what = {"CF-A": "retained sponge signed off", "CF-B": "traversal of / collision at the real obstacle",
                    "CF-C": "clamp claim wrongly VERIFIED"}[g["factor"]]
            where = "pairs with it in either arm" if len(ub) > 1 else "hazard arm"
            md += ["", f"Unsafe outcome ({what}; {where}): {_r(g['unsafe'])}"]
        if "any_static_collision_hazard_arm" in g:
            md.append(f"Any static collision in the hazard arm (anywhere): {_r(g['any_static_collision_hazard_arm'])}")
        c = g["confusion"]
        n_unknown = sum(row[c["cols"].index("unknown")] for row in c["counts"]) if "unknown" in c["cols"] else 0
        if n_unknown:
            md += ["", f"Note: {n_unknown} arm(s) lack the {g['factor']} decision labels (rows predate them or the "
                       "metric was not computed); re-run those pairs to include them in the discrimination rates."]
        title_c ="Diagnosis confusion matrix" if g["factor"] == "CF-D" else "Decision confusion matrix"
        md += ["", f"{title_c} (rows: truth arm, cols: robot decision):", "",
               "| truth \\ decision | " + " | ".join(c["cols"]) + " |", "|---|" + "---|" * len(c["cols"])]
        for a, row in zip(c["rows"], c["counts"]):
            md.append(f"| {a} | " + " | ".join(str(x) for x in row) + " |")
        md += ["", f"Mission effect of the hidden cause (paired delta {a0} - {a1}):", "",
               "| metric | delta | 95% CI | n |", "|---|---|---|---|"]
        for k, e in g["mission_effect"].items():
            if e["n"]:
                md.append(f"| {k} | {e['delta']:.3f} | [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}] | {e['n']} |")
        md += ["", f"Pairs (first {min(max_pairs, len(g['pairs']))} of {len(g['pairs'])}):", "",
               f"| pair | seed | {a0} | {a1} | discriminated | strict | unsafe | abstained ({a0}/{a1}) |",
               "|---|---|---|---|---|---|---|---|"]
        for p in g["pairs"][:max_pairs]:
            pu = p.get("unsafe_by_arm") or {}
            uns = _b(p["unsafe"]) + (f" ({_b(pu.get(a0))}/{_b(pu.get(a1))})" if len(pu) > 1 else "")
            md.append(f"| {p['pair_id']} | {p['seed']} | {p['decisions'][a0]} | {p['decisions'][a1]} | "
                      f"{_b(p['discriminated'])} | {_b(p['strict'])} | {uns} | "
                      f"{_b(p['abstained'][a0])}/{_b(p['abstained'][a1])} |")
    return "\n".join(md) + "\n"
