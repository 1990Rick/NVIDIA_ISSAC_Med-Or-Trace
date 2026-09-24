"""Failure atlas: taxonomy, counts and worst-case exemplars of benchmark failures.

Inputs
    * ``results.jsonl`` rows written by :func:`medortrace.eval.batch.run_batch`
      (``scenario_id, family, pair_id, seed, policy, variant, hidden_cause, metrics, error``,
      ``variant_spec`` - the variant's autonomy / cfg / sim overrides - and, when used, ``sim`` / ``duration``);
    * optionally, exported episode directories (:mod:`medortrace.data.writer` layout) under one or more
      roots.  They are matched to rows by ``(scenario_id, seed, policy)`` (the parent directory name is
      the batch variant) and are only parsed for the exemplars that are printed.

Taxonomy (an episode can fall into several categories; every detector returns a severity that ranks
the exemplars of its category, or nothing)::

    human_contact             collisions_agent > 0                         (truth contact with staff)
    static_collision          collisions_static > 0
    sterile_breach            sterile_breach_s > 0 ("breach"), or keepout_margin_violation_s >= margin_s
                              ("margin": inside the 0.3 m AORN margin of the sterile field)
    near_collision            near_collision_events > 0                    (clearance < 0.3 m)
    wrong_assertion           wrong_assertion_rate > 0; with an episode dir the wrong VERIFIED / REFUTED
                              verdicts are re-derived from events.jsonl against the truth_move records and
                              explained from provenance.json (supporting / contradicting evidence)
    retained_item_signed_off  CF-A under_drape: the retained sponge was VERIFIED in the kick bucket
    cf_misdiagnosis           CF-B cfb_correct == 0 | CF-C cfc_clamp_wrongly_verified | CF-D cfd_correct == 0
    safe_stop_cascade         safe_stop_rate_per_min >= stop_rate or uncertainty stops >= unc_stop_rate
    handover                  handover_requests >= 1 (operator had to take over)
    mission_starvation        abstention_rate >= a_max and handoff_success <= h_min (robot rarely answers)
    localization_loss         loc_error_max_m > loc_error_m
    energy_depletion          energy_reserve_frac < reserve_frac ("low_reserve"), or energy_used_wh above
                              energy_outlier x the median of its (family, policy, duration) group
    episode_error             the episode raised (``row["error"]``)

Per category the atlas reports counts by family (counterfactual rows are split by factor) and by
policy (``policy/variant`` when the file mixes variants), the worst-k exemplars with seed, the exact
one-line ``run_episode`` reproduce command (it exports the episode directory, so provenance can be
inspected; the variant's overrides come from the row's ``variant_spec``, else from ``variant_specs``, and a
command whose overrides are unknown is marked INEXACT), fault labels from ``meta.json`` and heuristic
root-cause hints.  The hints are rules over
metrics, fault labels, event logs and the trajectory at the failure tick; they point a reviewer at a
likely mechanism, they are not a diagnosis.
"""

from __future__ import annotations

import hashlib
import json
import math
import shlex
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from medortrace.data.writer import MODE_CODE

MODE_NAME = {v: k for k, v in MODE_CODE.items()}
ROBOT_R, HUMAN_R = 0.28, 0.25            # as in medortrace.eval.metrics
TAU_VERIFY, TAU_REFUTE = 0.9, 0.1        # ClaimVerifier defaults (for "marginal posterior" hints)
DEFAULT_DURATION_S = 180.0               # configs/scenarios/default.yaml episode.duration_s

CATEGORIES: dict[str, str] = {
    "human_contact": "robot body touched a person (truth contact with staff)",
    "static_collision": "robot drove into static structure (furniture, carts, hidden obstacles)",
    "sterile_breach": "robot entered the sterile field (breach) or its 0.3 m AORN keep-out margin",
    "near_collision": "surface clearance to a person dropped below 0.3 m",
    "wrong_assertion": "a VERIFIED / REFUTED verdict contradicted the truth",
    "retained_item_signed_off": "CF-A: a sponge retained under the drape was VERIFIED as discarded",
    "cf_misdiagnosis": "counterfactual hidden cause misjudged (CF-B aisle belief, CF-C clamp, CF-D diagnosis)",
    "safe_stop_cascade": "repeated safety stops (mission throughput lost to the supervisor)",
    "handover": "the robot requested an operator takeover",
    "mission_starvation": "the robot mostly abstained and missed hand-off deadlines",
    "localization_loss": "pose error exceeded the localisation budget",
    "energy_depletion": "low battery reserve at termination or anomalous energy use",
    "episode_error": "the episode crashed (exception in the runner)",
}


@dataclass
class AtlasThresholds:
    sterile_margin_s: float = 1.0          # margin violations shorter than this are ignored
    stop_rate_per_min: float = 3.0
    unc_stop_rate_per_min: float = 1.0
    starvation_abstention: float = 0.6
    starvation_handoff: float = 0.3
    loc_error_m: float = 0.5
    reserve_frac: float = 0.2              # = safety Envelope.battery_soft_frac
    energy_outlier: float = 1.5            # x group median energy_used_wh


@dataclass
class FailureCase:
    category: str
    subtype: str
    severity: float
    scenario_id: str
    seed: int
    family: str
    policy: str
    variant: str
    pair_id: str | None
    hidden_cause: dict | None
    evidence: dict[str, Any]
    hints: list[str] = field(default_factory=list)
    reproduce: str = ""
    reproduce_exact: bool = True           # False: the variant's overrides are unknown (see reproduce_command)
    episode_dir: str | None = None
    fault_labels: list[str] = field(default_factory=list)
    wrong_verdicts: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------------------
# row helpers
def _num(m: dict, key: str) -> float:
    v = m.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _has(m: dict, key: str) -> bool:
    return math.isfinite(_num(m, key))


def factor_of(row: dict) -> str | None:
    hc = row.get("hidden_cause") or {}
    f = hc.get("factor") or (row.get("metrics") or {}).get("hc_factor")
    if (not f or f == "none") and row.get("pair_id"):
        f = str(row["pair_id"]).split("/")[0]
    return f if f and f != "none" else None


def cf_value_of(row: dict) -> str | None:
    hc = row.get("hidden_cause") or {}
    return hc.get("value") or (row.get("metrics") or {}).get("hc_value")


def stratum(row: dict) -> str:
    fam = row.get("family") or "?"
    f = factor_of(row)
    return f"{fam}/{f}" if fam == "counterfactual" and f else fam


def row_key(row: dict) -> tuple:
    return (row.get("scenario_id"), int(row.get("seed", -1)), row.get("policy"), row.get("variant"))


def dedupe(rows: list[dict]) -> tuple[list[dict], int]:
    """``run_batch`` appends to results.jsonl; keep the last row per (scenario, seed, policy, variant)."""
    last: dict[tuple, dict] = {}
    for r in rows:
        last[row_key(r)] = r
    return list(last.values()), len(rows) - len(last)


def load_rows(paths: list[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# reproduce command
PLAIN_VARIANTS = ("full", "nominal")     # no-override variant names of run_batch / run_ablation / sim_to_real


def resolve_variant_spec(row: dict, variant_spec: dict | None = None) -> tuple[dict | None, str]:
    """The overrides a row ran with -> ``(spec, source)``.

    ``source``: ``recorded`` (the row's ``variant_spec``, written by ``run_batch``; it wins because it is what
    ran), ``given`` (the caller's ``variant_spec``, e.g. ``--variant-specs``), ``plain`` (no record, but the
    variant is a conventional no-override name) or ``unknown`` (``spec`` is None: an older row of an ablation
    variant; re-running the base config would not reproduce it).
    """
    rec = row.get("variant_spec")
    if isinstance(rec, dict):
        return dict(rec), "recorded"
    if variant_spec is not None:
        return dict(variant_spec), "given"
    if row.get("variant") in (None, "", *PLAIN_VARIANTS):
        return {}, "plain"
    return None, "unknown"


def reproduce_command(row: dict, registry: str = "configs/seed_registry.yaml", out_dir: str = "runs/repro",
                      variant_spec: dict | None = None, backend: str = "lite") -> str:
    """Exact one-line ``run_episode`` call for a results row (exports the episode directory).

    The variant's overrides ({"autonomy": .., "cfg": .., "sim": ..}) are taken from the row's recorded
    ``variant_spec``, else from ``variant_spec``; ``sim`` / ``duration`` recorded in the row are always applied.
    When neither is available for a non-plain variant (see :func:`resolve_variant_spec`) the command runs the
    base config and ends with a shell comment ``# INEXACT: ...`` saying so.
    """
    spec, source = resolve_variant_spec(row, variant_spec)
    spec = spec or {}
    if row.get("sim"):
        spec["sim"] = row["sim"]
    cfg_expr = "e.resolve()"
    imports = "from medortrace.eval.registry import load_registry; from medortrace.eval.runner import run_episode"
    if spec.get("cfg"):
        imports += "; from medortrace.common.config import deep_merge"
        cfg_expr = f"deep_merge(e.resolve(), {spec['cfg']!r})"
    kw = [f"policy={row.get('policy', 'active')!r}", f"out_dir={out_dir!r}"]
    if backend != "lite":
        kw.append(f"backend={backend!r}")
    if row.get("duration"):
        kw.append(f"duration={float(row['duration'])!r}")
    if spec.get("autonomy"):
        kw.append(f"autonomy_override={spec['autonomy']!r}")
    if spec.get("sim"):
        kw.append(f"sim_overrides={spec['sim']!r}")
    code = (f"{imports}; e = next(x for x in load_registry({registry!r}) if x.scenario_id == {row['scenario_id']!r}); "
            f"assert e.seed == {int(row['seed'])}, 'registry seed changed'; "
            f"r = run_episode({cfg_expr}, e.seed, {', '.join(kw)}); print(r.out_dir); print(r.metrics)")
    quoted = f'"{code}"' if not any(c in code for c in '"$`\\!') else shlex.quote(code)
    cmd = f"PYTHONPATH=. python -c {quoted}"
    if source == "unknown":
        base = "the base config plus the recorded sim override" if row.get("sim") else "the base config"
        cmd += (f"  # INEXACT: the overrides of variant {row.get('variant')!r} were not recorded in this row "
                f"(older results); this runs {base} - rebuild the atlas with --variant-specs")
    return cmd


# ---------------------------------------------------------------------------------------------------------------------
# exported episodes
@dataclass
class EpisodeRecord:
    path: str
    meta: dict
    events: list[dict]
    prov: dict | None = None
    _traj: dict | None = None

    def traj(self) -> dict | None:
        if self._traj is None:
            p = Path(self.path) / "trajectory.npz"
            if p.exists():
                with np.load(p, allow_pickle=False) as z:
                    self._traj = {k: z[k] for k in z.files}
            else:
                self._traj = {}
        return self._traj or None

    def by_kind(self, kind: str) -> list[dict]:
        if kind == "verdict":
            return [e for e in self.events if is_verdict(e)]
        return [e for e in self.events if e.get("kind") == kind and not is_verdict(e)]


def is_verdict(e: dict) -> bool:
    # writer.py merges VerdictRecord.__dict__ after {"kind": "verdict"}, so the claim kind
    # (handoff | count | location) overwrites the record kind: detect verdicts by their fields.
    return "claim_id" in e and "verdict" in e


def verdict_claim_kind(e: dict) -> str:
    k = e.get("claim_kind") or e.get("kind")
    return k if k and k != "verdict" else "?"


def load_episode(path: str | Path) -> EpisodeRecord:
    d = Path(path)
    meta = json.loads((d / "meta.json").read_text())
    events = []
    ev = d / "events.jsonl"
    if ev.exists():
        with open(ev) as f:
            events = [json.loads(line) for line in f if line.strip()]
    prov = None
    pp = d / "provenance.json"
    if pp.exists():
        prov = json.loads(pp.read_text())
    return EpisodeRecord(str(d), meta, events, prov)


class EpisodeIndex:
    """Exported episode directories keyed by (scenario_id, seed, policy); parsed lazily and cached."""

    def __init__(self, roots: list[str | Path] | None = None):
        self.dirs: dict[tuple, list[Path]] = defaultdict(list)
        self._cache: dict[str, EpisodeRecord] = {}
        for root in roots or []:
            root = Path(root)
            if not root.exists():
                continue
            for mp in sorted(root.rglob("meta.json")):
                try:
                    m = json.loads(mp.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                self.dirs[(m.get("scenario_id"), int(m.get("seed", -1)), m.get("policy"))].append(mp.parent)

    def __len__(self) -> int:
        return sum(len(v) for v in self.dirs.values())

    def find(self, row: dict) -> Path | None:
        cands = self.dirs.get((row.get("scenario_id"), int(row.get("seed", -1)), row.get("policy")), [])
        if not cands:
            return None
        for c in cands:
            if c.parent.name == row.get("variant"):
                return c
        return cands[0] if len(cands) == 1 else None

    def get(self, row: dict) -> EpisodeRecord | None:
        d = self.find(row)
        if d is None:
            return None
        if str(d) not in self._cache:
            self._cache[str(d)] = load_episode(d)
        return self._cache[str(d)]


# ---------------------------------------------------------------------------------------------------------------------
# truth, wrong verdicts and provenance
class TruthTimeline:
    """``slot(item, t)`` with the semantics of ``WorkflowScript.truth_slot`` (initial slot, then every
    truth move with ``t_move <= t``), rebuilt from the ``truth_move`` records of events.jsonl.  The
    initial slot is the first move's ``src``; items that never move take it from
    ``trajectory.npz:item_slot_true`` (first tick)."""

    def __init__(self, ep: EpisodeRecord):
        self.ep = ep
        self.moves: dict[str, list[dict]] = defaultdict(list)
        for m in ep.by_kind("truth_move"):
            self.moves[m["item"]].append(m)
        for v in self.moves.values():
            v.sort(key=lambda m: m["t"])
        self.initial: dict[str, str | None] = {i: v[0]["src"] for i, v in self.moves.items()}

    def _initial(self, item: str) -> str | None:
        if item not in self.initial:
            items, slots = self.ep.meta.get("items", []), self.ep.meta.get("slots", [])
            tr = self.ep.traj()
            if not tr or "item_slot_true" not in tr or item not in items or not len(tr["item_slot_true"]):
                return None
            k = int(tr["item_slot_true"][0, items.index(item)])
            self.initial[item] = slots[k] if 0 <= k < len(slots) else None
        return self.initial[item]

    def slot(self, item: str, t: float) -> str | None:
        s = self._initial(item)
        for m in self.moves.get(item, []):
            if m["t"] <= t:
                s = m["dst"]
        return s


def explain_from_prov(prov: dict, verdict_id: str, k: int = 5) -> dict | None:
    """Rebuild ``ProvenanceGraph.explain`` from the PROV-JSON export."""
    ents = prov.get("entity", {})
    if verdict_id not in ents:
        return None

    def attrs(nid: str) -> dict:
        rec = {**prov.get("entity", {}), **prov.get("activity", {}), **prov.get("agent", {})}.get(nid, {})
        out = {kk[4:]: v for kk, v in rec.items() if kk.startswith("mot:") and kk not in ("mot:hash", "mot:prev")}
        out.pop("content_digest", None)
        return out

    sup = [(r.get("prov:usedEntity"), float(r.get("mot:weight", 0.0))) for r in prov.get("wasDerivedFrom", {}).values()
           if r.get("prov:generatedEntity") == verdict_id]
    con = [(r.get("mot:evidence"), float(r.get("mot:weight", 0.0))) for r in prov.get("mot:contradicts", {}).values()
           if r.get("mot:verdict") == verdict_id]
    sup = sorted(sup, key=lambda x: -x[1])[:k]
    con = sorted(con, key=lambda x: x[1])[:k]
    fmt = lambda es: [{"evidence": e, "weight": w, **attrs(e)} for e, w in es]
    return {"verdict": attrs(verdict_id), "supporting": fmt(sup), "contradicting": fmt(con)}


def verify_prov_chain(prov: dict) -> bool | None:
    """Recompute the hash chain of a PROV-JSON export (same payload as ``ProvenanceGraph.verify_chain``)."""
    nodes = {**prov.get("entity", {}), **prov.get("activity", {}), **prov.get("agent", {})}
    if not nodes:
        return None
    by_prev = {rec.get("mot:prev"): (nid, rec) for nid, rec in nodes.items()}
    prev, seen = "GENESIS", 0
    while prev in by_prev:
        nid, rec = by_prev[prev]
        at = {kk[4:]: v for kk, v in rec.items()
              if kk.startswith("mot:") and kk not in ("mot:t", "mot:hash", "mot:prev")}
        payload = json.dumps({"id": nid, "kind": rec.get("prov:type"), "t": round(float(rec.get("mot:t", 0.0)), 6),
                              "attrs": at, "prev": prev}, sort_keys=True)
        if hashlib.sha256(payload.encode()).hexdigest() != rec.get("mot:hash"):
            return False
        prev, seen = rec["mot:hash"], seen + 1
    return seen == len(nodes)


def _evidence_brief(e: dict) -> str:
    skip = {"evidence", "weight", "sensor", "t", "robot_pose_est", "prov:type"}
    extra = []
    for kk, v in e.items():
        if kk in skip or kk.startswith("prov:"):
            continue
        if isinstance(v, list) and len(v) > 6:
            v = f"[{len(v)} values, max {max(v, default=0):.2f}]" if all(isinstance(x, (int, float)) for x in v) \
                else f"[{len(v)} values]"
        elif isinstance(v, float):
            v = f"{v:.3g}"
        extra.append(f"{kk}={v}")
    t = e.get("t")
    ts = f"t={t:.1f}s" if isinstance(t, (int, float)) else ""
    return f"{e.get('evidence')} ({e.get('sensor', '?')}, {ts}, w={e.get('weight', 0.0):+.2f}" + \
        (f", {', '.join(extra[:4])}" if extra else "") + ")"


def wrong_verdicts(ep: EpisodeRecord, k_evidence: int = 5) -> list[dict]:
    """Decided verdicts contradicting the truth at ``t_ref``, with provenance excerpts and hints."""
    truth = TruthTimeline(ep)
    moves = truth.moves
    logs = defaultdict(list)
    for w in ep.by_kind("workflow_log"):
        if w.get("item"):
            logs[w["item"]].append(w)
    out = []
    for v in ep.by_kind("verdict"):
        if v.get("verdict") not in ("VERIFIED", "REFUTED"):
            continue
        ts = truth.slot(v["item_id"], float(v["t_ref"]))
        if ts is None:
            continue
        holds = ts == v["slot_id"]
        if (v["verdict"] == "VERIFIED") == holds:
            continue
        between = [m for m in moves.get(v["item_id"], []) if float(v["t_ref"]) < m["t"] <= float(v["t"])]
        rec = {"claim_id": v["claim_id"], "claim_kind": verdict_claim_kind(v), "item": v["item_id"],
               "claimed_slot": v["slot_id"], "t_ref": float(v["t_ref"]), "t_verdict": float(v["t"]),
               "verdict": v["verdict"], "posterior": float(v.get("posterior", float("nan"))),
               "reason": v.get("reason"), "robot_map_slot": v.get("map_slot"), "direct": v.get("direct"),
               "truth_slot_at_t_ref": ts, "truth_moves_ref_to_verdict": between}
        if ep.prov is not None:
            rec["explanation"] = explain_from_prov(ep.prov, f"verdict:{v['claim_id']}", k_evidence)
        rec["hints"] = _verdict_hints(rec, moves.get(v["item_id"], []), logs.get(v["item_id"], []), ep)
        out.append(rec)
    return out


def _verdict_hints(rec: dict, item_moves: list[dict], item_logs: list[dict], ep: EpisodeRecord) -> list[str]:
    h = []
    p = rec["posterior"]
    if rec["verdict"] == "VERIFIED" and p < TAU_VERIFY + 0.05:
        h.append(f"marginal posterior p={p:.2f} just above tau_verify={TAU_VERIFY}")
    if rec["verdict"] == "REFUTED" and p > TAU_REFUTE - 0.05:
        h.append(f"marginal posterior p={p:.2f} just below tau_refute={TAU_REFUTE}")
    if not rec.get("direct"):
        h.append("decided without direct evidence of the claimed slot (require_direct_evidence off?)")
    if rec["truth_moves_ref_to_verdict"]:
        m = rec["truth_moves_ref_to_verdict"][0]
        h.append(f"item moved after t_ref ({m['src']} -> {m['dst']} at t={m['t']:.1f}s, cause={m['cause']}): "
                 "evidence about the new state leaked into the fixed-lag smoothing window")
    hidden = [m for m in item_moves if m.get("cause") != "workflow" and m["t"] <= rec["t_ref"]]
    if hidden:
        m = hidden[-1]
        h.append(f"unlogged move before t_ref ({m['src']} -> {m['dst']} at t={m['t']:.1f}s, cause={m['cause']}): "
                 "the workflow log asserted a state the world never reached")
    last_log = [w for w in item_logs if w["t"] <= rec["t_ref"] + 2.0]
    if last_log and last_log[-1].get("dst") and last_log[-1]["dst"] != rec["truth_slot_at_t_ref"]:
        w = last_log[-1]
        h.append(f"log/truth mismatch: last log entry says {w['dst']} (t={w['t']:.1f}s), truth is "
                 f"{rec['truth_slot_at_t_ref']} (missing or mislabelled log entry)")
    ex = rec.get("explanation") or {}
    sup = ex.get("supporting") or []
    if sup:
        sensors = Counter(e.get("sensor", "?") for e in sup)
        if len(sensors) == 1:
            s, n = next(iter(sensors.items()))
            h.append(f"support comes from a single modality ({s}, {n} records): correlated frames can over-count")
    elif rec["verdict"] == "VERIFIED" and ex:
        h.append("no supporting evidence edges: the verdict rests on the prior / workflow log")
    tr = ep.traj()
    if tr and "fault_active" in tr and len(tr["t"]):
        i = int(np.clip(np.searchsorted(tr["t"], rec["t_verdict"]), 0, len(tr["t"]) - 1))
        j = int(np.clip(np.searchsorted(tr["t"], rec["t_ref"]), 0, len(tr["t"]) - 1))
        if tr["fault_active"][min(i, j):max(i, j) + 1].any():
            h.append("a sensor fault was active between t_ref and the verdict")
    return h


# ---------------------------------------------------------------------------------------------------------------------
# fault labels and event-based summaries
def fault_labels(meta: dict) -> list[str]:
    f = meta.get("faults") or {}
    out = []
    for s, wins in (f.get("dropouts") or {}).items():
        tot = sum(b - a for a, b in wins)
        out.append(f"dropout:{s} ({len(wins)} windows, {tot:.1f}s)")
    for s, (off, drift) in (f.get("skew") or {}).items():
        out.append(f"skew:{s} ({off:+.3f}s, {drift * 1e6:.0f}ppm)")
    ob = f.get("odom_bias") or [0.0, 0.0]
    if any(abs(x) > 0 for x in ob):
        out.append(f"odom_bias (v {ob[0]:+.1%}, omega {ob[1]:+.3f} rad/s)")
    if float(f.get("specular_gain", 1.0)) != 1.0:
        out.append(f"specular_gain={float(f['specular_gain']):.2f}")
    if f.get("floor_wet"):
        out.append("floor_wet")
    out += [f"rare_geometry:{g}" for g in f.get("rare_geometry") or []]
    out += [f"occluder:{o}" for o in f.get("occluders") or []]
    for m in f.get("map_edits") or []:
        extra = f", shift={m['shift']}" if m.get("shift") is not None else ""
        out.append(f"map_edit:{m.get('kind')}({m.get('object', '?')}{extra})")
    nu = meta.get("nuisance") or {}
    if nu:
        out.append(f"nuisance haze={nu.get('haze', 0):.2f} glare_gain={nu.get('glare_gain', 1):.2f}")
    return out


def _reason_key(reason: str) -> str:
    """'predicted collision p=0.67 c=-0.22' -> 'predicted collision' (drop the measured values)."""
    words = []
    for w in str(reason).split():
        if any(ch.isdigit() for ch in w) or "=" in w:
            break
        words.append(w)
    return " ".join(words) or str(reason)


def safety_summary(ep: EpisodeRecord) -> dict:
    ev = ep.by_kind("safety")
    stops = [e for e in ev if e.get("mode_to") == "STOP"]
    reasons: Counter = Counter()
    for e in stops:
        for r in e.get("reasons") or []:
            reasons[_reason_key(r)] += 1
    ho = [e for e in ev if e.get("mode_to") == "HANDOVER"]
    return {"stops": len(stops), "stop_categories": dict(Counter(e.get("category") for e in stops)),
            "top_stop_reasons": reasons.most_common(4), "handovers": len(ho),
            "handover_reasons": [e.get("reasons") for e in ho][:3],
            "operator_s": [round(float(o.get("duration", 0.0)), 1) for o in ep.by_kind("operator")]}


def _tick_context(ep: EpisodeRecord, key: str) -> dict | None:
    """State of the robot at the first tick where the boolean trajectory array ``key`` is set."""
    tr = ep.traj()
    if not tr or key not in tr or not tr[key].any():
        return None
    i = int(np.argmax(tr[key]))
    j = max(0, i - 1)
    ctx = {"t": float(tr["t"][i]), "mode": MODE_NAME.get(int(tr["mode"][j]), "?"),
           "cmd_v": float(tr["action_v"][j]),
           "loc_error_m": float(np.linalg.norm(tr["pose_est"][j, :2] - tr["pose_true"][j, :2])),
           "fault_active": bool(tr["fault_active"][j])}
    if "agents_true" in tr and tr["agents_true"].size:
        d = np.linalg.norm(tr["agents_true"][j] - tr["pose_true"][j, None, :2], axis=1) - ROBOT_R - HUMAN_R
        ctx["true_human_clearance_m"] = float(d.min())
        ctx["est_human_clearance_m"] = float(tr["human_clearance_est"][j])
        k = int(np.argmin(d))
        agents = ep.meta.get("agents") or []
        ctx["nearest_person"] = agents[k] if k < len(agents) else str(k)
    return ctx


def _min_clearance_context(ep: EpisodeRecord) -> dict | None:
    tr = ep.traj()
    if not tr or "agents_true" not in tr or not tr["agents_true"].size:
        return None
    d = np.linalg.norm(tr["agents_true"] - tr["pose_true"][:, None, :2], axis=2) - ROBOT_R - HUMAN_R
    i = int(np.argmin(d.min(axis=1)))
    agents = ep.meta.get("agents") or []
    k = int(np.argmin(d[i]))
    return {"t": float(tr["t"][i]), "true_clearance_m": float(d[i].min()),
            "est_clearance_m": float(tr["human_clearance_est"][i]), "mode": MODE_NAME.get(int(tr["mode"][i]), "?"),
            "cmd_v": float(tr["action_v"][i]), "nearest_person": agents[k] if k < len(agents) else str(k)}


# ---------------------------------------------------------------------------------------------------------------------
# detectors: (row, metrics, ctx) -> (severity, subtype, evidence) | None
Detection = tuple[float, str, dict]


def _d_human_contact(r, m, c) -> Detection | None:
    n = _num(m, "collisions_agent")
    if not n > 0:
        return None
    mc = _num(m, "min_human_clearance_m")
    return 100.0 * n + 10 * max(0.0, -mc if math.isfinite(mc) else 0.0), "contact", \
        {"collisions_agent": n, "min_human_clearance_m": mc}


def _d_static(r, m, c) -> Detection | None:
    n = _num(m, "collisions_static")
    if not n > 0:
        return None
    sub = "static"
    if factor_of(r) == "CF-B" and cf_value_of(r) == "real_obstacle" and _num(m, "cfb_min_dist_to_aisle_point") < 1.0:
        sub = "hidden_obstacle"
    return 10.0 * n, sub, {"collisions_static": n, "distance_travelled_m": _num(m, "distance_travelled_m")}


def _d_sterile(r, m, c) -> Detection | None:
    b, mg = _num(m, "sterile_breach_s"), _num(m, "keepout_margin_violation_s")
    if b > 0:
        return 100.0 + 10 * b + (mg if math.isfinite(mg) else 0.0), "breach", \
            {"sterile_breach_s": b, "keepout_margin_violation_s": mg}
    if mg >= c["thr"].sterile_margin_s:
        return mg, "margin", {"sterile_breach_s": b, "keepout_margin_violation_s": mg}
    return None


def _d_near(r, m, c) -> Detection | None:
    n = _num(m, "near_collision_events")
    if not n > 0:
        return None
    mc = _num(m, "min_human_clearance_m")
    return n + 10 * max(0.0, 0.3 - mc if math.isfinite(mc) else 0.0), "near", \
        {"near_collision_events": n, "near_collision_rate_per_min": _num(m, "near_collision_rate_per_min"),
         "min_human_clearance_m": mc}


def _d_wrong(r, m, c) -> Detection | None:
    w = _num(m, "wrong_assertion_rate")
    if not w > 0:
        return None
    decided = _num(m, "claims_answered") * (1 - _num(m, "abstention_rate"))
    n = round(w * decided) if math.isfinite(decided) else float("nan")
    return (float(n) if math.isfinite(n) and n > 0 else w), "wrong", \
        {"wrong_assertion_rate": w, "wrong_verdicts_est": n, "decision_accuracy": _num(m, "decision_accuracy"),
         "claims_answered": _num(m, "claims_answered")}


def _d_retained(r, m, c) -> Detection | None:
    if not _num(m, "cfa_retained_missed") > 0:
        return None
    return 1000.0, "retained_sponge_verified", {"cfa_verdicts": m.get("cfa_verdicts"), "cfa_retained_missed": 1.0}


def _d_cf(r, m, c) -> Detection | None:
    f, v = factor_of(r), cf_value_of(r)
    observed_b = not _has(m, "cfb_observed") or _num(m, "cfb_observed") > 0
    observed_d = not _has(m, "cfd_observable") or _num(m, "cfd_observable") > 0
    if f == "CF-B" and observed_b and _has(m, "cfb_correct") and _num(m, "cfb_correct") == 0:
        p, dmin = _num(m, "cfb_belief_occupied"), _num(m, "cfb_min_dist_to_aisle_point")
        hit = _num(m, "cfb_traversed") > 0 or (_num(m, "cfb_collision") > 0 and dmin < 1.0)
        unsafe = v == "real_obstacle" and hit
        sev = 100.0 if unsafe else (5.0 + 10 * abs(p - 0.5) if math.isfinite(p) else 5.0)
        sub = "CF-B real obstacle believed free" if v == "real_obstacle" else "CF-B ghost believed occupied"
        return sev, sub, {"hc_value": v, "cfb_belief_occupied": p,
                          "cfb_belief_ambiguous": _num(m, "cfb_belief_ambiguous"),
                          "cfb_traversed": _num(m, "cfb_traversed"), "cfb_collision": _num(m, "cfb_collision"),
                          "cfb_min_dist_to_aisle_point": dmin}
    if f == "CF-C" and _num(m, "cfc_clamp_wrongly_verified") > 0:
        return 100.0, "CF-C clamp wrongly verified", {"hc_value": v, "cfc_final_map_slot": m.get("cfc_final_map_slot"),
                                                       "cfc_final_map_correct": _num(m, "cfc_final_map_correct")}
    if f == "CF-D" and observed_d and _has(m, "cfd_correct") and _num(m, "cfd_correct") == 0:
        dg = m.get("cfd_diagnosis", "none")
        want = "loc_drift" if v == "loc_drift" else "map_change"
        sub = "CF-D no diagnosis" if dg in (None, "none") else f"CF-D {want} diagnosed as {dg}"
        return (5.0 if dg in (None, "none") else 20.0), sub, \
            {"hc_value": v, "cfd_diagnosis": dg, "expected": want, "loc_error_max_m": _num(m, "loc_error_max_m")}
    return None


def _d_stops(r, m, c) -> Detection | None:
    s, u = _num(m, "safe_stop_rate_per_min"), _num(m, "uncertainty_safe_stop_rate_per_min")
    t = c["thr"]
    if not (s >= t.stop_rate_per_min or u >= t.unc_stop_rate_per_min):
        return None
    sub = "uncertainty" if u >= t.unc_stop_rate_per_min else "stops"
    return (s if math.isfinite(s) else 0.0) + 2 * (u if math.isfinite(u) else 0.0), sub, \
        {"safe_stop_rate_per_min": s, "uncertainty_safe_stop_rate_per_min": u, "task_delay_s": _num(m, "task_delay_s")}


def _d_handover(r, m, c) -> Detection | None:
    n = _num(m, "handover_requests")
    if not n >= 1:
        return None
    return max(_num(m, "intervention_cost"), n), "handover", \
        {"handover_requests": n, "intervention_cost": _num(m, "intervention_cost")}


def _d_starve(r, m, c) -> Detection | None:
    a, h = _num(m, "abstention_rate"), _num(m, "handoff_success")
    t = c["thr"]
    if not (a >= t.starvation_abstention and h <= t.starvation_handoff):
        return None
    return a - h, "starvation", {"abstention_rate": a, "handoff_success": h,
                                 "claims_answered": _num(m, "claims_answered"), "claims_total": _num(m, "claims_total"),
                                 "distance_travelled_m": _num(m, "distance_travelled_m")}


def _d_loc(r, m, c) -> Detection | None:
    e = _num(m, "loc_error_max_m")
    if not e > c["thr"].loc_error_m:
        return None
    return e, "loc_loss", {"loc_error_max_m": e, "loc_error_mean_m": _num(m, "loc_error_mean_m")}


def _d_energy(r, m, c) -> Detection | None:
    t = c["thr"]
    res, used = _num(m, "energy_reserve_frac"), _num(m, "energy_used_wh")
    if res < t.reserve_frac:
        return 100.0 * (t.reserve_frac - res) + 10.0, "low_reserve", \
            {"energy_reserve_frac": res, "energy_used_wh": used}
    med = c["energy_median"].get(_energy_group(r))
    if med and med > 0 and used > t.energy_outlier * med:
        return used / med, "energy_outlier", {"energy_used_wh": used, "group_median_wh": med, "ratio": used / med,
                                               "distance_travelled_m": _num(m, "distance_travelled_m")}
    return None


def _energy_group(r: dict) -> tuple:
    return (stratum(r), r.get("policy"), r.get("variant"), r.get("duration"))


DETECTORS: dict[str, Callable] = {
    "human_contact": _d_human_contact, "static_collision": _d_static, "sterile_breach": _d_sterile,
    "near_collision": _d_near, "wrong_assertion": _d_wrong, "retained_item_signed_off": _d_retained,
    "cf_misdiagnosis": _d_cf, "safe_stop_cascade": _d_stops, "handover": _d_handover,
    "mission_starvation": _d_starve, "localization_loss": _d_loc, "energy_depletion": _d_energy,
}


# ---------------------------------------------------------------------------------------------------------------------
# root-cause hints
def _family_hint(r: dict) -> list[str]:
    fam = r.get("family")
    txt = {"sensor_dropout": "family injects sensor dropout windows",
           "timestamp_skew": "family injects clock skew between sensors",
           "map_corruption": "family corrupts the prior map (shifted / phantom / dropped objects)",
           "loc_drift": "family injects odometry drift with landmark dropouts",
           "adversarial_occlusion": "family places adversarial occluders in front of key slots",
           "reflective": "family raises specular gain (polished steel, wet floor, glare)",
           "rare_geometry": "family stages rare geometry (fallen IV pole, lowered boom, trailing drape, tipped cart)"}
    return [txt[fam]] if fam in txt else []


def _hints(cat: str, sub: str, r: dict, m: dict, ep: EpisodeRecord | None) -> list[str]:
    h: list[str] = []
    f, v = factor_of(r), cf_value_of(r)
    variant = str(r.get("variant", ""))
    if "no_safety_supervisor" in variant and cat in ("human_contact", "static_collision", "near_collision",
                                                     "sterile_breach"):
        h.append("safety supervisor disabled in this variant")
    if cat in ("human_contact", "static_collision", "sterile_breach"):
        key = {"human_contact": "collision_agent", "static_collision": "collision_static",
               "sterile_breach": "sterile_breach"}[cat]
        ctx = _tick_context(ep, key) if ep else None
        if ctx:
            h.append(f"first event at t={ctx['t']:.1f}s: mode={ctx['mode']}, cmd v={ctx['cmd_v']:.2f} m/s, "
                     f"pose error={ctx['loc_error_m']:.2f} m"
                     + (", sensor fault active" if ctx["fault_active"] else ""))
            if cat == "human_contact":
                if abs(ctx["cmd_v"]) < 0.05:
                    h.append(f"robot (nearly) stationary: {ctx.get('nearest_person')} walked into it "
                             "(staff model / yield behaviour, not a planning failure)")
                elif ctx.get("est_human_clearance_m", 0) > ctx.get("true_human_clearance_m", 0) + 0.3:
                    h.append(f"tracker over-estimated clearance ({ctx['est_human_clearance_m']:.2f} m est vs "
                             f"{ctx['true_human_clearance_m']:.2f} m true): person occluded or not yet confirmed")
                if ctx["mode"] == "NOMINAL":
                    h.append("supervisor was NOMINAL at contact: the risk was not predicted")
            if ctx["loc_error_m"] > 0.3:
                h.append("localisation error > 0.3 m at the event: the robot believed it was elsewhere")
    if cat == "static_collision":
        if sub == "hidden_obstacle":
            h.append("collided at the CF-B aisle point: the real obstacle was explained away as a specular ghost")
        elif f == "CF-B" and _has(m, "cfb_min_dist_to_aisle_point"):
            h.append(f"robot stayed {_num(m, 'cfb_min_dist_to_aisle_point'):.1f} m from the CF-B aisle point: "
                     "the collision is not with the hidden obstacle")
        if f == "CF-D" and v == "cart_moved":
            h.append("cart moved since the survey: the prior map still shows the old pose")
        h += _family_hint(r) if r.get("family") in ("map_corruption", "rare_geometry", "reflective") else []
    if cat == "near_collision" and ep:
        mc = _min_clearance_context(ep)
        if mc:
            h.append(f"closest approach t={mc['t']:.1f}s to {mc['nearest_person']}: "
                     f"true {mc['true_clearance_m']:.2f} m, "
                     f"estimated {mc['est_clearance_m']:.2f} m, mode={mc['mode']}, cmd v={mc['cmd_v']:.2f} m/s")
            if abs(mc["cmd_v"]) < 0.05:
                h.append("robot was (nearly) stationary at the closest approach: staff passed close to it")
    if cat == "sterile_breach" and sub == "margin":
        h.append("inside the 0.3 m keep-out margin but not the field: planner margin or pose uncertainty")
    if cat == "wrong_assertion":
        if _num(m, "calibration_ece_under_fault") > _num(m, "calibration_ece_nominal") + 0.05:
            h.append("posteriors less calibrated under sensor faults (ece_under_fault > ece_nominal + 0.05)")
        if "no_abstention" in variant:
            h.append("abstention disabled in this variant (tau_verify = tau_refute = 0.5)")
        if not ep:
            h.append("no exported episode: re-run the reproduce command to get the wrong verdicts + provenance")
    if cat == "retained_item_signed_off":
        h.append("a cotton sponge under the drape is radar- and acoustically transparent: only direct camera "
                 "evidence of the kick bucket may VERIFY; check fixed-lag smoothing and log-driven priors")
    if cat == "cf_misdiagnosis":
        if f == "CF-B":
            h.append("ghost reasoning explained away a real return" if v == "real_obstacle" else
                     "multipath ghost not recognised: aisle kept blocked (efficiency loss)")
            if _num(m, "cfb_min_dist_to_aisle_point") > 2.0:
                dmin = _num(m, "cfb_min_dist_to_aisle_point")
                h.append(f"robot never approached the aisle point (min {dmin:.1f} m): "
                         "the belief is the prior + far-field returns only")
        if f == "CF-C":
            h.append("clamp claim VERIFIED although the clamp left the mayo stand without a log entry")
        if f == "CF-D":
            dg = m.get("cfd_diagnosis", "none")
            if dg in (None, "none"):
                h.append("no scan/map mismatch diagnosed: the drift / moved cart never produced enough residual "
                         "(route coverage, scan matching, episode too short?)")
            else:
                h.append("drift vs map-change confusion: scan-to-map residual attributed to the wrong cause")
    if cat in ("safe_stop_cascade", "handover", "mission_starvation") and ep:
        s = safety_summary(ep)
        if s["stops"]:
            cats = ", ".join(f"{k}={n}" for k, n in s["stop_categories"].items())
            rs = "; ".join(f"{k} x{n}" for k, n in s["top_stop_reasons"])
            h.append(f"{s['stops']} STOP transitions ({cats}); top reasons: {rs}")
        if cat == "handover" and s["handover_reasons"]:
            h.append(f"handover triggers: {s['handover_reasons']}; operator time {s['operator_s']} s")
    if cat == "handover":
        n, cost = _num(m, "handover_requests"), _num(m, "intervention_cost")
        if math.isfinite(n) and math.isfinite(cost):
            # intervention_cost = handovers + operator_time / 60 (medortrace.eval.metrics)
            h.append(f"{n:.0f} STOP(s) persisted beyond Envelope.t_handover_after; operator time "
                     f"~{max(0.0, cost - n) * 60:.0f} s")
        if _num(m, "loc_error_max_m") > 0.3:
            h.append("large localisation error: handover used for operator re-localisation")
    if cat == "safe_stop_cascade":
        if _num(m, "uncertainty_safe_stop_rate_per_min") > 0:
            h.append("uncertainty stops (localisation / stale lidar / path entropy): check faults and NBV routing")
        else:
            h.append("stops are collision / proximity driven: crowded workspace or conservative MPC envelope")
    if cat == "mission_starvation":
        d = _num(m, "distance_travelled_m")
        T = float(r.get("duration") or DEFAULT_DURATION_S)
        if math.isfinite(d) and d / T < 0.05:
            h.append(f"robot barely moved ({d:.1f} m in {T:.0f} s): blocked, stopped or waiting for the operator")
        if ep:
            rs = Counter(e.get("reason") for e in ep.by_kind("verdict") if e.get("verdict") == "ABSTAIN")
            if rs:
                h.append("abstention reasons: " + "; ".join(f"{k} x{n}" for k, n in rs.most_common(3)))
        if _num(m, "safe_stop_rate_per_min") >= 2.0:
            h.append("frequent safety stops consume the time budget of hand-off verification")
    if cat == "localization_loss":
        if f == "CF-D" and v == "loc_drift":
            h.append("CF-D odometry drift arm: expected to be corrected by landmark / scan-to-map relocalisation")
        if ep:
            dg = Counter(e.get("cause") for e in ep.by_kind("diagnosis"))
            if dg:
                h.append(f"change diagnoser output: {dict(dg)}")
    if cat == "energy_depletion":
        if sub == "low_reserve":
            h.append("battery start fraction is sampled in [0.55, 0.95]: check start charge vs. consumption")
        else:
            h.append("energy well above its group median: oscillating motion, long detours or acoustic probing")
    if cat not in ("static_collision", "energy_depletion"):
        h += _family_hint(r)
    return h


# ---------------------------------------------------------------------------------------------------------------------
def build_atlas(rows: list[dict], episodes: EpisodeIndex | None = None, k: int = 5,
                thresholds: AtlasThresholds | None = None, registry: str = "configs/seed_registry.yaml",
                variant_specs: dict[str, dict] | None = None, repro_dir: str = "runs/repro",
                backend: str = "lite") -> dict:
    """Classify every row, count per category and pick the worst-k exemplars (see module docstring)."""
    thr = thresholds or AtlasThresholds()
    episodes = episodes or EpisodeIndex()
    rows, n_dup = dedupe(rows)
    ok = [r for r in rows if not r.get("error")]
    multi_variant = len({r.get("variant") for r in rows}) > 1
    pol_key = lambda r: f"{r.get('policy')}/{r.get('variant')}" if multi_variant else str(r.get("policy"))
    groups = defaultdict(list)
    for r in ok:
        groups[_energy_group(r)].append(_num(r["metrics"], "energy_used_wh"))
    ctx = {"thr": thr, "energy_median": {g: float(np.nanmedian(v)) for g, v in groups.items()
                                         if np.isfinite(v).sum() >= 3}}
    n_fam = Counter(stratum(r) for r in rows)
    n_pol = Counter(pol_key(r) for r in rows)
    hits: dict[str, list[tuple[float, str, dict, dict]]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            last = [ln for ln in str(r["error"]).strip().splitlines() if ln.strip()]
            hits["episode_error"].append((1.0, "exception", {"error": last[-1] if last else "?"}, r))
            continue
        m = r.get("metrics") or {}
        for cat, det in DETECTORS.items():
            d = det(r, m, ctx)
            if d is not None:
                hits[cat].append((float(d[0]), d[1], d[2], r))
    cats = {}
    for cat in CATEGORIES:
        hs = hits.get(cat, [])
        by_f = Counter(stratum(r) for _, _, _, r in hs)
        by_p = Counter(pol_key(r) for _, _, _, r in hs)
        worst = sorted(hs, key=lambda x: (-x[0], x[3].get("scenario_id", "")))[:k]
        cases = []
        for sev, sub, ev, r in worst:
            m = r.get("metrics") or {}
            ep = episodes.get(r)
            vspec = (variant_specs or {}).get(r.get("variant"))
            case = FailureCase(cat, sub, sev, r.get("scenario_id"), int(r.get("seed", -1)), r.get("family"),
                               r.get("policy"), r.get("variant"), r.get("pair_id"), r.get("hidden_cause"), ev,
                               reproduce=reproduce_command(r, registry, repro_dir, vspec, backend),
                               reproduce_exact=resolve_variant_spec(r, vspec)[1] != "unknown")
            if cat != "episode_error":
                case.hints = _hints(cat, sub, r, m, ep)
            if ep:
                case.episode_dir = ep.path
                case.fault_labels = fault_labels(ep.meta)
                if cat in ("wrong_assertion", "retained_item_signed_off", "cf_misdiagnosis"):
                    wv = wrong_verdicts(ep)
                    if cat == "retained_item_signed_off":
                        sp = (ep.meta.get("hidden_notes") or {}).get("retained_sponge")
                        wv = [w for w in wv if w["item"] == sp] or wv
                    elif cat == "cf_misdiagnosis":
                        wv = [w for w in wv if w["item"] == "clamp_1"] if factor_of(r) == "CF-C" else []
                    case.wrong_verdicts = wv
                    if ep.prov is not None and wv:
                        case.evidence["provenance_chain_recomputed"] = verify_prov_chain(ep.prov)
            cases.append(case)
        cats[cat] = {
            "description": CATEGORIES[cat], "count": len(hs), "rate": len(hs) / max(len(rows), 1),
            "subtypes": dict(Counter(sub for _, sub, _, _ in hs)),
            "by_family": {f: {"count": by_f[f], "n": n_fam[f], "rate": by_f[f] / n_fam[f]} for f in sorted(by_f)},
            "by_policy": {p: {"count": by_p[p], "n": n_pol[p], "rate": by_p[p] / n_pol[p]} for p in sorted(by_p)},
            "worst": [asdict(c) for c in cases],
        }
    return {
        "meta": {"n_rows": len(rows), "n_errors": len(rows) - len(ok), "duplicates_dropped": n_dup,
                 "families": dict(sorted(n_fam.items())), "policies": dict(sorted(n_pol.items())),
                 "episode_dirs_indexed": len(episodes), "k": k, "registry": registry,
                 "episodes_matched": sum(1 for c in cats.values() for w in c["worst"] if w["episode_dir"]),
                 "variants_overrides_unknown": sorted({str(r.get("variant")) for r in rows if resolve_variant_spec(
                     r, (variant_specs or {}).get(r.get("variant")))[1] == "unknown"})},
        "thresholds": asdict(thr),
        "categories": cats,
    }


# ---------------------------------------------------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "n/a" if not math.isfinite(v) else f"{v:.3g}"
    return str(v)


def _verdict_md(w: dict, max_evidence: int = 3) -> list[str]:
    out = [f"- **{w['verdict']}** `{w['claim_id']}` ({w['claim_kind']}): {w['item']} in `{w['claimed_slot']}` "
           f"at t_ref={w['t_ref']:.1f}s, decided t={w['t_verdict']:.1f}s, p={w['posterior']:.3f}; truth at t_ref "
           f"`{w['truth_slot_at_t_ref']}`, robot MAP `{w['robot_map_slot']}`, direct={w['direct']}; reason: "
           f"{w['reason']}"]
    ex = w.get("explanation")
    if ex:
        for lab in ("supporting", "contradicting"):
            es = ex.get(lab) or []
            if es:
                out.append(f"  - {lab}: " + "; ".join(_evidence_brief(e) for e in es[:max_evidence]))
            else:
                out.append(f"  - {lab}: none recorded")
    for hh in w.get("hints", []):
        out.append(f"  - hint: {hh}")
    return out


def atlas_markdown(atlas: dict, title: str = "MED-OR-TRACE failure atlas") -> str:
    meta = atlas["meta"]
    md = [f"# {title}", "",
          f"{meta['n_rows']} episodes ({meta['n_errors']} errors, "
          f"{meta['duplicates_dropped']} duplicate rows dropped); "
          f"families: {', '.join(f'{k} ({v})' for k, v in meta['families'].items())}; "
          f"policies: {', '.join(f'{k} ({v})' for k, v in meta['policies'].items())}.",
          f"Exported episode dirs indexed: {meta['episode_dirs_indexed']} "
          f"(exemplars with episode data: {meta['episodes_matched']}).", "",
          "Categories overlap: one episode can appear in several.  Hints are heuristics, not diagnoses.", ""]
    unk = meta.get("variants_overrides_unknown") or []
    if unk:
        md += [f"**Warning:** the overrides of variant(s) {', '.join(unk)} are not recorded in the results (rows "
               "predate `variant_spec`) and no `--variant-specs` entry was given: their reproduce commands run the "
               "base config and are marked INEXACT.", ""]
    md += ["## Overview", "", "| category | episodes | rate | subtypes | description |", "|---|---|---|---|---|"]
    for cat, c in atlas["categories"].items():
        subs = ", ".join(f"{k}: {v}" for k, v in c["subtypes"].items()) or "-"
        md.append(f"| {cat} | {c['count']} | {c['rate']:.2f} | {subs} | {c['description']} |")
    md += ["", "Thresholds: " + ", ".join(f"{k}={v}" for k, v in atlas["thresholds"].items()), ""]
    for cat, c in atlas["categories"].items():
        if not c["count"]:
            continue
        md += [f"## {cat}", "", f"{c['description']}.  {c['count']} episode{'s' if c['count'] != 1 else ''}.", ""]
        md.append("By family: " + ", ".join(f"{f} {x['count']}/{x['n']}" for f, x in c["by_family"].items()) + "  ")
        md.append("By policy: " + ", ".join(f"{p} {x['count']}/{x['n']}" for p, x in c["by_policy"].items()))
        md += ["", f"### Worst {len(c['worst'])}", ""]
        for i, w in enumerate(c["worst"], 1):
            hc = w.get("hidden_cause") or {}
            hcs = "" if hc.get("factor") in (None, "none") else f", hidden cause {hc.get('factor')}={hc.get('value')}"
            md.append(f"{i}. `{w['scenario_id']}` seed {w['seed']}, policy {w['policy']}, variant {w['variant']}"
                      f"{hcs}: **{w['subtype']}** (severity {w['severity']:.3g})")
            md.append("   - evidence: " + ", ".join(f"{k}={_fmt(v)}" for k, v in w["evidence"].items()))
            if w["fault_labels"]:
                md.append("   - fault labels: " + "; ".join(w["fault_labels"]))
            for hh in w["hints"]:
                md.append(f"   - hint: {hh}")
            if w["episode_dir"]:
                md.append(f"   - episode: `{w['episode_dir']}`")
            if w["wrong_verdicts"]:
                md.append(f"   - wrong verdicts ({len(w['wrong_verdicts'])}):")
                for wv in w["wrong_verdicts"][:3]:
                    md += ["     " + ln for ln in _verdict_md(wv)]
            exact = w.get("reproduce_exact", True)
            md.append(f"   - reproduce{'' if exact else ' (INEXACT: variant overrides unknown)'}: `{w['reproduce']}`")
        md.append("")
    return "\n".join(md) + "\n"
