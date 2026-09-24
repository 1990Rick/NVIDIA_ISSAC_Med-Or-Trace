#!/usr/bin/env python3
"""Sim-to-real sensitivity: how much do the benchmark metrics and conclusions depend on simulator parameters?

One-at-a-time (OAT) perturbations of *simulator* parameters, applied through
``run_episode(sim_overrides=...)`` (batch variants ``{"sim": {...}}``): the world changes, the autonomy
stack keeps its nominal models, because those models are what the robot *believes* - a real sensor
that differs from them is exactly the sim-to-real gap being probed.

    python scripts/sim_to_real_sensitivity.py --n-seeds 3 --workers 3 --out runs/s2r
    python scripts/sim_to_real_sensitivity.py --families nominal --n-seeds 1 --params camera_pd0 haze \\
        --levels 0.25 --duration 20 --workers 2 --out runs/s2r_smoke
    python scripts/sim_to_real_sensitivity.py --analyze-only --out runs/s2r      # re-analyse results.jsonl

Design
  * fixed registry seeds: the first ``--n-seeds`` entries of each family in ``--split`` (deterministic
    registry order) or explicit ``--scenarios``.  Every perturbation runs on the same seeds, for the
    active policy and a baseline (common random numbers: differences are paired);
  * parameter p (see ``PARAMS``) is scaled by ``1 + level`` for each ``--levels`` entry and clipped to
    its valid range; sampling ranges ([lo, hi], e.g. haze) are scaled at both ends and p is the range
    midpoint.  The *realised* relative change dp/p (after clipping) is recorded in ``design.json``;
  * normalised sensitivity (elasticity) of metric M, active policy::

        S = (dM / M0) / (dp / p0),   dM = mean_seeds(M_perturbed) - mean_seeds(M_nominal)

    and, when both a negative and a positive level exist, the central estimate
    ``S_c = ((M+ - M-) / M0) / ((p+ - p-) / p0)``.  95% CIs bootstrap over seeds (paired).  Metrics
    with |M0| < 1e-9 (e.g. zero collisions) have no elasticity; the absolute delta is reported;
  * conclusion flips: per metric, the paired delta active - baseline under nominal vs. perturbed
    simulation; a flip is flagged when the direction-adjusted sign changes ("active better" <->
    "baseline better") and is *significant* when both bootstrap CIs exclude 0 (needs >= 3 seeds).

Writes ``results.jsonl`` (run_batch rows), ``design.json``, ``sensitivity.json`` and ``sensitivity.md``.
Cost: seeds x policies x (1 + params x levels) episodes - print it with ``--dry-run``.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from medortrace.common.config import get
from medortrace.eval.aggregate import DIRECTION, PRIMARY, SECONDARY
from medortrace.sim.sensors_lite import CameraConfig, LidarConfig

EPS = 1e-9


@dataclass
class SimParam:
    name: str
    path: str                         # dotted path in the scenario config
    default: float | list             # used when the scenario config does not set it (lite simulator default)
    lo: float = 0.0                   # valid range (values are clipped)
    hi: float = math.inf
    is_range: bool = False            # a [lo, hi] sampling range: both ends scaled, p = midpoint
    doc: str = ""


PARAMS: dict[str, SimParam] = {p.name: p for p in [
    SimParam("camera_pd0", "sensors.camera.pd0", CameraConfig().pd0, 0.0, 1.0,
             doc="camera detection probability at full visibility"),
    SimParam("lidar_range_sigma", "sensors.lidar.range_sigma", LidarConfig().range_sigma,
             doc="lidar range noise std [m]"),
    SimParam("specular_gain", "faults.specular_gain", 1.0, doc="multipath / specular ghost gain"),
    SimParam("glare_gain", "nuisance.glare_gain", [0.8, 1.3], is_range=True, doc="surgical-light glare gain range"),
    SimParam("agents_A_robot", "agents.A_robot", 4.0, doc="staff social-force repulsion from the robot"),
    SimParam("agents_yield_distance", "agents.yield_distance", 1.3, doc="distance at which staff yield [m]"),
    SimParam("robot_max_acc", "robot.max_acc", 0.6, doc="drive acceleration limit [m/s^2]"),
    SimParam("detector_logit_scale", "sensors.camera.logit_scale", CameraConfig().logit_scale,
             doc="detector class-margin scale (confusion of the camera front-end surrogate)"),
    SimParam("workflow_p_log_missing", "workflow.p_log_missing", 0.05, 0.0, 1.0,
             doc="probability a workflow event is not logged"),
    SimParam("haze", "nuisance.haze", [0.0, 0.15], 0.0, 0.9, is_range=True, doc="haze / smoke range"),
]}


def _set_dotted(d: dict, path: str, value) -> dict:
    cur = d
    parts = path.split(".")
    for k in parts[:-1]:
        cur = cur.setdefault(k, {})
    cur[parts[-1]] = value
    return d


def _scalar(v) -> float:
    return float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)


def perturb(param: SimParam, cfg: dict, level: float) -> tuple[dict, float, float]:
    """-> (sim override, p0, p1) for scaling ``param`` by ``1 + level`` in scenario ``cfg``."""
    base = get(cfg, param.path, param.default)
    if param.is_range:
        new = [float(np.clip(x * (1 + level), param.lo, param.hi)) for x in base]
    else:
        new = float(np.clip(float(base) * (1 + level), param.lo, param.hi))
    return _set_dotted({}, param.path, new), _scalar(base), _scalar(new)


def _param_doc(p: SimParam) -> dict:
    """JSON-safe description (strict JSON has no Infinity)."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in asdict(p).items()}


def variant_name(param: str, level: float) -> str:
    return f"{param}{level:+.2f}"


# ---------------------------------------------------------------------------------------------------------------------
# analysis
def _boot_idx(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def _ci(samples: np.ndarray) -> list[float]:
    s = samples[np.isfinite(samples)]
    if not len(s):
        return [float("nan"), float("nan")]
    return [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))]


def elasticity(m0: np.ndarray, m1: np.ndarray, rel: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    """S = (mean(m1) - mean(m0)) / mean(m0) / mean(rel), paired bootstrap over seeds."""
    ok = np.isfinite(m0) & np.isfinite(m1) & np.isfinite(rel)
    m0, m1, rel = m0[ok], m1[ok], rel[ok]
    n = len(m0)
    if n == 0:
        return {"n": 0, "S": None, "ci95": [None, None], "M0": None, "dM": None}
    dM = float(m1.mean() - m0.mean())
    out = {"n": n, "M0": float(m0.mean()), "dM": dM, "rel_dp": float(rel.mean())}
    if abs(m0.mean()) < EPS or abs(rel.mean()) < EPS:
        return {**out, "S": None, "ci95": [None, None]}
    idx = _boot_idx(n, B, seed)
    b0, b1, br = m0[idx].mean(1), m1[idx].mean(1), rel[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = np.where((np.abs(b0) > EPS) & (np.abs(br) > EPS), (b1 - b0) / b0 / br, np.nan)
    return {**out, "S": float(dM / m0.mean() / rel.mean()), "ci95": _ci(sb)}


def central_elasticity(m0: np.ndarray, mm: np.ndarray, mp: np.ndarray, rm: np.ndarray, rp: np.ndarray,
                       B: int = 2000, seed: int = 0) -> dict:
    ok = np.isfinite(m0) & np.isfinite(mm) & np.isfinite(mp) & np.isfinite(rm) & np.isfinite(rp)
    m0, mm, mp, rm, rp = m0[ok], mm[ok], mp[ok], rm[ok], rp[ok]
    n = len(m0)
    if n == 0 or abs(m0.mean()) < EPS or abs(rp.mean() - rm.mean()) < EPS:
        return {"n": n, "S": None, "ci95": [None, None]}
    idx = _boot_idx(n, B, seed)
    b0 = m0[idx].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mp[idx].mean(1) - mm[idx].mean(1)) / b0 / (rp[idx].mean(1) - rm[idx].mean(1))
    sb[np.abs(b0) < EPS] = np.nan
    return {"n": n, "S": float((mp.mean() - mm.mean()) / m0.mean() / (rp.mean() - rm.mean())), "ci95": _ci(sb)}


def paired(d: np.ndarray, B: int = 2000, seed: int = 0) -> dict:
    d = d[np.isfinite(d)]
    if not len(d):
        return {"n": 0, "delta": None, "ci95": [None, None]}
    bs = d[_boot_idx(len(d), B, seed)].mean(1)
    return {"n": int(len(d)), "delta": float(d.mean()), "ci95": _ci(bs)}


def _excludes_zero(ci: list) -> bool:
    return all(x is not None and math.isfinite(x) for x in ci) and (ci[0] > 0 or ci[1] < 0)


def analyze(rows: list[dict], design: dict, active: str = "active", baseline: str | None = "fixed_route",
            metrics: list[str] | None = None, B: int = 2000) -> dict:
    metrics = metrics or PRIMARY + SECONDARY
    rows_by: dict[tuple, dict] = {}
    for r in rows:                      # last row wins (results.jsonl is appended to)
        if not r.get("error"):
            rows_by[(r["scenario_id"], r["policy"], r["variant"])] = r["metrics"]
    seeds = sorted({s for (s, p, v) in rows_by if p == active and v == "nominal"})
    dz = design.get("entries", {})

    def M(s: str, p: str, v: str, m: str) -> float:
        x = rows_by.get((s, p, v), {}).get(m)
        return float(x) if x is not None else float("nan")
    res: dict = {"seeds": seeds, "active": active, "baseline": baseline, "params": {}, "flips": []}
    nominal_cmp = {}
    if baseline:
        for m in metrics:
            nominal_cmp[m] = paired(np.array([M(s, active, "nominal", m) - M(s, baseline, "nominal", m)
                                              for s in seeds]), B)
        res["nominal_comparison"] = nominal_cmp
    for pname, levels in design.get("levels_by_param", {}).items():
        pres: dict = {"path": PARAMS[pname].path if pname in PARAMS else "?", "levels": {}, "central": {}}
        for lev in levels:
            v = variant_name(pname, lev)
            ss = [s for s in seeds if (s, active, v) in rows_by]
            rel = np.array([dz.get(s, {}).get(v, {}).get("rel", np.nan) for s in ss], dtype=float)
            lres = {"variant": v, "rel_dp": float(np.nanmean(rel)) if np.isfinite(rel).any() else None,
                    "p0": sorted({dz.get(s, {}).get(v, {}).get("p0") for s in ss} - {None}),
                    "p1": sorted({dz.get(s, {}).get(v, {}).get("p1") for s in ss} - {None}), "metrics": {}}
            for m in metrics:
                e = elasticity(np.array([M(s, active, "nominal", m) for s in ss]),
                               np.array([M(s, active, v, m) for s in ss]), rel, B)
                if baseline:
                    pc = paired(np.array([M(s, active, v, m) - M(s, baseline, v, m) for s in ss]), B)
                    e["active_minus_baseline"] = pc
                    nc = nominal_cmp[m]
                    if pc["delta"] is not None and nc["delta"] is not None:
                        dirn = DIRECTION.get(m, 1)
                        a, b = dirn * nc["delta"], dirn * pc["delta"]
                        if a * b < 0:
                            sig = nc["n"] >= 3 and pc["n"] >= 3 and _excludes_zero(nc["ci95"]) and \
                                _excludes_zero(pc["ci95"])
                            res["flips"].append({"param": pname, "level": lev, "metric": m,
                                                 "nominal_delta": nc["delta"], "perturbed_delta": pc["delta"],
                                                 "nominal_ci95": nc["ci95"], "perturbed_ci95": pc["ci95"],
                                                 "active_better_nominal": a > 0, "significant": bool(sig),
                                                 "n": min(nc["n"], pc["n"])})
                lres["metrics"][m] = e
            pres["levels"][f"{lev:+.2f}"] = lres
        neg = [x for x in levels if x < 0]
        pos = [x for x in levels if x > 0]
        if neg and pos:
            lm, lp = max(neg), min(pos)            # the two levels closest to nominal
            vm, vp = variant_name(pname, lm), variant_name(pname, lp)
            ss = [s for s in seeds if (s, active, vm) in rows_by and (s, active, vp) in rows_by]
            rm = np.array([dz.get(s, {}).get(vm, {}).get("rel", np.nan) for s in ss], dtype=float)
            rp = np.array([dz.get(s, {}).get(vp, {}).get("rel", np.nan) for s in ss], dtype=float)
            for m in metrics:
                pres["central"][m] = central_elasticity(np.array([M(s, active, "nominal", m) for s in ss]),
                                                        np.array([M(s, active, vm, m) for s in ss]),
                                                        np.array([M(s, active, vp, m) for s in ss]), rm, rp, B)
        res["params"][pname] = pres
    return res


def headline(pres: dict, m: str) -> tuple[float | None, list, str]:
    """Central elasticity when available, else the largest-|level| one-sided estimate."""
    c = pres["central"].get(m)
    if c and c.get("S") is not None:
        return c["S"], c["ci95"], "central"
    for lev, lr in sorted(pres["levels"].items(), key=lambda kv: -abs(float(kv[0]))):
        e = lr["metrics"].get(m, {})
        if e.get("S") is not None:
            return e["S"], e["ci95"], lev
    return None, [None, None], ""


def _f(x, nd=2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def markdown(res: dict, design: dict) -> str:
    md = ["# Sim-to-real sensitivity (one-at-a-time simulator perturbations)", "",
          f"Seeds: {len(res['seeds'])} registry entries ({', '.join(res['seeds'][:8])}"
          f"{', ...' if len(res['seeds']) > 8 else ''}); policies: {res['active']}"
          f"{' vs ' + res['baseline'] if res['baseline'] else ''}; duration: {design.get('duration') or 'registry'}.",
          "The autonomy stack keeps its nominal models; only the simulated world is perturbed.", "",
          "S = (dM/M0)/(dp/p0) for the active policy (central difference when both signs were run; otherwise the "
          "one-sided level shown).  |S| > 1: the metric is elastic in that parameter.  95% bootstrap CIs over "
          "seeds are in sensitivity.json.", ""]
    params = list(res["params"])
    for title, ms in (("Primary metrics", PRIMARY), ("Secondary metrics", SECONDARY)):
        md += [f"## {title}", "", "| parameter | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
        for p in params:
            cells = []
            for m in ms:
                S, ci, how = headline(res["params"][p], m)
                if S is None:
                    lv = next(iter(res["params"][p]["levels"].values()), {"metrics": {}})
                    dM = lv["metrics"].get(m, {}).get("dM")
                    cells.append(f"dM={_f(dM, 3)}" if dM is not None else "n/a")
                else:
                    cells.append(("**" if abs(S) > 1 else "") + f"{S:+.2f}" + ("**" if abs(S) > 1 else ""))
            md.append(f"| {p} | " + " | ".join(cells) + " |")
        md.append("")
    rank = []
    for p in params:
        vals = [abs(headline(res["params"][p], m)[0]) for m in PRIMARY if headline(res["params"][p], m)[0] is not None]
        rank.append((max(vals) if vals else float("nan"), p))
    rank.sort(key=lambda x: -x[0] if math.isfinite(x[0]) else 0)
    md += ["## Most sim-sensitive parameters (max |S| over primary metrics)", ""]
    md += [f"{i}. {p}: {_f(s)}" for i, (s, p) in enumerate(rank, 1)]
    md += ["", "## Conclusion flips (active vs baseline)", ""]
    if not res["baseline"]:
        md.append("No baseline policy was run.")
    elif not res["flips"]:
        md.append("No metric changes its active-vs-baseline conclusion under any perturbation.")
    else:
        md += ["| parameter | level | metric | nominal delta | perturbed delta | significant | n |",
               "|---|---|---|---|---|---|---|"]
        for f in res["flips"]:
            sig = "yes" if f["significant"] else ("no (n<3)" if f["n"] < 3 else "no")
            md.append(f"| {f['param']} | {f['level']:+.2f} | {f['metric']} | {_f(f['nominal_delta'], 3)} | "
                      f"{_f(f['perturbed_delta'], 3)} | {sig} | {f['n']} |")
    if res["baseline"] and res.get("nominal_comparison"):
        md += ["", f"Nominal paired delta {res['active']} - {res['baseline']}:", "",
               "| metric | delta | 95% CI | n |", "|---|---|---|---|"]
        for m, d in res["nominal_comparison"].items():
            if d["n"]:
                md.append(f"| {m} | {_f(d['delta'], 3)} | [{_f(d['ci95'][0], 3)}, {_f(d['ci95'][1], 3)}] | {d['n']} |")
    md += ["", "## Perturbations", "", "| parameter | config path | levels | p0 | realised dp/p | description |",
           "|---|---|---|---|---|---|"]
    for p in params:
        pr = res["params"][p]
        lv = pr["levels"]
        p0 = sorted({x for lr in lv.values() for x in lr["p0"]})
        md.append(f"| {p} | `{pr['path']}` | {', '.join(lv)} | {', '.join(_f(x, 4) for x in p0)} | "
                  f"{', '.join(_f(lr['rel_dp'], 3) for lr in lv.values())} | {PARAMS[p].doc if p in PARAMS else ''} |")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------------------------------------------------
def select_entries(reg, families: list[str], split: str | None, n_seeds: int, scenarios: list[str] | None):
    if scenarios:
        by = {e.scenario_id: e for e in reg}
        missing = [s for s in scenarios if s not in by]
        if missing:
            raise SystemExit(f"unknown scenario ids: {missing}")
        return [by[s] for s in scenarios]
    out = []
    for fam in families:
        out += [e for e in reg if e.family == fam and (split is None or e.split == split)][:n_seeds]
    return out


def main() -> None:
    from medortrace.eval.aggregate import load_results
    from medortrace.eval.registry import FAMILIES, load_registry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="configs/seed_registry.yaml")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-seeds", type=int, default=2, help="registry entries per family")
    ap.add_argument("--scenarios", nargs="*", default=None, help="explicit scenario ids (overrides families)")
    ap.add_argument("--params", nargs="+", default=list(PARAMS), choices=list(PARAMS))
    ap.add_argument("--levels", nargs="+", type=float, default=[-0.25, 0.25], help="relative parameter changes")
    ap.add_argument("--active", default="active")
    ap.add_argument("--baseline", default="fixed_route", help="baseline policy ('none' to skip flip analysis)")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default="runs/sim_to_real")
    ap.add_argument("--analyze-only", action="store_true", help="re-analyse <out>/results.jsonl + design.json")
    ap.add_argument("--dry-run", action="store_true", help="print the run plan and exit")
    a = ap.parse_args()
    out = Path(a.out)
    baseline = None if a.baseline.lower() == "none" else a.baseline
    if not a.analyze_only:
        from medortrace.eval.batch import run_batch
        reg = load_registry(a.registry)
        entries = select_entries(reg, a.families, a.split, a.n_seeds, a.scenarios)
        policies = [a.active] + ([baseline] if baseline else [])
        levels = [x for x in a.levels if abs(x) > 1e-12]
        design = {"registry": a.registry, "duration": a.duration,
                  "params": {p: _param_doc(PARAMS[p]) for p in a.params},
                  "levels_by_param": {p: levels for p in a.params}, "policies": policies, "entries": {}}
        groups: dict[str, dict] = {}
        for e in entries:
            cfg = e.resolve()
            variants = {"nominal": {}}
            design["entries"][e.scenario_id] = {}
            for p in a.params:
                for lev in levels:
                    ov, p0, p1 = perturb(PARAMS[p], cfg, lev)
                    v = variant_name(p, lev)
                    variants[v] = {"sim": ov}
                    design["entries"][e.scenario_id][v] = {"param": p, "level": lev, "p0": p0, "p1": p1,
                                                            "rel": (p1 - p0) / p0 if abs(p0) > EPS else None}
            key = json.dumps(variants, sort_keys=True)   # entries sharing base values share one batch
            groups.setdefault(key, {"variants": variants, "entries": []})["entries"].append(e)
        n_runs = len(entries) * len(policies) * (1 + len(a.params) * len(levels))
        print(f"{len(entries)} seeds x {len(policies)} policies x {1 + len(a.params) * len(levels)} variants = "
              f"{n_runs} episodes ({len(groups)} batch group(s)), duration={a.duration or 'registry'}")
        if a.dry_run:
            return
        out.mkdir(parents=True, exist_ok=True)
        (out / "design.json").write_text(json.dumps(design, indent=1))
        for g in groups.values():
            run_batch(g["entries"], policies, g["variants"], workers=a.workers, out_jsonl=out / "results.jsonl",
                      duration=a.duration)
    design = json.loads((out / "design.json").read_text())
    rows = load_results(out / "results.jsonl")
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"WARNING: {len(errs)} episodes raised; first:\n{errs[0]['error']}")
    pols = design.get("policies", [a.active, baseline])
    res = analyze(rows, design, active=pols[0], baseline=pols[1] if len(pols) > 1 else None, B=a.bootstrap)
    res["n_errors"] = len(errs)
    (out / "sensitivity.json").write_text(json.dumps(res, indent=1))
    md = markdown(res, design)
    (out / "sensitivity.md").write_text(md)
    print(md)
    print(f"wrote {out / 'sensitivity.md'} and {out / 'sensitivity.json'}")


if __name__ == "__main__":
    main()
