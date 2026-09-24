"""Aggregation, confidence intervals and paired comparisons for benchmark results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRIMARY = ["min_human_clearance_m", "near_collision_rate_per_min", "task_delay_s", "handoff_success",
           "human_path_disruption_m", "uncertainty_safe_stop_rate_per_min"]
SECONDARY = ["intervention_cost", "energy_reserve_frac", "energy_used_wh", "calibration_ece",
             "calibration_ece_under_fault", "correct_abstention_frac", "wrong_assertion_rate",
             "decision_accuracy", "abstention_rate", "brier"]
SAFETY = ["collisions_agent", "collisions_static", "sterile_breach_s", "keepout_margin_violation_s",
          "near_collision_events", "handover_requests"]
# direction: +1 higher is better, -1 lower is better
DIRECTION = {"min_human_clearance_m": 1, "near_collision_rate_per_min": -1, "task_delay_s": -1, "handoff_success": 1,
             "human_path_disruption_m": -1, "uncertainty_safe_stop_rate_per_min": -1, "intervention_cost": -1,
             "energy_reserve_frac": 1, "energy_used_wh": -1, "calibration_ece": -1, "calibration_ece_under_fault": -1,
             "correct_abstention_frac": 1, "wrong_assertion_rate": -1, "decision_accuracy": 1, "abstention_rate": -1,
             "brier": -1, "collisions_agent": -1, "collisions_static": -1, "sterile_breach_s": -1,
             "keepout_margin_violation_s": -1, "near_collision_events": -1, "handover_requests": -1}


def load_results(path: str | Path, dedupe: bool = True) -> list[dict]:
    """Rows of a results.jsonl; with ``dedupe`` a re-run appended for the same
    (scenario, policy, variant, backend) replaces the earlier row (last wins)."""
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not dedupe:
        return rows
    last = {}
    for i, r in enumerate(rows):
        last[(r.get("scenario_id"), r.get("policy"), r.get("variant"), r.get("backend"))] = i
    return [rows[i] for i in sorted(last.values())]


def bootstrap_ci(x: np.ndarray, n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def summarize(rows: list[dict], group_keys=("family", "policy"), metrics=None) -> list[dict]:
    metrics = metrics or PRIMARY + SECONDARY + SAFETY
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in group_keys)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rec = dict(zip(group_keys, key))
        rec["n"] = len(rs)
        for m in metrics:
            mean, lo, hi = bootstrap_ci(np.array([_num(r["metrics"].get(m)) for r in rs], dtype=float))
            rec[m] = {"mean": mean, "ci95": [lo, hi]}
        out.append(rec)
    return out


def _num(v) -> float:
    """Metric value as float; None (NaN written to JSON as null), strings and bools-as-missing -> NaN."""
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def paired_delta(rows: list[dict], a: str, b: str, key: str = "policy", metrics=None, seed: int = 0) -> dict:
    """Paired bootstrap of metric(a) - metric(b) over matching scenario_ids."""
    metrics = metrics or PRIMARY + SECONDARY + SAFETY
    by = defaultdict(dict)
    for r in rows:
        by[r["scenario_id"]][r[key]] = r["metrics"]
    common = [s for s, d in by.items() if a in d and b in d]
    rng = np.random.default_rng(seed)
    res = {"n_pairs": len(common)}
    for m in metrics:
        d = np.array([_num(by[s][a].get(m)) - _num(by[s][b].get(m)) for s in common], dtype=float)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            res[m] = None
            continue
        bs = rng.choice(d, size=(2000, len(d)), replace=True).mean(1)
        # two-sided bootstrap p-value for H0: mean delta = 0
        p = float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        better = DIRECTION.get(m, 1) * d.mean() > 0
        res[m] = {"delta": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                  "p_boot": p, "a_better": bool(better)}
    return res


def markdown_table(summary: list[dict], metrics: list[str], group_keys=("family", "policy")) -> str:
    head = "| " + " | ".join(list(group_keys) + ["n"] + metrics) + " |"
    sep = "|" + "---|" * (len(group_keys) + 1 + len(metrics))
    lines = [head, sep]
    for rec in summary:
        cells = [str(rec[k]) for k in group_keys] + [str(rec["n"])]
        for m in metrics:
            v = rec[m]
            cells.append(
                "n/a" if not np.isfinite(v["mean"]) else f"{v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
