#!/usr/bin/env python3
"""Ablation study of the active autonomy stack.

Each variant removes one capability; all variants run on the *same* registry
entries so differences are paired.

    python scripts/run_ablation.py --split val --limit 30 --workers 8 --out runs/ablation
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from medortrace.eval.aggregate import PRIMARY, SAFETY, SECONDARY, load_results, paired_delta
from medortrace.eval.batch import run_batch
from medortrace.eval.registry import load_registry, select

VARIANTS = {
    "full": {},
    "no_radar": {"autonomy": {"modalities": ["lidar", "camera", "acoustic"]}},
    "no_acoustic": {"autonomy": {"modalities": ["lidar", "camera", "radar"]}},
    "camera_lidar_only": {"autonomy": {"modalities": ["lidar", "camera"]}},
    "no_ghost_reasoning": {"autonomy": {"use_ghost_reasoning": False}},
    "no_workflow_log": {"autonomy": {"use_workflow_log": False}},
    "no_temporal_model": {"autonomy": {"use_temporal_model": False}},
    "deterministic_map": {"autonomy": {"deterministic_map": True}},
    "no_time_sync": {"autonomy": {"use_time_sync": False}},
    "no_scan_matching": {"autonomy": {"use_scan_matching": False}},
    "no_safety_supervisor": {"autonomy": {"use_safety_supervisor": False}},
    "no_abstention": {"cfg": {"verifier": {"require_direct_evidence": False, "tau_verify": 0.5, "tau_refute": 0.5}}},
}

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--registry", default="configs/seed_registry.yaml")
ap.add_argument("--split", default="val")
ap.add_argument("--family", default=None)
ap.add_argument("--limit", type=int, default=30)
ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
ap.add_argument("--duration", type=float, default=None)
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--out", default="runs/ablation")
a = ap.parse_args()

reg = load_registry(a.registry)
sel = select(reg, a.family, a.split)
# stratify: spread the limit over strata (each stress family; each counterfactual
# factor, taking whole matched pairs so both arms of a pair are always present)


def _stratum(e) -> str:
    return e.family if e.family != "counterfactual" else (e.pair_id or "cf").split("/")[0]


strata = sorted({_stratum(e) for e in sel})
per = max(1, a.limit // max(len(strata), 1))
picked = []
for s_ in strata:
    es = [x for x in sel if _stratum(x) == s_]
    if es and es[0].pair_id:
        pairs = sorted({x.pair_id for x in es})[: max(1, per // 2)]
        picked += [x for x in es if x.pair_id in pairs]
    else:
        picked += es[:per]
sel = picked
print(f"{len(sel)} scenarios x {len(a.variants)} variants")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
run_batch(sel, ("active",), {k: VARIANTS[k] for k in a.variants}, workers=a.workers,
          out_jsonl=out / "results.jsonl", duration=a.duration)
rows = [r for r in load_results(out / "results.jsonl") if not r.get("error")]
report = {}
md = ["# Ablation study (paired: variant - full)", ""]
for v in a.variants:
    if v == "full":
        continue
    d = paired_delta(rows, v, "full", key="variant")
    report[v] = d
    md += [
        f"## {v} (n={d['n_pairs']})",
        "",
        "| metric | delta | 95% CI | p_boot | variant better |",
        "|---|---|---|---|---|",
    ]
    for m in PRIMARY + SECONDARY + SAFETY:
        x = d.get(m)
        if x:
            ci = f"[{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}]"
            md.append(f"| {m} | {x['delta']:.4f} | {ci} | {x['p_boot']:.3f} | {x['a_better']} |")
    md.append("")
(out / "ablation.json").write_text(json.dumps(report, indent=1))
(out / "ablation.md").write_text("\n".join(md))
print("\n".join(md))
