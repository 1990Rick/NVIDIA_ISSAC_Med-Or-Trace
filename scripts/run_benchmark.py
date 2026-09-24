#!/usr/bin/env python3
"""Run the benchmark: active (ours) vs fixed-route inspection vs passive sensing.

    python scripts/run_benchmark.py --split test --limit-per-family 20 --workers 8 --out runs/bench
    python scripts/run_benchmark.py --family counterfactual --policies active --export

Writes ``results.jsonl`` (one row per episode x policy), ``summary.json`` and
``summary.md`` (mean and bootstrap 95% CI per family x policy) and, with
``--export``, trajectory-level datasets per episode (see docs/dataset_schema.md).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401

from medortrace.eval.aggregate import PRIMARY, SAFETY, SECONDARY, load_results, markdown_table, paired_delta, summarize
from medortrace.eval.batch import run_batch
from medortrace.eval.registry import load_registry, select

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--registry", default="configs/seed_registry.yaml")
ap.add_argument("--family", default=None)
ap.add_argument("--split", default="test")
ap.add_argument("--factor", default=None, help="CF-A|CF-B|CF-C|CF-D")
ap.add_argument("--limit-per-family", type=int, default=None)
ap.add_argument("--policies", nargs="+", default=["active", "fixed_route", "passive"])
ap.add_argument("--duration", type=float, default=None)
ap.add_argument("--workers", type=int, default=None)
ap.add_argument("--backend", default="lite", choices=["lite", "isaac"])
ap.add_argument("--export", action="store_true", help="write per-episode trajectory datasets")
ap.add_argument("--save-raw", action="store_true")
ap.add_argument("--out", default="runs/benchmark")
a = ap.parse_args()

reg = load_registry(a.registry)
sel = select(reg, a.family, a.split, a.factor)
if a.limit_per_family:
    per = defaultdict(list)
    for e in sel:
        key = e.family if e.family != "counterfactual" else (e.pair_id or "").split("/")[0]
        per[key].append(e)
    sel = []
    for key, es in per.items():
        if es and es[0].pair_id:
            pairs = sorted({e.pair_id for e in es})[: a.limit_per_family]
            sel += [e for e in es if e.pair_id in pairs]
        else:
            sel += es[: a.limit_per_family]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
print(f"{len(sel)} scenarios x {len(a.policies)} policies")
run_batch(sel, a.policies, workers=a.workers, out_jsonl=out / "results.jsonl",
          export_dir=str(out / "episodes") if a.export else None, duration=a.duration, backend=a.backend,
          save_raw=a.save_raw)
rows = [r for r in load_results(out / "results.jsonl") if not r.get("error")]
summ = summarize(rows)
(out / "summary.json").write_text(json.dumps(summ, indent=1))
md = ["# MED-OR-TRACE benchmark summary", "", "## Primary metrics", markdown_table(summ, PRIMARY), "",
      "## Secondary metrics", markdown_table(summ, SECONDARY), "", "## Safety counters", markdown_table(summ, SAFETY)]
for base in [p for p in a.policies if p != "active"]:
    if "active" in a.policies:
        d = paired_delta(rows, "active", base)
        md += ["", f"## Paired delta: active - {base} (n={d['n_pairs']})", "", "| metric | delta | 95% CI | p_boot | active better |",
               "|---|---|---|---|---|"]
        for m in PRIMARY + SECONDARY + SAFETY:
            v = d.get(m)
            if v:
                md.append(f"| {m} | {v['delta']:.4f} | [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {v['p_boot']:.3f} | {v['a_better']} |")
(out / "summary.md").write_text("\n".join(md) + "\n")
print((out / "summary.md").read_text())
