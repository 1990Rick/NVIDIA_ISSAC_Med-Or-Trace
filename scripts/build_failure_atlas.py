#!/usr/bin/env python3
"""Build the failure atlas (taxonomy, counts, worst-k exemplars, reproduce commands) of a benchmark run.

    python scripts/build_failure_atlas.py runs/bench/results.jsonl
    python scripts/build_failure_atlas.py runs/bench/results.jsonl --episodes runs/bench/episodes --k 8
    python scripts/build_failure_atlas.py runs/ablation/results.jsonl --variant-specs ablation_variants.json \\
        --set stop_rate_per_min=4 loc_error_m=0.3

Exported episode directories (``run_benchmark.py --export``) are picked up from ``<results dir>/episodes``
by default; with them the atlas adds fault labels, trajectory context at the failure tick, the wrong
verdicts re-derived from events.jsonl and their provenance explanation.  Reproduce commands take each
variant's overrides from the rows' ``variant_spec`` (recorded by ``run_batch``); ``--variant-specs`` is only
needed for older results without it, whose ablation commands are otherwise marked INEXACT.  Writes
``failure_atlas.md`` and ``failure_atlas.json`` next to the results (or to ``--out``).
See ``medortrace/eval/failure_atlas.py``.
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path

import _bootstrap  # noqa: F401

from medortrace.eval.failure_atlas import AtlasThresholds, EpisodeIndex, atlas_markdown, build_atlas, load_rows


def _thresholds(pairs: list[str]) -> AtlasThresholds:
    thr = AtlasThresholds()
    names = {f.name for f in fields(thr)}
    for kv in pairs:
        k, _, v = kv.partition("=")
        if k not in names:
            raise SystemExit(f"unknown threshold {k!r}; known: {sorted(names)}")
        setattr(thr, k, float(v))
    return thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results.jsonl file(s) from run_batch")
    ap.add_argument("--episodes", nargs="*", default=None,
                    help="root(s) of exported episode dirs (default: <results dir>/episodes if it exists)")
    ap.add_argument("--k", type=int, default=5, help="exemplars per category")
    ap.add_argument("--registry", default="configs/seed_registry.yaml", help="registry used by reproduce commands")
    ap.add_argument("--repro-dir", default="runs/repro", help="out_dir of the reproduce commands")
    ap.add_argument("--variant-specs", default=None,
                    help="JSON file {variant: {autonomy:.., cfg:.., sim:..}}: overrides for rows that predate the "
                         "recorded variant_spec (older results); recorded specs take precedence")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override AtlasThresholds fields")
    ap.add_argument("--out", default=None, help="output directory (default: directory of the first results file)")
    a = ap.parse_args()

    rows = load_rows(a.results)
    roots = a.episodes
    if roots is None:
        roots = [str(Path(p).parent / "episodes") for p in a.results if (Path(p).parent / "episodes").exists()]
    specs = json.loads(Path(a.variant_specs).read_text()) if a.variant_specs else None
    atlas = build_atlas(rows, EpisodeIndex(roots), k=a.k, thresholds=_thresholds(a.set), registry=a.registry,
                        variant_specs=specs, repro_dir=a.repro_dir)
    atlas["meta"]["sources"] = [str(p) for p in a.results]
    atlas["meta"]["episode_roots"] = [str(r) for r in roots]
    out = Path(a.out or Path(a.results[0]).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "failure_atlas.json").write_text(json.dumps(atlas, indent=1, default=str))
    md = atlas_markdown(atlas)
    (out / "failure_atlas.md").write_text(md)
    print(md)
    print(f"wrote {out / 'failure_atlas.md'} and {out / 'failure_atlas.json'}")


if __name__ == "__main__":
    main()
