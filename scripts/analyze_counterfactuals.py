#!/usr/bin/env python3
"""Pair-level analysis of the counterfactual families (CF-A..CF-D) of a benchmark run.

    python scripts/analyze_counterfactuals.py runs/bench/results.jsonl
    python scripts/analyze_counterfactuals.py runs/cf_check/results.jsonl --out runs/cf_check --max-pairs 20

Reports per factor x policy x variant: discrimination rate (the robot's outcome differs correctly between
the two arms of a matched pair), decision / diagnosis confusion matrices, unsafe-outcome rates (CF-A
retained sponge signed off, CF-B traversal of / collision at the real obstacle, CF-C clamp wrongly
verified), abstention and appropriate abstention, and the mission-level effect of the hidden cause.
Writes ``counterfactual_analysis.md`` and ``.json``.  See ``medortrace/eval/counterfactual_analysis.py``.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from medortrace.eval.counterfactual_analysis import analysis_markdown, analyze
from medortrace.eval.failure_atlas import load_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results.jsonl file(s) from run_batch")
    ap.add_argument("--out", default=None, help="output directory (default: directory of the first results file)")
    ap.add_argument("--max-pairs", type=int, default=40, help="pairs listed per factor in the markdown")
    a = ap.parse_args()
    res = analyze(load_rows(a.results))
    res["meta"]["sources"] = [str(p) for p in a.results]
    if not res["groups"]:
        print("no complete counterfactual pairs found (rows need pair_id and a CF-A..CF-D hidden cause)")
    out = Path(a.out or Path(a.results[0]).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "counterfactual_analysis.json").write_text(json.dumps(res, indent=1, default=str))
    md = analysis_markdown(res, max_pairs=a.max_pairs)
    (out / "counterfactual_analysis.md").write_text(md)
    print(md)
    print(f"wrote {out / 'counterfactual_analysis.md'} and {out / 'counterfactual_analysis.json'}")


if __name__ == "__main__":
    main()
