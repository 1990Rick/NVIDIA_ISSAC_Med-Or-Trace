#!/usr/bin/env python3
"""Tune the next-best-view objective weights with the Cross-Entropy Method (lite simulator).

    python scripts/train_nbv_policy.py --iters 8 --pop 10 --elite 3 --episodes 8 --duration 60 --workers 3
    python scripts/train_nbv_policy.py --iters 1 --pop 2 --elite 1 --episodes 1 --duration 20 --workers 2   # smoke

Candidates are evaluated on TRAIN-split registry entries with common random numbers (every candidate of
an iteration runs on the same seeds); see ``medortrace/eval/policy_search.py`` for the algorithm and the
objective.  Objective coefficients are configurable::

    --objective-weights handoff_success=1.0 1-abstention_rate=0.5 wrong_assertion_rate=-2 ...
    --objective-file my_objective.yaml          # {objective: {metric: coef, ...}} or a flat mapping

Outputs
  * ``--output`` (default ``configs/policies/nbv_trained.yaml``): ``weights:`` (loadable by the stack via
    ``autonomy.nbv_weights: policies/nbv_trained.yaml``) + ``provenance:`` (seeds per iteration, objective
    history, search config and its hash, base weights, git commit);
  * ``--out`` run directory: ``history.jsonl`` (one record per iteration: candidates, per-episode objectives,
    elites, mu/sigma) and ``results.jsonl`` (one row per episode x candidate with metrics and objective terms).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml

from medortrace.common.config import REPO_ROOT, load_yaml
from medortrace.eval.policy_search import DEFAULT_OBJECTIVE, CemConfig, CemSearch, load_base_weights
from medortrace.eval.registry import load_registry, select

SMOKE_EPISODE_BUDGET = 50      # below this many evaluated episodes the output is flagged as a smoke run


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=str(REPO_ROOT / "configs/seed_registry.yaml"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--family", default=None)
    ap.add_argument("--factor", default=None, help="CF-A|CF-B|CF-C|CF-D")
    ap.add_argument("--base", default="policies/nbv_default.yaml", help="policy file with the initial weights")
    ap.add_argument("--keys", nargs="+", default=None, help="weights to search (default: all)")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=8, help="registry entries per iteration (shared by candidates)")
    ap.add_argument("--duration", type=float, default=60.0, help="episode sim time (s); <=0: scenario default")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--sigma0", type=float, default=0.5)
    ap.add_argument("--sigma-min", type=float, default=0.05)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--extra-noise", type=float, default=0.1)
    ap.add_argument("--no-include-mean", action="store_true", help="do not re-evaluate the mean as cand_0")
    ap.add_argument("--fixed-entries", action="store_true", help="same registry subset every iteration")
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--final", choices=["mean", "best"], default="mean")
    ap.add_argument("--error-penalty", type=float, default=-10.0)
    ap.add_argument("--objective-weights", nargs="*", default=None, metavar="TERM=COEF",
                    help="override/add objective terms ('1-<metric>' uses 1-metric; COEF 0 drops a term)")
    ap.add_argument("--objective-file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/policy_search")
    ap.add_argument("--output", default=str(REPO_ROOT / "configs/policies/nbv_trained.yaml"))
    return ap.parse_args(argv)


def objective_terms(a: argparse.Namespace) -> dict[str, float]:
    terms = dict(DEFAULT_OBJECTIVE)
    if a.objective_file:
        doc = load_yaml(a.objective_file)
        terms = {k: float(v) for k, v in (doc.get("objective", doc) or {}).items()}
    for kv in a.objective_weights or []:
        k, _, v = kv.partition("=")
        if not _:
            raise SystemExit(f"--objective-weights expects TERM=COEF, got {kv!r}")
        terms[k.strip()] = float(v)
    return {k: v for k, v in terms.items() if v != 0.0}


def repo_relative(path: str | Path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                               timeout=10)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv=None) -> int:
    a = parse_args(argv)
    reg = load_registry(a.registry)
    pool = select(reg, a.family, a.split, a.factor)
    base = load_base_weights(a.base)
    cfg = CemConfig(iters=a.iters, pop=a.pop, elite=a.elite, episodes=a.episodes,
                    duration=a.duration if a.duration and a.duration > 0 else None, workers=a.workers, keys=a.keys,
                    sigma0=a.sigma0, sigma_min=a.sigma_min, alpha=a.alpha, extra_noise=a.extra_noise,
                    include_mean=not a.no_include_mean, resample_entries=not a.fixed_entries,
                    stratify=not a.no_stratify, seed=a.seed, split=a.split, family=a.family, factor=a.factor,
                    error_penalty=a.error_penalty, final=a.final, objective=objective_terms(a))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("history.jsonl", "results.jsonl"):
        (out / f).unlink(missing_ok=True)
    print(f"[train_nbv_policy] pool={len(pool)} {a.split} entries, keys={cfg.keys or sorted(base)}")
    print(f"[train_nbv_policy] objective={json.dumps(cfg.objective)}")
    res = CemSearch(pool, base, cfg, out_dir=out).run()

    n_eval = sum(len(r.entries) * len(r.candidates) for r in res.history)
    smoke = n_eval < SMOKE_EPISODE_BUDGET
    prov = res.provenance({
        "generator": "scripts/train_nbv_policy.py",
        "command": " ".join([Path(sys.argv[0]).name] + (argv if argv is not None else sys.argv[1:])),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "registry": repo_relative(a.registry),
        "base_policy": a.base,
        "run_dir": str(out),
        "smoke_scale": smoke,
    })
    doc = {"weights": {k: float(v) for k, v in res.weights.items()}, "provenance": prov}
    head = ["# Next-best-view objective weights tuned by scripts/train_nbv_policy.py (CEM, lite simulator).",
            "# Use with: autonomy: {nbv_weights: policies/nbv_trained.yaml}",
            f"# cfg_hash {res.cfg_hash}; {len(res.history)} iterations, {n_eval} episodes evaluated."]
    if smoke:
        head.append(f"# WARNING: smoke-scale search (< {SMOKE_EPISODE_BUDGET} episodes) - NOT a tuned policy.")
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        f.write("\n".join(head) + "\n")
        yaml.safe_dump(doc, f, sort_keys=False, width=120)
    (out / "result.json").write_text(json.dumps(doc, indent=1))
    print(f"[train_nbv_policy] objective history: "
          f"{[round(h['best'], 3) for h in prov['objective_history']]} (best per iteration)")
    print(f"[train_nbv_policy] weights ({a.final}): " + ", ".join(f"{k}={v:.4g}" for k, v in res.weights.items()))
    print(f"[train_nbv_policy] wrote {a.output} and {out}/history.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
