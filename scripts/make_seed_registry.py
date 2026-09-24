#!/usr/bin/env python3
"""Generate the scenario seed registry (configs/seed_registry.yaml).

    python scripts/make_seed_registry.py --per-family 100 --pairs-per-cf 50
    -> 8 families x 100 + 4 factors x 50 pairs x 2 arms = 1200 episodes
"""
import argparse
from collections import Counter

import _bootstrap  # noqa: F401

from medortrace.eval.registry import build_registry, save_registry

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--per-family", type=int, default=100)
ap.add_argument("--pairs-per-cf", type=int, default=50)
ap.add_argument("--master-seed", type=int, default=20260924)
ap.add_argument("--duration", type=float, default=None, help="override episode duration (s)")
ap.add_argument("--out", default="configs/seed_registry.yaml")
a = ap.parse_args()
entries = build_registry(a.per_family, a.pairs_per_cf, a.master_seed, a.duration)
save_registry(entries, a.out, {"master_seed": a.master_seed, "per_family": a.per_family,
                               "pairs_per_cf": a.pairs_per_cf, "duration_override": a.duration})
print(f"wrote {len(entries)} entries to {a.out}")
print("families:", dict(Counter(e.family for e in entries)))
print("splits:", dict(Counter(e.split for e in entries)))
