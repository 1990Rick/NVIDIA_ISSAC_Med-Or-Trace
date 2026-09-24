#!/usr/bin/env python3
"""Export the reproducible USD scene family for registry entries.

    python scripts/export_usd_scenes.py --split test --limit 20 --out assets/scenes

Writes ``assets/robot/medortrace_rig.usda`` once and one ``<scenario_id>.usda``
per entry (true world at t=0, with the episode's nuisance-perturbed materials
and causal labels), plus ``index.json``.
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from medortrace.common.config import CONFIG_DIR, load_yaml
from medortrace.eval.registry import load_registry, select
from medortrace.sim.episode import build_episode
from medortrace.usd.robot_rig import build_rig
from medortrace.usd.scene_builder import build_stage

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--registry", default="configs/seed_registry.yaml")
ap.add_argument("--split", default=None)
ap.add_argument("--family", default=None)
ap.add_argument("--limit", type=int, default=10)
ap.add_argument("--out", default="assets/scenes")
a = ap.parse_args()
out = Path(a.out)
rig_path = out.parent / "robot" / "medortrace_rig.usda"
build_rig(rig_path, load_yaml(CONFIG_DIR / "robot" / "rig.yaml"))
index = []
for e in select(load_registry(a.registry), a.family, a.split, limit=a.limit):
    ep = build_episode(e.resolve(), e.seed)
    p = out / f"{e.scenario_id}.usda"
    build_stage(ep.spec, ep.materials, p, robot_rig=f"../robot/{rig_path.name}")
    index.append({"scenario_id": e.scenario_id, "usd": str(p), "seed": e.seed, "family": e.family,
                  "hidden_cause": ep.spec.hidden_cause, "cfg_hash": e.cfg_hash})
    print("wrote", p)
(out / "index.json").write_text(json.dumps(index, indent=1))
