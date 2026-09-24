"""Argument helpers shared by the Isaac Sim entry scripts (pure Python: no omni imports).

Scenario selection resolves to ``(cfg, seed, entry)`` *before* ``SimulationApp``
starts, so a typo in a scenario id fails in a second instead of after Kit boots.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import _bootstrap  # noqa: F401

from medortrace.common.config import deep_merge, load_config
from medortrace.eval.registry import RegistryEntry, load_registry

DEFAULT_REGISTRY = str(_bootstrap.ROOT / "configs" / "seed_registry.yaml")


def add_scenario_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("scenario (registry entry, or config + seed)")
    g.add_argument("--registry", default=DEFAULT_REGISTRY)
    g.add_argument("--scenario-id", default=None, help="registry entry, e.g. cf_b__p0000__real_obstacle")
    g.add_argument("--config", default=None, help="scenario config, e.g. scenarios/cf_b.yaml (with --seed)")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--override", default=None, help='JSON deep-merged into the config, e.g. \'{"faults": {...}}\'')


def add_isaac_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("Isaac Sim")
    g.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--prims-api", choices=["stable", "experimental"], default=os.environ.get(
        "MEDORTRACE_ISAAC_PRIMS", "stable"), help="prim wrappers (see medortrace/isaac/compat.py)")
    g.add_argument("--physics-hz", type=float, default=120.0)


def detector_arg(s: str) -> str:
    if s == "gt_surrogate":
        return s
    if s.startswith("model:"):
        p = Path(s.split(":", 1)[1]).expanduser()
        if not p.is_file():
            raise argparse.ArgumentTypeError(f"detector checkpoint not found: {p}")
        return f"model:{p.resolve()}"
    raise argparse.ArgumentTypeError("expected gt_surrogate or model:<path>")


def resolve_scenario(a: argparse.Namespace) -> tuple[dict, int, RegistryEntry | None]:
    entry = None
    if a.scenario_id:
        found = [e for e in load_registry(a.registry) if e.scenario_id == a.scenario_id]
        if not found:
            raise SystemExit(f"scenario id {a.scenario_id!r} not in {a.registry}")
        entry = found[0]
        cfg, seed = entry.resolve(), int(entry.seed if a.seed is None else a.seed)
    elif a.config:
        if a.seed is None:
            raise SystemExit("--config needs --seed")
        cfg, seed = load_config(a.config), int(a.seed)
        cfg["scenario_id"] = f"{Path(a.config).stem}__s{seed}"
    else:
        raise SystemExit("give --scenario-id or --config/--seed")
    if a.override:
        cfg = deep_merge(cfg, json.loads(a.override))
    return cfg, seed, entry


def set_prims_api(name: str) -> None:
    """Must run before ``medortrace.isaac.compat`` is imported (it reads the variable once)."""
    os.environ["MEDORTRACE_ISAAC_PRIMS"] = name
