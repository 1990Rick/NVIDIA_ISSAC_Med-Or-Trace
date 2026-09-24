"""Scenario seed registry.

The registry is the single index of every experiment episode.  Each entry
fully determines an episode (config file + overrides + seed), so results are
reproducible on any backend and any machine::

    - scenario_id: cf_a__p0007__under_drape
      family: counterfactual
      config: scenarios/cf_a.yaml
      seed: 1830274461
      overrides: {hidden_cause: {factor: CF-A, value: under_drape}}
      pair_id: CF-A/p0007           # matched counterfactual pair (same seed)
      split: test                   # train | val | test
      cfg_hash: 9c1e...

Splits are assigned by hashing the *pair* (or scenario) id so that both arms
of a counterfactual pair always land in the same split (no leakage).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from medortrace.common.config import config_hash, load_config
from medortrace.common.rng import stable_hash

FAMILIES = {
    "nominal": "scenarios/nominal.yaml",
    "sensor_dropout": "scenarios/sensor_dropout.yaml",
    "timestamp_skew": "scenarios/timestamp_skew.yaml",
    "map_corruption": "scenarios/map_corruption.yaml",
    "loc_drift": "scenarios/loc_drift.yaml",
    "adversarial_occlusion": "scenarios/adversarial_occlusion.yaml",
    "reflective": "scenarios/reflective.yaml",
    "rare_geometry": "scenarios/rare_geometry.yaml",
}
COUNTERFACTUALS = {
    "CF-A": ("scenarios/cf_a.yaml", ["under_drape", "kick_bucket"]),
    "CF-B": ("scenarios/cf_b.yaml", ["specular_ghost", "real_obstacle"]),
    "CF-C": ("scenarios/cf_c.yaml", ["dropped_floor", "handed_off"]),
    "CF-D": ("scenarios/cf_d.yaml", ["loc_drift", "cart_moved"]),
}
RARE = ["iv_pole_fallen", "boom_lowered", "drape_trailing", "cart_tipped"]


@dataclass
class RegistryEntry:
    scenario_id: str
    family: str
    config: str
    seed: int
    overrides: dict = field(default_factory=dict)
    pair_id: str | None = None
    split: str = "train"
    cfg_hash: str = ""

    def resolve(self) -> dict:
        cfg = load_config(self.config, self.overrides)
        cfg["scenario_id"] = self.scenario_id
        return cfg


def _split(key: str, fr=(0.7, 0.15, 0.15)) -> str:
    u = (stable_hash("split:" + key) % 10_000) / 10_000
    return "train" if u < fr[0] else ("val" if u < fr[0] + fr[1] else "test")


def build_registry(per_family: int = 100, pairs_per_cf: int = 50, master_seed: int = 20260924,
                   duration_s: float | None = None) -> list[RegistryEntry]:
    rng = np.random.default_rng(master_seed)
    out: list[RegistryEntry] = []
    common = {"episode": {"duration_s": duration_s}} if duration_s else {}
    for fam, path in FAMILIES.items():
        for i in range(per_family):
            seed = int(rng.integers(1, 2**31 - 1))
            ov = dict(common)
            if fam == "rare_geometry":
                # each rare episode stages 1-2 rare conditions (combinatorial coverage)
                k = int(rng.integers(1, 3))
                ov = {**ov, "faults": {"rare_geometry": sorted(rng.choice(RARE, size=k, replace=False).tolist())}}
            sid = f"{fam}__{i:04d}"
            e = RegistryEntry(sid, fam, path, seed, ov, None, _split(sid))
            e.cfg_hash = config_hash(e.resolve())
            out.append(e)
    for fac, (path, values) in COUNTERFACTUALS.items():
        for i in range(pairs_per_cf):
            seed = int(rng.integers(1, 2**31 - 1))
            pid = f"{fac}/p{i:04d}"
            for v in values:
                sid = f"{fac.lower().replace('-', '_')}__p{i:04d}__{v}"
                ov = {**common, "hidden_cause": {"factor": fac, "value": v}}
                e = RegistryEntry(sid, "counterfactual", path, seed, ov, pid, _split(pid))
                e.cfg_hash = config_hash(e.resolve())
                out.append(e)
    return out


def save_registry(entries: list[RegistryEntry], path: str | Path, meta: dict | None = None) -> None:
    doc = {"schema": "medortrace.seed_registry/1", "meta": meta or {},
           "entries": [asdict(e) for e in entries]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, width=120)


def load_registry(path: str | Path) -> list[RegistryEntry]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    return [RegistryEntry(**e) for e in doc["entries"]]


def select(entries: list[RegistryEntry], family: str | None = None, split: str | None = None,
           factor: str | None = None, limit: int | None = None) -> list[RegistryEntry]:
    out = [e for e in entries if (family is None or e.family == family) and (split is None or e.split == split)
           and (factor is None or (e.pair_id or "").startswith(factor))]
    return out[:limit] if limit else out
