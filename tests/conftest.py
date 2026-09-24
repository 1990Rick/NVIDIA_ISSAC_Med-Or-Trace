"""Shared fixtures for the MED-OR-TRACE test-suite.

Design notes
------------
* The repository root is put on ``sys.path`` so ``pytest`` works both with and
  without ``PYTHONPATH=.``.
* Fixtures here are *factories* or cheap fresh objects (function scope): many
  tests mutate episodes / beliefs, so nothing mutable is shared between tests.
* Nothing is ``autouse``: the training-component tests in this directory are
  unaffected by this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medortrace.common.config import load_config  # noqa: E402
from medortrace.common.geometry import OrientedBox  # noqa: E402
from medortrace.sim.episode import Episode, build_episode  # noqa: E402
from medortrace.sim.raycast import RayScene  # noqa: E402

CF_CONFIGS = {
    "CF-A": "scenarios/cf_a.yaml",
    "CF-B": "scenarios/cf_b.yaml",
    "CF-C": "scenarios/cf_c.yaml",
    "CF-D": "scenarios/cf_d.yaml",
}


@pytest.fixture
def default_cfg() -> dict:
    """A fresh copy of ``configs/scenarios/default.yaml``."""
    return load_config()


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def _make_scene(boxes=(), cylinders=(), ceiling: float = 3.0) -> RayScene:
    """``boxes``: OrientedBox list; ``cylinders``: (x, y, r, h) tuples."""
    boxes = list(boxes)
    cyl = np.array(list(cylinders), dtype=float).reshape(-1, 4)
    return RayScene(
        np.array([b.center for b in boxes], dtype=float).reshape(-1, 3),
        np.array([b.half for b in boxes], dtype=float).reshape(-1, 3),
        np.array([b.yaw for b in boxes], dtype=float),
        cyl[:, :2].copy(), cyl[:, 2].copy(), cyl[:, 3].copy(), ceiling)


@pytest.fixture
def make_scene():
    """Factory building a :class:`RayScene` from boxes and (x, y, r, h) cylinders."""
    return _make_scene


@pytest.fixture
def box():
    """Factory: ``box((cx, cy, cz), (hx, hy, hz), yaw=0)`` -> OrientedBox."""
    return lambda c, h, yaw=0.0: OrientedBox(np.array(c, float), np.array(h, float), yaw)


@pytest.fixture
def cf_episode():
    """Factory: ``cf_episode(factor, value, seed, **episode_overrides)`` -> fresh Episode."""

    def build(factor: str, value: str, seed: int, override: dict | None = None) -> Episode:
        ov = {"hidden_cause": {"factor": factor, "value": value}}
        if override:
            ov.update(override)
        return build_episode(load_config(CF_CONFIGS[factor], ov), seed)

    return build
