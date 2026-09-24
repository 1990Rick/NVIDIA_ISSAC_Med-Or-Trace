"""Reproducible, independent random streams.

Every episode is fully determined by a single 64-bit *scenario seed* taken from
the seed registry.  From it we spawn independent child streams per subsystem so
that, e.g., changing the number of lidar rays does not perturb staff
trajectories.  This is what makes counterfactual pairs possible: two episodes
share every stream except the one that encodes the hidden cause.
"""

from __future__ import annotations

import hashlib

import numpy as np

STREAMS = (
    "layout",       # room geometry, furniture placement
    "materials",    # surface property perturbations (nuisance)
    "clutter",      # distractor objects (nuisance)
    "agents",       # staff trajectories & behaviour
    "workflow",     # surgical workflow event timing
    "hidden",       # hidden-cause variable (counterfactual factor)
    "sensors",      # measurement noise
    "faults",       # dropout, skew, drift, corruption schedule
    "robot",        # actuation noise
    "policy",       # stochastic planner sampling (MPPI, NBV candidates)
    "visibility",   # lighting / haze / glare
)


def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


class RngStreams:
    """A bundle of named ``numpy.random.Generator`` streams derived from one seed."""

    def __init__(self, seed: int, overrides: dict[str, int] | None = None):
        self.seed = int(seed)
        self.overrides = dict(overrides or {})
        self._gens: dict[str, np.random.Generator] = {}
        for name in STREAMS:
            self._gens[name] = self._make(name)

    def _make(self, name: str) -> np.random.Generator:
        sub = self.overrides.get(name, self.seed)
        return np.random.default_rng(np.random.SeedSequence([int(sub) & (2**63 - 1), stable_hash(name)]))

    def __getitem__(self, name: str) -> np.random.Generator:
        if name not in self._gens:
            self._gens[name] = self._make(name)
        return self._gens[name]

    def fork(self, name: str, tag: str) -> np.random.Generator:
        """Deterministic child generator (independent of consumption of ``name``)."""
        sub = self.overrides.get(name, self.seed)
        return np.random.default_rng(
            np.random.SeedSequence([int(sub) & (2**63 - 1), stable_hash(name), stable_hash(tag)])
        )
