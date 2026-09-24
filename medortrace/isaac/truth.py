"""Ground-truth custody placement shared by the Isaac backend and the synthetic-data generator.

The lite simulator decides *where* an item physically is from its truth slot
(``LiteBackend.slot_position`` / ``item_position``): surface-like slots get a
small per-item offset drawn from the ``layout`` stream fork ``item_offsets``,
hand slots follow the holder 0.3 m towards the OR table at 1.0 m height, and
``elsewhere`` means out of the room.  This module reproduces those rules
exactly (the fork is independent of stream consumption, so the offsets are
bit-identical), which keeps a registry seed the same experiment in the lite
backend, the Isaac backend and the Replicator dataset.

Pure numpy; importable without Isaac Sim.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from medortrace.world.scene import SceneSpec

OFFSET_SLOT_KINDS = ("surface", "container", "floor", "under_drape")
HIDDEN_Z = -5.0          # parking height for items that are "elsewhere" (out of the room)


def item_offsets(ep) -> dict[str, np.ndarray]:
    """Per-item in-slot offsets, identical to ``LiteBackend._make_item_offsets``."""
    rng = ep.streams.fork("layout", "item_offsets")
    return {i.id: np.array([*rng.uniform(-0.12, 0.12, 2), 0.0]) for i in ep.spec.items}


def hand_position(spec: SceneSpec, holder_xy: np.ndarray) -> np.ndarray:
    toward = spec.object("or_table").box.center[:2] - np.asarray(holder_xy, float)
    toward = toward / (np.linalg.norm(toward) + 1e-9)
    return np.array([*(np.asarray(holder_xy, float) + 0.3 * toward), 1.0])


def slot_position(spec: SceneSpec, sid: str, staff_xy: Mapping[str, np.ndarray]) -> np.ndarray:
    """Nominal slot position; hand slots are resolved from the holder's current position."""
    s = spec.slot(sid)
    if s.kind == "hand":
        return hand_position(spec, staff_xy[s.anchor])
    return np.asarray(s.position, float)


def item_position(spec: SceneSpec, iid: str, sid: str, offsets: Mapping[str, np.ndarray],
                  staff_xy: Mapping[str, np.ndarray]) -> np.ndarray:
    """World position of an item's *support point* (NaN when elsewhere), as in the lite backend."""
    if sid == "elsewhere":
        return np.full(3, np.nan)
    p = slot_position(spec, sid, staff_xy).copy()
    s = spec.slot(sid)
    if s.kind in OFFSET_SLOT_KINDS:
        p = p + offsets[iid] * min(1.0, s.radius / 0.2)
    return p


def item_prim_center(spec: SceneSpec, iid: str, support: np.ndarray) -> np.ndarray:
    """USD prim translate for an item resting at ``support`` (box centre = support + half height)."""
    if not np.all(np.isfinite(support)):
        return np.array([0.0, 0.0, HIDDEN_Z])
    half_h = 0.5 * float(spec.item(iid).size[2])
    return np.asarray(support, float) + np.array([0.0, 0.0, half_h])


def staff_positions(population) -> dict[str, np.ndarray]:
    """``StaffPopulation`` -> {name: xy}."""
    return {a.spec.name: a.pos.copy() for a in population.agents}


class CustodyTimeline:
    """Replays ``workflow.truth`` forward in time (monotone ``advance``)."""

    def __init__(self, workflow):
        self.truth = sorted(workflow.truth, key=lambda m: m.t)
        self.slots = dict(workflow.initial)
        self._ptr = 0
        self.t = -np.inf

    def advance(self, t: float) -> list[str]:
        """Apply all moves with ``m.t <= t``; returns the ids of items whose slot changed."""
        if t < self.t:
            raise ValueError(f"CustodyTimeline can only advance forward ({t} < {self.t})")
        changed = []
        while self._ptr < len(self.truth) and self.truth[self._ptr].t <= t:
            m = self.truth[self._ptr]
            self.slots[m.item_id] = m.dst
            changed.append(m.item_id)
            self._ptr += 1
        self.t = t
        return changed
