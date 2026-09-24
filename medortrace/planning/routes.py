"""Baseline goal generators: fixed-route inspection and passive sensing."""

from __future__ import annotations

import numpy as np

from medortrace.planning.costmap import Costmap
from medortrace.world.scene import Slot, SterileZone


def fixed_route(zones: list[SterileZone], slots: list[Slot], cm: Costmap, standoff: float = 0.45) -> list[np.ndarray]:
    """Clockwise ring of inspection poses around the sterile zones plus
    containers/counters, each facing the nearest slot.  Belief-agnostic."""
    pts = []
    for z in zones:
        c = z.box.corners_xy()
        ctr = z.box.center[:2]
        for p in c:
            d = p - ctr
            q = p + d / np.linalg.norm(d) * (z.keepout_margin + standoff + 0.2)
            pts.append(q)
    for s in slots:
        if s.kind in ("container", "surface") and not s.sterile:
            off = np.array([0.0, -0.8 if s.position[1] > cm.grid.shape[1] * cm.grid.res / 2 else 0.8])
            pts.append(s.position[:2] + off)
    ctr = np.mean([z.box.center[:2] for z in zones], axis=0)
    pts = sorted(pts, key=lambda p: np.arctan2(p[1] - ctr[1], p[0] - ctr[0]))
    route = []
    for p in pts:
        if not cm.grid.in_bounds(p[None])[0] or cm.lookup(p[None], "lethal")[0]:
            continue
        pos = np.array([s.position[:2] for s in slots if np.all(np.isfinite(s.position))])
        near = pos[np.argmin(np.linalg.norm(pos - p, axis=1))]
        route.append(np.array([p[0], p[1], np.arctan2(near[1] - p[1], near[0] - p[0])]))
    return route


def passive_vantage(zones: list[SterileZone], cm: Costmap, room: tuple) -> np.ndarray:
    """A single fixed vantage point with the best view of the sterile field."""
    ctr = np.mean([z.box.center[:2] for z in zones], axis=0)
    best, bd = None, -1
    W, D, _ = room
    for x in np.linspace(0.8, W - 0.8, 12):
        for y in np.linspace(0.8, D - 0.8, 12):
            p = np.array([x, y])
            if cm.lookup(p[None], "lethal")[0] or cm.lookup(p[None], "edt")[0] < 0.6:
                continue
            d = np.linalg.norm(p - ctr)
            score = -abs(d - 2.6)
            if score > bd:
                best, bd = p, score
    if best is None:
        best = np.array([1.0, 1.0])
    return np.array([best[0], best[1], np.arctan2(ctr[1] - best[1], ctr[0] - best[0])])
