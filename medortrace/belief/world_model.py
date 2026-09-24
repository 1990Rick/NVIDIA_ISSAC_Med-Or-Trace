"""Temporal world model: change detection and cause diagnosis.

A persistent mismatch between lidar and the prior map has (at least) two
physically different causes that look alike in a single scan (CF-D):

* **localisation drift** - every static surface appears displaced by the
  same rigid transform; landmark innovations (NIS) grow; residual points are
  spread over the whole field of view;
* **map change** - one movable object (a cart) was moved since the survey;
  landmarks remain consistent; residual points concentrate near one object.

``ChangeDiagnoser`` accumulates evidence for both hypotheses over a window
and emits a diagnosis with a probability.  On *map change* it re-anchors the
object (and its slots) in the robot's belief; on *drift* it inflates the
localisation covariance so that the supervisor and verifier react.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from medortrace.common.geometry import OrientedBox
from medortrace.world.scene import SceneObject, Slot


@dataclass
class Diagnosis:
    t: float
    cause: str                    # none | loc_drift | map_change
    prob: float
    object: str | None = None
    shift: np.ndarray | None = None


class ChangeDiagnoser:
    def __init__(self, prior_map: list[SceneObject], window: int = 25, min_frac: float = 0.06):
        self.prior = prior_map
        self.window = window
        self.min_frac = min_frac
        self.hist: deque = deque(maxlen=window)
        self.diagnoses: list[Diagnosis] = []
        self.applied: set[str] = set()

    def observe(self, t: float, residual_pts: np.ndarray, n_static: int, nis_avg: float) -> Diagnosis | None:
        frac = len(residual_pts) / max(n_static, 1)
        spread = float(np.linalg.norm(residual_pts[:, :2].std(0))) if len(residual_pts) > 3 else 0.0
        near_obj = None
        conc = 0.0
        if len(residual_pts) > 5:
            best = 0
            for o in self.prior:
                if not o.movable:
                    continue
                d = np.linalg.norm(residual_pts[:, :2] - o.box.center[:2], axis=1)
                k = int((d < 1.2).sum())
                if k > best:
                    best, near_obj = k, o
            conc = best / len(residual_pts)
        self.hist.append((t, frac, spread, conc, nis_avg, near_obj.name if near_obj else None,
                          residual_pts[:, :2].mean(0) if len(residual_pts) else None))
        if len(self.hist) < self.window // 2:
            return None
        fr = np.array([h[1] for h in self.hist])
        if np.median(fr) < self.min_frac:
            return None
        nis = np.median([h[4] for h in self.hist])
        concm = np.median([h[3] for h in self.hist])
        spreadm = np.median([h[2] for h in self.hist])
        # likelihood-style scoring of the two hypotheses
        s_drift = 1.5 * np.clip((nis - 3.0) / 6.0, 0, 2) + np.clip((spreadm - 1.0) / 1.0, 0, 1.5) + (1 - concm)
        s_change = 2.0 * concm + np.clip((4.0 - nis) / 2.0, 0, 1) + np.clip((1.2 - spreadm) / 0.6, 0, 1)
        p_change = float(np.exp(s_change) / (np.exp(s_change) + np.exp(s_drift)))
        names = [h[5] for h in self.hist if h[5]]
        obj = max(set(names), key=names.count) if names else None
        if p_change > 0.7 and obj:
            cents = np.array([h[6] for h in self.hist if h[5] == obj and h[6] is not None])
            o = next(o for o in self.prior if o.name == obj)
            shift = cents.mean(0) - o.box.center[:2] if len(cents) else np.zeros(2)
            dg = Diagnosis(t, "map_change", p_change, obj, shift)
        elif p_change < 0.3:
            dg = Diagnosis(t, "loc_drift", 1 - p_change)
        else:
            return None
        self.diagnoses.append(dg)
        return dg

    def reanchor(self, dg: Diagnosis, slots: list[Slot]) -> list[SceneObject]:
        """Move a changed object in the prior map (and its slots) by the estimated shift."""
        if dg.object is None or dg.object in self.applied:
            return self.prior
        # residual points see only the visible faces: shift estimate is biased
        # toward the robot; clip to a plausible cart displacement
        sh = np.clip(dg.shift, -1.0, 1.0)
        for o in self.prior:
            if o.name == dg.object:
                o.box = OrientedBox(o.box.center + np.array([sh[0], sh[1], 0.0]), o.box.half, o.box.yaw)
                o.tags.append("reanchored")
        for s in slots:
            if s.anchor == dg.object:
                s.position = s.position + np.array([sh[0], sh[1], 0.0])
        self.applied.add(dg.object)
        return self.prior

    def summary(self) -> dict:
        if not self.diagnoses:
            return {"cause": "none", "prob": 0.0}
        causes = [d.cause for d in self.diagnoses]
        c = max(set(causes), key=causes.count)
        return {"cause": c, "prob": float(np.mean([d.prob for d in self.diagnoses if d.cause == c])),
                "first_t": float(next(d.t for d in self.diagnoses if d.cause == c)), "n": len(self.diagnoses)}
