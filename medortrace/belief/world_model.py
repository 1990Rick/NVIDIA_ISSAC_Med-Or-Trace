"""Temporal world model: scan-to-map consistency, change detection and cause
diagnosis (localisation drift vs. map change).

A persistent mismatch between lidar and the prior map has (at least) two
physically different causes that look alike in a single scan (CF-D):

* **localisation drift** - *every* static surface appears displaced by the
  same rigid transform.  A rigid scan-to-map alignment (small search over
  dx, dy, dtheta against the prior-map distance field) removes most of the
  residual;
* **map change** - one movable object (a cart) was moved since the survey.
  No rigid transform explains the residual; the unexplained points
  concentrate around one movable object while the rest of the scan fits.

``ChangeDiagnoser`` accumulates both statistics over a sliding window and
emits a diagnosis with a probability.  On *drift* the aligned pose is
returned as a pseudo-measurement for the EKF (scan-to-map localisation);
on *map change* the object and its slots are re-anchored in the belief.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from medortrace.common.geometry import OrientedBox
from medortrace.planning.grid import GridSpec
from medortrace.world.scene import SceneObject, Slot


@dataclass
class Diagnosis:
    t: float
    cause: str                    # none | loc_drift | map_change
    prob: float
    object: str | None = None
    shift: np.ndarray | None = None


@dataclass
class ScanMatch:
    cost0: float
    cost: float
    dx: float
    dy: float
    dth: float

    @property
    def gain(self) -> float:
        return (self.cost0 - self.cost) / max(self.cost0, 1e-6)

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.dx, self.dy) + 0.5 * abs(self.dth))


def scan_match(pts: np.ndarray, pivot: np.ndarray, edt: np.ndarray, grid: GridSpec, max_xy: float = 0.35,
               step_xy: float = 0.07, max_th: float = np.deg2rad(6.0), step_th: float = np.deg2rad(1.0),
               clip: float = 0.4) -> ScanMatch:
    """Exhaustive rigid alignment (rotation about ``pivot``) of world points to the map EDT."""
    xs = np.arange(-max_xy, max_xy + 1e-9, step_xy)
    ths = np.arange(-max_th, max_th + 1e-9, step_th)
    P = pts[:, :2] - pivot
    best = (np.inf, 0.0, 0.0, 0.0)
    cost0 = None
    for th in ths:
        c, s = np.cos(th), np.sin(th)
        R = P @ np.array([[c, s], [-s, c]]) + pivot                      # rotated points
        cand = R[None, None] + np.stack(np.meshgrid(xs, xs, indexing="ij"), -1)[:, :, None, :]
        cell = np.floor((cand - grid.origin) / grid.res).astype(int)
        cell[..., 0] = np.clip(cell[..., 0], 0, grid.shape[0] - 1)
        cell[..., 1] = np.clip(cell[..., 1], 0, grid.shape[1] - 1)
        d = np.minimum(edt[cell[..., 0], cell[..., 1]], clip).mean(axis=-1)     # (nx, ny)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < best[0]:
            best = (float(d[i, j]), float(xs[i]), float(xs[j]), float(th))
        if abs(th) < 1e-9:
            k0 = int(np.argmin(np.abs(xs)))
            cost0 = float(d[k0, k0])
    return ScanMatch(cost0 if cost0 is not None else best[0], *best)


class ChangeDiagnoser:
    def __init__(self, prior_map: list[SceneObject], window: int = 10, min_residual_frac: float = 0.02):
        self.prior = prior_map
        self.window = window
        self.min_frac = min_residual_frac
        self.hist: deque = deque(maxlen=window)
        self.diagnoses: list[Diagnosis] = []
        self.applied: set[str] = set()
        self.last_match: ScanMatch | None = None

    def observe(self, t: float, static_pts: np.ndarray, residual_mask: np.ndarray, pose: np.ndarray,
                edt: np.ndarray, grid: GridSpec, nis_avg: float, rng: np.random.Generator) -> Diagnosis | None:
        band = (static_pts[:, 2] > 0.15) & (static_pts[:, 2] < 1.6)
        pts = static_pts[band]
        res = residual_mask[band]
        if len(pts) < 50:
            return None
        sub = rng.choice(len(pts), size=min(300, len(pts)), replace=False)
        m = scan_match(pts[sub], pose[:2], edt, grid)
        self.last_match = m
        frac = float(res.mean())
        # concentration of the unexplained points around one movable prior object
        rp = pts[res]
        conc, near = 0.0, None
        if len(rp) > 5:
            best = 0
            for o in self.prior:
                if not o.movable:
                    continue
                k = int((np.linalg.norm(rp[:, :2] - o.box.center[:2], axis=1) < 1.2).sum())
                if k > best:
                    best, near = k, o
            conc = best / len(rp)
        self.hist.append({"t": t, "frac": frac, "gain": m.gain, "mag": m.magnitude, "conc": conc,
                          "obj": near.name if near else None,
                          "cent": rp[:, :2].mean(0) if len(rp) else None, "nis": nis_avg})
        if len(self.hist) < max(3, self.window // 2):
            return None
        H = list(self.hist)
        frac_m = float(np.median([h["frac"] for h in H]))
        gain = float(np.median([h["gain"] for h in H]))
        mag = float(np.median([h["mag"] for h in H]))
        # a mismatch exists if points are unexplained, or if the scan keeps needing
        # a rigid correction (drift being continuously compensated by scan matching)
        if frac_m < self.min_frac and not (gain > 0.3 and mag > 0.05):
            return None
        concm = float(np.median([h["conc"] for h in H]))
        # log-odds style evidence for drift vs map change
        s_drift = 3.0 * np.clip((gain - 0.2) / 0.3, 0, 1.5) + 2.0 * np.clip((mag - 0.05) / 0.15, 0, 1) \
            + 0.5 * np.clip((np.median([h["nis"] for h in H]) - 4.0) / 6.0, 0, 1)
        s_change = 2.5 * np.clip((0.25 - gain) / 0.2, 0, 1.2) + 2.0 * concm
        p_change = float(1 / (1 + np.exp(s_drift - s_change)))
        names = [h["obj"] for h in H if h["obj"]]
        obj = max(set(names), key=names.count) if names else None
        if p_change > 0.75 and obj:
            cents = np.array([h["cent"] for h in H if h["obj"] == obj and h["cent"] is not None])
            o = next(o for o in self.prior if o.name == obj)
            shift = cents.mean(0) - o.box.center[:2] if len(cents) else np.zeros(2)
            dg = Diagnosis(t, "map_change", p_change, obj, shift)
        elif p_change < 0.25:
            dg = Diagnosis(t, "loc_drift", 1 - p_change)
        else:
            return None
        self.diagnoses.append(dg)
        return dg

    def pose_correction(self, pose: np.ndarray, min_gain: float = 0.35, min_mag: float = 0.05) -> np.ndarray | None:
        """Aligned pose from the last scan match (pseudo-measurement for the EKF)."""
        m = self.last_match
        if m is None or m.gain < min_gain or m.magnitude < min_mag:
            return None
        # the match transforms *points*; the robot pose moves by the same rigid motion
        return np.array([pose[0] + m.dx, pose[1] + m.dy, pose[2] + m.dth])

    def reanchor(self, dg: Diagnosis, slots: list[Slot]) -> list[SceneObject]:
        """Move a changed object in the prior map (and its slots) by the estimated shift."""
        if dg.object is None or dg.object in self.applied:
            return self.prior
        # residual points see only the visible faces: the centroid shift is biased
        # towards the robot; clip to a plausible cart displacement
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
