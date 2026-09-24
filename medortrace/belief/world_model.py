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


def _majority(xs: list):
    """Most frequent element; ties go to the most recent one (deterministic,
    unlike ``max(set(xs), key=xs.count)`` whose tie-break follows the
    per-process string hash seed)."""
    if not xs:
        return None
    counts: dict = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return next(x for x in reversed(xs) if counts[x] == best)


class ChangeDiagnoser:
    def __init__(self, prior_map: list[SceneObject], window: int = 10, min_residual_frac: float = 0.02):
        self.prior = prior_map
        self.window = window
        self.min_frac = min_residual_frac
        self.hist: deque = deque(maxlen=window)
        self.res_buf: deque = deque(maxlen=window)        # recent residual points (world xy)
        self.diagnoses: list[Diagnosis] = []
        self.applied: set[str] = set()
        self.last_match: ScanMatch | None = None

    def observe(self, t: float, static_pts: np.ndarray, residual_mask: np.ndarray, pose: np.ndarray,
                edt: np.ndarray, grid: GridSpec, nis_avg: float, rng: np.random.Generator,
                occ=None) -> Diagnosis | None:
        band = (static_pts[:, 2] > 0.15) & (static_pts[:, 2] < 1.6)
        pts = static_pts[band]
        res = residual_mask[band]
        if len(pts) < 50:
            return None
        sub = rng.choice(len(pts), size=min(300, len(pts)), replace=False)
        m = scan_match(pts[sub], pose[:2], edt, grid)
        self.last_match = m
        frac = float(res.mean())
        # excess concentration of the unexplained points around one movable prior
        # object: the share of residual points near it minus the share of *all*
        # scan points near it.  A pose error leaves residuals on every surface in
        # view (excess ~ 0 even when the robot looks mostly at one cart); a moved
        # object concentrates them (excess >> 0).
        rp = pts[res]
        self.res_buf.append(rp[:, :2].copy())
        conc, near, cent = 0.0, None, None
        if len(rp) > 5:
            for o in self.prior:
                if not o.movable:
                    continue
                in_r = np.linalg.norm(rp[:, :2] - o.box.center[:2], axis=1) < 1.2
                if in_r.sum() < 5:
                    continue
                in_all = np.linalg.norm(pts[:, :2] - o.box.center[:2], axis=1) < 1.2
                ex = float(in_r.mean() - in_all.mean())
                if ex > conc:
                    conc, near, cent = ex, o, rp[in_r, :2].mean(0)
        self.hist.append({"t": t, "frac": frac, "gain": m.gain, "mag": m.magnitude, "conc": conc,
                          "obj": near.name if near else None, "cent": cent, "nis": nis_avg})
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
        s_change = 2.5 * np.clip((0.25 - gain) / 0.2, 0, 1.2) + 3.0 * concm
        p_change = float(1 / (1 + np.exp(s_drift - s_change)))
        names = [h["obj"] for h in H if h["obj"]]
        obj = _majority(names)
        if p_change > 0.75 and obj:
            # unexplained points *near* an object (clutter, specular ghosts, a person,
            # items) do not show that it moved; a move also leaves its surveyed
            # footprint observably empty.  Without that absence evidence there is no
            # map-change diagnosis (and hence no permanent re-anchoring).
            if occ is not None and self.vacated_fraction(obj, occ) < self.min_vacated:
                return None
            # ... and the residual returns must trace the object's own outline at a
            # displaced pose (a mirror-finish object also looks "vacated" - its
            # specular ghosts land behind it - but ghosts, clutter and people do not
            # form its outline).  The fitted shift replaces the centroid estimate,
            # which is biased towards the faces the robot happens to see.
            shift, support, n_on = self.fit_object_shift(obj)
            if support < self.min_outline_support or n_on < self.min_outline_points or np.hypot(*shift) < 0.15:
                return None
            dg = Diagnosis(t, "map_change", p_change, obj, shift)
        elif p_change < 0.25:
            dg = Diagnosis(t, "loc_drift", 1 - p_change)
        else:
            return None
        self.diagnoses.append(dg)
        return dg

    min_vacated = 0.25     # fraction of the surveyed footprint interior that must be observed free
    min_outline_support = 0.5   # fraction of nearby residual points on the displaced outline
    min_outline_points = 15

    def fit_object_shift(self, obj: str, max_shift: float = 1.0, tol: float = 0.07) -> tuple[np.ndarray, float, int]:
        """Register the object's surveyed 2D outline to the buffered residual points.

        Grid search over translations (the orientation is kept); score = number of
        residual points within ``tol`` of the shifted outline, minus points deep
        inside it (a solid object cannot return from its interior).  Returns the
        best shift, the fraction of nearby residual points it explains, and the
        count of explained points."""
        o = next((o for o in self.prior if o.name == obj), None)
        if o is None or not self.res_buf:
            return np.zeros(2), 0.0, 0
        P = np.concatenate(list(self.res_buf), axis=0)
        c0 = o.box.center[:2]
        P = P[np.linalg.norm(P - c0, axis=1) < max_shift + np.hypot(*o.box.half[:2]) + tol]
        if len(P) == 0:
            return np.zeros(2), 0.0, 0
        R = np.array([[np.cos(o.box.yaw), -np.sin(o.box.yaw)], [np.sin(o.box.yaw), np.cos(o.box.yaw)]])
        L = (P - c0) @ R                                            # points in the object frame
        h = o.box.half[:2]

        def score(s_local):
            q = np.abs(L[None] - s_local[:, None, :]) - h           # (N_s, N_p, 2)
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=2)
            inside = np.minimum(np.max(q, axis=2), 0.0)             # <= 0 inside
            d = outside + inside                                    # signed distance to the outline
            on = np.abs(d) < tol
            deep = d < -2 * tol
            return on.sum(1) - deep.sum(1), on.sum(1)

        best = (-np.inf, np.zeros(2), 0)
        for step, span, ctr in ((0.1, max_shift, np.zeros(2)), (0.02, 0.1, None)):
            base = best[1] if ctr is None else ctr
            g = np.arange(-span, span + 1e-9, step)
            S = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2) + base
            S = S[np.linalg.norm(S, axis=1) <= max_shift]
            sc, on = score(S)
            k = int(np.argmax(sc))
            if sc[k] > best[0]:
                best = (float(sc[k]), S[k], int(on[k]))
        s_local, n_on = best[1], best[2]
        return R @ s_local, n_on / len(P), n_on

    def vacated_fraction(self, obj: str, occ) -> float:
        """Fraction of the object's surveyed footprint interior (shrunk by one cell)
        whose voxel column, at the object's height, has been observed free: a lidar
        ray passed through where the object should be."""
        o = next((o for o in self.prior if o.name == obj), None)
        if o is None:
            return 0.0
        b, r = o.box, occ.res
        g = occ.grid2d
        c = g.world_to_cell(b.corners_xy())
        i0, i1 = max(0, c[:, 0].min() - 1), min(g.shape[0], c[:, 0].max() + 2)
        j0, j1 = max(0, c[:, 1].min() - 1), min(g.shape[1], c[:, 1].max() + 2)
        I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        ctr = g.cell_to_world(np.stack([I.ravel(), J.ravel()], 1))
        inner = OrientedBox(b.center, np.array([max(b.half[0] - r, 0.5 * r), max(b.half[1] - r, 0.5 * r),
                                                b.half[2]]), b.yaw)
        m = inner.contains_xy(ctr)
        if not m.any():
            return 0.0
        k0 = max(1, int(np.ceil(max(b.z_min, 0.15) / r)))
        k1 = min(occ.nz, max(k0 + 1, int(np.ceil(min(b.z_max, 1.5) / r))))
        ii, jj = I.ravel()[m], J.ravel()[m]
        p = occ.prob()[ii, jj, k0:k1]
        seen = occ.observed[ii, jj, k0:k1]
        free_col = (seen & (p < 0.35)).any(axis=1)
        return float(free_col.mean())

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
        c = _majority(causes)
        objs = [d.object for d in self.diagnoses if d.cause == c and d.object]
        return {"cause": c, "prob": float(np.mean([d.prob for d in self.diagnoses if d.cause == c])),
                "first_t": float(next(d.t for d in self.diagnoses if d.cause == c)), "n": len(self.diagnoses),
                "object": _majority(objs)}
