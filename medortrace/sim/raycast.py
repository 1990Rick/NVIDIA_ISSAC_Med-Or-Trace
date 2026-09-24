"""Vectorised analytic ray casting against yaw-oriented boxes, vertical
cylinders (people) and the floor/ceiling planes.

This is the geometric core of the lite simulator's lidar, radar, camera
visibility and acoustic models.  In Isaac Sim the equivalent work is done by
the RTX sensor pipeline (OptiX) or PhysX scene queries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR_ID = -2
CEIL_ID = -3
NO_HIT = -1


@dataclass
class RayScene:
    box_center: np.ndarray    # (B,3)
    box_half: np.ndarray      # (B,3)
    box_yaw: np.ndarray       # (B,)
    cyl_xy: np.ndarray        # (C,2)
    cyl_r: np.ndarray         # (C,)
    cyl_h: np.ndarray         # (C,)
    ceiling: float = 3.0

    @property
    def n_boxes(self) -> int:
        return len(self.box_center)


@dataclass
class RayHits:
    t: np.ndarray             # (R,) distance, inf if none
    obj: np.ndarray           # (R,) index: [0,B) boxes, [B,B+C) cylinders, FLOOR_ID, CEIL_ID, NO_HIT
    normal: np.ndarray        # (R,3)


def cast(scene: RayScene, O: np.ndarray, Dr: np.ndarray, t_max: float = 30.0,
         exclude: np.ndarray | None = None) -> RayHits:
    """Cast rays O + t*Dr (Dr unit).  ``exclude`` is a bool mask over objects (B+C).

    Fully vectorised over rays x objects (slab test for boxes, quadratic for
    cylinders); memory is O(R*(B+C)).
    """
    R = len(O)
    eps = 1e-9
    B = scene.n_boxes
    C = len(scene.cyl_xy)
    best_t = np.full(R, np.inf)
    best_o = np.full(R, NO_HIT, dtype=int)
    best_n = np.zeros((R, 3))
    if R == 0:
        return RayHits(best_t, best_o, best_n)
    if B:
        cy, sy = np.cos(-scene.box_yaw), np.sin(-scene.box_yaw)                  # (B,)
        p = O[:, None, :] - scene.box_center[None]                                # R,B,3
        px = cy * p[..., 0] - sy * p[..., 1]
        py = sy * p[..., 0] + cy * p[..., 1]
        dx = cy[None] * Dr[:, None, 0] - sy[None] * Dr[:, None, 1]
        dy = sy[None] * Dr[:, None, 0] + cy[None] * Dr[:, None, 1]
        dz = np.broadcast_to(Dr[:, None, 2], dx.shape)
        P = np.stack([px, py, p[..., 2]], -1)
        Dl = np.stack([dx, dy, dz], -1)
        Dl = np.where(np.abs(Dl) < eps, eps, Dl)
        h = scene.box_half[None]
        t1 = (-h - P) / Dl
        t2 = (h - P) / Dl
        tn = np.minimum(t1, t2)
        tf = np.maximum(t1, t2)
        tmin = tn.max(-1)
        tmax = tf.min(-1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 1e-4) & (tmin < t_max)
        if exclude is not None:
            hit &= ~exclude[None, :B]
        tb = np.where(hit, tmin, np.inf)
        bi = tb.argmin(1)
        bt = tb[np.arange(R), bi]
        ok = np.isfinite(bt)
        if ok.any():
            r_ok = np.where(ok)[0]
            b_ok = bi[ok]
            ax = tn[r_ok, b_ok].argmax(-1)
            nl = np.zeros((len(r_ok), 3))
            nl[np.arange(len(r_ok)), ax] = -np.sign(Dl[r_ok, b_ok, ax])
            yw = scene.box_yaw[b_ok]
            cw, sw = np.cos(yw), np.sin(yw)
            nw = np.stack([cw * nl[:, 0] - sw * nl[:, 1], sw * nl[:, 0] + cw * nl[:, 1], nl[:, 2]], 1)
            best_t[r_ok] = bt[ok]
            best_o[r_ok] = b_ok
            best_n[r_ok] = nw
    if C:
        p = O[:, None, :2] - scene.cyl_xy[None]                                   # R,C,2
        d = Dr[:, None, :2]
        a = (d * d).sum(-1) + eps
        bq = 2 * (p * d).sum(-1)
        cq = (p * p).sum(-1) - scene.cyl_r[None] ** 2
        disc = bq * bq - 4 * a * cq
        okd = disc >= 0
        sq = np.sqrt(np.where(okd, disc, 0.0))
        t = (-bq - sq) / (2 * a)
        z = O[:, None, 2] + t * Dr[:, None, 2]
        hit = okd & (t > 1e-4) & (z >= 0) & (z <= scene.cyl_h[None]) & (t < t_max)
        if exclude is not None:
            hit &= ~exclude[None, B:B + C]
        tc = np.where(hit, t, np.inf)
        ci = tc.argmin(1)
        ct = tc[np.arange(R), ci]
        better = ct < best_t
        if better.any():
            r_ok = np.where(better)[0]
            hp = p[r_ok, ci[better]] + ct[better, None] * d[r_ok, 0]
            n2 = hp / (np.linalg.norm(hp, axis=1, keepdims=True) + eps)
            best_t[r_ok] = ct[better]
            best_o[r_ok] = B + ci[better]
            best_n[r_ok] = np.concatenate([n2, np.zeros((len(r_ok), 1))], 1)
    # floor and ceiling
    with np.errstate(divide="ignore", invalid="ignore"):
        tfl = np.where(Dr[:, 2] < -eps, -O[:, 2] / Dr[:, 2], np.inf)
        tce = np.where(Dr[:, 2] > eps, (scene.ceiling - O[:, 2]) / Dr[:, 2], np.inf)
    hitf = (tfl < best_t) & (tfl < t_max)
    best_t = np.where(hitf, tfl, best_t)
    best_o = np.where(hitf, FLOOR_ID, best_o)
    best_n = np.where(hitf[:, None], np.array([0, 0, 1.0]), best_n)
    hitc = (tce < best_t) & (tce < t_max)
    best_t = np.where(hitc, tce, best_t)
    best_o = np.where(hitc, CEIL_ID, best_o)
    best_n = np.where(hitc[:, None], np.array([0, 0, -1.0]), best_n)
    return RayHits(best_t, best_o, best_n)


def segment_occluded(scene: RayScene, a: np.ndarray, b: np.ndarray, exclude: np.ndarray | None = None,
                     tol: float = 0.05) -> np.ndarray:
    """For segments a->b (N,3), True where something blocks the segment before b."""
    d = b - a
    L = np.linalg.norm(d, axis=1)
    Dr = d / (L[:, None] + 1e-12)
    h = cast(scene, a, Dr, t_max=float(L.max()) + 1.0 if len(L) else 1.0, exclude=exclude)
    blocked = (h.t < L - tol) & (h.obj != FLOOR_ID)
    return blocked
