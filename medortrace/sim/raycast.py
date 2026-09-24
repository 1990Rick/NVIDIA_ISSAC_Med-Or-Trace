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
        ox = O[:, 0:1] - scene.box_center[None, :, 0]
        oy = O[:, 1:2] - scene.box_center[None, :, 1]
        pz = O[:, 2:3] - scene.box_center[None, :, 2]
        px = cy * ox - sy * oy
        py = sy * ox + cy * oy
        dx = cy[None] * Dr[:, 0:1] - sy[None] * Dr[:, 1:2]
        dy = sy[None] * Dr[:, 0:1] + cy[None] * Dr[:, 1:2]
        dz = np.broadcast_to(Dr[:, 2:3], dx.shape)
        tmin = np.full(dx.shape, -np.inf)
        tmax = np.full(dx.shape, np.inf)
        axis_of_min = np.zeros(dx.shape, dtype=np.int8)
        sign_of_min = np.zeros(dx.shape)
        for a, (pp, dd) in enumerate(((px, dx), (py, dy), (pz, dz))):
            h = scene.box_half[None, :, a]
            dd = np.where(np.abs(dd) < eps, eps, dd)
            inv = 1.0 / dd
            t1 = (-h - pp) * inv
            t2 = (h - pp) * inv
            tn = np.minimum(t1, t2)
            tf = np.maximum(t1, t2)
            upd = tn > tmin
            tmin = np.where(upd, tn, tmin)
            axis_of_min = np.where(upd, a, axis_of_min)
            sign_of_min = np.where(upd, -np.sign(dd), sign_of_min)
            tmax = np.minimum(tmax, tf)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 1e-4) & (tmin < t_max)
        if exclude is not None:
            hit &= ~exclude[None, :B]
        tb = np.where(hit, tmin, np.inf)
        bi = tb.argmin(1)
        ar = np.arange(R)
        bt = tb[ar, bi]
        ok = np.isfinite(bt)
        if ok.any():
            r_ok = np.where(ok)[0]
            b_ok = bi[ok]
            ax = axis_of_min[r_ok, b_ok]
            sg = sign_of_min[r_ok, b_ok]
            nl = np.zeros((len(r_ok), 3))
            nl[np.arange(len(r_ok)), ax] = sg
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


def segments_blocked(scene: RayScene, a: np.ndarray, b: np.ndarray, own: np.ndarray | None = None,
                     exclude: np.ndarray | None = None, tol: float = 0.05, own_tol: float = 0.6) -> np.ndarray:
    """Batched occlusion test for many segments with per-segment "own" objects.

    One vectorised cast for all segments.  A segment is blocked if the first
    hit lies before its end point (minus ``tol``), unless that first hit is the
    segment's own object (``own[i]``, e.g. the table an item rests on or the
    person holding it) within ``own_tol`` of the end point.  ``exclude`` is a
    global mask of objects that are transparent for every segment.
    """
    if len(a) == 0:
        return np.zeros(0, dtype=bool)
    d = b - a
    L = np.linalg.norm(d, axis=1)
    Dr = d / (L[:, None] + 1e-12)
    h = cast(scene, a, Dr, t_max=float(L.max()) + 1.0, exclude=exclude)
    blocked = (h.t < L - tol) & (h.obj != FLOOR_ID)
    if own is not None:
        blocked &= ~((h.obj == own) & (h.t > L - own_tol))
    return blocked
