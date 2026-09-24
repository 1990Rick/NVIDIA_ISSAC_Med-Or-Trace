"""Risk-aware MPPI controller for a differential-drive base.

For K sampled control sequences over horizon H we roll out the unicycle
model and evaluate:

    J = w_track  * tracking/progress to the reference path
      + w_obs    * obstacle proximity (EDT from belief costmap; lethal = inf)
      + w_keep   * sterile keep-out (hard constraint)
      + w_soft   * uncertainty / ambiguous-return soft cost
      + w_risk   * CVaR_alpha over sampled human futures of a clearance cost
      + w_ctrl   * control effort and smoothness (energy proxy)

The human term uses M sampled futures per tracked person (from the tracker's
posterior) and takes the Conditional Value-at-Risk over those futures, so the
controller is conservative against the worst (1-alpha) tail of plausible
human motion rather than the mean prediction.

Outputs include the chosen command and risk diagnostics consumed by the
safety supervisor: minimum predicted clearance and collision probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medortrace.common.geometry import wrap_angle
from medortrace.planning.costmap import Costmap


@dataclass
class MpcResult:
    v: float
    omega: float
    min_pred_clearance: float
    collision_prob: float
    cvar: float
    feasible: bool
    best_traj: np.ndarray


class MppiController:
    def __init__(self, horizon: int = 20, dt: float = 0.1, samples: int = 120, lam: float = 1.0,
                 sigma=(0.25, 0.6), alpha: float = 0.8, weights: dict | None = None,
                 human_radius: float = 0.3, max_v: float = 0.7, max_w: float = 1.2):
        self.H, self.dt, self.K = horizon, dt, samples
        self.lam = lam
        self.sigma = np.array(sigma)
        self.alpha = alpha
        w = {"track": 3.0, "heading": 0.5, "obs": 6.0, "soft": 0.4, "risk": 8.0, "ctrl": 0.2, "smooth": 0.05,
             "progress": 2.0}
        w.update(weights or {})
        self.w = w
        self.r_h = human_radius
        self.max_v, self.max_w = max_v, max_w
        self.U = np.zeros((horizon, 2))

    def rollout(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """U: (K,H,2) -> states (K,H,3)."""
        K = U.shape[0]
        X = np.zeros((K, self.H, 3))
        s = np.repeat(x0[None], K, 0).astype(float)
        for h in range(self.H):
            v, w = U[:, h, 0], U[:, h, 1]
            s = s + np.stack([v * np.cos(s[:, 2]) * self.dt, v * np.sin(s[:, 2]) * self.dt, w * self.dt], 1)
            X[:, h] = s
        return X

    def solve(self, x0: np.ndarray, path: np.ndarray, cm: Costmap, human_samples: np.ndarray,
              robot_radius: float, rng: np.random.Generator, speed_scale: float = 1.0,
              goal_heading: float | None = None) -> MpcResult:
        H, K = self.H, self.K
        vmax = self.max_v * speed_scale
        # temporally correlated exploration noise: sample 5 knots and interpolate
        knots = rng.normal(0, 1, (K, 5, 2)) * self.sigma
        noise = np.einsum("hk,nkd->nhd", self._interp_matrix(H, 5), knots)
        U = self.U[None] + noise
        # a few straight "go" candidates towards the path direction help cold starts
        U[2:6, :, 0] = np.linspace(0.25, 1.0, 4)[:, None] * vmax
        U[0] = self.U                         # keep the nominal sequence
        U[1] = 0.0                            # always consider stopping
        U[:, :, 0] = np.clip(U[:, :, 0], -0.15, vmax)
        U[:, :, 1] = np.clip(U[:, :, 1], -self.max_w, self.max_w)
        X = self.rollout(x0, U)
        xy = X[:, :, :2]
        cost = np.zeros(K)
        # --- path tracking --------------------------------------------------
        if path is not None and len(path):
            # distances to the path *polyline* (smoothed paths keep only corner
            # vertices, so vertex distances would penalise the straight runs)
            dproj, rem = polyline_projection(xy, path)                          # K,H
            cost += self.w["track"] * dproj.mean(axis=1)
            goal = path[-1]
            start_goal = np.linalg.norm(x0[:2] - goal)
            # progress = cost-to-go along the path from the end of the horizon
            # (distance back to the path + remaining arc length).  Euclidean
            # distance to the goal would pull the rollouts straight at it and
            # fight the path wherever it detours (kick buckets, sterile
            # keep-out) until the robot stalls.
            cost += self.w["progress"] * (dproj[:, -1] + rem[:, -1])
            if goal_heading is not None and start_goal < 0.4:
                cost += self.w["heading"] * np.abs(wrap_angle(X[:, -1, 2] - goal_heading))
            elif len(path) > 1:
                look = lookahead_point(x0[:2], path, 0.8)
                desired = np.arctan2(look[1] - x0[1], look[0] - x0[0])
                cost += 0.3 * self.w["heading"] * np.abs(wrap_angle(X[:, :5, 2] - desired)).mean(axis=1)
        # --- static obstacles / keep-out ---------------------------------
        edt = cm.lookup(xy, "edt")
        keep = cm.lookup(xy, "keepout")
        soft = cm.lookup(xy, "soft")
        clearance_static = edt - robot_radius
        # recovery-aware feasibility: if the robot already starts too close
        # (e.g. after localisation correction) allow motions that do not worsen it
        c0 = float(cm.lookup(x0[None, :2], "edt")[0]) - robot_radius
        thr = min(0.02, c0 - 0.02)
        in_keep0 = bool(cm.lookup(x0[None, :2], "keepout")[0])
        infeasible = (clearance_static < thr).any(axis=1)
        if not in_keep0:
            infeasible |= keep.any(axis=1)
        else:
            cost += 20.0 * keep.mean(axis=1)
        cost += self.w["obs"] * np.exp(-np.clip(clearance_static, 0, None) / 0.12).mean(axis=1)
        cost += self.w["soft"] * soft.mean(axis=1)
        cost += np.where(infeasible, 1e4, 0.0)
        # --- humans: CVaR over sampled futures ---------------------------
        min_clear = np.full(K, np.inf)
        coll_frac = np.zeros(K)
        cvar = np.zeros(K)
        if human_samples.size:
            T, M = human_samples.shape[:2]
            hs = human_samples[:, :, :H]                                         # T,M,H,2
            dd = np.linalg.norm(xy[:, None, None, :, :] - hs[None], axis=4)      # K,T,M,H
            c = dd - robot_radius - self.r_h
            per = np.exp(-np.clip(c, 0, None) / 0.35).sum(axis=3) + 50.0 * (c < 0.1).sum(axis=3)   # K,T,M
            per = per.sum(axis=1)                                                # K,M
            q = np.quantile(per, self.alpha, axis=1)
            tail = np.where(per >= q[:, None], per, np.nan)
            cvar = np.nanmean(tail, axis=1)
            cost += self.w["risk"] * cvar / H
            cm_ = c.min(axis=(1, 3))                                             # K,M
            min_clear = np.median(cm_, axis=1)
            coll_frac = (cm_ < 0.0).mean(axis=1)
        # --- control effort / smoothness (energy proxy) ------------------
        cost += self.w["ctrl"] * (U[:, :, 0] ** 2 + 0.3 * U[:, :, 1] ** 2).mean(axis=1)
        cost += self.w["smooth"] * (np.diff(U, axis=1) ** 2).sum(axis=(1, 2))
        # --- MPPI weighting (temperature adapted to the cost spread) ------
        beta = cost.min()
        finite = cost[cost < 1e4]
        lam = self.lam * max(1e-3, float(np.median(finite) - beta)) * 0.2 if len(finite) > 2 else self.lam
        wts = np.exp(-(cost - beta) / lam)
        wts /= wts.sum()
        U_new = (wts[:, None, None] * U).sum(axis=0)
        # never execute an infeasible average: fall back to best feasible sample
        Xn = self.rollout(x0, U_new[None])
        ok_new = not ((cm.lookup(Xn[0, :, :2], "edt") - robot_radius < thr).any()
                      or (not in_keep0 and cm.lookup(Xn[0, :, :2], "keepout").any()))
        best = int(np.argmin(cost))
        if not ok_new:
            U_new = U[best]
            Xn = X[best:best + 1]
        feasible = bool(cost[best] < 1e4)
        if not feasible:
            U_new = np.zeros_like(U_new)
            Xn = self.rollout(x0, U_new[None])
        self.U = np.vstack([U_new[1:], U_new[-1:]])
        return MpcResult(float(U_new[0, 0]), float(U_new[0, 1]),
                         float(min_clear[best]) if np.isfinite(min_clear[best]) else 10.0,
                         float(coll_frac[best]), float(cvar[best]) if human_samples.size else 0.0,
                         feasible, Xn[0])

    @staticmethod
    def _interp_matrix(H: int, n_knots: int) -> np.ndarray:
        tk = np.linspace(0, H - 1, n_knots)
        M = np.zeros((H, n_knots))
        for h in range(H):
            j = min(int(np.searchsorted(tk, h, side="right")) - 1, n_knots - 2)
            w = (h - tk[j]) / (tk[j + 1] - tk[j])
            M[h, j], M[h, j + 1] = 1 - w, w
        return M

    def reset(self) -> None:
        self.U[:] = 0.0


def polyline_projection(pts: np.ndarray, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point projection of ``pts`` (...,2) onto the polyline ``path`` (P,2).

    Returns (distance to the polyline, remaining arc length from the projected
    point to the end of the path), both shaped like ``pts[..., 0]``.
    """
    if len(path) == 1:
        d = np.linalg.norm(pts - path[0], axis=-1)
        return d, np.zeros_like(d)
    a, ab = path[:-1], np.diff(path, axis=0)                                # S,2
    L2 = (ab ** 2).sum(axis=1)
    seg = np.sqrt(L2)
    rem_after = np.cumsum(seg[::-1])[::-1] - seg                           # arc length after segment i
    ap = pts[..., None, :] - a                                              # ...,S,2
    tt = np.clip((ap * ab).sum(axis=-1) / np.maximum(L2, 1e-12), 0.0, 1.0)  # ...,S
    dist = np.linalg.norm(ap - tt[..., None] * ab, axis=-1)                 # ...,S
    k = dist.argmin(axis=-1)
    dmin = np.take_along_axis(dist, k[..., None], -1)[..., 0]
    tk = np.take_along_axis(tt, k[..., None], -1)[..., 0]
    return dmin, rem_after[k] + (1.0 - tk) * seg[k]


def lookahead_point(x: np.ndarray, path: np.ndarray, dist: float) -> np.ndarray:
    """Point ``dist`` metres further along ``path`` than the projection of ``x``."""
    if len(path) == 1:
        return path[0]
    _, rem0 = polyline_projection(x[None], path)
    target = max(float(rem0[0]) - dist, 0.0)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    rem_v = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]                          # remaining arc at each vertex
    i = int(np.searchsorted(-rem_v, -target, side="left"))                  # first vertex with rem <= target
    if i == 0:
        return path[0]
    if i >= len(path):
        return path[-1]
    f = (rem_v[i - 1] - target) / max(rem_v[i - 1] - rem_v[i], 1e-12)
    return path[i - 1] + f * (path[i] - path[i - 1])
