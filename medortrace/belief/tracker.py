"""Multi-target tracking of people (and moving carts) from lidar clusters and
Doppler radar, with identity maintenance for staff roles.

Constant-velocity Kalman filters; radar updates use the nonlinear measurement
[x, y, radial_velocity] (EKF), which lets the tracker estimate velocity from a
single radar frame - valuable when the camera is blinded or lidar is dropped.
Tracks are predicted forward as *samples* for the risk-aware MPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    x: np.ndarray                 # [px, py, vx, vy]
    P: np.ndarray
    last_t: float
    hits: int = 1
    identity: str | None = None
    last_seen: float = 0.0
    max_speed: float = 0.0
    history: list = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    @property
    def person_like(self) -> bool:
        """Identity-matched staff or anything that has been seen moving."""
        return self.identity is not None or self.max_speed > 0.25


class PeopleTracker:
    def __init__(self, staff_homes: dict[str, np.ndarray] | None = None, q_acc: float = 1.2,
                 r_lidar: float = 0.12, r_radar_pos: float = 0.25, r_radar_vel: float = 0.08,
                 gate: float = 11.8, max_age: float = 1.5):
        self.tracks: list[Track] = []
        self.q = q_acc
        self.r_l = r_lidar
        self.r_rp = r_radar_pos
        self.r_rv = r_radar_vel
        self.gate = gate
        self.max_age = max_age
        self._next = 0
        self.staff_homes = staff_homes or {}
        self.sterile_names: set[str] = set()
        self.last_t = 0.0

    def predict(self, t: float) -> None:
        for tr in self.tracks:
            dt = t - tr.last_t
            if dt <= 0:
                continue
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + G @ (np.eye(2) * self.q ** 2) @ G.T
            tr.last_t = t
            sp = np.linalg.norm(tr.x[2:])
            if sp > 2.0:                       # people rarely exceed 2 m/s indoors
                tr.x[2:] *= 2.0 / sp
        self.tracks = [tr for tr in self.tracks if tr.P[0, 0] < 4.0]
        self.last_t = t

    def _new(self, xy: np.ndarray, t: float, v: np.ndarray | None = None) -> None:
        P = np.diag([0.1, 0.1, 1.0, 1.0])
        x = np.array([xy[0], xy[1], 0.0, 0.0]) if v is None else np.array([xy[0], xy[1], v[0], v[1]])
        self.tracks.append(Track(self._next, x, P, t, last_seen=t))
        self._next += 1

    def _after_update(self, tr: Track, t: float) -> None:
        tr.hits += 1
        tr.last_seen = t
        if tr.hits > 4:
            tr.max_speed = max(tr.max_speed, float(np.linalg.norm(tr.x[2:])))

    def update_positions(self, meas: np.ndarray, t: float, r: float | None = None) -> None:
        """Associate position measurements (M,2) with tracks (Hungarian, Mahalanobis gate)."""
        if len(meas) == 0:
            return
        r = self.r_l if r is None else r
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
        R = np.eye(2) * r ** 2
        n, m = len(self.tracks), len(meas)
        used = set()
        if n:
            C = np.full((n, m), 1e6)
            for i, tr in enumerate(self.tracks):
                S = H @ tr.P @ H.T + R
                d = meas - tr.x[:2]
                C[i] = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
            ri, ci = linear_sum_assignment(np.minimum(C, 1e6))
            for i, j in zip(ri, ci):
                if C[i, j] > self.gate:
                    continue
                tr = self.tracks[i]
                S = H @ tr.P @ H.T + R
                K = tr.P @ H.T @ np.linalg.inv(S)
                tr.x = tr.x + K @ (meas[j] - tr.x[:2])
                tr.P = (np.eye(4) - K @ H) @ tr.P
                self._after_update(tr, t)
                used.add(j)
        for j in range(m):
            if j not in used and all(np.linalg.norm(meas[j] - tr.x[:2]) > 0.6 for tr in self.tracks):
                self._new(meas[j], t)

    def update_radar(self, dets_world: list[tuple[np.ndarray, float, np.ndarray]], t: float,
                     robot_vel_world: np.ndarray) -> None:
        """dets: (xy, radial_velocity_rel, unit_los(2)).  EKF with range-rate."""
        for xy, vr, u in dets_world:
            best, bd = None, self.gate
            for tr in self.tracks:
                S = tr.P[:2, :2] + np.eye(2) * self.r_rp ** 2
                d = xy - tr.x[:2]
                md = float(d @ np.linalg.solve(S, d))
                if md < bd:
                    best, bd = tr, md
            v_abs = vr + float(robot_vel_world @ u)
            if best is None:
                if abs(v_abs) > 0.15 and all(np.linalg.norm(xy - tr.x[:2]) > 0.6 for tr in self.tracks):
                    self._new(xy, t, v=u * v_abs)
                continue
            H = np.zeros((3, 4))
            H[0, 0] = H[1, 1] = 1
            H[2, 2:] = u
            z = np.array([xy[0], xy[1], v_abs])
            R = np.diag([self.r_rp ** 2, self.r_rp ** 2, self.r_rv ** 2])
            S = H @ best.P @ H.T + R
            K = best.P @ H.T @ np.linalg.inv(S)
            best.x = best.x + K @ (z - H @ best.x)
            best.P = (np.eye(4) - K @ H) @ best.P
            self._after_update(best, t)

    def prune(self, t: float) -> None:
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_age]
        self.assign_identities()

    def assign_identities(self, sterile: set[str] | None = None) -> None:
        """Keep role identities (surgeon, circulator, ...) by Hungarian matching.

        Scrubbed staff are anchored to their stations (0.8 m gate around the
        station); roaming staff are matched to their last tracked position.
        """
        if not self.staff_homes or not self.tracks:
            return
        sterile = self.sterile_names if sterile is None else sterile
        names = list(self.staff_homes)
        prev = {tr.identity: tr for tr in self.tracks if tr.identity}
        ref = np.array([prev[nm].x[:2] if (nm in prev and nm not in sterile) else self.staff_homes[nm]
                        for nm in names])
        gate = np.array([0.8 if nm in sterile else 1.5 for nm in names])
        C = np.linalg.norm(np.array([tr.x[:2] for tr in self.tracks])[:, None, :] - ref[None], axis=2)
        Cg = np.where(C < gate[None], C, 1e3)
        ri, ci = linear_sum_assignment(Cg)
        for tr in self.tracks:
            tr.identity = None
        for i, j in zip(ri, ci):
            if Cg[i, j] < 1e3:
                self.tracks[i].identity = names[j]
                if names[j] not in sterile:
                    self.staff_homes[names[j]] = self.tracks[i].x[:2].copy()

    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]

    def people(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed and t.person_like]

    def predict_samples(self, horizon: int, dt: float, n_samples: int, rng: np.random.Generator,
                        accel_sigma: float = 0.8, tracks: list[Track] | None = None) -> np.ndarray:
        """(T, S, H, 2) sampled future positions of (person-like) confirmed tracks."""
        trs = self.people() if tracks is None else tracks
        if not trs:
            return np.zeros((0, n_samples, horizon, 2))
        out = np.zeros((len(trs), n_samples, horizon, 2))
        for k, tr in enumerate(trs):
            x0 = rng.multivariate_normal(tr.x, tr.P + np.eye(4) * 1e-6, size=n_samples)
            p = x0[:, :2].copy()
            v = x0[:, 2:].copy()
            acc = rng.normal(0, accel_sigma * dt, (horizon,) + v.shape)
            vs = v[None] + np.cumsum(acc, axis=0)
            out[k] = np.transpose(p[None] + np.cumsum(vs * dt, axis=0), (1, 0, 2))
        return out
