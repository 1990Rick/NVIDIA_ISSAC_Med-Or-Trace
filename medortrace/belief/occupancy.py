"""Probabilistic 3D/4D occupancy belief.

Three co-registered layers over a voxel grid (default 0.1 m):

* ``static``    - log-odds occupancy of structure, initialised from the prior
                  map at *moderate* confidence (maps can be stale), updated by
                  lidar ray casting and relaxed back towards the prior with a
                  long time constant (the 4th, temporal dimension);
* ``ambiguous`` - evidence from returns suspected to be multipath ghosts or
                  haze; kept separate so it is *visible to the planner* as an
                  uncertainty (treated as soft cost, never silently dropped);
* ``dynamic``   - a 2D layer of recently observed moving-object occupancy with
                  a short time constant (people, carts in motion).

Entropy of the static layer plus ambiguous mass forms the *uncertainty field*
used by active perception and published on ROS 2.
"""

from __future__ import annotations

import numpy as np

from medortrace.planning.grid import GridSpec


def logit(p):
    return np.log(p / (1 - p))


def binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class OccupancyBelief:
    FLOOR_Z = 0.03
    L_OCC = 0.85
    L_FREE = -0.4
    L_MIN, L_MAX = -4.0, 4.0

    def __init__(self, room: tuple[float, float, float], res: float = 0.1, z_max: float = 2.0,
                 tau_static: float = 120.0, tau_dynamic: float = 2.0, tau_ambiguous: float = 20.0,
                 deterministic: bool = False):
        W, D, _ = room
        self.res = res
        self.nx, self.ny, self.nz = int(np.ceil(W / res)), int(np.ceil(D / res)), int(np.ceil(z_max / res))
        self.grid2d = GridSpec(np.array([0.0, 0.0]), res, (self.nx, self.ny))
        self.static = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.prior = np.zeros_like(self.static)
        self.ambiguous = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.dynamic = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.observed = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        self.tau_s, self.tau_d, self.tau_a = tau_static, tau_dynamic, tau_ambiguous
        self.deterministic = deterministic

    # ------------------------------------------------------------------
    def set_prior_from_boxes(self, boxes, l_occ: float = 1.2, l_free: float = -1.5) -> None:
        cx = (np.arange(self.nx) + 0.5) * self.res
        cy = (np.arange(self.ny) + 0.5) * self.res
        cz = (np.arange(self.nz) + 0.5) * self.res
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel()], 1)
        prior = np.full(self.static.shape, l_free, dtype=np.float32)
        for b in boxes:
            inside = (b.distance_xy(pts) <= self.res * 0.5).reshape(self.nx, self.ny)
            zmask = (cz >= b.z_min - self.res * 0.5) & (cz <= b.z_max + self.res * 0.5)
            prior[inside[:, :, None] & zmask[None, None, :]] = l_occ
        self.prior = prior
        self.static = prior.copy()
        self._prior_version = getattr(self, "_prior_version", 0) + 1

    def _idx(self, pts: np.ndarray):
        ijk = np.floor(pts / self.res).astype(int)
        ok = (ijk[:, 0] >= 0) & (ijk[:, 1] >= 0) & (ijk[:, 2] >= 0) & \
             (ijk[:, 0] < self.nx) & (ijk[:, 1] < self.ny) & (ijk[:, 2] < self.nz)
        return ijk, ok

    # ------------------------------------------------------------------
    def integrate_scan(self, origin: np.ndarray, endpoints: np.ndarray, hit_weight: np.ndarray,
                       ghost_prob: np.ndarray, dynamic_mask: np.ndarray, free_rays: np.ndarray | None = None,
                       carve_max: float = 8.0, step: float = 0.15, carve_limit: np.ndarray | None = None) -> None:
        """Update from world-frame endpoints.

        ``hit_weight`` in [0,1] scales the occupied update (1 - ghost prob),
        ``dynamic_mask`` routes points to the dynamic layer, ``free_rays`` are
        world directions of max-range (no-return) rays used for carving only.
        """
        segs = [endpoints]
        if free_rays is not None and len(free_rays):
            segs.append(origin + free_rays * carve_max)
        allend = np.concatenate(segs, 0) if len(segs) > 1 else endpoints
        d = allend - origin
        L = np.linalg.norm(d, axis=1)
        u = d / (L[:, None] + 1e-9)
        Lc = np.minimum(L - self.res, carve_max)
        if carve_limit is not None:
            # suspected multipath: carve free space only up to the mirror surface
            Lc[: len(endpoints)] = np.minimum(Lc[: len(endpoints)], carve_limit - self.res)
        n_s = int(np.ceil(carve_max / step))
        ts = (np.arange(n_s) + 0.5) * step
        samp = origin + u[:, None, :] * ts[None, :, None]
        valid = ts[None, :] < Lc[:, None]
        # ghost-suspected rays still carve up to the *mirror surface*, not to the ghost
        pts = samp[valid]
        ijk, ok = self._idx(pts)
        ijk = ijk[ok]
        if len(ijk):
            np.add.at(self.static, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), self.L_FREE)
            self.observed[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
            # looking *through* a suspected ghost location resolves it as free
            self.ambiguous[ijk[:, 0], ijk[:, 1]] *= np.float32(0.8)
        # occupied endpoints
        n_end = len(endpoints)
        if n_end:
            ijk, ok = self._idx(endpoints)
            # floor returns are free-space evidence only; at the lidar's grazing angles
            # floor points scatter by millimetres, so 3 cm keeps low obstacles (a fallen
            # IV pole is 6 cm tall) out of the floor class
            floor = endpoints[:, 2] < self.FLOOR_Z
            st = ok & ~dynamic_mask & ~floor
            w = hit_weight[st]
            np.add.at(self.static, (ijk[st, 0], ijk[st, 1], ijk[st, 2]), (self.L_OCC * w).astype(np.float32))
            self.observed[ijk[st, 0], ijk[st, 1], ijk[st, 2]] = True
            amb = ok & (ghost_prob > 0.2) & ~floor
            np.add.at(self.ambiguous, (ijk[amb, 0], ijk[amb, 1]), ghost_prob[amb].astype(np.float32) * 0.5)
            dy = ok & dynamic_mask
            np.add.at(self.dynamic, (ijk[dy, 0], ijk[dy, 1]), 0.6)
        np.clip(self.static, self.L_MIN, self.L_MAX, out=self.static)
        np.clip(self.ambiguous, 0, 3, out=self.ambiguous)
        np.clip(self.dynamic, 0, 3, out=self.dynamic)

    def decay(self, dt: float) -> None:
        """Temporal relaxation (4D belief): evidence ages towards the prior."""
        a = np.float32(np.exp(-dt / self.tau_s))
        self.static = self.prior + (self.static - self.prior) * a
        self.dynamic *= np.float32(np.exp(-dt / self.tau_d))
        self.ambiguous *= np.float32(np.exp(-dt / self.tau_a))

    # ------------------------------------------------------------------
    def prob(self) -> np.ndarray:
        p = 1 / (1 + np.exp(-self.static))
        if self.deterministic:
            return (p > 0.5).astype(np.float32)
        return p

    def column_occupancy(self, z_lo: float = 0.1, z_hi: float = 1.6) -> np.ndarray:
        """Max occupancy over the voxel layers covering [z_lo, z_hi).  The lowest layer
        (0-res) is included only when z_lo < res: floor returns never mark it occupied
        (FLOOR_Z), so there it holds genuine low obstacles only."""
        k0, k1 = max(0, int(np.floor(z_lo / self.res + 1e-9))), int(np.ceil(z_hi / self.res))
        return self.prob()[:, :, k0:k1].max(axis=2)

    def uncertainty_field(self, z_lo: float = 0.1, z_hi: float = 1.6) -> np.ndarray:
        """2D map in bits: entropy of the column occupancy (max over the robot's
        height band) plus the ambiguous (ghost-suspect) mass, capped at 1 each."""
        if self.deterministic:
            return np.zeros((self.nx, self.ny), dtype=np.float32)
        p = self.column_occupancy(z_lo, z_hi)
        return (binary_entropy(p) + np.clip(self.ambiguous, 0, 1)).astype(np.float32)

    def excess_uncertainty_field(self, z_lo: float = 0.1, z_hi: float = 1.6) -> np.ndarray:
        """2D map in bits of uncertainty *beyond the surveyed prior*: column entropy
        minus the prior's column entropy (clipped at 0) plus the ambiguous mass.

        A voxel the lidar has never observed (e.g. low voxels next to the robot,
        below the lowest ring) keeps the prior's entropy - H(sigmoid(l_free)) =
        0.68 bits for surveyed free space - which is expected, not a hazard.
        Conflicting evidence (p -> 0.5) or ghost-suspect returns raise it.
        """
        if self.deterministic:
            return np.zeros((self.nx, self.ny), dtype=np.float32)
        k0, k1 = max(1, int(np.ceil(z_lo / self.res - 1e-9))), int(np.ceil(z_hi / self.res))
        key = (k0, k1)
        if getattr(self, "_prior_col_key", None) != (key, getattr(self, "_prior_version", 0)):
            p0 = (1 / (1 + np.exp(-self.prior[:, :, k0:k1]))).max(axis=2)
            self._prior_col_h = binary_entropy(p0)
            self._prior_col_key = (key, getattr(self, "_prior_version", 0))
        h = binary_entropy(self.column_occupancy(z_lo, z_hi))
        return (np.clip(h - self._prior_col_h, 0, 1) + np.clip(self.ambiguous, 0, 1)).astype(np.float32)

    def unknown_fraction(self) -> float:
        return float(1.0 - self.observed[:, :, 1:16].mean())

    def blocked_segments(self, a: np.ndarray, b: np.ndarray, p_block: float = 0.7, step: float = 0.1) -> np.ndarray:
        """Probability-thresholded voxel ray march along segments a->b (N,3)."""
        d = b - a
        L = np.linalg.norm(d, axis=1)
        n = int(np.ceil(max(L.max() if len(L) else 0, step) / step))
        ts = (np.arange(1, n) * step)
        pts = a[:, None, :] + (d / (L[:, None] + 1e-9))[:, None, :] * ts[None, :, None]
        valid = ts[None, :] < (L[:, None] - 0.25)
        ijk, ok = self._idx(pts.reshape(-1, 3))
        ok = ok & valid.ravel()
        blk = np.zeros(len(ok), dtype=bool)
        idx = np.where(ok)[0]
        if len(idx):
            lo = self.static[ijk[idx, 0], ijk[idx, 1], ijk[idx, 2]]
            blk[idx] = lo > logit(p_block)
        return blk.reshape(len(a), -1).any(axis=1)
