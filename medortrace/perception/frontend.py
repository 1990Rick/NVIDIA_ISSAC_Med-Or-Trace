"""Perception front end: raw sensor messages -> typed proposals.

* Lidar: world-frame points, multipath/haze ghost probability from prior-map
  reasoning, map-residual (change) points, person-candidate clusters.
* Camera: calibrated class posteriors (temperature scaling of the learned
  detector's logits), world positions, slot association.
* Radar: moving detections for tracking; static low-RCS returns for the
  metallic-item-behind-fabric channel.

In Isaac Sim the "learned" parts are real networks trained on Replicator
data (see scripts/train_detector.py); here the interfaces are identical and
the lite simulator emits the logits of a surrogate model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from medortrace.common.msgs import CameraFrame, LidarScan, RadarFrame
from medortrace.planning.grid import GridSpec, rasterize_boxes
from medortrace.sim.raycast import NO_HIT, RayScene, cast
from medortrace.world.materials import MATERIALS
from medortrace.world.scene import SceneObject, Slot

CLASSES = ["sponge", "clamp", "needle_driver", "specimen", "implant_box"]


def softmax(z, T=1.0):
    z = np.asarray(z) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class LidarProposal:
    origin: np.ndarray
    points: np.ndarray             # (N,3) world
    ghost_prob: np.ndarray         # (N,)
    carve_limit: np.ndarray        # (N,) metres
    residual: np.ndarray           # (N,) bool: unexplained by prior map
    person_clusters: np.ndarray    # (M,2) centroids
    cluster_labels: np.ndarray     # (N,) -1 or cluster id
    free_dirs: np.ndarray          # (F,3) world dirs of no-return rays
    gt_ghost: np.ndarray | None = None


class LidarFrontEnd:
    def __init__(self, prior_map: list[SceneObject], room: tuple[float, float, float], use_ghost_reasoning: bool = True):
        self.objs = prior_map
        self.scene = RayScene(np.array([o.box.center for o in prior_map]), np.array([o.box.half for o in prior_map]),
                              np.array([o.box.yaw for o in prior_map]), np.zeros((0, 2)), np.zeros(0), np.zeros(0),
                              room[2])
        self.spec = np.array([MATERIALS[o.material].specularity for o in prior_map])
        self.trans = np.array([MATERIALS[o.material].transmissivity for o in prior_map])
        self.is_wall = np.array([o.kind == "wall" for o in prior_map])
        self.room = room
        self.use_ghost = use_ghost_reasoning
        W, D, _ = room
        self.g = GridSpec(np.array([0.0, 0.0]), 0.05, (int(np.ceil(W / 0.05)), int(np.ceil(D / 0.05))))
        occ = rasterize_boxes(self.g, [o.box for o in prior_map], inflate=self.g.res, z_band=(0.05, 1.8))
        self.map_edt = distance_transform_edt(~occ) * self.g.res

    def rebuild(self, prior_map: list[SceneObject]) -> None:
        self.__init__(prior_map, self.room, self.use_ghost)

    def process(self, scan: LidarScan, pose: np.ndarray, mount_x: float = 0.1) -> LidarProposal:
        x, y, th = pose
        c, s = np.cos(th), np.sin(th)
        origin = np.array([x + mount_x * c, y + mount_x * s, scan.sensor_height])
        fin = np.isfinite(scan.ranges)
        d_s = scan.directions
        d_w = np.stack([c * d_s[:, 0] - s * d_s[:, 1], s * d_s[:, 0] + c * d_s[:, 1], d_s[:, 2]], 1)
        rng = scan.ranges[fin]
        dirs = d_w[fin]
        pts = origin + dirs * rng[:, None]
        n = len(pts)
        ghost = np.zeros(n)
        carve = rng.copy()
        W, D, H = self.room
        outside = (pts[:, 0] < -0.15) | (pts[:, 1] < -0.15) | (pts[:, 0] > W + 0.15) | (pts[:, 1] > D + 0.15) | (pts[:, 2] < -0.15)
        if self.use_ghost and n:
            h = cast(self.scene, np.repeat(origin[None], n, 0), dirs, t_max=25.0)
            behind = np.isfinite(h.t) & (rng > h.t + 0.3) & (h.obj >= 0)
            oi = np.where(h.obj >= 0, h.obj, 0)
            p_spec = np.clip(self.spec[oi] * 1.1, 0, 0.95)
            p = np.where(self.is_wall[oi], 0.95, np.maximum(p_spec, 0.5 * (1 - self.trans[oi])))
            ghost = np.where(behind, p, 0.0)
            carve = np.where(behind, h.t, rng)
            # floor mirror (wet floor): points below the floor plane
            below = pts[:, 2] < -0.05
            ghost = np.where(below, 0.95, ghost)
        ghost = np.where(outside, 1.0, ghost)
        # map residual: static-looking points not explained by the prior map
        cell = self.g.world_to_cell(pts[:, :2])
        dmap = self.map_edt[cell[:, 0], cell[:, 1]]
        band = (pts[:, 2] > 0.1) & (pts[:, 2] < 1.8)
        residual = band & (dmap > 0.2) & (ghost < 0.5) & ~outside
        # person candidates: residual points clustered on a 0.15 m grid
        labels = -np.ones(n, dtype=int)
        cents = []
        if residual.any():
            cg = 0.15
            ij = np.floor(pts[residual, :2] / cg).astype(int)
            ij -= ij.min(0)
            shape = ij.max(0) + 1
            img = np.zeros(shape, dtype=bool)
            img[ij[:, 0], ij[:, 1]] = True
            lab, nlab = label(img, structure=np.ones((3, 3)))
            pl = lab[ij[:, 0], ij[:, 1]] - 1
            ridx = np.where(residual)[0]
            for k in range(nlab):
                m = pl == k
                if m.sum() < 4:
                    continue
                P = pts[ridx[m]]
                ext = P[:, :2].max(0) - P[:, :2].min(0)
                if ext.max() > 1.1:
                    continue
                labels[ridx[m]] = len(cents)
                cents.append(P[:, :2].mean(0))
        return LidarProposal(origin, pts, ghost, carve, residual, np.array(cents).reshape(-1, 2), labels,
                             d_w[~fin], scan.gt_is_ghost[fin] if scan.gt_is_ghost is not None else None)


@dataclass
class CameraProposal:
    world: np.ndarray              # (N,3)
    probs: np.ndarray              # (N,C) calibrated
    slot_idx: np.ndarray           # (N,) associated slot or -1
    tag: list


class CameraFrontEnd:
    def __init__(self, slots: list[Slot], temperature: float = 1.0, mount_height: float = 1.45):
        self.slots = slots
        self.T = temperature
        self.h = mount_height

    def process(self, frame: CameraFrame, pose: np.ndarray) -> CameraProposal:
        x, y, th = pose
        cam = np.array([x, y, self.h])
        n = len(frame.detections)
        W = np.zeros((n, 3))
        P = np.zeros((n, len(CLASSES)))
        tags = []
        for k, d in enumerate(frame.detections):
            az = th + d.bearing
            W[k] = cam + d.range * np.array([np.cos(d.elevation) * np.cos(az), np.cos(d.elevation) * np.sin(az),
                                             np.sin(d.elevation)])
            P[k] = softmax(d.logits, self.T)
            tags.append(d.item_id_hint)
        pos = np.array([s.position if np.all(np.isfinite(s.position)) else [1e3, 1e3, 1e3] for s in self.slots])
        rad = np.array([s.radius for s in self.slots])
        idx = -np.ones(n, dtype=int)
        for k in range(n):
            dist = np.linalg.norm(pos[:, :2] - W[k, :2], axis=1) + 0.5 * np.abs(pos[:, 2] - W[k, 2])
            tol = rad + 0.15 + 0.04 * frame.detections[k].range
            ok = dist < tol
            for j, s in enumerate(self.slots):
                if s.hidden_from_camera:
                    ok[j] = False
            if ok.any():
                idx[k] = int(np.argmin(np.where(ok, dist / tol, np.inf)))
        return CameraProposal(W, P, idx, tags)


def radar_to_world(frame: RadarFrame, pose: np.ndarray, mount_height: float = 0.6):
    x, y, th = pose
    out = []
    for d in frame.detections:
        az = th + d.azimuth
        u = np.array([np.cos(az), np.sin(az)])
        xy = np.array([x, y]) + d.range * np.cos(d.elevation) * u
        z = mount_height + d.range * np.sin(d.elevation)
        out.append((xy, z, d.radial_velocity, u, d.rcs_dbsm))
    return out
