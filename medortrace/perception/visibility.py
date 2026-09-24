"""Belief-side sensor visibility model.

Predicts, from the robot's *estimated* state and *believed* geometry (prior
map + lidar-built occupancy + tracked people), the probability that each slot
would yield a detection for a given camera pose.  The same function is used
(a) to interpret an actual frame (negative evidence only where the slot was
predicted visible), and (b) by active perception to score candidate views.
Keeping them identical is what makes negative evidence *defensible*.
"""

from __future__ import annotations

import numpy as np

from medortrace.belief.occupancy import OccupancyBelief
from medortrace.sim.raycast import RayScene, segments_blocked
from medortrace.world.scene import SceneObject, Slot

CLASS_SIZE = {"sponge": 0.1, "clamp": 0.16, "needle_driver": 0.18, "specimen": 0.08, "implant_box": 0.25}


class VisibilityModel:
    def __init__(self, prior_map: list[SceneObject], slots: list[Slot], cam_cfg: dict, room_height: float = 3.0):
        self.objs = [o for o in prior_map]
        self.slots = slots
        self.idx = {o.name: i for i, o in enumerate(self.objs)}
        self.box_c = np.array([o.box.center for o in self.objs])
        self.box_h = np.array([o.box.half for o in self.objs])
        self.box_y = np.array([o.box.yaw for o in self.objs])
        self.h_cam = float(cam_cfg.get("mount_height", 1.45))
        self.pitch = np.deg2rad(float(cam_cfg.get("pitch_deg", -25.0)))
        self.hfov = np.deg2rad(float(cam_cfg.get("hfov_deg", 90.0)))
        self.vfov = np.deg2rad(float(cam_cfg.get("vfov_deg", 65.0)))
        self.max_range = float(cam_cfg.get("max_range", 5.0))
        self.pd0 = float(cam_cfg.get("pd0", 0.92))
        self.ceiling = room_height
        self.anchor_excl = []
        for s in slots:
            ex = np.zeros(len(self.objs), dtype=bool)
            if s.anchor in self.idx:
                ex[self.idx[s.anchor]] = True
            if s.anchor == "back_table" and "back_table_drape" in self.idx:
                ex[self.idx["back_table_drape"]] = True
            self.anchor_excl.append(ex)

    def scene(self, people_xy: np.ndarray) -> RayScene:
        n = len(people_xy)
        return RayScene(self.box_c, self.box_h, self.box_y, people_xy.reshape(-1, 2), np.full(n, 0.25),
                        np.full(n, 1.75), self.ceiling)

    def slot_visibility(self, cam_xy: np.ndarray, cam_yaw: float, people_xy: np.ndarray,
                        occ: OccupancyBelief | None = None, hand_owner: dict[int, int] | None = None) -> np.ndarray:
        """Geometric visible fraction per slot in [0,1] (0 outside FOV / hidden)."""
        S = len(self.slots)
        vis = np.zeros(S)
        cam = np.array([cam_xy[0], cam_xy[1], self.h_cam])
        cand, pts, owners = [], [], []
        for k, s in enumerate(self.slots):
            if s.kind == "elsewhere" or s.hidden_from_camera or not np.all(np.isfinite(s.position)):
                continue
            d = s.position - cam
            r = float(np.linalg.norm(d))
            if r > self.max_range or r < 0.2:
                continue
            b = (np.arctan2(d[1], d[0]) - cam_yaw + np.pi) % (2 * np.pi) - np.pi
            e = np.arctan2(d[2], np.hypot(d[0], d[1]))
            if abs(b) > self.hfov / 2 or abs(e - self.pitch) > self.vfov / 2:
                continue
            if s.needs_top_view and e > np.deg2rad(-38.0):
                continue
            for off in ((0, 0, 0.03), (0.05, 0.0, 0.03), (-0.05, 0.0, 0.03)):
                pts.append(s.position + np.array(off))
                owners.append(k)
            cand.append(k)
        if not cand:
            return vis
        pts = np.array(pts)
        owners = np.array(owners)
        scn = self.scene(people_xy)
        own = np.full(len(pts), -99)
        for k in cand:
            m = owners == k
            a_idx = np.where(self.anchor_excl[k])[0]
            if len(a_idx):
                own[m] = a_idx[0]
            if hand_owner and k in hand_owner and hand_owner[k] is not None:
                own[m] = len(self.objs) + hand_owner[k]
        blocked = segments_blocked(scn, np.repeat(cam[None], len(pts), 0), pts, own=own)
        if occ is not None:
            blocked |= occ.blocked_segments(np.repeat(cam[None], len(pts), 0), pts)
        for k in cand:
            vis[k] = 1.0 - blocked[owners == k].mean()
        return vis

    def slot_pd_classes(self, cam_xy, cam_yaw, people_xy, occ, classes, hand_owner=None, haze: float = 0.0):
        """One visibility computation, per-class detection probabilities."""
        vis = self.slot_visibility(cam_xy, cam_yaw, people_xy, occ, hand_owner)
        cam = np.array([cam_xy[0], cam_xy[1], self.h_cam])
        pos = np.array([s.position if np.all(np.isfinite(s.position)) else [1e3, 1e3, 0] for s in self.slots])
        r = np.linalg.norm(pos - cam, axis=1)
        base = self.pd0 * vis * np.clip(1.2 - r / self.max_range, 0.1, 1.0) * (1 - haze)
        return {c: base * np.clip(CLASS_SIZE.get(c, 0.1) / 0.08, 0.5, 1.0) for c in classes}

    def slot_pd(self, cam_xy, cam_yaw, people_xy, occ=None, hand_owner=None, cls: str = "sponge",
                haze: float = 0.0) -> np.ndarray:
        vis = self.slot_visibility(cam_xy, cam_yaw, people_xy, occ, hand_owner)
        cam = np.array([cam_xy[0], cam_xy[1], self.h_cam])
        pos = np.array([s.position if np.all(np.isfinite(s.position)) else [1e3, 1e3, 0] for s in self.slots])
        r = np.linalg.norm(pos - cam, axis=1)
        size = CLASS_SIZE.get(cls, 0.1)
        rf = np.clip(1.2 - r / self.max_range, 0.1, 1.0) * np.clip(size / 0.08, 0.5, 1.0)
        return self.pd0 * vis * rf * (1 - haze)
