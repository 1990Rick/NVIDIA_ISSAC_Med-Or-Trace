"""Active perception: next-best-view selection.

Candidate viewpoints are sampled around slots whose items matter now
(pending claims, high entropy, high criticality) and around high-uncertainty
map regions.  Each admissible candidate (free, outside sterile keep-out,
reachable, not crowding people) is scored as

    V(c) = w_eig    * sum_i crit_i * urgency_i * EIG_i(c)       (item custody)
         + w_modal  * EIG over hidden slots via radar/acoustic  (non-visual)
         + w_unc    * uncertainty mass in the camera frustum     (map)
         + w_watch  * sum_s lambda_s * pd_s(c)                   (anticipation)
         - w_path   * path length  - w_turn * |heading change|
         - w_risk   * human-proximity risk at the viewpoint (now)
         - w_traffic* staff traffic density at the viewpoint (time-averaged)
         - w_sterile* proximity to the sterile boundary

The anticipation term values viewpoints that keep the slots where the next
hand-offs are likely in view: lambda_s is the expected rate of logged events at
slot s (a workflow prior decayed towards the rate observed in the log so far,
maintained by the stack) and pd_s(c) the predicted detection probability from
the candidate.  Without it the planner is myopic - it chases open claims and
is not in position when the next hand-off happens (a parked camera with an
overview of the field then verifies more hand-offs).  Overview candidates on a
ring around the activity centroid are added to the sampled viewpoints.

EIG_i is the exact expected entropy reduction of item i's categorical belief
for a binary detect / no-detect outcome at each visible slot (summed over
slots, an upper bound that is cheap and monotone).  Weights are loaded from a
YAML policy file and can be optimised with ``scripts/train_nbv_policy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medortrace.belief.items import ItemBelief
from medortrace.belief.occupancy import OccupancyBelief
from medortrace.perception.visibility import VisibilityModel
from medortrace.planning.costmap import Costmap
from medortrace.planning.grid import dijkstra_field, extract_path, nearest_free, smooth_path

DEFAULT_WEIGHTS = {"w_eig": 4.0, "w_modal": 2.0, "w_unc": 0.02, "w_path": 0.5, "w_turn": 0.15,
                   "w_risk": 2.0, "w_sterile": 0.5, "w_urgent": 3.0, "w_commit": 1.0, "w_watch": 1.5,
                   "w_traffic": 3.0}


@dataclass
class ViewGoal:
    pose: np.ndarray               # x, y, yaw
    path: np.ndarray               # (P,2)
    score: float
    target_slot: str | None
    probe_region: str | None
    breakdown: dict


def _entropy(b):
    b = np.clip(b, 1e-9, 1)
    return -(b * np.log2(b)).sum(axis=-1)


def expected_info_gain(B: np.ndarray, pd: np.ndarray, fp: float = 0.02) -> np.ndarray:
    """B: (I,S) item beliefs; pd: (I,S) detection prob per item/slot -> (I,) summed EIG."""
    I, S = B.shape
    H0 = _entropy(B)
    gain = np.zeros(I)
    for s in np.where((pd > 0.03).any(axis=0))[0]:
        L1 = np.full((I, S), fp)
        L1[:, s] = pd[:, s] + fp
        L0 = 1 - L1
        P1 = (B * L1).sum(1)
        P0 = 1 - P1
        post1 = B * L1 / P1[:, None]
        post0 = B * L0 / np.maximum(P0[:, None], 1e-9)
        gain += H0 - (P1 * _entropy(post1) + P0 * _entropy(post0))
    return np.maximum(gain, 0)


class NextBestView:
    def __init__(self, vis: VisibilityModel, weights: dict | None = None, robot_radius: float = 0.28,
                 n_per_target: int = 8, n_explore: int = 8, modalities=("lidar", "camera", "radar", "acoustic")):
        self.vis = vis
        self.w = dict(DEFAULT_WEIGHTS)
        self.w.update(weights or {})
        self.r = robot_radius
        self.n_t = n_per_target
        self.n_e = n_explore
        self.modalities = set(modalities)
        self.person_gate = 1.1          # m, centre-to-centre

    def plan(self, pose: np.ndarray, belief: ItemBelief, occ: OccupancyBelief, cm: Costmap,
             people_xy: np.ndarray, urgency_items: dict[str, float], urgent_slots: dict[str, float],
             rng: np.random.Generator, haze: float = 0.0, incumbent: "ViewGoal | None" = None,
             activity: dict[str, float] | None = None, traffic: np.ndarray | None = None) -> ViewGoal | None:
        slots = belief.slots
        S = len(slots)
        item_ids = list(belief.items)
        B = belief.matrix()
        crit = np.array([belief.items[i].spec.criticality for i in item_ids])
        urg = np.array([1.0 + urgency_items.get(i, 0.0) for i in item_ids])
        H = _entropy(B)
        # slot value: probability mass of uncertain / urgent items
        slot_val = ((crit * urg * np.maximum(H, 0.15))[:, None] * B).sum(0)
        for sid, u in urgent_slots.items():
            if sid in belief.sidx:
                slot_val[belief.sidx[sid]] += self.w["w_urgent"] * u
        for k, s in enumerate(slots):
            if s.kind == "elsewhere" or not np.all(np.isfinite(s.position)):
                slot_val[k] = 0.0
        targets = [k for k in np.argsort(-slot_val)[:6] if slot_val[k] > 1e-3]
        cands = []
        for k in targets:
            s = slots[k]
            # open containers: the camera (h=1.45 m, pitch -25 deg, vfov 65 deg) must look
            # down steeper than 38 deg but stay inside the frustum -> ~0.75-1.4 m standoff
            rmin, rmax = (0.75, 1.4) if s.needs_top_view else (0.9, 2.4)
            if s.hidden_from_camera:
                rmin, rmax = (0.9, 3.0)
            for _ in range(self.n_t):
                a = rng.uniform(-np.pi, np.pi)
                rr = rng.uniform(rmin, rmax)
                p = s.position[:2] + rr * np.array([np.cos(a), np.sin(a)])
                yaw = np.arctan2(s.position[1] - p[1], s.position[0] - p[0])
                cands.append((p, yaw, k))
        unc = occ.uncertainty_field() + 0.0
        flat = unc.ravel()
        if flat.sum() > 0:
            pick = rng.choice(len(flat), size=self.n_e, p=flat / flat.sum())
            for c in pick:
                ij = np.array(np.unravel_index(c, unc.shape))
                p = occ.grid2d.cell_to_world(ij)[0]
                off = rng.normal(0, 1.0, 2)
                q = p + off
                cands.append((q, np.arctan2(p[1] - q[1], p[0] - q[0]), None))
        lam = np.zeros(S)
        for sid, rate in (activity or {}).items():
            if sid in belief.sidx:
                lam[belief.sidx[sid]] = rate
        if lam.sum() > 0:
            # overview viewpoints on a ring around the activity centroid, facing it
            pos = np.array([s.position[:2] if np.all(np.isfinite(s.position[:2])) else [np.nan, np.nan]
                            for s in slots])
            ok = np.isfinite(pos[:, 0]) & (lam > 0)
            ctr = (lam[ok, None] * pos[ok]).sum(0) / lam[ok].sum()
            for a in np.linspace(-np.pi, np.pi, 8, endpoint=False) + rng.uniform(0, np.pi / 4):
                for rr in (2.2, 2.8):
                    q = ctr + rr * np.array([np.cos(a), np.sin(a)])
                    cands.append((q, float(np.arctan2(ctr[1] - q[1], ctr[0] - q[0])), None))
        cands.append((pose[:2].copy(), pose[2], None))            # staying is always an option
        inc_idx = None
        if incumbent is not None:
            # hysteresis: the current goal competes with a commitment bonus
            k_inc = belief.sidx.get(incumbent.target_slot) if incumbent.target_slot else None
            cands.append((incumbent.pose[:2].copy(), incumbent.pose[2], k_inc))
            inc_idx = len(cands) - 1
        # --- admissibility ------------------------------------------------
        adm = []
        for ci_, (p, yaw, k) in enumerate(cands):
            is_inc = ci_ == inc_idx
            if not cm.grid.in_bounds(p[None])[0]:
                continue
            if cm.lookup(p[None], "lethal")[0] or cm.lookup(p[None], "edt")[0] < self.r + 0.1:
                continue
            # a viewpoint must not put the robot inside the supervisor's hard human-
            # clearance limit (0.45 m surface-to-surface = 0.98 m centre-to-centre)
            if len(people_xy) and np.min(np.linalg.norm(people_xy - p, axis=1)) < self.person_gate:
                continue
            adm.append((p, yaw, k, is_inc))
        if not adm:
            return None
        # --- scoring --------------------------------------------------------
        start = nearest_free(cm.lethal, tuple(cm.grid.world_to_cell(pose[None, :2])[0]))
        if start is None:
            return None
        dist, parent = dijkstra_field(cm.soft * 0.3, cm.lethal, start)
        best = None
        best_goal_cell = None
        cls_list = [belief.items[i].spec.cls for i in item_ids]
        metallic = np.array([belief.items[i].spec.metallic for i in item_ids])
        for p, yaw, k, is_inc in adm:
            goal = nearest_free(cm.lethal, tuple(cm.grid.world_to_cell(p[None])[0]))
            if goal is None or not np.isfinite(dist[goal]):
                continue
            plen = float(dist[goal]) * cm.grid.res
            if "camera" in self.modalities:
                pd_by_cls = self.vis.slot_pd_classes(p, yaw, people_xy, occ, set(cls_list), haze=haze)
                pdm = np.array([pd_by_cls[c] for c in cls_list])
            else:
                pdm = np.zeros((len(item_ids), S))
            eig = expected_info_gain(B, pdm)
            v_eig = float((crit * urg * eig).sum())
            v_watch = float((lam * pdm.max(axis=0)).sum()) if lam.any() else 0.0
            # non-visual modalities for hidden slots (radar/acoustic: metallic only)
            pdx = np.zeros((len(item_ids), S))
            probe = None
            for j, s in enumerate(slots):
                if not s.hidden_from_camera or s.kind == "elsewhere" or not np.all(np.isfinite(s.position)):
                    continue
                r = np.linalg.norm(s.position[:2] - p)
                if "acoustic" in self.modalities and s.acoustic_region and r < 3.2:
                    pdx[:, j] = np.maximum(pdx[:, j], np.where(metallic, 0.7, 0.03) * (1 - r / 3.5))
                    probe = s.acoustic_region
                if "radar" in self.modalities and s.kind == "under_drape" and r < 6.0:
                    pdx[:, j] = np.maximum(pdx[:, j], np.where(metallic, 0.3, 0.0))
            v_modal = float((crit * urg * expected_info_gain(B, pdx)).sum()) if pdx.any() else 0.0
            # exploration: uncertainty mass near the candidate that is *new*
            # relative to what the robot can already sense from where it is
            v_unc = self._frustum_uncertainty(p, yaw, unc, occ, exclude_center=pose[:2])
            turn = abs((yaw - pose[2] + np.pi) % (2 * np.pi) - np.pi)
            risk = 0.0
            if len(people_xy):
                dmin = np.min(np.linalg.norm(people_xy - p, axis=1))
                risk = float(np.exp(-(dmin - self.person_gate) / 0.4))
                # straight-line corridor to the viewpoint passing close to people
                seg = np.linspace(pose[:2], p, 8)
                dpath = np.min(np.linalg.norm(seg[:, None, :] - people_xy[None], axis=2))
                risk += float(np.exp(-(dpath - 0.7) / 0.3))
            # expected staff presence at the viewpoint over the recent past: a robot
            # parked in a walkway is approached and forces people around it
            traffic_at = 0.0
            if traffic is not None:
                ci = cm.grid.world_to_cell(p[None])[0]
                if 0 <= ci[0] < traffic.shape[0] and 0 <= ci[1] < traffic.shape[1]:
                    traffic_at = float(min(traffic[ci[0], ci[1]], 1.0))
            dz = float(cm.lookup(p[None], "zone_dist")[0])
            sterile_pen = float(np.exp(-(dz - 0.5) / 0.3))
            score = (self.w["w_eig"] * v_eig + self.w["w_modal"] * v_modal + self.w["w_unc"] * v_unc
                     + self.w.get("w_watch", 0.0) * v_watch
                     - self.w["w_path"] * plen - self.w["w_turn"] * turn - self.w["w_risk"] * risk
                     - self.w.get("w_traffic", 0.0) * traffic_at
                     - self.w["w_sterile"] * sterile_pen) + (self.w["w_commit"] if is_inc else 0.0)
            if best is None or score > best.score:
                best_goal_cell = goal
                best = ViewGoal(np.array([p[0], p[1], yaw]), np.zeros((0, 2)), score,
                                slots[k].id if k is not None else None,
                                probe if v_modal > 1e-3 else None,
                                {"eig": v_eig, "modal": v_modal, "unc": v_unc, "watch": v_watch, "path": plen,
                                 "risk": risk, "traffic": traffic_at,
                                 "sterile": sterile_pen})
        if best is not None:
            cells = extract_path(parent, best_goal_cell)
            path = cm.grid.cell_to_world(np.array(cells))
            path = np.vstack([pose[None, :2], path, best.pose[None, :2]])
            best.path = smooth_path(path, cm.lethal, cm.grid)
        return best

    def _frustum_uncertainty(self, p, yaw, unc, occ, exclude_center=None, exclude_r: float = 3.0) -> float:
        g = occ.grid2d
        c = g.world_to_cell(p[None])[0]
        r = int(3.0 / g.res)
        i0, i1 = max(0, c[0] - r), min(g.shape[0], c[0] + r)
        j0, j1 = max(0, c[1] - r), min(g.shape[1], c[1] + r)
        sub = unc[i0:i1:2, j0:j1:2]
        if sub.size == 0:
            return 0.0
        I, J = np.meshgrid(np.arange(i0, i1, 2), np.arange(j0, j1, 2), indexing="ij")
        xy = g.cell_to_world(np.stack([I.ravel(), J.ravel()], 1)).reshape(I.shape + (2,))
        d = xy - p
        ang = np.abs((np.arctan2(d[..., 1], d[..., 0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        m = (ang < np.pi / 4) & (np.linalg.norm(d, axis=-1) < 3.0)
        if exclude_center is not None:
            m &= np.linalg.norm(xy - exclude_center, axis=-1) > exclude_r
        return float(sub[m].sum())
