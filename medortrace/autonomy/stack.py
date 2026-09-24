"""The MED-OR-TRACE autonomy stack (backend- and middleware-agnostic).

    SensorBundle --> [time sync] --> [EKF localisation] --> pose history
                      |                                     |
                      v                                     v
                [lidar FE] --ghost/residual--> [3D/4D occupancy] --> costmap
                [lidar FE, radar] -----------> [people tracker] --> human futures
                [camera FE, radar, acoustic] -> [item custody belief] <-- [workflow log]
                                                     |
                           [provenance graph] <------+----> [claim verifier: VERIFIED/REFUTED/ABSTAIN]
                                                     |
             [change diagnoser (drift vs map change)]|
                                                     v
          [policy: active NBV | fixed route | passive] --> goal/path
                                                     v
                                  [risk-aware MPPI (CVaR over human futures)]
                                                     v
                                      [safety supervisor] --> VelocityCommand

``AutonomyStack.step(bundle)`` is called at the control rate by the episode
runner (lite or Isaac Sim) or by the ROS 2 node ``medortrace_ros/autonomy_node``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from medortrace.belief.items import ItemBelief
from medortrace.belief.localization import EkfLocalizer
from medortrace.belief.occupancy import OccupancyBelief
from medortrace.belief.scene_graph import build_scene_graph
from medortrace.belief.tracker import PeopleTracker
from medortrace.belief.world_model import ChangeDiagnoser
from medortrace.common.config import load_yaml
from medortrace.common.geometry import wrap_angle
from medortrace.common.msgs import SensorBundle, VelocityCommand, WorkflowEventType
from medortrace.perception.frontend import CLASSES, CameraFrontEnd, LidarFrontEnd, radar_to_world
from medortrace.perception.sync import PoseHistory, TimeSyncMonitor
from medortrace.perception.visibility import VisibilityModel
from medortrace.planning.costmap import Costmap
from medortrace.planning.mpc import MppiController
from medortrace.planning.nbv import NextBestView
from medortrace.planning.routes import fixed_route, passive_vantage
from medortrace.provenance.graph import ProvenanceGraph
from medortrace.provenance.verifier import ClaimVerifier
from medortrace.safety.supervisor import Envelope, Mode, SafetyInputs, SafetySupervisor
from medortrace.sim.raycast import RayScene, segment_occluded, segments_blocked
from medortrace.world.materials import MATERIALS
from medortrace.world.scene import ItemSpec, Landmark, SceneObject, Slot, StaffSpec, SterileZone
from medortrace.world.workflow import count_claims, handoff_claim


@dataclass
class StackInputs:
    """Everything the robot legitimately knows before the episode starts."""

    room: tuple[float, float, float]
    prior_map: list[SceneObject]
    slots: list[Slot]
    zones: list[SterileZone]
    landmarks: list[Landmark]
    items: list[ItemSpec]
    initial_placement: dict[str, str]       # pre-operative count sheet
    staff: list[StaffSpec]
    start_pose: np.ndarray
    dock: np.ndarray
    duration: float
    claim_grace: float = 25.0               # verification window of a handoff claim (s)
    count_grace: float = 90.0               # ... of a count claim (a surgical count takes minutes)


@dataclass
class StepTelemetry:
    t: float
    pose_est: np.ndarray
    pose_std: float
    nis: float
    mode: str
    cmd: tuple[float, float]
    mpc_min_clear: float
    mpc_coll_prob: float
    goal: np.ndarray | None
    target_slot: str | None
    probe: str | None
    item_entropy: dict[str, float]
    path_entropy: float
    n_tracks: int
    human_clearance_est: float
    verdicts: list = field(default_factory=list)
    ghost_stats: tuple[int, int, int, int] | None = None   # tp, fp, fn, tn (supervision only)


class AutonomyStack:
    SUSPECT_HOLD_S = 15.0      # displaced-anchor cue: withhold negative camera evidence this long

    def __init__(self, inputs: StackInputs, cfg: dict):
        self.cfg = cfg
        a = cfg.get("autonomy", {})
        self.policy = a.get("policy", "active")
        self.modalities = set(a.get("modalities", ["lidar", "camera", "radar", "acoustic"]))
        self.use_log = a.get("use_workflow_log", True)
        self.inp = inp = copy.deepcopy(inputs)
        rcfg = cfg.get("robot", {})
        self.radius = float(rcfg.get("footprint_radius", 0.28))
        self.height = float(rcfg.get("height_m", 1.55))
        self.battery_cap = float(rcfg.get("battery_capacity_wh", 480.0))
        sc = cfg.get("sensors", {})
        self.cam_cfg = sc.get("camera", {})
        self.lidar_mount_x = float(sc.get("lidar", {}).get("mount_x", 0.1))
        self.radar_h = float(sc.get("radar", {}).get("mount_height", 0.6))
        # --- belief layers ------------------------------------------------
        self.sync = TimeSyncMonitor(enabled=a.get("use_time_sync", True))
        self.poses = PoseHistory()
        self.ekf = EkfLocalizer(inp.start_pose, {lm.id: lm.position for lm in inp.landmarks})
        self.occ = OccupancyBelief(inp.room, deterministic=a.get("deterministic_map", False),
                                   tau_static=120.0 if a.get("use_temporal_model", True) else 1e9)
        self.occ.set_prior_from_boxes([o.box for o in inp.prior_map])
        self.lidar_fe = LidarFrontEnd(inp.prior_map, inp.room, use_ghost_reasoning=a.get("use_ghost_reasoning", True))
        calib = {}
        try:
            calib = load_yaml(a.get("detector_calibration", "perception/detector_calibration.yaml"))
        except FileNotFoundError:
            pass
        self.cam_fe = CameraFrontEnd(inp.slots, temperature=float(calib.get("temperature", 1.0)),
                                     mount_height=float(self.cam_cfg.get("mount_height", 1.45)))
        homes = {s.name: s.home.copy() for s in inp.staff}
        self.tracker = PeopleTracker(staff_homes=homes)
        self.sterile_staff = {s.name for s in inp.staff if s.sterile}
        self.tracker.sterile_names = self.sterile_staff
        self.dwell_s = float(a.get("viewpoint_dwell_s", 2.0))
        self.use_scan_matching = a.get("use_scan_matching", True)
        self._arrived_t = None
        self.items = ItemBelief(
            inp.items,
            inp.slots,
            inp.initial_placement,
            params={
                **cfg.get("belief", {}),
                "use_temporal_model": a.get("use_temporal_model", True),
                **({"soft_confusion": calib["soft_confusion"]} if "soft_confusion" in calib else {}),
            },
        )
        self.prov = ProvenanceGraph()
        self.verifier = ClaimVerifier(self.items, self.prov, **cfg.get("verifier", {}))
        self.diag = ChangeDiagnoser(inp.prior_map)
        self._movable = {o.name for o in inp.prior_map if o.movable}
        self._suspect_slot_t: dict[int, float] = {}   # slot index -> last displaced-anchor cue
        self.vis = VisibilityModel(inp.prior_map, inp.slots, self.cam_cfg, inp.room[2])
        nbv_w = {}
        wpath = a.get("nbv_weights")
        if wpath:
            try:
                nbv_w = load_yaml(wpath).get("weights", {})
            except FileNotFoundError:
                nbv_w = {}
        nbv_w.update(a.get("nbv_weight_overrides", {}))
        self.nbv = NextBestView(self.vis, nbv_w, self.radius, modalities=self.modalities)
        self.mpc = MppiController(max_v=float(rcfg.get("max_v", 0.7)), max_w=float(rcfg.get("max_omega", 1.2)),
                                  weights=cfg.get("mpc", {}).get("weights"))
        self.sup = SafetySupervisor(Envelope.from_dict(cfg.get("safety", {}).get("envelope")),
                                    enabled=a.get("use_safety_supervisor", True))
        self.rng = np.random.default_rng(int(cfg.get("_policy_seed", 0)))
        # --- runtime state --------------------------------------------------
        self.t = 0.0
        self.last_lidar_t = 0.0
        self.cm: Costmap | None = None
        self.cm_t = -1e9
        self.goal = None
        self.goal_t = -1e9
        self.route: list[np.ndarray] = []
        self.route_i = 0
        self.probe = None
        self.pending_counts: list = []
        self.log_seen: list = []
        self.telemetry: list[StepTelemetry] = []
        self.operator_request_open = False
        self.docking = False
        self._ghost_conf = np.zeros(4, dtype=int)
        self._seq = 0
        self._path_entropy = 0.0
        self._last_retreat = None
        self.belief_history: list = []
        self.belief_history_t: list = []
        self._prior_scene = RayScene(np.array([o.box.center for o in inp.prior_map]),
                                     np.array([o.box.half for o in inp.prior_map]),
                                     np.array([o.box.yaw for o in inp.prior_map]),
                                     np.zeros((0, 2)), np.zeros(0), np.zeros(0), inp.room[2])
        self.metal_prior_idx = {o.name: i for i, o in enumerate(inp.prior_map)}

    # ==================================================================
    def step(self, b: SensorBundle, dt: float) -> VelocityCommand:
        self.t = t = b.t
        verdicts = []
        # ---------------- localisation ------------------------------------
        self.ekf.predict(b.odom, b.imu, dt)
        if b.landmarks is not None:
            tl = self.sync.correct("landmarks", b.landmarks.header.stamp, b.landmarks.header.recv_stamp)
            if tl is not None:
                self.ekf.update_landmarks(b.landmarks, t)
        pose = self.ekf.x.copy()
        self.poses.add(t, pose)
        # ---------------- lidar -------------------------------------------------
        if b.lidar is not None and "lidar" in self.modalities:
            tm = self.sync.correct("lidar", b.lidar.header.stamp, b.lidar.header.recv_stamp)
            if tm is not None:
                p_meas = self.poses.at(tm)
                lp = self.lidar_fe.process(b.lidar, p_meas, self.lidar_mount_x)
                self.tracker.predict(t)
                persons = self._person_candidates(lp)
                self.tracker.update_positions(persons, t)
                dyn = self._dynamic_mask(lp)
                w = 1.0 - lp.ghost_prob
                self.occ.integrate_scan(lp.origin, lp.points, w, lp.ghost_prob, dyn, lp.free_dirs[::3],
                                        carve_limit=lp.carve_limit)
                self.last_lidar_t = t
                self._lidar_frames = getattr(self, "_lidar_frames", 0) + 1
                if self._lidar_frames % 2 == 0:   # 2.5 Hz scan-to-map consistency check
                    static = ~dyn & (lp.ghost_prob < 0.5)
                    dg = self.diag.observe(t, lp.points[static], lp.residual[static], p_meas,
                                           self.lidar_fe.map_edt, self.lidar_fe.g, self.ekf.nis_avg, self.rng)
                    if dg is not None:
                        self._act_on_diagnosis(dg)
                    corr = self.diag.pose_correction(pose)
                    recent_change = any(d.cause == "map_change" and t - d.t < 10 for d in self.diag.diagnoses)
                    if corr is not None and self.use_scan_matching and not recent_change:
                        self.ekf.update_pose(corr, np.diag([0.06, 0.06, 0.025]) ** 2)
                if lp.gt_ghost is not None:
                    pred = lp.ghost_prob > 0.5
                    g = lp.gt_ghost.astype(bool)
                    self._ghost_conf += np.array(
                        [(pred & g).sum(), (pred & ~g).sum(), (~pred & g).sum(), (~pred & ~g).sum()]
                    )
                if self._seq % 5 == 0:
                    self.prov.add_evidence(
                        f"lidar:{b.lidar.header.seq}",
                        t,
                        "lidar",
                        {
                            "n_points": len(lp.points),
                            "n_ghost_suspect": int((lp.ghost_prob > 0.5).sum()),
                            "n_residual": int(lp.residual.sum()),
                        },
                        pose,
                    )
        self.occ.decay(dt)
        # ---------------- radar --------------------------------------------------
        metal_hits = np.zeros(len(self.inp.slots))
        radar_pd = np.zeros(len(self.inp.slots))
        if b.radar is not None and "radar" in self.modalities:
            tm = self.sync.correct("radar", b.radar.header.stamp, b.radar.header.recv_stamp)
            if tm is not None:
                p_meas = self.poses.at(tm)
                rv = b.odom.v * np.array([np.cos(pose[2]), np.sin(pose[2])]) if b.odom else np.zeros(2)
                dets = radar_to_world(b.radar, p_meas, self.radar_h)
                moving = [(xy, vr, u) for xy, z, vr, u, rcs in dets if abs(vr + rv @ u) > 0.15 and rcs > -8]
                self.tracker.predict(t)
                self.tracker.update_radar(moving, t, rv)
                static_metal = [(xy, z) for xy, z, vr, u, rcs in dets if abs(vr + rv @ u) <= 0.15 and -20 < rcs < -4]
                radar_pd, metal_hits = self._radar_slot_evidence(p_meas, static_metal)
        self.tracker.prune(t)
        tracks = self.tracker.confirmed()
        people_xy = self._occluding_people(tracks)
        self.items.update_hand_positions({tr.identity: tr.x[:2] for tr in tracks if tr.identity})
        # ---------------- item belief: predict + workflow --------------------
        self.items.predict(dt)
        for ev in b.workflow:
            self._on_workflow(ev, t)
        for due, ev in list(self.pending_counts):
            if t >= due:
                for c in count_claims(ev, [e for e in self.log_seen if e.type != WorkflowEventType.COUNT],
                                      self.inp.initial_placement, self.inp.duration, self.inp.count_grace):
                    self.verifier.add_claim(c)
                self.pending_counts.remove((due, ev))
        # ---------------- camera -------------------------------------------
        loc_ok = self.ekf.pos_std < 0.3
        if b.camera is not None and "camera" in self.modalities and loc_ok:
            tm = self.sync.correct("camera", b.camera.header.stamp, b.camera.header.recv_stamp)
            if tm is not None:
                self._camera_update(b, self.poses.at(tm), people_xy, tracks, t)
        if radar_pd.any():
            eid = f"radar:{b.radar.header.seq}"
            self.prov.add_evidence(eid, t, "radar", {"metal_hits": metal_hits.tolist()}, pose)
            self.items.update_radar_fabric(radar_pd, metal_hits, eid, t)
        if b.acoustic is not None and "acoustic" in self.modalities:
            self._acoustic_update(b, pose, t)
        # ---------------- verification --------------------------------------
        self.verifier.degraded = self.sup.degraded() and self.sup.mode != Mode.CAUTION
        verdicts = self.verifier.step(t)
        # ---------------- planning & control ---------------------------------
        if self.cm is None or t - self.cm_t > 0.5:
            self.cm = Costmap(self.occ, self.inp.zones, self.radius, robot_height=self.height)
            self.cm_t = t
        cmd = self._plan_and_control(pose, people_xy, tracks, b, t, dt)
        # ---------------- telemetry ------------------------------------------
        if self._seq % 10 == 0:
            self.belief_history.append(self.items.matrix().astype(np.float32))
            self.belief_history_t.append(t)
        self._seq += 1
        ent = {i: self.items.entropy(i) for i in self.items.items}
        hc = self._human_clearance(pose, tracks)
        self.telemetry.append(StepTelemetry(
            t, pose.copy(), self.ekf.pos_std, self.ekf.nis_avg, self.sup.mode.value, (cmd.v, cmd.omega),
            self._last_mpc.min_pred_clearance if self._last_mpc else 10.0,
            self._last_mpc.collision_prob if self._last_mpc else 0.0,
            None if self.goal is None else self.goal.pose.copy(), getattr(self.goal, "target_slot", None),
            cmd.acoustic_probe_target, ent, self._path_entropy, len(tracks), hc, verdicts))
        return cmd

    # ==================================================================
    def _person_candidates(self, lp) -> np.ndarray:
        cents = lp.person_clusters
        if len(cents) == 0:
            return cents
        keep = ~self.inp_keepout_core(cents) | self._near_sterile_staff(cents)
        return cents[keep]

    def inp_keepout_core(self, xy):
        m = np.zeros(len(xy), dtype=bool)
        for z in self.inp.zones:
            m |= z.box.contains_xy(xy, margin=-0.1)
        return m

    def _near_sterile_staff(self, xy):
        homes = np.array(
            [self.tracker.staff_homes[n] for n in self.sterile_staff if n in self.tracker.staff_homes]
        ).reshape(-1, 2)
        if len(homes) == 0:
            return np.zeros(len(xy), dtype=bool)
        return np.min(np.linalg.norm(xy[:, None] - homes[None], axis=2), axis=1) < 0.6

    def _occluding_people(self, tracks) -> np.ndarray:
        """Tracked people plus the scrubbed team at their stations (prior knowledge:
        scrubbed staff stand at the table even when the tracker cannot see them)."""
        # person_like: has an identity or has *ever* moved like a person - a
        # nurse standing still is still a person (a cart that never moves is not)
        sel = [tr for tr in tracks if tr.person_like]
        pts = [tr.x[:2] for tr in sel]
        names = [tr.identity for tr in sel]
        for nm in self.sterile_staff:
            h = self.tracker.staff_homes.get(nm)
            if h is not None and nm not in names and all(np.linalg.norm(h - q) > 0.5 for q in pts):
                pts.append(h)
                names.append(nm)
        self._people_names = names
        return np.array(pts).reshape(-1, 2)

    def _dynamic_mask(self, lp) -> np.ndarray:
        dyn = np.zeros(len(lp.points), dtype=bool)
        if not len(lp.person_clusters):
            return dyn
        trs = self.tracker.tracks
        for k, c in enumerate(lp.person_clusters):
            for tr in trs:
                if np.linalg.norm(tr.x[:2] - c) < 0.6 and tr.person_like:
                    dyn |= lp.cluster_labels == k
                    break
        return dyn

    def _human_clearance(self, pose, tracks) -> float:
        people = [tr for tr in tracks if tr.person_like]
        if not people:
            return 10.0
        return float(min(np.linalg.norm(tr.x[:2] - pose[:2]) for tr in people) - self.radius - 0.25)

    # ------------------------------------------------------------------
    def _on_workflow(self, ev, t: float) -> None:
        self.log_seen.append(ev)
        eid = f"wf:{ev.event_id}"
        self.prov.add_evidence(eid, t, "workflow", {"type": ev.type.value, "item": ev.item_id, "src": ev.src,
                                                    "dst": ev.dst, "reported_t": ev.t, "confidence": ev.confidence},
                               attributed_to=ev.reporter)
        if ev.type == WorkflowEventType.COUNT:
            self.pending_counts.append((t + 3.0, ev))
            return
        if ev.item_id:
            if self.use_log:
                self.items.apply_workflow_event(ev.item_id, ev.src, ev.dst, ev.confidence, eid, t)
            self.verifier.note_item_event(ev.item_id, ev.t)
        c = handoff_claim(ev, self.inp.duration, self.inp.claim_grace)
        if c:
            self.verifier.add_claim(c)

    def _camera_update(self, b, p_meas, people_xy, tracks, t) -> None:
        cp = self.cam_fe.process(b.camera, p_meas)
        S = len(self.inp.slots)
        hand_owner = {}
        names = list(getattr(self, "_people_names", []))
        for k, s in enumerate(self.inp.slots):
            if s.kind == "hand":
                hand_owner[k] = names.index(s.anchor) if s.anchor in names else None
                if hand_owner[k] is None:
                    s.position = np.array([np.nan, np.nan, np.nan])
        classes = {st.spec.cls for st in self.items.items.values()}
        haze_est = 0.05
        pdc = self.vis.slot_pd_classes(p_meas[:2], p_meas[2], people_xy, self.occ, classes, hand_owner, haze_est)
        # displaced-anchor guard: a confident detection outside every slot's
        # association gate but close to a slot on a *movable* prior object (a
        # cart moved since the survey) means that slot's surveyed position may
        # be stale.  Until the change diagnoser re-anchors the object, "nothing
        # seen at the surveyed position" is not evidence of absence there.
        for k in np.flatnonzero(cp.slot_idx < 0):
            if cp.probs[k].max() < 0.5 and not cp.tag[k]:
                continue
            for j, s in enumerate(self.inp.slots):
                if s.anchor not in self._movable or s.hidden_from_camera or not np.all(np.isfinite(s.position)):
                    continue
                dxy = np.linalg.norm(s.position[:2] - cp.world[k, :2])
                if dxy < 1.2 and abs(s.position[2] - cp.world[k, 2]) < 0.35:
                    self._suspect_slot_t[j] = t
        for j, ts in self._suspect_slot_t.items():
            if t - ts < self.SUSPECT_HOLD_S:
                for c in pdc:
                    pdc[c][j] = 0.0
        counts = {c: np.zeros(S) for c in CLASSES}
        tag_reads = []
        for k in range(len(cp.slot_idx)):
            j = cp.slot_idx[k]
            if j < 0:
                continue
            for ci, c in enumerate(CLASSES):
                counts[c][j] += cp.probs[k, ci]
            if cp.tag[k] and cp.tag[k] in self.items.items:
                tag_reads.append((cp.tag[k], int(j)))
        glare = {i: 0.85 * MATERIALS[st.spec.material].glare for i, st in self.items.items.items()}
        eid = f"camera:{b.camera.header.seq}"
        self.prov.add_evidence(eid, t, "camera", {"n_det": len(cp.slot_idx), "assoc": [int(j) for j in cp.slot_idx],
                                                  "tags": [x for x in cp.tag if x]}, p_meas)
        self.items.update_camera_classes(pdc, counts, tag_reads, eid, t, glare)

    def _radar_slot_evidence(self, p_meas, static_metal):
        S = len(self.inp.slots)
        pd = np.zeros(S)
        hits = np.zeros(S)
        origin = np.array([p_meas[0], p_meas[1], self.radar_h])
        excl_base = np.array([MATERIALS[o.material].radar_penetrable for o in self.inp.prior_map])
        pos = np.array([s.position if np.all(np.isfinite(s.position)) else [1e3, 1e3, 1e3] for s in self.inp.slots])
        cand = []
        for k, s in enumerate(self.inp.slots):
            if s.kind not in ("under_drape", "floor") or not np.all(np.isfinite(s.position)):
                continue
            d = s.position - origin
            r = np.linalg.norm(d)
            az = wrap_angle(np.arctan2(d[1], d[0]) - p_meas[2])
            if r > 12.0 or abs(az) > np.deg2rad(60):
                continue
            cand.append((k, s, r))
        if cand:
            own = np.array([self.metal_prior_idx.get(s.anchor, -99) for k, s, r in cand])
            blk = segments_blocked(self._prior_scene, np.repeat(origin[None], len(cand), 0),
                                   np.array([s.position for k, s, r in cand]), own=own, exclude=excl_base,
                                   tol=0.15, own_tol=1.0)
            for (k, s, r), b in zip(cand, blk):
                if b:
                    continue
                snr = -12.0 - 40 * np.log10(max(r, 0.5)) + 35.0 - (6.0 if s.kind == "under_drape" else 0.0)
                pd[k] = min(1 / (1 + np.exp(-(snr - 3.0) / 2.0)), 0.95) / max(self.items.radar_pd, 1e-3)
        # associate each static metallic return to its nearest slot of any kind;
        # only returns whose nearest slot is an evidence slot count as hits
        for xy, _z in static_metal:
            dd = np.linalg.norm(pos[:, :2] - xy, axis=1)
            j = int(np.argmin(dd))
            if pd[j] > 0 and dd[j] < self.inp.slots[j].radius + 0.2:
                hits[j] += 1
        return pd, hits

    def _acoustic_update(self, b, pose, t) -> None:
        for echo in b.acoustic.echoes:
            region = echo.target_region
            ks = [k for k, s in enumerate(self.inp.slots) if s.acoustic_region == region]
            if not ks:
                continue
            s0 = self.inp.slots[ks[0]]
            origin = np.array([pose[0], pose[1], 1.2])
            r = float(np.linalg.norm(s0.position - origin))
            occl = bool(segment_occluded(self._prior_scene, origin[None], s0.position[None], tol=0.3)[0])
            anchor = next((o for o in self.inp.prior_map if o.name == s0.anchor), None)
            base = MATERIALS[anchor.material].acoustic_reflectivity if anchor else 0.5
            att = (1.0 / (1.0 + 0.3 * r * r)) * (0.45 if occl else 1.0)
            refl = {i: MATERIALS[st.spec.material].acoustic_reflectivity for i, st in self.items.items.items()}

            def expected(others, include=None, base=base, refl=refl, att=att):
                e = 0.15 * base + sum(0.6 * refl[j] * p for j, p in others.items())
                if include is not None:
                    e += 0.6 * refl[include]
                return e * att

            eid = f"acoustic:{b.acoustic.header.seq}"
            self.prov.add_evidence(eid, t, "acoustic", {"region": region, "energy": echo.energy,
                                                        "nlos": echo.path_occluded}, pose)
            old = self.items.ac_sigma
            self.items.ac_sigma = old * (1.8 if occl else 1.0)
            self.items.update_acoustic(ks, echo.energy, expected, eid, t)
            self.items.ac_sigma = old

    def _act_on_diagnosis(self, dg) -> None:
        if dg.cause == "map_change":
            if dg.object not in self.diag.applied:
                self.diag.reanchor(dg, self.inp.slots)
                self.lidar_fe.rebuild(self.inp.prior_map)
                self.vis = VisibilityModel(self.inp.prior_map, self.inp.slots, self.cam_cfg, self.inp.room[2])
                self.nbv.vis = self.vis
                self.occ.set_prior_from_boxes([o.box for o in self.inp.prior_map])
                # re-anchored slots are trusted again
                for j, sl in enumerate(self.inp.slots):
                    if sl.anchor == dg.object:
                        self._suspect_slot_t.pop(j, None)
                self.prov.add_evidence(f"diag:{len(self.diag.diagnoses)}", dg.t, "world_model",
                                       {"cause": "map_change", "object": dg.object, "p": dg.prob})
        elif dg.cause == "loc_drift":
            # admit that we are (partly) lost: inflate covariance so that scan
            # matching / landmarks re-anchor the estimate and the supervisor reacts
            self.ekf.P[:2, :2] += np.eye(2) * 0.004
            self.ekf.P[2, 2] += 0.001
            self.prov.add_evidence(f"diag:{len(self.diag.diagnoses)}", dg.t, "world_model",
                                   {"cause": "loc_drift", "p": dg.prob})

    # ------------------------------------------------------------------
    _last_mpc = None

    def _plan_and_control(self, pose, people_xy, tracks, b, t, dt) -> VelocityCommand:
        cm = self.cm
        batt_frac = (b.battery_wh or self.battery_cap) / self.battery_cap
        if batt_frac < 0.12 and not self.docking:
            self.docking = True
        at_goal = self.goal is not None and np.linalg.norm(self.goal.pose[:2] - pose[:2]) < 0.2
        if at_goal and self._arrived_t is None:
            self._arrived_t = t
        if not at_goal:
            self._arrived_t = None
        dwell_done = self._arrived_t is not None and t - self._arrived_t >= self.dwell_s
        replan = self.goal is None or dwell_done or (t - self.goal_t > 3.0 and not at_goal)
        if self.docking:
            if self.goal is None or self.goal.target_slot != "__dock__":
                self.goal = self._goal_from_pose(np.array([*self.inp.dock, 0.0]), pose, "__dock__")
                self.goal_t = t
        elif self.policy == "active":
            if replan:
                g = self.nbv.plan(pose, self.items, self.occ, cm, people_xy, self.verifier.urgency(t),
                                  self.verifier.urgent_slots(t), self.rng,
                                  incumbent=None if (self.goal is None or dwell_done) else self.goal)
                if g is not None:
                    if self.goal is None or g is not self.goal:
                        self._arrived_t = None
                    self.goal = g
                    self.goal_t = t
        elif self.policy == "fixed_route":
            if not self.route:
                self.route = fixed_route(self.inp.zones, self.inp.slots, cm)
            if self.route and (self.goal is None or np.linalg.norm(self.goal.pose[:2] - pose[:2]) < 0.25
                               or t - self.goal_t > 25.0):
                p = self.route[self.route_i % len(self.route)]
                self.route_i += 1
                self.goal = self._goal_from_pose(p, pose, None)
                self.goal_t = t
        elif self.policy == "passive":
            if self.goal is None:
                self.goal = self._goal_from_pose(passive_vantage(self.inp.zones, cm, self.inp.room), pose, None)
                self.goal_t = t
        # acoustic probing (active: planner's choice; fixed-route: nearest region in range)
        probe = None
        if "acoustic" in self.modalities:
            if self.policy == "active" and self.goal is not None:
                probe = self.goal.probe_region
            elif self.policy == "fixed_route":
                regs = [(np.linalg.norm(s.position[:2] - pose[:2]), s.acoustic_region) for s in self.inp.slots
                        if s.acoustic_region]
                regs = [r for r in regs if r[0] < 3.2]
                probe = min(regs)[1] if regs else None
        # path uncertainty ahead (for the supervisor)
        path = self.goal.path if self.goal is not None else np.array([pose[:2]])
        ahead = path[:6] if len(path) else np.array([pose[:2]])
        self._path_entropy = float(np.mean(self._unc_at(ahead, excess=True)))
        # MPC
        hs = self.tracker.predict_samples(self.mpc.H, self.mpc.dt, 12, self.rng)
        speed_scale = 1.0
        goal_heading = self.goal.pose[2] if self.goal is not None else None
        if self.goal is not None and np.linalg.norm(self.goal.pose[:2] - pose[:2]) < 0.2 and \
                abs(wrap_angle(goal_heading - pose[2])) < 0.2:
            res = None
            v, w = 0.0, 0.0
            if self.policy == "passive":
                w = 0.25 * np.sin(0.15 * t)   # slow pan to cover the field
        else:
            res = self.mpc.solve(pose, path, cm, hs, self.radius, self.rng, speed_scale, goal_heading)
            v, w = res.v, res.omega
            if self.goal is not None and np.linalg.norm(self.goal.pose[:2] - pose[:2]) < 0.2:
                v, w = 0.0, float(np.clip(1.5 * wrap_angle(goal_heading - pose[2]), -0.8, 0.8))
        self._last_mpc = res
        # safety supervisor
        si = SafetyInputs(t=t, collision_prob=res.collision_prob if res else 0.0,
                          pred_clearance=res.min_pred_clearance if res else 10.0,
                          human_clearance=self._human_clearance(pose, tracks), loc_std=self.ekf.pos_std,
                          nis=self.ekf.nis_avg, lidar_age=t - self.last_lidar_t if "lidar" in self.modalities else 0.0,
                          path_entropy=self._path_entropy, skew_uncorrectable=False,
                          contact_force=b.contact.force_n if b.contact else 0.0, battery_frac=batt_frac,
                          operator_ack=getattr(self, "_operator_ack", False))
        self._operator_ack = False
        mode = self.sup.update(si)
        if mode == Mode.HANDOVER:
            self.operator_request_open = True
        retreat = self._retreat_cmd(pose, tracks) if mode == Mode.RETREAT else None
        v, w = self.sup.gate(v, w, retreat)
        return VelocityCommand(float(v), float(w), acoustic_probe_target=probe)

    def _unc_at(self, xy, excess: bool = False):
        u = self.occ.excess_uncertainty_field() if excess else self.occ.uncertainty_field()
        c = self.occ.grid2d.world_to_cell(xy)
        return u[c[:, 0], c[:, 1]]

    def _goal_from_pose(self, p, pose, tag):
        from medortrace.planning.grid import dijkstra_field, extract_path, nearest_free, smooth_path
        from medortrace.planning.nbv import ViewGoal
        cm = self.cm
        s = nearest_free(cm.lethal, tuple(cm.grid.world_to_cell(pose[None, :2])[0]))
        g = nearest_free(cm.lethal, tuple(cm.grid.world_to_cell(p[None, :2])[0]))
        path = np.array([pose[:2], p[:2]])
        if s is not None and g is not None:
            dist, parent = dijkstra_field(cm.soft * 0.3, cm.lethal, s)
            if np.isfinite(dist[g]):
                cells = extract_path(parent, g)
                path = smooth_path(
                    np.vstack([pose[None, :2], cm.grid.cell_to_world(np.array(cells))]), cm.lethal, cm.grid
                )
        return ViewGoal(np.asarray(p, float), path, 0.0, tag, None, {})

    def _retreat_cmd(self, pose, tracks):
        pts = [tr.x[:2] for tr in tracks]
        if not pts:
            return (-0.1, 0.0)
        near = min(pts, key=lambda q: np.linalg.norm(q - pose[:2]))
        away = np.arctan2(pose[1] - near[1], pose[0] - near[0])
        err = wrap_angle(away - pose[2])
        heading = np.array([np.cos(pose[2]), np.sin(pose[2])])

        def clear(xy) -> bool:
            return self.cm is None or float(self.cm.lookup(xy[None], "edt")[0]) > self.radius + 0.05

        back_ok = clear(pose[:2] - 0.3 * heading)
        front_ok = clear(pose[:2] + 0.3 * heading)
        if abs(err) < np.pi / 2:
            # move away from the person only if the way ahead is clear; otherwise turn in place
            return (0.15 if front_ok else 0.0, float(np.clip(1.2 * err, -0.6, 0.6)))
        return (-0.12 if back_ok else 0.0, float(np.clip(1.2 * wrap_angle(err + np.pi), -0.6, 0.6)))

    # ------------------------------------------------------------------
    def operator_intervention(self, pose_hint: np.ndarray | None, t: float) -> None:
        """Operator acknowledged a HANDOVER; optional manual re-localisation."""
        if pose_hint is not None:
            self.ekf.x = np.asarray(pose_hint, float).copy()
            self.ekf.P = np.diag([0.03, 0.03, 0.01]) ** 2
            self.ekf.nis_hist.clear()
        self._operator_ack = True
        self.operator_request_open = False
        self.prov.add_evidence(
            f"operator:{t:.1f}",
            t,
            "operator",
            {"action": "ack+relocalise" if pose_hint is not None else "ack"},
            attributed_to="operator",
        )

    def finalize(self, t: float) -> list:
        """Close all open claims at episode end (ABSTAIN for anything undecided)."""
        out = []
        for oc in list(self.verifier.open.values()):
            oc.claim.t_due = min(oc.claim.t_due, t)
        out += self.verifier.step(max(t, max([oc.claim.t_due for oc in self.verifier.open.values()], default=t)))
        return out

    def scene_graph(self) -> dict:
        return build_scene_graph(self.t, self.inp.prior_map, self.inp.slots, self.items, self.tracker.confirmed(),
                                 self.ekf.x, self.ekf.P, self.inp.zones)

    def ghost_confusion(self) -> np.ndarray:
        return self._ghost_conf.copy()


def stack_inputs_from_episode(ep) -> StackInputs:
    """What the robot is told before an episode (never the true world)."""
    sv = ep.survey_spec
    return StackInputs(
        room=sv.room, prior_map=copy.deepcopy(ep.prior_map), slots=copy.deepcopy(sv.slots),
        zones=copy.deepcopy(sv.sterile_zones), landmarks=copy.deepcopy(sv.landmarks), items=copy.deepcopy(sv.items),
        initial_placement=dict(ep.workflow.initial), staff=copy.deepcopy(sv.staff),
        start_pose=np.array([sv.robot_start[0], sv.robot_start[1], sv.robot_start[2]]), dock=sv.dock.copy(),
        duration=ep.workflow.duration, claim_grace=float(ep.cfg.get("workflow", {}).get("claim_grace_s", 25.0)),
        count_grace=float(ep.cfg.get("workflow", {}).get("count_grace_s", 90.0)))
