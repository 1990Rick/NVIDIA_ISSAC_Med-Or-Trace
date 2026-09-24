"""Lite backend: fast analytic simulator implementing :class:`SimBackend`.

Used for CI, large-sweep benchmarking, policy search and ablations.  It runs
~10-50x real time on one CPU core.  The Isaac Sim backend
(``medortrace.isaac.backend``) produces the same message types from RTX
sensors and PhysX.
"""

from __future__ import annotations

import numpy as np

from medortrace.common.geometry import wrap_angle
from medortrace.common.msgs import (
    AcousticFrame,
    CameraFrame,
    ContactState,
    Header,
    ImuSample,
    LandmarkFrame,
    LidarScan,
    RadarFrame,
    SensorBundle,
    VelocityCommand,
    WheelOdometry,
)
from medortrace.sim.backend import SimBackend, TruthSnapshot
from medortrace.sim.episode import Episode
from medortrace.sim.raycast import RayScene
from medortrace.sim.sensors_lite import (
    AcousticConfig,
    CameraConfig,
    LidarConfig,
    RadarConfig,
    simulate_acoustic,
    simulate_camera,
    simulate_landmarks,
    simulate_lidar,
    simulate_radar,
)
from medortrace.world.agents import StaffPopulation
from medortrace.world.materials import MATERIALS


class RobotParams:
    def __init__(self, cfg: dict):
        r = cfg.get("robot", {})
        self.radius = float(r.get("footprint_radius", 0.28))
        self.max_v = float(r.get("max_v", 0.7))
        self.max_w = float(r.get("max_omega", 1.2))
        self.acc = float(r.get("max_acc", 0.6))
        self.alpha = float(r.get("max_alpha", 1.8))
        self.mass = float(r.get("mass_kg", 55.0))
        self.battery_wh = float(r.get("battery_capacity_wh", 480.0))
        self.p_idle_w = float(r.get("power_idle_w", 38.0))
        self.p_sensors_w = float(r.get("power_sensors_w", 27.0))
        self.p_acoustic_w = float(r.get("power_acoustic_w", 4.0))
        self.c_rr = float(r.get("rolling_resistance", 0.02))
        self.eta = float(r.get("drivetrain_efficiency", 0.75))
        self.height = float(r.get("height_m", 1.55))


class LiteBackend(SimBackend):
    def __init__(self, cfg: dict | None = None):
        self._cfg = cfg or {}
        self._t = 0.0
        self._dt = 0.1

    # ------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self._t

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self, episode: Episode) -> SensorBundle:
        self.episode = ep = episode
        cfg = ep.cfg
        self._dt = float(cfg.get("episode", {}).get("dt", 0.1))
        self._t = 0.0
        self.rp = RobotParams(cfg)
        self.rng = ep.streams["sensors"]
        self.robot_rng = ep.streams["robot"]
        sc = cfg.get("sensors", {})
        self.lidar_cfg = LidarConfig(**sc.get("lidar", {}))
        self.cam_cfg = CameraConfig(**sc.get("camera", {}))
        self.radar_cfg = RadarConfig(**sc.get("radar", {}))
        self.ac_cfg = AcousticConfig(**sc.get("acoustic", {}))
        self.rates = {"lidar": 5.0, "camera": 5.0, "radar": 10.0, "landmarks": 5.0, "acoustic": 2.0}
        self.rates.update(sc.get("rates_hz", {}))
        self._last = {k: -1e9 for k in self.rates}
        self._seq = 0
        spec = ep.spec
        self.pose = np.array([spec.robot_start[0], spec.robot_start[1], spec.robot_start[2]])
        self.vel = np.zeros(2)
        self.prev_vel = np.zeros(2)
        self.battery = self.rp.battery_wh * float(
            ep.streams["robot"].uniform(*cfg.get("robot", {}).get("battery_start_frac", [0.55, 0.95]))
        )
        self.energy_used = 0.0
        self.actual = StaffPopulation(spec, ep.workflow.staff_tasks, ep.streams, robot_aware=True,
                                      params=cfg.get("agents"))
        self.shadow = StaffPopulation(spec, ep.workflow.staff_tasks, ep.streams, robot_aware=False,
                                      params=cfg.get("agents"))
        self.static_objs = spec.objects
        self.obj_materials = [ep.materials[o.material] for o in self.static_objs]
        self.obj_tags = [list(o.tags) for o in self.static_objs]
        self._box_c = np.array([o.box.center for o in self.static_objs])
        self._box_h = np.array([o.box.half for o in self.static_objs])
        self._box_y = np.array([o.box.yaw for o in self.static_objs])
        self._obj_index = {o.name: i for i, o in enumerate(self.static_objs)}
        self._wf_ptr = 0
        self._truth_ptr = 0
        self.item_slot = dict(ep.workflow.initial)
        self._item_offsets = self._make_item_offsets()
        self.collision_agent = False
        self.collision_static = False
        self.contact_force = 0.0
        self._acoustic_target: str | None = None
        self._pending_wf = sorted(ep.workflow.log, key=lambda e: e.t)
        self._wf_latency = {e.event_id: float(ep.streams["workflow"].uniform(0.2, 2.0)) for e in self._pending_wf}
        return self._sense(first=True)

    def _make_item_offsets(self) -> dict[str, np.ndarray]:
        rng = self.episode.streams.fork("layout", "item_offsets")
        return {i.id: np.array([*rng.uniform(-0.12, 0.12, 2), 0.0]) for i in self.episode.spec.items}

    # ------------------------------------------------------------------
    def _ray_scene(self) -> RayScene:
        pos = self.actual.positions()
        n = len(pos)
        return RayScene(self._box_c, self._box_h, self._box_y, pos,
                        np.full(n, 0.25), np.full(n, 1.75), ceiling=self.episode.spec.room[2])

    def slot_position(self, sid: str) -> np.ndarray:
        s = self.episode.spec.slot(sid)
        if s.kind == "hand":
            a = self.actual.get(s.anchor)
            toward = self.episode.spec.object("or_table").box.center[:2] - a.pos
            toward = toward / (np.linalg.norm(toward) + 1e-9)
            return np.array([*(a.pos + 0.3 * toward), 1.0])
        return s.position

    def item_position(self, iid: str) -> np.ndarray:
        sid = self.item_slot[iid]
        if sid == "elsewhere":
            return np.full(3, np.nan)
        p = self.slot_position(sid).copy()
        s = self.episode.spec.slot(sid)
        if s.kind in ("surface", "container", "floor", "under_drape"):
            p = p + self._item_offsets[iid] * min(1.0, s.radius / 0.2)
        return p

    # ------------------------------------------------------------------
    def step(self, cmd: VelocityCommand) -> SensorBundle:
        dt = self._dt
        rp = self.rp
        # --- robot dynamics (unicycle with acceleration limits) ---------------
        v_c = float(np.clip(cmd.v, -0.3, rp.max_v))
        w_c = float(np.clip(cmd.omega, -rp.max_w, rp.max_w))
        self.prev_vel = self.vel.copy()
        dv = np.clip(v_c - self.vel[0], -rp.acc * dt, rp.acc * dt)
        dw = np.clip(w_c - self.vel[1], -rp.alpha * dt, rp.alpha * dt)
        self.vel = self.vel + np.array([dv, dw])
        self.vel += self.robot_rng.normal(0, [0.005, 0.005])
        new_pose = self.pose.copy()
        new_pose[2] = wrap_angle(self.pose[2] + self.vel[1] * dt)
        new_pose[0] += self.vel[0] * dt * np.cos(self.pose[2] + 0.5 * self.vel[1] * dt)
        new_pose[1] += self.vel[0] * dt * np.sin(self.pose[2] + 0.5 * self.vel[1] * dt)
        self.collision_static = False
        self.collision_agent = False
        self.contact_force = 0.0
        for o in self.static_objs:
            # overhead structure above the mast height does not collide
            if o.box.z_min > rp.height or o.box.z_max < 0.02:
                continue
            if o.box.distance_xy(new_pose[None, :2])[0] < rp.radius:
                self.collision_static = True
                break
        ap = self.actual.positions()
        if len(ap) and np.min(np.linalg.norm(ap - new_pose[:2], axis=1)) < rp.radius + 0.25:
            self.collision_agent = True
        if self.collision_static or self.collision_agent:
            self.contact_force = float(min(400.0, rp.mass * abs(self.vel[0]) / dt * 0.5 + 20.0))
            self.vel[:] = 0.0
            new_pose[:2] = self.pose[:2]
        self.pose = new_pose
        # --- energy -----------------------------------------------------------
        a = (self.vel[0] - self.prev_vel[0]) / dt
        p_motion = (rp.c_rr * rp.mass * 9.81 * abs(self.vel[0]) + rp.mass * max(0.0, a * self.vel[0])
                    + 3.0 * abs(self.vel[1])) / rp.eta
        p = rp.p_idle_w + rp.p_sensors_w + p_motion + (rp.p_acoustic_w if cmd.acoustic_probe_target else 0.0)
        e = p * dt / 3600.0
        self.battery = max(0.0, self.battery - e)
        self.energy_used += e
        # --- staff --------------------------------------------------------
        rv_world = self.vel[0] * np.array([np.cos(self.pose[2]), np.sin(self.pose[2])])
        self.actual.step(self._t, dt, self.pose[:2], rv_world)
        self.shadow.step(self._t, dt, None, None)
        self._t += dt
        # --- item custody truth ------------------------------------------
        truth = self.episode.workflow.truth
        while self._truth_ptr < len(truth) and truth[self._truth_ptr].t <= self._t:
            m = truth[self._truth_ptr]
            self.item_slot[m.item_id] = m.dst
            self._truth_ptr += 1
        self._acoustic_target = cmd.acoustic_probe_target
        return self._sense()

    # ------------------------------------------------------------------
    def _due(self, name: str) -> bool:
        if self._t - self._last[name] >= 1.0 / self.rates[name] - 1e-6:
            self._last[name] = self._t
            return True
        return False

    def _hdr(self, sensor: str, frame: str) -> Header:
        self._seq += 1
        return Header(self.episode.faults.stamp(sensor, self._t), self._t, frame, self._seq)

    def _sense(self, first: bool = False) -> SensorBundle:
        ep = self.episode
        fm = ep.faults
        t = self._t
        b = SensorBundle(t=t)
        scene = self._ray_scene()
        x, y, th = self.pose
        haze = float(ep.spec.nuisance.get("haze", 0.0))
        mats_all = (
            self.obj_materials
            + [MATERIALS["human"]] * len(scene.cyl_xy)
            + [ep.materials["floor_vinyl"], ep.materials["painted_wall"]]
        )
        tags_all = self.obj_tags + [[] for _ in range(len(scene.cyl_xy))]
        # --- lidar ----------------------------------------------------------
        if self._due("lidar") and not fm.dropped("lidar", t):
            lc = self.lidar_cfg
            origin = np.array([x + lc.mount_x * np.cos(th), y + lc.mount_x * np.sin(th), lc.mount_height])
            ranges, d_s, ring, inten, ghost, obj = simulate_lidar(
                scene, mats_all, tags_all, origin, th, lc, self.rng, fm.specular_gain, fm.floor_wet, haze)
            fin = np.isfinite(ranges)
            pts = d_s[fin] * ranges[fin, None]
            b.lidar = LidarScan(self._hdr("lidar", "lidar_link"), pts, inten[fin], ring[fin], d_s, ranges,
                                sensor_height=lc.mount_height, gt_is_ghost=ghost, gt_object_id=obj)
        # --- camera ---------------------------------------------------------
        if self._due("camera") and not fm.dropped("camera", t):
            cc = self.cam_cfg
            cam = np.array([x, y, cc.mount_height])
            items = []
            excl_map = {}
            nobj = scene.n_boxes + len(scene.cyl_xy)
            names = self.actual.names()
            for it in ep.spec.items:
                sid = self.item_slot[it.id]
                if sid == "elsewhere":
                    continue
                s = ep.spec.slot(sid)
                ex = np.zeros(nobj, dtype=bool)
                if s.kind == "hand":
                    ex[scene.n_boxes + names.index(s.anchor)] = True
                elif s.anchor in self._obj_index:
                    ex[self._obj_index[s.anchor]] = True
                    if s.anchor == "back_table":
                        ex[self._obj_index["back_table_drape"]] = True
                excl_map[it.id] = ex
                items.append({"id": it.id, "cls": it.cls, "pos": self.item_position(it.id), "size": it.size,
                              "material": ep.materials[it.material], "slot": s, "tag_readable": it.tag_readable})
            dets = simulate_camera(scene, cam, th, cc, items, self.rng, ep.spec.nuisance.get("glare_gain", 1.0),
                                   haze, exclude_map=excl_map)
            b.camera = CameraFrame(self._hdr("camera", "camera_link"), dets, np.deg2rad(cc.hfov_deg), cc.max_range)
        # --- radar ------------------------------------------------------------
        if self._due("radar") and not fm.dropped("radar", t):
            rc = self.radar_cfg
            origin = np.array([x, y, rc.mount_height])
            targets = []
            for k, a in enumerate(self.actual.agents):
                targets.append({"pos": np.array([a.pos[0], a.pos[1], 1.0]), "vel": a.vel, "rcs_dbsm": 0.0,
                                "id": a.spec.name, "self_index": scene.n_boxes + k})
            for it in ep.spec.items:
                if not it.metallic:
                    continue
                sid = self.item_slot[it.id]
                if sid == "elsewhere":
                    continue
                s = ep.spec.slot(sid)
                si = self._obj_index.get(s.anchor)
                targets.append({"pos": self.item_position(it.id), "vel": np.zeros(2), "rcs_dbsm": -12.0,
                                "id": it.id, "self_index": si, "behind_fabric": s.kind == "under_drape"})
            rv_world = self.vel[0] * np.array([np.cos(th), np.sin(th)])
            dets = simulate_radar(scene, self.obj_materials, origin, th, rv_world, targets, rc, self.rng)
            b.radar = RadarFrame(self._hdr("radar", "radar_link"), dets)
        # --- acoustic probe ---------------------------------------------------
        if self._acoustic_target and self._due("acoustic") and not fm.dropped("acoustic", t):
            region = self._acoustic_target
            slots = [s for s in ep.spec.slots if s.acoustic_region == region]
            if slots:
                s0 = slots[0]
                contents = [ep.materials[i.material] for i in ep.spec.items
                            if self.item_slot[i.id] in {s.id for s in slots}]
                base = ep.materials[ep.spec.object(s0.anchor).material].acoustic_reflectivity \
                    if s0.anchor in self._obj_index else 0.5
                echo = simulate_acoustic(scene, np.array([x, y, 1.2]), region, s0.position, base, contents,
                                         self.ac_cfg, self.rng)
                if echo is not None:
                    b.acoustic = AcousticFrame(self._hdr("acoustic", "acoustic_link"), [echo])
        # --- landmarks --------------------------------------------------------
        if self._due("landmarks") and not fm.dropped("landmarks", t):
            obs = simulate_landmarks(scene, np.array([x, y, 1.45]), th, ep.spec.landmarks, self.rng)
            b.landmarks = LandmarkFrame(self._hdr("landmarks", "camera_link"), obs)
        # --- proprioception -------------------------------------------------
        if not fm.dropped("odom", t):
            vb, wb = (fm.odom_bias if t >= fm.odom_bias_start else (0.0, 0.0))
            b.odom = WheelOdometry(self._hdr("odom", "base_link"),
                                   self.vel[0] * (1 + vb) + float(self.rng.normal(0, 0.01)),
                                   self.vel[1] + wb + float(self.rng.normal(0, 0.005)))
        if not fm.dropped("imu", t):
            acc = (self.vel[0] - self.prev_vel[0]) / self._dt
            for k in range(10):
                h = self._hdr("imu", "imu_link")
                h.stamp -= (9 - k) * self._dt / 10
                b.imu.append(
                    ImuSample(
                        h,
                        np.array([acc, self.vel[0] * self.vel[1], 9.81]) + self.rng.normal(0, 0.05, 3),
                        np.array([0.0, 0.0, self.vel[1]]) + self.rng.normal(0, 0.003, 3),
                    )
                )
        b.contact = ContactState(self._hdr("contact", "bumper"), self.contact_force > 0, self.contact_force)
        # --- workflow log (delivered with latency) --------------------------------
        while self._wf_ptr < len(self._pending_wf) and \
                self._pending_wf[self._wf_ptr].t + self._wf_latency[self._pending_wf[self._wf_ptr].event_id] <= t:
            b.workflow.append(self._pending_wf[self._wf_ptr])
            self._wf_ptr += 1
        b.battery_wh = self.battery
        return b

    # ------------------------------------------------------------------
    def truth(self) -> TruthSnapshot:
        ep = self.episode
        return TruthSnapshot(
            t=self._t, robot_pose=self.pose.copy(), robot_vel=self.vel.copy(),
            agent_names=self.actual.names(), agent_pos=self.actual.positions().copy(),
            agent_vel=self.actual.velocities().copy(), shadow_pos=self.shadow.positions().copy(),
            item_slots=dict(self.item_slot), item_pos={i.id: self.item_position(i.id) for i in ep.spec.items},
            collision_agent=self.collision_agent, collision_static=self.collision_static,
            contact_force=self.contact_force, battery_wh=self.battery, energy_used_wh=self.energy_used,
            in_keepout=bool(ep.spec.in_keepout(self.pose[None, :2], extra=-ep.spec.sterile_zones[0].keepout_margin)[0]),
            fault_active=ep.faults.active(self._t),
        )
