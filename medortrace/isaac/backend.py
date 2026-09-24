"""Isaac Sim backend implementing :class:`medortrace.sim.backend.SimBackend`.

Pipeline per control tick (10 Hz by default):

1. apply the gated velocity command to the wheel drives (PhysX articulation);
2. advance staff behaviour (``StaffPopulation`` actual + robot-free shadow) and
   move the USD staff proxies;
3. apply ground-truth custody moves (workflow truth incl. hidden causes) by
   teleporting item rigid bodies to their slot / the holder's hand;
4. step PhysX ``physics_substeps`` times (120 Hz), rendering on the last
   substep so RTX sensors produce data;
5. read RTX/physics sensors at their configured rates, apply the episode's
   fault model (dropouts, clock skew, odometry bias) and return a
   :class:`SensorBundle` identical in type to the lite backend's.

The *same* ``Episode`` (scene spec, workflow, faults, prior map) drives both
backends; the USD stage is authored from it on reset.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from medortrace.common.config import CONFIG_DIR, load_yaml
from medortrace.common.geometry import wrap_angle
from medortrace.common.msgs import Header, SensorBundle, VelocityCommand, WheelOdometry
from medortrace.isaac.compat import world_cls, xform_prim_cls
from medortrace.sim.backend import SimBackend, TruthSnapshot
from medortrace.sim.episode import Episode
from medortrace.usd.robot_rig import build_rig
from medortrace.usd.scene_builder import build_stage
from medortrace.world.agents import StaffPopulation


class IsaacBackend(SimBackend):
    def __init__(self, cfg: dict | None = None, physics_hz: float = 120.0, staff_mode: str = "capsule",
                 detector: str = "gt_surrogate", work_dir: str | None = None):
        self._cfg = cfg or {}
        self.physics_hz = physics_hz
        self.staff_mode = staff_mode
        self.detector = detector
        self.work_dir = Path(work_dir or tempfile.mkdtemp(prefix="medortrace_isaac_"))
        self._t = 0.0
        self._dt = 0.1
        self.world = None

    @property
    def t(self) -> float:
        return self._t

    @property
    def dt(self) -> float:
        return self._dt

    # ------------------------------------------------------------------
    def reset(self, episode: Episode) -> SensorBundle:
        import omni.usd
        from medortrace.isaac.robot import RobotController
        from medortrace.isaac.sensors import (AcousticAdapter, CameraAdapter, ContactAdapter, ImuAdapter,
                                              LandmarkAdapter, RtxLidarAdapter, RtxRadarAdapter)
        from medortrace.isaac.staff import StaffDriver
        from medortrace.sim.lite_backend import RobotParams

        self.episode = ep = episode
        cfg = ep.cfg
        self._dt = float(cfg.get("episode", {}).get("dt", 0.1))
        self._t = 0.0
        rig = load_yaml(CONFIG_DIR / "robot" / "rig.yaml")
        rig_path = self.work_dir / "robot" / "medortrace_rig.usda"
        build_rig(rig_path, rig)
        scene_path = self.work_dir / "scenes" / f"{ep.spec.scenario_id or 'episode'}.usda"
        build_stage(ep.spec, ep.materials, scene_path, robot_rig=f"../robot/{rig_path.name}")
        omni.usd.get_context().open_stage(str(scene_path))
        World = world_cls()
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / self.physics_hz, rendering_dt=self._dt)
        self.substeps = max(1, int(round(self.physics_hz * self._dt)))
        self.robot = RobotController("/World/Robot", rig, cfg)
        self.world.reset()
        self.robot.initialize()
        self.rp = RobotParams(cfg)
        sc = cfg.get("sensors", {})
        self.rates = {"lidar": 5.0, "camera": 5.0, "radar": 10.0, "landmarks": 5.0, "acoustic": 2.0}
        self.rates.update(sc.get("rates_hz", {}))
        self._last = {k: -1e9 for k in self.rates}
        base = "/World/Robot/base_link"
        self.lidar = RtxLidarAdapter(f"{base}/lidar_link")
        self.radar = RtxRadarAdapter(f"{base}/radar_link")
        self.camera = CameraAdapter(f"{base}/camera_link/rgb", detector=self.detector, rng=ep.streams["sensors"])
        self.acoustic = AcousticAdapter(f"{base}/acoustic_link")
        self.imu = ImuAdapter(f"{base}/imu_link")
        self.contact = ContactAdapter(base)
        self.landmarks = LandmarkAdapter({lm.id: lm.position for lm in ep.spec.landmarks})
        self.rng = ep.streams["sensors"]
        self.actual = StaffPopulation(ep.spec, ep.workflow.staff_tasks, ep.streams, True, cfg.get("agents"))
        self.shadow = StaffPopulation(ep.spec, ep.workflow.staff_tasks, ep.streams, False, cfg.get("agents"))
        self.staff = StaffDriver(self.actual.names(), self.staff_mode)
        XP = xform_prim_cls()
        self.items = {i.id: XP(f"/World/Items/{i.id}") for i in ep.spec.items}
        self.base_prim = XP(base)
        self.item_slot = dict(ep.workflow.initial)
        self._truth_ptr = 0
        self._wf_ptr = 0
        self._pending_wf = sorted(ep.workflow.log, key=lambda e: e.t)
        self._wf_latency = {e.event_id: float(ep.streams["workflow"].uniform(0.2, 2.0)) for e in self._pending_wf}
        self._acoustic_target = None
        self._prev_v = 0.0
        self._seq = 0
        self._place_items()
        return self._sense()

    # ------------------------------------------------------------------
    def _pose(self) -> np.ndarray:
        if hasattr(self.base_prim, "get_world_poses"):
            p, q = self.base_prim.get_world_poses()
            p, q = np.asarray(p)[0], np.asarray(q)[0]
        else:
            p, q = self.base_prim.get_world_pose()
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([p[0], p[1], yaw])

    def _slot_pos(self, sid: str) -> np.ndarray:
        s = self.episode.spec.slot(sid)
        if s.kind == "hand":
            a = self.actual.get(s.anchor)
            toward = self.episode.spec.object("or_table").box.center[:2] - a.pos
            toward /= np.linalg.norm(toward) + 1e-9
            return np.array([*(a.pos + 0.3 * toward), 1.0])
        return s.position

    def _place_items(self) -> None:
        for iid, prim in self.items.items():
            sid = self.item_slot[iid]
            p = np.array([0.0, 0.0, -5.0]) if sid == "elsewhere" else self._slot_pos(sid) + np.array([0, 0, 0.02])
            q = np.array([1.0, 0, 0, 0])
            if hasattr(prim, "set_world_poses"):
                prim.set_world_poses(positions=p[None], orientations=q[None])
            else:
                prim.set_world_pose(position=p, orientation=q)

    def step(self, cmd: VelocityCommand) -> SensorBundle:
        dt = self._dt
        self.robot.command(cmd.v, cmd.omega, dt)
        pose = self._pose()
        rv = self.robot.v_cmd * np.array([np.cos(pose[2]), np.sin(pose[2])])
        self.actual.step(self._t, dt, pose[:2], rv)
        self.shadow.step(self._t, dt, None, None)
        self.staff.apply(self.actual.positions(), np.array([a.heading for a in self.actual.agents]))
        truth = self.episode.workflow.truth
        while self._truth_ptr < len(truth) and truth[self._truth_ptr].t <= self._t + dt:
            m = truth[self._truth_ptr]
            self.item_slot[m.item_id] = m.dst
            self._truth_ptr += 1
        self._place_items()
        for k in range(self.substeps):
            self.world.step(render=(k == self.substeps - 1))
        self._t += dt
        self._acoustic_target = cmd.acoustic_probe_target
        self.battery = self.robot.update_energy(dt, self.rp.p_acoustic_w if cmd.acoustic_probe_target else 0.0)
        return self._sense()

    def _due(self, name: str) -> bool:
        if self._t - self._last[name] >= 1.0 / self.rates[name] - 1e-6:
            self._last[name] = self._t
            return True
        return False

    def _sense(self) -> SensorBundle:
        ep = self.episode
        fm = ep.faults
        t = self._t
        stamp = fm.stamp
        b = SensorBundle(t=t)
        pose = self._pose()
        if self._due("lidar") and not fm.dropped("lidar", t):
            b.lidar = self.lidar.read(t, stamp)
        if self._due("camera") and not fm.dropped("camera", t):
            b.camera = self.camera.read(t, stamp, np.deg2rad(-25.0))
        if self._due("radar") and not fm.dropped("radar", t):
            b.radar = self.radar.read(t, stamp)
        if self._acoustic_target and self._due("acoustic") and not fm.dropped("acoustic", t):
            slots = [s for s in ep.spec.slots if s.acoustic_region == self._acoustic_target]
            if slots:
                ids = {s.id for s in slots}
                contents = [ep.materials[i.material].acoustic_reflectivity for i in ep.spec.items
                            if self.item_slot[i.id] in ids]
                anchor = ep.spec.object(slots[0].anchor) if slots[0].anchor in [o.name for o in ep.spec.objects] else None
                base_r = ep.materials[anchor.material].acoustic_reflectivity if anchor else 0.5
                b.acoustic = self.acoustic.read(t, stamp, np.array([pose[0], pose[1], 1.2]), self._acoustic_target,
                                                slots[0].position, base_r, contents, self.rng)
        if self._due("landmarks") and not fm.dropped("landmarks", t):
            b.landmarks = self.landmarks.read(t, stamp, np.array([pose[0], pose[1], 1.45]), pose[2], self.rng)
        if not fm.dropped("imu", t):
            b.imu = self.imu.read(t, stamp)
        if not fm.dropped("odom", t):
            vb, wb = fm.odom_bias if t >= fm.odom_bias_start else (0.0, 0.0)
            try:
                jv = np.asarray(self.robot.art.get_joint_velocities()).reshape(-1)
                d = self.robot._dof
                wl, wr = jv[d.get("left_wheel_joint", 0)], jv[d.get("right_wheel_joint", 1)]
                v = self.robot.r * (wl + wr) / 2
                w = self.robot.r * (wr - wl) / self.robot.b
            except Exception:
                v, w = self.robot.v_cmd, self.robot.w_cmd
            self._seq += 1
            b.odom = WheelOdometry(Header(stamp("odom", t), t, "base_link", self._seq),
                                   v * (1 + vb) + float(self.rng.normal(0, 0.01)), w + wb + float(self.rng.normal(0, 0.005)))
        b.contact = self.contact.read(t, stamp, self.robot.efforts())
        while self._wf_ptr < len(self._pending_wf) and \
                self._pending_wf[self._wf_ptr].t + self._wf_latency[self._pending_wf[self._wf_ptr].event_id] <= t:
            b.workflow.append(self._pending_wf[self._wf_ptr])
            self._wf_ptr += 1
        b.battery_wh = getattr(self, "battery", self.rp.battery_wh)
        return b

    def truth(self) -> TruthSnapshot:
        ep = self.episode
        pose = self._pose()
        ap = self.actual.positions()
        d = np.linalg.norm(ap - pose[:2], axis=1) if len(ap) else np.array([10.0])
        c = self.contact.s.get_current_frame() if hasattr(self, "contact") else {}
        return TruthSnapshot(
            t=self._t, robot_pose=pose, robot_vel=np.array([self.robot.v_cmd, self.robot.w_cmd]),
            agent_names=self.actual.names(), agent_pos=ap.copy(), agent_vel=self.actual.velocities().copy(),
            shadow_pos=self.shadow.positions().copy(), item_slots=dict(self.item_slot),
            item_pos={i: self._slot_pos(s) if s != "elsewhere" else np.full(3, np.nan) for i, s in self.item_slot.items()},
            collision_agent=bool(np.min(d) < self.rp.radius + 0.25),
            collision_static=bool(c.get("in_contact", False)) and not bool(np.min(d) < self.rp.radius + 0.35),
            contact_force=float(c.get("force", 0.0)), battery_wh=float(getattr(self, "battery", 0.0)),
            energy_used_wh=float(self.robot.energy_used),
            in_keepout=bool(ep.spec.in_keepout(pose[None, :2], extra=-ep.spec.sterile_zones[0].keepout_margin)[0]),
            fault_active=ep.faults.active(self._t))

    def close(self) -> None:
        if self.world is not None:
            self.world.stop()
        _ = wrap_angle  # keep import (used by subclasses)
