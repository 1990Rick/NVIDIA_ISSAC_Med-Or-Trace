"""Isaac Sim backend implementing :class:`medortrace.sim.backend.SimBackend`.

Pipeline per control tick (10 Hz by default):

1. apply the gated velocity command to the wheel drives (PhysX articulation);
   with ``drive="external"`` the wheels are driven by the ROS 2 OmniGraph
   (``/medortrace/cmd_vel``) instead and the command's twist is ignored;
2. advance staff behaviour (``StaffPopulation`` actual + robot-free shadow) and
   move the USD staff proxies;
3. apply ground-truth custody moves (workflow truth incl. hidden causes) by
   teleporting the (kinematic) item bodies to their slot / the holder's hand,
   with the lite simulator's placement rules (``medortrace.isaac.truth``) and
   toggling visibility for items entering/leaving the room;
4. advance the world by one control period with a single rendered
   ``World.step``: PhysX runs ``rendering_dt / physics_dt`` (= 12) substeps
   internally, and RTX sensors see exactly one control period per frame;
5. read RTX/physics sensors at their configured rates, apply the episode's
   fault model (dropouts, clock skew, odometry bias) and return a
   :class:`SensorBundle` identical in type to the lite backend's.

The *same* ``Episode`` (scene spec, workflow, faults, prior map) drives both
backends; the USD stage is authored from it on reset.  Timing faults
(dropouts, skew, odometry bias) are applied to the readings in step 5; the
reflective faults (``specular_gain``, ``floor_wet``) are material edits of the
opened stage (:func:`medortrace.isaac.sensors.apply_reflective_faults`,
recorded in ``reflective_faults``).  RNG parity: the battery start charge,
workflow-log latencies and item offsets consume the same streams / forks as
``LiteBackend``.

The radar is auxiliary: if it cannot be created on the running release the
episode continues without it (``radar = None``) and the reason is recorded in
``sensor_warnings`` together with the adapters' own warnings (e.g. a custom
RTX profile that the release did not apply).

Backend options (``staff_mode``, ``detector``, ``drive``, ...) are not part of
the scenario config, so a registry entry keeps its ``cfg_hash`` on either
backend.  ``eval.runner.make_backend`` constructs ``IsaacBackend(cfg)``; entry
scripts set process-wide options with :meth:`IsaacBackend.configure` and can
register :meth:`IsaacBackend.add_reset_hook` callbacks (e.g. building the ROS 2
graph once the sensors' render products exist).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from medortrace.common.config import CONFIG_DIR, load_yaml
from medortrace.common.msgs import Header, SensorBundle, VelocityCommand, WheelOdometry
from medortrace.isaac.compat import XformGroup, world_cls
from medortrace.isaac.truth import CustodyTimeline, item_offsets, item_position, item_prim_center
from medortrace.sim.backend import SimBackend, TruthSnapshot
from medortrace.sim.episode import Episode
from medortrace.usd.robot_rig import build_rig
from medortrace.usd.scene_builder import build_stage, safe
from medortrace.world.agents import StaffPopulation

ROBOT_PRIM = "/World/Robot"
PHYSICS_SCENE = "/World/PhysicsScene"     # authored by medortrace.usd.scene_builder
DEFAULT_OPTIONS = {
    "staff_mode": "capsule",        # capsule | people
    "detector": "gt_surrogate",     # gt_surrogate | model:<path>
    "physics_hz": 120.0,
    "work_dir": None,               # where the episode's USD stage + rig are authored
    "drive": "internal",            # internal (stack commands) | external (ROS 2 cmd_vel graph)
    "items_kinematic": True,        # custody moves are ground-truth teleports
    "normalize_nonvisual": True,    # map non-vocabulary RTX material tokens to their aliases
}


class IsaacBackend(SimBackend):
    options: dict = dict(DEFAULT_OPTIONS)
    reset_hooks: list[Callable[["IsaacBackend"], None]] = []

    @classmethod
    def configure(cls, **kw) -> None:
        unknown = set(kw) - set(DEFAULT_OPTIONS)
        if unknown:
            raise ValueError(f"unknown IsaacBackend options {sorted(unknown)}; known: {sorted(DEFAULT_OPTIONS)}")
        cls.options = {**cls.options, **kw}

    @classmethod
    def add_reset_hook(cls, fn: Callable[["IsaacBackend"], None]) -> None:
        cls.reset_hooks = [*cls.reset_hooks, fn]

    def __init__(self, cfg: dict | None = None, **overrides):
        opts = {**self.options, **overrides}
        unknown = set(opts) - set(DEFAULT_OPTIONS)
        if unknown:
            raise ValueError(f"unknown IsaacBackend options {sorted(unknown)}")
        self._cfg = cfg or {}
        self.opts = opts
        self.physics_hz = float(opts["physics_hz"])
        self.staff_mode = opts["staff_mode"]
        self.detector = opts["detector"]
        self.drive = opts["drive"]
        self.work_dir = Path(opts["work_dir"] or tempfile.mkdtemp(prefix="medortrace_isaac_"))
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
        from pxr import UsdGeom

        from medortrace.isaac.compat import open_stage, usd_stage
        from medortrace.isaac.robot import RobotController, find_articulation_root
        from medortrace.isaac.sensors import (
            AcousticAdapter,
            CameraAdapter,
            ContactAdapter,
            ImuAdapter,
            LandmarkAdapter,
            RtxLidarAdapter,
            RtxRadarAdapter,
            apply_reflective_faults,
            normalize_nonvisual_tokens,
        )
        from medortrace.isaac.staff import StaffDriver
        from medortrace.sim.lite_backend import RobotParams

        self.episode = ep = episode
        cfg = ep.cfg
        spec = ep.spec
        self._dt = float(cfg.get("episode", {}).get("dt", 0.1))
        self._t = 0.0
        rig = load_yaml(CONFIG_DIR / "robot" / "rig.yaml")
        phys = load_yaml(CONFIG_DIR / "sensors" / "physics_sensors.yaml")
        rig_path = self.work_dir / "robot" / "medortrace_rig.usda"
        build_rig(rig_path, rig)
        self.scene_path = self.work_dir / "scenes" / f"{spec.scenario_id or 'episode'}.usda"
        build_stage(spec, ep.materials, self.scene_path, robot_rig=f"../robot/{rig_path.name}")
        open_stage(str(self.scene_path))
        stage = usd_stage()
        self.item_ids = [i.id for i in spec.items]
        self.item_paths = [f"/World/Items/{safe(i)}" for i in self.item_ids]
        if self.opts["items_kinematic"]:
            for p in self.item_paths:
                a = stage.GetPrimAtPath(p).GetAttribute("physics:kinematicEnabled")
                if a and a.IsValid():
                    a.Set(True)
        # the episode's reflective faults (lite: simulate_lidar specular_gain / floor_wet) as material edits
        self.reflective_faults = apply_reflective_faults(stage, ep.materials, ep.faults.specular_gain,
                                                         ep.faults.floor_wet)
        self.nonvisual_fixes = normalize_nonvisual_tokens(stage) if self.opts["normalize_nonvisual"] else []
        self.base_path = find_articulation_root(stage, ROBOT_PRIM)
        World = world_cls()
        if hasattr(World, "clear_instance"):
            World.clear_instance()      # World is a singleton: drop the previous episode's instance
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0 / self.physics_hz, rendering_dt=self._dt,
                           physics_prim_path=PHYSICS_SCENE)      # reuse the authored scene, no second one
        self.substeps = max(1, int(round(self.physics_hz * self._dt)))   # executed inside World.step
        base = self.base_path
        self.robot = RobotController(base, rig, cfg)
        imu_cfg = phys.get("imu", {})
        bump = phys.get("contact", {}).get("bumper", {})
        self.imu = ImuAdapter(f"{base}/imu_link", float(imu_cfg.get("rate_hz", 100)),
                              int(imu_cfg.get("linear_acceleration_filter_size", 4)))
        self.contact = ContactAdapter(base, float(bump.get("radius_m", 0.3)), float(bump.get("threshold_n", 1.0)))
        self.world.reset()
        self.robot.initialize()
        self.imu.initialize()
        self.contact.initialize()
        self.rp = RobotParams(cfg)
        frac = float(ep.streams["robot"].uniform(*cfg.get("robot", {}).get("battery_start_frac", [0.55, 0.95])))
        self.robot.battery_wh = self.rp.battery_wh * frac      # same draw as LiteBackend.reset
        self.battery = self.robot.battery_wh
        sc = cfg.get("sensors", {})
        self.rates = {"lidar": 5.0, "camera": 5.0, "radar": 10.0, "landmarks": 5.0, "acoustic": 2.0}
        self.rates.update(sc.get("rates_hz", {}))
        self._last = {k: -1e9 for k in self.rates}
        frames = rig.get("frames", {})
        lc = sc.get("lidar", {})
        self.lidar = RtxLidarAdapter(f"{base}/lidar_link", mount_height=float(lc.get("mount_height", 0.9)),
                                     az_res_deg=float(lc.get("az_res_deg", 2.0)),
                                     max_range=float(lc.get("max_range", 20.0)))
        self.sensor_warnings = list(self.lidar.warnings)
        rx = frames.get("radar_link", {}).get("xyz", [0.25, 0.0, 0.6])
        try:
            self.radar = RtxRadarAdapter(f"{base}/radar_link", mount_offset=(float(rx[0]), float(rx[1]), 0.0))
            self.sensor_warnings += self.radar.warnings
        except Exception as e:  # radar is an auxiliary modality: run without it rather than abort the episode
            self.radar = None
            msg = f"RTX radar unavailable, episode runs without radar: {type(e).__name__}: {e}"
            print(f"[medortrace] WARNING: {msg}")
            self.sensor_warnings.append(msg)
        items_info = {i.id: {"cls": i.cls, "size": tuple(i.size), "glare": float(ep.materials[i.material].glare),
                             "tag_readable": bool(i.tag_readable), "prim": p}
                      for i, p in zip(spec.items, self.item_paths)}
        self.camera = CameraAdapter(f"{base}/camera_link/rgb", detector=self.detector, items=items_info,
                                    rng=ep.streams["sensors"], sensor_cfg=sc.get("camera", {}), nuisance=spec.nuisance)
        self.acoustic = AcousticAdapter(f"{base}/acoustic_link")
        self.landmarks = LandmarkAdapter({lm.id: lm.position for lm in spec.landmarks})
        self.rng = ep.streams["sensors"]
        self.actual = StaffPopulation(spec, ep.workflow.staff_tasks, ep.streams, True, cfg.get("agents"))
        self.shadow = StaffPopulation(spec, ep.workflow.staff_tasks, ep.streams, False, cfg.get("agents"))
        self.staff = StaffDriver(self.actual.names(), self.staff_mode, roles={s.name: s.role for s in spec.staff})
        self.items = XformGroup(self.item_paths)
        self._imageable = {iid: UsdGeom.Imageable(stage.GetPrimAtPath(p)) for iid, p in zip(self.item_ids,
                                                                                             self.item_paths)}
        self.base_prim = XformGroup([base])
        self.offsets = item_offsets(ep)
        self.custody = CustodyTimeline(ep.workflow)
        self.custody.advance(0.0)
        self.item_slot = self.custody.slots
        self._pending_wf = sorted(ep.workflow.log, key=lambda e: e.t)
        self._wf_latency = {e.event_id: float(ep.streams["workflow"].uniform(0.2, 2.0)) for e in self._pending_wf}
        self._wf_ptr = 0
        self._acoustic_target = None
        self._seq = 0
        self._place_items(force=True)
        for hook in type(self).reset_hooks:
            hook(self)
        return self._sense()

    # ------------------------------------------------------------------
    def _pose(self) -> np.ndarray:
        try:
            return self.robot.world_pose()
        except Exception:   # physics view not ready: read the USD/Fabric pose of the base link
            from medortrace.isaac.robot import yaw_from_quat_wxyz
            p, q = self.base_prim.get_world_poses()
            return np.array([p[0, 0], p[0, 1], yaw_from_quat_wxyz(q[0])])

    def _staff_xy(self) -> dict[str, np.ndarray]:
        return {a.spec.name: a.pos for a in self.actual.agents}

    def item_position(self, iid: str) -> np.ndarray:
        return item_position(self.episode.spec, iid, self.item_slot[iid], self.offsets, self._staff_xy())

    def _place_items(self, changed: list[str] | tuple = (), force: bool = False) -> None:
        spec = self.episode.spec
        idx, centers = [], []
        for k, iid in enumerate(self.item_ids):
            sid = self.item_slot[iid]
            in_hand = sid != "elsewhere" and spec.slot(sid).kind == "hand"
            if not (force or in_hand or iid in changed):
                continue
            support = self.item_position(iid)
            idx.append(k)
            centers.append(item_prim_center(spec, iid, support))
            if force or iid in changed:
                if np.all(np.isfinite(support)):
                    self._imageable[iid].MakeVisible()
                else:
                    self._imageable[iid].MakeInvisible()
        if idx:
            q = np.tile([1.0, 0.0, 0.0, 0.0], (len(idx), 1))
            self.items.set_world_poses(np.array(centers), q, indices=idx)

    def step(self, cmd: VelocityCommand) -> SensorBundle:
        dt = self._dt
        if self.drive == "internal":
            self.robot.command(cmd.v, cmd.omega, dt)
        pose = self._pose()
        v_meas, _ = self.robot.measured_twist()
        rv = v_meas * np.array([np.cos(pose[2]), np.sin(pose[2])])
        self.actual.step(self._t, dt, pose[:2], rv)
        self.shadow.step(self._t, dt, None, None)
        vel = self.actual.velocities()
        self.staff.apply(self.actual.positions(), np.array([a.heading for a in self.actual.agents]),
                         np.linalg.norm(vel, axis=1) if len(vel) else None)
        changed = self.custody.advance(self._t + dt)
        self._place_items(changed)
        self.world.step(render=True)
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
            b.camera = self.camera.read(t, stamp)
        if self.radar is not None and self._due("radar") and not fm.dropped("radar", t):
            b.radar = self.radar.read(t, stamp)
        if self._acoustic_target and self._due("acoustic") and not fm.dropped("acoustic", t):
            slots = [s for s in ep.spec.slots if s.acoustic_region == self._acoustic_target]
            if slots:
                ids = {s.id for s in slots}
                contents = [ep.materials[i.material].acoustic_reflectivity for i in ep.spec.items
                            if self.item_slot[i.id] in ids]
                names = {o.name for o in ep.spec.objects}
                anchor = ep.spec.object(slots[0].anchor) if slots[0].anchor in names else None
                base_r = ep.materials[anchor.material].acoustic_reflectivity if anchor else 0.5
                b.acoustic = self.acoustic.read(t, stamp, np.array([pose[0], pose[1], 1.2]), self._acoustic_target,
                                                slots[0].position, base_r, contents, self.rng)
        if self._due("landmarks") and not fm.dropped("landmarks", t):
            b.landmarks = self.landmarks.read(t, stamp, np.array([pose[0], pose[1], 1.45]), pose[2], self.rng)
        if not fm.dropped("imu", t):
            b.imu = self.imu.read(t, stamp)
        if not fm.dropped("odom", t):
            vb, wb = fm.odom_bias if t >= fm.odom_bias_start else (0.0, 0.0)
            v, w = self.robot.measured_twist()
            self._seq += 1
            b.odom = WheelOdometry(Header(stamp("odom", t), t, "base_link", self._seq),
                                   v * (1 + vb) + float(self.rng.normal(0, 0.01)),
                                   w + wb + float(self.rng.normal(0, 0.005)))
        b.contact = self.contact.read(t, stamp, self.robot.arm_efforts())
        while self._wf_ptr < len(self._pending_wf) and \
                self._pending_wf[self._wf_ptr].t + self._wf_latency[self._pending_wf[self._wf_ptr].event_id] <= t:
            b.workflow.append(self._pending_wf[self._wf_ptr])
            self._wf_ptr += 1
        b.battery_wh = self.battery
        return b

    def truth(self) -> TruthSnapshot:
        ep = self.episode
        pose = self._pose()
        ap = self.actual.positions()
        d = np.linalg.norm(ap - pose[:2], axis=1) if len(ap) else np.array([10.0])
        c = self.contact.frame()
        near_agent = bool(np.min(d) < self.rp.radius + 0.35)
        return TruthSnapshot(
            t=self._t, robot_pose=pose, robot_vel=np.array(self.robot.measured_twist()),
            agent_names=self.actual.names(), agent_pos=ap.copy(), agent_vel=self.actual.velocities().copy(),
            shadow_pos=self.shadow.positions().copy(), item_slots=dict(self.item_slot),
            item_pos={i: self.item_position(i) for i in self.item_ids},
            collision_agent=bool(np.min(d) < self.rp.radius + 0.25),
            collision_static=bool(c.get("in_contact", False)) and not near_agent,
            contact_force=float(c.get("force", 0.0)), battery_wh=float(self.battery),
            energy_used_wh=float(self.robot.energy_used),
            in_keepout=bool(ep.spec.in_keepout(pose[None, :2], extra=-ep.spec.sterile_zones[0].keepout_margin)[0]),
            fault_active=ep.faults.active(self._t))

    def close(self) -> None:
        if self.world is not None:
            self.world.stop()
            World = type(self.world)
            if hasattr(World, "clear_instance"):
                World.clear_instance()
            self.world = None
