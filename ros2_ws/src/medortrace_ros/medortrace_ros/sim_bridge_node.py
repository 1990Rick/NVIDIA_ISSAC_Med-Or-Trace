"""ROS 2 bridge that drives a MED-OR-TRACE simulator backend.

    /medortrace/cmd_vel, /medortrace/acoustic/probe_target --> SimBackend.step(VelocityCommand)
    SimBackend SensorBundle --> /clock, /tf (odom->base_link), lidar PointCloud2, camera detections,
        radar, acoustic, landmarks, IMU, odometry, contact, battery, workflow events
        + latched /medortrace/mission and (simulation only) /medortrace/sim/ground_truth/odom

``--backend lite`` (default) runs the analytic simulator, so the complete ROS 2
graph - bridge, autonomy node, workflow gateway, operator console, RViz - runs
on a laptop without Isaac Sim.  ``--backend isaac`` runs the same bridge inside
Isaac Sim's Python (``medortrace.isaac.backend.IsaacBackend``, which applies
the command to the wheel drives itself); with ``--isaac-graph`` the OmniGraph
of ``medortrace.isaac.ros2_bridge`` is added through
``IsaacBackend.add_reset_hook(ros2_reset_hook(drive_from_cmd_vel=False))`` for
RTX lidar / RGB / depth / camera_info, and the topics it publishes itself
(``ISAAC_GRAPH_GROUPS``: /clock, lidar points) are left to it.

TF with the OmniGraph.  The graph's ``PublishTF`` publishes ``base_link``
under the stage's world frame (``World``), and its ``ComputeOdom`` /
``PublishOdom`` report the noise-free chassis pose.  Next to the autonomy
node's ``map->odom`` that would give ``base_link`` a second parent (or leave
``map->odom`` dangling).  :func:`handover_base_tf` therefore deletes those
nodes (``GRAPH_NODES_OWNED_BY_BRIDGE``) and the bridge keeps publishing
``/medortrace/odom`` and ``odom->base_link`` from the backend's wheel
odometry, as in every other mode: the tree is always
``map -> odom -> base_link -> *_link`` (the graph's ``PublishSensorTF``
provides the sensor frames under ``base_link``).

``create(backend)`` is the ``--bridge medortrace_ros.sim_bridge_node:create``
factory of ``scripts/isaac/ros2_sim.py`` (Isaac Sim driven by an external
``/medortrace/cmd_vel`` through the OmniGraph): it applies
:func:`handover_base_tf` and returns a callable that publishes every
``SensorBundle``'s remaining topics (odometry + odom->base_link TF, custom
messages, mission, ground truth).

Timing.  The bridge owns simulated time and publishes ``/clock`` (offset by
``clock_offset_s`` so that time 0 is never ambiguous with "no clock yet";
the offset is the mission origin ``t0``).  In ``lockstep`` mode the simulator
advances one step (``episode.dt``) only after the autonomy node answered the
previous step with a command (or after ``lockstep_timeout_s`` of wall time),
so the result does not depend on CPU speed; otherwise it free-runs at
``rate_factor`` x real time.  Before the first step it waits until
``/medortrace/cmd_vel`` has a publisher (the autonomy node creates it once its
stack is built).

The episode (scene, workflow, faults, seed streams) comes from
``medortrace.sim.episode.build_episode`` with the same config merges as the
in-process runner, so a registry seed is the same experiment over ROS.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from medortrace.common.geometry import wrap_angle
from medortrace.common.msgs import SensorBundle, VelocityCommand
from medortrace_ros.mission import episode_config, mission_from_episode

ALL_GROUPS = ("clock", "tf", "lidar", "camera", "radar", "acoustic", "landmarks", "imu", "odom", "contact",
              "battery", "workflow", "mission", "ground_truth")
ISAAC_GRAPH_GROUPS = ("clock", "lidar")      # published by the OmniGraph when --isaac-graph
# OmniGraph nodes (medortrace.isaac.ros2_bridge.graph_spec) whose output the bridge publishes instead: the world-
# parented base_link TF and the noise-free odometry (see the module docstring, "TF with the OmniGraph")
GRAPH_NODES_OWNED_BY_BRIDGE = ("PublishTF", "ComputeOdom", "PublishOdom")


def integrate_odom(pose: np.ndarray, v: float, omega: float, dt: float) -> np.ndarray:
    """Dead-reckon wheel odometry (midpoint heading) in the odom frame (REP 105: origin = start pose)."""
    th = pose[2] + 0.5 * omega * dt
    out = pose + np.array([v * dt * np.cos(th), v * dt * np.sin(th), omega * dt])
    out[2] = float(wrap_angle(out[2]))
    return out


def handover_base_tf(backend) -> list[str]:  # pragma: no cover - requires Isaac Sim
    """Delete the OmniGraph nodes the bridge replaces (``GRAPH_NODES_OWNED_BY_BRIDGE``); returns their names.

    Also usable as an ``IsaacBackend.add_reset_hook`` callback registered after ``ros2_reset_hook`` (hooks
    run in registration order).  Idempotent: nodes that are already gone are skipped.
    """
    import omni.graph.core as og
    import omni.usd

    from medortrace.isaac.ros2_bridge import GRAPH_PATH

    path = getattr(backend, "ros2_graph", None) or GRAPH_PATH
    stage = omni.usd.get_context().get_stage()
    present = [n for n in GRAPH_NODES_OWNED_BY_BRIDGE if stage.GetPrimAtPath(f"{path}/{n}").IsValid()]
    if present:
        og.Controller.edit(path, {og.Controller.Keys.DELETE_NODES: present})
        print(f"[medortrace] OmniGraph {path}: removed {present} (odom + odom->base_link come from the bridge)")
    return present


class SimBridgeCore:
    """ROS-free part: episode + backend + command latch + wheel-odometry integration."""

    def __init__(self, scenario: str | dict, seed: int, backend: str = "lite", duration: float | None = None,
                 policy: str | None = None, autonomy_override: dict | None = None, clock_offset: float = 100.0):
        from medortrace.eval.runner import make_backend
        from medortrace.sim.episode import build_episode

        self.cfg = episode_config(scenario, seed, policy, duration, autonomy_override)
        self.seed = int(seed)
        self.ep = build_episode(self.cfg, self.seed)
        self.backend_name = backend
        self.backend = make_backend(backend, self.cfg)
        self.t0 = float(clock_offset)
        self.duration = float(self.ep.workflow.duration)
        self.cmd = VelocityCommand()
        self.odom_pose = np.zeros(3)          # odom frame = where the robot started (REP 105)
        self.n_steps = 0
        self.bundle: SensorBundle = self.backend.reset(self.ep)

    @property
    def t(self) -> float:
        return float(self.backend.t)

    @property
    def dt(self) -> float:
        return float(self.backend.dt)

    @property
    def finished(self) -> bool:
        return self.backend.t >= self.duration - 1e-9

    def set_command(self, v: float | None = None, omega: float | None = None, probe: str | None = "__keep__") -> None:
        if v is not None:
            self.cmd.v = float(v)
        if omega is not None:
            self.cmd.omega = float(omega)
        if probe != "__keep__":
            self.cmd.acoustic_probe_target = probe or None

    def step(self) -> SensorBundle:
        self.bundle = self.backend.step(self.cmd)
        self.n_steps += 1
        o = self.bundle.odom
        if o is not None:                      # dead-reckoned wheel odometry (drifts with the odom bias fault)
            self.odom_pose = integrate_odom(self.odom_pose, o.v, o.omega, self.dt)
        return self.bundle

    def truth(self):
        return self.backend.truth()

    def mission(self) -> dict:
        return mission_from_episode(self.ep, self.cfg, t0=self.t0)

    def close(self) -> None:
        self.backend.close()


# =====================================================================================================================
class BundlePublisher:
    """Publishes a SensorBundle topic by topic (+ /clock, odom->base_link TF, mission, ground truth)."""

    def __init__(self, node, groups: set[str], t0: float, capacity_wh: float = 480.0, include_gt: bool = True,
                 qos_file: str | None = None):
        from diagnostic_msgs.msg import DiagnosticArray
        from medortrace_msgs.msg import (
            AcousticFrame,
            CameraDetectionArray,
            ContactState,
            LandmarkObservationArray,
            RadarDetectionArray,
        )
        from medortrace_msgs.msg import WorkflowEvent as RosWorkflowEvent
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import BatteryState, Imu, PointCloud2
        from std_msgs.msg import String

        from medortrace_ros import convert as cv
        from medortrace_ros.qos import load_qos_config
        from medortrace_ros.topics import TOPICS

        self.node, self.cv, self.groups, self.t0 = node, cv, groups, float(t0)
        self.capacity_wh, self.include_gt = float(capacity_wh), include_gt
        self.qcfg = load_qos_config(qos_file)
        types = {"clock": Clock, "lidar_points": PointCloud2, "camera_detections": CameraDetectionArray,
                 "radar": RadarDetectionArray, "acoustic": AcousticFrame, "landmarks": LandmarkObservationArray,
                 "imu": Imu, "odom": Odometry, "contact": ContactState, "battery": BatteryState,
                 "workflow": RosWorkflowEvent, "mission": String, "ground_truth": Odometry,
                 "diagnostics": DiagnosticArray}
        group_of = {"lidar_points": "lidar", "camera_detections": "camera"}
        self.pub = {k: node.create_publisher(t, TOPICS[k], self.q(k)) for k, t in types.items()
                    if group_of.get(k, k) in groups or k == "diagnostics"}
        self.tf = None
        if "tf" in groups:
            from tf2_ros import TransformBroadcaster
            self.tf = TransformBroadcaster(node)

    def q(self, key: str):
        from medortrace_ros.qos import qos_profile
        return qos_profile(key, self.qcfg)

    def publish_mission(self, mission: dict) -> None:
        if "mission" in self.pub:
            from std_msgs.msg import String
            self.pub["mission"].publish(String(data=json.dumps(mission, separators=(",", ":"))))

    def publish_clock(self, t: float) -> None:
        if "clock" in self.pub:
            self.pub["clock"].publish(self.cv.clock_msg(t + self.t0))

    def publish(self, b: SensorBundle, t: float, odom_pose: np.ndarray | None = None, truth=None) -> None:
        """``t``: simulation time of the bundle; ``odom_pose``: odom-frame pose; ``truth``: TruthSnapshot."""
        cv, t0, pub = self.cv, self.t0, self.pub
        self.publish_clock(t)                               # time first: receivers stamp recv >= sensor stamp
        if self.tf is not None and odom_pose is not None:
            self.tf.sendTransform(cv.transform_msg("odom", "base_link", odom_pose, t, t0))
        if b.odom is not None and "odom" in pub and odom_pose is not None:
            pub["odom"].publish(cv.odom_to_ros(b.odom, odom_pose, t0))
        if "imu" in pub:
            for s in b.imu:
                pub["imu"].publish(cv.imu_to_ros(s, t0))
        if b.lidar is not None and "lidar_points" in pub:
            pub["lidar_points"].publish(cv.lidar_to_pointcloud2(b.lidar, t0, self.include_gt))
        if b.camera is not None and "camera_detections" in pub:
            pub["camera_detections"].publish(cv.camera_to_ros(b.camera, t0))
        if b.radar is not None and "radar" in pub:
            pub["radar"].publish(cv.radar_to_ros(b.radar, t0))
        if b.acoustic is not None and "acoustic" in pub:
            pub["acoustic"].publish(cv.acoustic_to_ros(b.acoustic, t0))
        if b.landmarks is not None and "landmarks" in pub:
            pub["landmarks"].publish(cv.landmarks_to_ros(b.landmarks, t0))
        if b.contact is not None and "contact" in pub:
            pub["contact"].publish(cv.contact_to_ros(b.contact, t0))
        if b.battery_wh is not None and "battery" in pub:
            pub["battery"].publish(cv.battery_to_ros(b.battery_wh, t, t0, self.capacity_wh))
        if "workflow" in pub:
            for ev in b.workflow:
                pub["workflow"].publish(cv.workflow_to_ros(ev, t0, "sim"))
        if truth is not None and "ground_truth" in pub:
            m = cv.odom_to_ros(_zero_odom(t), truth.robot_pose, t0, "map", "base_link_gt")
            m.twist.twist.linear.x, m.twist.twist.angular.z = float(truth.robot_vel[0]), float(truth.robot_vel[1])
            pub["ground_truth"].publish(m)

    def publish_diagnostics(self, t: float, values: dict, warn: bool = False) -> None:
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

        st = DiagnosticStatus(level=DiagnosticStatus.WARN if warn else DiagnosticStatus.OK,
                              name="medortrace/sim_bridge", message="running", hardware_id="medortrace_sim",
                              values=[KeyValue(key=k, value=str(v)) for k, v in values.items()])
        self.pub["diagnostics"].publish(DiagnosticArray(header=self.cv.ros_header(t, "", self.t0), status=[st]))


def _zero_odom(t: float):
    from medortrace.common.msgs import Header, WheelOdometry
    return WheelOdometry(Header(t, t, "base_link"), 0.0, 0.0)


class SimBridgeNode:
    """Steps a SimBridgeCore from a wall-clock timer (lockstep with the autonomy node) and publishes."""

    def __init__(self, node, core: SimBridgeCore, groups: set[str], lockstep: bool = True, rate_factor: float = 1.0,
                 lockstep_timeout_s: float = 2.0, wait_for_autonomy: bool = True, include_gt: bool = True,
                 qos_file: str | None = None):
        from geometry_msgs.msg import Twist
        from std_msgs.msg import String

        from medortrace_ros.topics import TOPICS

        self.node, self.core, self.groups = node, core, groups
        self.log = node.get_logger()
        self.lockstep, self.rate_factor = lockstep, rate_factor
        self.lockstep_timeout, self.wait_for_autonomy = lockstep_timeout_s, wait_for_autonomy
        self.out = BundlePublisher(node, groups, core.t0, float(core.cfg.get("robot", {}).get(
            "battery_capacity_wh", 480.0)), include_gt, qos_file)
        node.create_subscription(Twist, TOPICS["cmd_vel"], self._on_cmd, self.out.q("cmd_vel"))
        node.create_subscription(String, TOPICS["acoustic_probe"], self._on_probe, self.out.q("acoustic_probe"))
        self._cmd_since_step = False
        self._stepping = False
        self._last_step_wall = time.monotonic()
        self._last_clock_wall = 0.0
        self._last_diag_wall = time.monotonic()
        self._diag_t = 0.0
        self._timeouts = 0
        self.done = False
        # ---- initial state (t = 0) ------------------------------------------------------------------------------
        self.out.publish_mission(core.mission())
        self._publish()
        period = core.dt / rate_factor if rate_factor > 0 else 0.001
        self.timer = node.create_timer(period, self._on_timer)
        self.log.info(f"sim bridge: backend={core.backend_name} scenario={core.ep.spec.scenario_id} seed={core.seed} "
                      f"duration={core.duration:.0f}s dt={core.dt} lockstep={lockstep} rate_factor={rate_factor} "
                      f"t0={core.t0} groups={sorted(groups)}")

    # ------------------------------------------------------------------
    def _on_cmd(self, msg) -> None:
        self.core.set_command(msg.linear.x, msg.angular.z)
        self._cmd_since_step = True

    def _on_probe(self, msg) -> None:
        self.core.set_command(probe=msg.data or None)

    def _autonomy_present(self) -> bool:
        from medortrace_ros.topics import TOPICS
        return self.node.count_publishers(TOPICS["cmd_vel"]) > 0

    def _publish(self) -> None:
        core = self.core
        self.out.publish(core.bundle, core.t, core.odom_pose, core.truth() if "ground_truth" in self.groups else None)

    def _on_timer(self) -> None:
        core = self.core
        wall = time.monotonic()
        if core.finished:
            if not self.done:
                self.done = True
                self.log.info(f"episode finished at t={core.t:.1f}s after {core.n_steps} steps "
                              f"({self._timeouts} lockstep timeouts)")
            return
        if not self._stepping:
            if self.wait_for_autonomy and not self._autonomy_present():
                if wall - self._last_clock_wall > 0.5:      # keep late joiners' clocks initialised
                    self._last_clock_wall = wall
                    self.out.publish_clock(core.t)
                return
            self._stepping = True                            # first step needs no command
        elif self.lockstep and not self._cmd_since_step:
            if wall - self._last_step_wall < self.lockstep_timeout:
                return
            self._timeouts += 1
            if self._timeouts in (1, 10, 100) or self._timeouts % 1000 == 0:
                self.log.warn(f"no command within {self.lockstep_timeout}s; stepping with the last one "
                              f"({self._timeouts} timeouts)")
        self._cmd_since_step = False
        self._last_step_wall = wall
        core.step()
        self._publish()
        if wall - self._last_diag_wall >= 1.0:
            rtf = (core.t - self._diag_t) / max(wall - self._last_diag_wall, 1e-6)
            self._last_diag_wall, self._diag_t = wall, core.t
            self.out.publish_diagnostics(core.t, {"sim_time": f"{core.t:.2f}", "steps": core.n_steps,
                                                  "real_time_factor": f"{rtf:.2f}",
                                                  "lockstep_timeouts": self._timeouts,
                                                  "backend": core.backend_name}, warn=self._timeouts > 0)


# =====================================================================================================================
def create(backend, groups: tuple[str, ...] | None = None, t0: float = 0.0):
    """``scripts/isaac/ros2_sim.py --bridge medortrace_ros.sim_bridge_node:create`` factory.

    There the OmniGraph publishes /clock (raw simulation time, hence ``t0 = 0``), the RTX lidar cloud and
    the sensor frames, and drives the wheels from /medortrace/cmd_vel.  Its world-parented base_link TF and
    odometry are removed (:func:`handover_base_tf`; the backend has been reset, so the graph exists); this
    sink publishes odometry + odom->base_link from the bundles' wheel odometry and the remaining
    (custom-message) topics of every bundle plus the mission and ground truth.
    """
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("medortrace_sim_bridge")
    ep = backend.episode
    grp = set(groups or ALL_GROUPS) - set(ISAAC_GRAPH_GROUPS)
    if grp & {"tf", "odom"}:
        handover_base_tf(backend)
    out = BundlePublisher(node, grp, t0, float(ep.cfg.get("robot", {}).get("battery_capacity_wh", 480.0)))
    out.publish_mission(mission_from_episode(ep, ep.cfg, t0=t0))
    odom_pose = np.zeros(3)

    def sink(bundle: SensorBundle) -> None:
        nonlocal odom_pose
        if bundle.odom is not None:
            odom_pose = integrate_odom(odom_pose, bundle.odom.v, bundle.odom.omega, float(backend.dt))
        out.publish(bundle, float(backend.t), odom_pose, backend.truth() if "ground_truth" in grp else None)
        rclpy.spin_once(node, timeout_sec=0.0)

    sink.node = node
    return sink


def _cli(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="MED-OR-TRACE simulator <-> ROS 2 bridge")
    ap.add_argument("--backend", choices=("lite", "isaac"), default=None)
    ap.add_argument("--scenario", default=None, help="scenario config, e.g. scenarios/nominal.yaml")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=None, help="episode length [s] (default: scenario)")
    ap.add_argument("--headless", action="store_true", help="Isaac Sim: no viewport")
    ap.add_argument("--isaac-graph", action="store_true", help="Isaac Sim: add the OmniGraph ROS 2 bridge")
    return ap.parse_known_args(argv)[0]


def main(args=None) -> None:
    argv = list(sys.argv if args is None else args)
    cli = _cli(argv[1:])
    app = None
    if cli.backend == "isaac":
        # Kit must be up before any omni / isaacsim import (and before rclpy in Isaac's bundled ROS libs)
        from medortrace.isaac.app import launch
        app = launch(headless=cli.headless, ros2=True)
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=argv)
    node = rclpy.create_node("medortrace_sim_bridge")

    def p(name, default):
        from medortrace_ros.node_utils import param
        return param(node, name, default)

    backend = cli.backend or str(p("backend", "lite"))
    scenario = cli.scenario or str(p("scenario", "scenarios/nominal.yaml"))
    seed = cli.seed if cli.seed is not None else int(p("seed", 0))
    duration = cli.duration if cli.duration is not None else (float(p("duration", 0.0)) or None)
    policy = str(p("policy", "")) or None
    override = json.loads(str(p("autonomy_overrides_json", "{}")) or "{}")
    isaac_graph = backend == "isaac" and bool(cli.isaac_graph or p("isaac_graph", False))
    groups = set(p("publish_groups", list(ALL_GROUPS)))
    if not bool(p("publish_ground_truth", True)):
        groups.discard("ground_truth")
    if not bool(p("publish_workflow", True)):
        groups.discard("workflow")
    clock_offset = float(p("clock_offset_s", 100.0))
    if isaac_graph:
        # the OmniGraph publishes /clock (raw simulation time), the RTX lidar cloud and the sensor frames; the
        # backend keeps driving the wheels from our command (drive_from_cmd_vel=False: one controller only) and
        # odometry + odom->base_link stay with the bridge (handover_base_tf runs after the graph is built)
        from medortrace.isaac.backend import IsaacBackend
        from medortrace.isaac.ros2_bridge import ros2_reset_hook

        IsaacBackend.add_reset_hook(ros2_reset_hook(drive_from_cmd_vel=False))
        if groups & {"tf", "odom"}:
            IsaacBackend.add_reset_hook(handover_base_tf)
        groups -= set(ISAAC_GRAPH_GROUPS)
        clock_offset = 0.0
    core = SimBridgeCore(scenario, seed, backend, duration, policy, override, clock_offset)
    bridge = SimBridgeNode(node, core, groups, lockstep=bool(p("lockstep", True)),
                           rate_factor=float(p("rate_factor", 1.0)),
                           lockstep_timeout_s=float(p("lockstep_timeout_s", 2.0)),
                           wait_for_autonomy=bool(p("wait_for_autonomy", True)),
                           include_gt=bool(p("lidar_ground_truth_fields", True)),
                           qos_file=str(p("qos_file", "")) or None)
    shutdown_on_finish = bool(p("shutdown_on_finish", False))
    try:
        # Isaac Sim: IsaacBackend.step() advances Kit (world.step renders), so no extra app.update() here
        while rclpy.ok() and not (shutdown_on_finish and bridge.done):
            rclpy.spin_once(node, timeout_sec=0.05)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        core.close()
        node.destroy_node()
        rclpy.try_shutdown()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
