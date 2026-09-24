"""ROS 2 node running the MED-OR-TRACE autonomy stack.

    sensor / workflow topics --callbacks--> SensorBundleAssembler --10 Hz timer--> AutonomyStack.step
        --> /medortrace/cmd_vel (Twist, already safety-gated) + /medortrace/acoustic/probe_target
        --> safety state, pose, map->odom TF (every tick); verdicts, provenance events (as produced);
            item beliefs, scene graph, uncertainty / occupancy grids, NBV goal + path (rate-limited)
    services: /medortrace/operator/ack, /medortrace/verification/verify_claim, /medortrace/verification/explain

The node is a thin shell around :class:`AutonomyRuntime`, which holds all the
logic and imports no ROS module: it can be driven from plain Python (tests,
rosbag-free replays) exactly as the node drives it.  The stack itself is the
unmodified ``medortrace.autonomy.stack.AutonomyStack`` that the in-process
episode runner uses, so simulation and physical-robot behaviour differ only
in where the ``SensorBundle`` comes from.

Mission (``mission_source`` parameter): ``topic`` waits for the latched
``/medortrace/mission`` (published by ``sim_bridge_node`` or a facility
server); ``scenario`` rebuilds it locally from ``scenario`` + ``seed``;
``file`` loads a JSON/YAML mission (physical prototype).  See ``mission.py``.

Time: the stack runs on mission time ``ros_time - t0``.  With ``use_sim_time``
the control timer follows ``/clock``; in lockstep with ``sim_bridge_node``
each simulator step triggers exactly one control tick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from medortrace.autonomy.stack import AutonomyStack, StackInputs
from medortrace.common.config import deep_merge, load_config
from medortrace.common.msgs import SensorBundle, VelocityCommand
from medortrace.provenance.verifier import VerdictRecord
from medortrace.world.workflow import Claim
from medortrace_ros.assembler import CHANNELS, AssemblerConfig, SensorBundleAssembler
from medortrace_ros.records import (
    NbvOut,
    ProvenanceOut,
    SafetyOut,
    VerdictOut,
    beliefs_out,
    nbv_out,
    occupancy_out,
    safety_out,
    uncertainty_out,
    verdict_out,
)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if hasattr(o, "value"):
        return o.value
    return str(o)


@dataclass
class TickResult:
    t: float
    dt: float
    cmd: VelocityCommand
    bundle: SensorBundle
    safety: SafetyOut
    verdicts: list[VerdictOut] = field(default_factory=list)
    provenance: list[ProvenanceOut] = field(default_factory=list)
    nbv: NbvOut | None = None          # only when the goal changed this tick


class AutonomyRuntime:
    """ROS-free core of the autonomy node: assembler -> stack -> publishable records."""

    def __init__(self, inputs: StackInputs, cfg: dict, asm_cfg: AssemblerConfig | None = None,
                 telemetry_keep: int = 600, evidence_k: int = 8):
        from medortrace_ros.records import ProvenanceCursor

        self.stack = AutonomyStack(inputs, cfg)
        self.asm = SensorBundleAssembler(asm_cfg)
        self.cursor = ProvenanceCursor()
        self.duration = float(inputs.duration)
        self.claim_grace = float(inputs.claim_grace)
        self.telemetry_keep = int(telemetry_keep)
        self.evidence_k = int(evidence_k)
        self.verdicts: list[VerdictOut] = []
        self.finalized = False
        self.n_ticks = 0
        self.odom_pose: np.ndarray | None = None      # latest odom-frame pose (for map->odom TF)
        self._goal = None
        self._n_queries = 0

    # ------------------------------------------------------------------
    def push(self, channel: str, msg, recv: float) -> None:
        self.asm.push(channel, msg, recv)

    def tick(self, now: float) -> TickResult:
        bundle, dt = self.asm.assemble(now)
        cmd = self.stack.step(bundle, dt)
        self.n_ticks += 1
        st = self.stack
        verdicts = [verdict_out(st, vr, self.evidence_k) for vr in st.telemetry[-1].verdicts]
        self.verdicts += verdicts
        nbv = None
        if st.goal is not None and st.goal is not self._goal:
            self._goal = st.goal
            nbv = nbv_out(st)
        res = TickResult(float(now), dt, cmd, bundle, safety_out(st, bundle), verdicts, self.cursor.poll(st.prov), nbv)
        self._trim()
        return res

    def _trim(self) -> None:
        """Bound memory on long runs (the stack keeps per-tick telemetry for offline datasets)."""
        k = self.telemetry_keep
        st = self.stack
        if len(st.telemetry) > 2 * k:
            del st.telemetry[:-k]
        if len(st.belief_history) > 2 * k:
            del st.belief_history[:-k]
            del st.belief_history_t[:-k]

    def mission_over(self, now: float) -> bool:
        return now >= self.duration - 1e-3

    def finalize(self, now: float) -> tuple[list[VerdictOut], list[ProvenanceOut]]:
        """Close all open claims (ABSTAIN for anything undecided), as the episode runner does."""
        if self.finalized:
            return [], []
        self.finalized = True
        recs = self.stack.finalize(max(now, self.duration) + self.claim_grace)
        out = [verdict_out(self.stack, vr, self.evidence_k) for vr in recs]
        self.verdicts += out
        return out, self.cursor.poll(self.stack.prov)

    # ---- services -----------------------------------------------------
    def operator_ack(self, pose_hint: np.ndarray | None, now: float) -> tuple[bool, str, str]:
        was_open = bool(self.stack.operator_request_open)
        self.stack.operator_intervention(None if pose_hint is None else np.asarray(pose_hint, float), now)
        msg = ("handover acknowledged; supervisor leaves HANDOVER on the next control tick" if was_open
               else "no handover pending; acknowledgement recorded")
        if pose_hint is not None:
            msg += "; pose re-initialised"
        return True, msg, self.stack.sup.mode.value

    def verify_claim(self, item_id: str, slot_id: str, t_ref: float | None, deadline_s: float | None,
                     claim_id: str | None) -> dict:
        st = self.stack
        ver = st.verifier
        if claim_id and claim_id in ver.done:
            vr = ver.done[claim_id]
            return {"accepted": True, "claim_id": claim_id, "decided": True, "message": "already decided",
                    "verdict": verdict_out(st, vr, self.evidence_k), "posterior": st.items.prob(vr.item_id, vr.slot_id)}
        if item_id not in st.items.items:
            return {"accepted": False, "claim_id": "", "message": f"unknown item {item_id!r}"}
        if slot_id not in st.items.sidx:
            return {"accepted": False, "claim_id": "", "message": f"unknown slot {slot_id!r}"}
        if claim_id and claim_id in ver.open:
            return {"accepted": True, "claim_id": claim_id, "decided": False, "message": "already pending",
                    "posterior": st.items.prob(item_id, slot_id)}
        self._n_queries += 1
        cid = claim_id or f"q_{self._n_queries:04d}_{item_id}"
        t_ref = float(st.t if t_ref is None else t_ref)
        grace = float(deadline_s) if deadline_s and deadline_s > 0 else self.claim_grace
        ver.add_claim(Claim(cid, t_ref, t_ref + grace, item_id, slot_id, source_event=None, kind="query"))
        note = "" if t_ref >= st.t - 1e-6 else " (t_ref in the past: smoothing starts from the current belief)"
        return {"accepted": True, "claim_id": cid, "decided": False, "message": "queued" + note,
                "posterior": st.items.prob(item_id, slot_id)}

    def explain(self, claim_id: str, k: int = 5) -> dict:
        st = self.stack
        vr: VerdictRecord | None = st.verifier.done.get(claim_id)
        if vr is None:
            state = "pending" if claim_id in st.verifier.open else "unknown"
            return {"found": False, "message": f"claim {claim_id!r} is {state}"}
        ex = st.prov.explain(f"verdict:{claim_id}", k or 5)
        chain = st.prov.handoff_chain(vr.item_id)
        return {"found": True, "message": "ok", "verdict": verdict_out(st, vr, self.evidence_k),
                "explanation_json": json.dumps(ex, default=_json_default, sort_keys=True),
                "custody_chain_json": json.dumps(chain, default=_json_default, sort_keys=True)}

    # ---- audit export -------------------------------------------------
    def export_audit(self, out_dir: str | Path) -> Path:
        """Provenance (PROV-JSON), verdicts, safety events, scene graph and middleware health."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        st = self.stack
        (d / "provenance.json").write_text(json.dumps(st.prov.to_prov_json(), default=_json_default))
        (d / "scene_graph.json").write_text(json.dumps(st.scene_graph(), default=_json_default))
        with open(d / "verdicts.jsonl", "w") as f:
            for v in self.verdicts:
                f.write(json.dumps(v.__dict__, default=_json_default) + "\n")
        with open(d / "safety_events.jsonl", "w") as f:
            for e in st.sup.events:
                f.write(json.dumps(e.__dict__, default=_json_default) + "\n")
        (d / "middleware_health.json").write_text(json.dumps(
            {"assembler": self.asm.health(st.t), "resets": self.asm.n_resets, "ticks": self.n_ticks,
             "provenance_chain_valid": st.prov.verify_chain()}, indent=1, default=_json_default))
        return d


# =====================================================================================================================
# ROS 2 shell
# =====================================================================================================================
class AutonomyNode:
    """rclpy wiring around :class:`AutonomyRuntime` (composition keeps this module importable without ROS)."""

    def __init__(self, node):
        from medortrace_ros.qos import load_qos_config

        self.node = node
        self.log = node.get_logger()
        self.qos_cfg = load_qos_config(self._param("qos_file", "") or None)
        self.rt: AutonomyRuntime | None = None
        self.t0: float | None = None
        self.finished = False
        self._last_pub: dict[str, float] = {}
        self._first_tick_wall = None
        p = self._param
        self.rate = float(p("control_rate_hz", 10.0))
        self.frames = {"map": p("map_frame", "map"), "odom": p("odom_frame", "odom"),
                       "base": p("base_frame", "base_link")}
        self.pub_rates = {"beliefs": float(p("beliefs_rate_hz", 2.0)),
                          "scene_graph": float(p("scene_graph_rate_hz", 1.0)),
                          "maps": float(p("maps_rate_hz", 1.0)), "diagnostics": float(p("diagnostics_rate_hz", 1.0))}
        self.required = list(p("required_channels", ["odom"]))
        self.ready_timeout = float(p("ready_timeout_s", 5.0))
        self.audit_dir = str(p("audit_dir", ""))
        self.stop_at_end = bool(p("stop_at_mission_end", True))
        self.publish_tf = bool(p("publish_tf", True))
        self.battery_cap = float(p("battery.capacity_wh", 480.0))
        self.radar_input = str(p("radar.input", "detections"))
        self.radar_opts = {"velocity_field": str(p("radar.velocity_field", "velocity")),
                           "rcs_field": str(p("radar.rcs_field", "intensity")),
                           "rcs_is_db": bool(p("radar.rcs_is_db", False))}
        self.lidar_height = float(p("lidar.sensor_height", -1.0))
        self.lidar_pattern = None
        regrid = bool(p("lidar.regrid_to_pattern", False))
        if bool(p("lidar.fill_no_return_rays", False)):            # pre-regrid name of the same switch
            self.log.warn("parameter lidar.fill_no_return_rays is deprecated; use lidar.regrid_to_pattern")
            regrid = True
        if regrid:
            # az_res_deg / max_range <= 0: the stack's sensors.lidar values of the mission cfg (set in _start)
            self.lidar_pattern = {"rings": int(p("lidar.pattern.rings", 16)),
                                  "elev_min_deg": float(p("lidar.pattern.elev_min_deg", -15.0)),
                                  "elev_max_deg": float(p("lidar.pattern.elev_max_deg", 15.0)),
                                  "az_res_deg": float(p("lidar.pattern.az_res_deg", 0.0)),
                                  "max_range": float(p("lidar.pattern.max_range", 0.0))}
        self.max_age = {c: float(p(f"max_age.{c}", v)) for c, v in AssemblerConfig().max_age_s.items()
                        if c != "workflow"}
        src = str(p("mission_source", "topic"))
        self.log.info(f"mission source: {src}")
        if src == "topic":
            from std_msgs.msg import String

            from medortrace_ros.qos import qos_profile
            from medortrace_ros.topics import TOPICS

            self._mission_sub = node.create_subscription(String, TOPICS["mission"], self._on_mission,
                                                         qos_profile("mission", self.qos_cfg))
        elif src == "scenario":
            from medortrace_ros.mission import mission_for_scenario

            dur = float(p("duration", 0.0)) or None
            # same origin as sim_bridge_node's /clock offset, so both sides agree on mission time
            self._start(mission_for_scenario(str(p("scenario", "scenarios/nominal.yaml")), int(p("seed", 0)),
                                             duration=dur, t0=float(p("clock_offset_s", 100.0))))
        elif src == "file":
            from medortrace_ros.mission import load_mission_file

            self._start(load_mission_file(str(p("mission_file", ""))))
        else:
            raise ValueError(f"mission_source must be topic|scenario|file, got {src!r}")

    # ------------------------------------------------------------------
    def _param(self, name: str, default):
        from medortrace_ros.node_utils import param
        return param(self.node, name, default)

    def _ros_now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _mission_now(self) -> float:
        return self._ros_now() - (self.t0 or 0.0)

    def _on_mission(self, msg) -> None:
        if self.rt is not None:
            return
        from medortrace_ros.mission import mission_from_json

        try:
            self._start(mission_from_json(msg.data))
        except Exception as e:  # noqa: BLE001 - keep the node alive, report the bad mission
            self.log.error(f"invalid mission: {e}")

    # ------------------------------------------------------------------
    def _start(self, mission: dict) -> None:
        from medortrace_ros.frames import lidar_mount
        from medortrace_ros.mission import stack_setup

        p = self._param
        base_name = str(p("base_config", "scenarios/default.yaml"))
        base = load_config(base_name) if base_name else {}
        overrides = json.loads(str(p("autonomy_overrides_json", "{}")) or "{}")
        pol = str(p("policy", ""))
        if pol:
            overrides = deep_merge(overrides, {"autonomy": {"policy": pol}})
        inputs, cfg = stack_setup(mission, base, overrides)
        lidar_cfg = cfg.get("sensors", {}).get("lidar", {})
        if self.lidar_height <= 0:
            self.lidar_height = float(lidar_cfg.get("mount_height", lidar_mount()[1]))
        if self.lidar_pattern is not None:
            from medortrace_ros.convert import pattern_ray_count, resolve_lidar_pattern

            self.lidar_pattern = resolve_lidar_pattern(self.lidar_pattern, lidar_cfg)
            self.log.info(f"lidar clouds re-binned onto {self.lidar_pattern} "
                          f"({pattern_ray_count(self.lidar_pattern)} rays per scan)")
        origin = str(p("time_origin", "mission"))
        now = self._ros_now()
        if origin == "zero":
            self.t0 = 0.0
        elif origin == "mission" and mission.get("t0") is not None:
            self.t0 = float(mission["t0"])
            if now > 0 and abs(now - self.t0) > 1e6:   # e.g. a simulation mission (t0=100) on a wall clock
                self.log.warn(f"mission t0={self.t0} is implausible for ROS time {now:.0f}; using the start time")
                self.t0 = now
        else:                                          # "start" (or mission without t0)
            self.t0 = now
        asm = AssemblerConfig.from_dict({"control_rate_hz": self.rate, "max_age_s": self.max_age})
        self.rt = AutonomyRuntime(inputs, cfg, asm, int(p("telemetry_keep", 600)), int(p("verdict_evidence_k", 8)))
        self.log.info(f"mission {mission.get('scenario_id')!r} seed={mission.get('seed')} t0={self.t0:.3f} "
                      f"duration={inputs.duration:.0f}s policy={self.rt.stack.policy} "
                      f"items={len(inputs.items)} slots={len(inputs.slots)}")
        if getattr(self, "_mission_sub", None) is None:
            # file / scenario missions: re-publish (latched) with the resolved origin so that the workflow
            # gateway and dashboards share this node's mission clock
            from std_msgs.msg import String

            from medortrace_ros.mission import mission_to_json
            from medortrace_ros.qos import qos_profile
            from medortrace_ros.topics import TOPICS

            self._mission_pub = self.node.create_publisher(String, TOPICS["mission"],
                                                           qos_profile("mission", self.qos_cfg))
            self._mission_pub.publish(String(data=mission_to_json({**mission, "t0": self.t0})))
        self._make_io()

    def _make_io(self) -> None:
        from diagnostic_msgs.msg import DiagnosticArray
        from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
        from medortrace_msgs.msg import (
            AcousticFrame,
            CameraDetectionArray,
            ClaimVerdict,
            ContactState,
            ItemBeliefArray,
            LandmarkObservationArray,
            NextBestView,
            ProvenanceEvent,
            RadarDetectionArray,
            SafetyState,
            SceneGraph,
            UncertaintyGrid,
        )
        from medortrace_msgs.msg import WorkflowEvent as RosWorkflowEvent
        from medortrace_msgs.srv import ExplainVerdict, OperatorAck, VerifyClaim
        from nav_msgs.msg import OccupancyGrid, Odometry, Path
        from sensor_msgs.msg import BatteryState, Imu, PointCloud2
        from std_msgs.msg import String

        from medortrace_ros import convert as cv
        from medortrace_ros.qos import qos_profile
        from medortrace_ros.topics import SERVICES, TOPICS

        n, q = self.node, (lambda k: qos_profile(k, self.qos_cfg))
        self.cv = cv
        pubs = {
            "item_beliefs": ItemBeliefArray, "verdicts": ClaimVerdict, "provenance": ProvenanceEvent,
            "safety": SafetyState, "scene_graph": SceneGraph, "uncertainty": UncertaintyGrid,
            "occupancy": OccupancyGrid, "nbv": NextBestView, "path": Path, "pose": PoseWithCovarianceStamped,
            "acoustic_probe": String, "diagnostics": DiagnosticArray,
        }
        self.pub = {k: n.create_publisher(t, TOPICS[k], q(k)) for k, t in pubs.items()}
        self.tf = None
        if self.publish_tf:
            from tf2_ros import TransformBroadcaster
            self.tf = TransformBroadcaster(n)
        t0 = self.t0
        h = self.lidar_height

        def sub(key, typ, conv, channel=None):
            def cb(msg):
                if self.rt is None or self.finished:
                    return
                recv = self._mission_now()
                try:
                    m = conv(msg, recv)
                except Exception as e:  # noqa: BLE001 - a malformed message must not kill the control loop
                    self.log.warn(f"dropping malformed {key} message: {e}", throttle_duration_sec=5.0)
                    return
                if m is not None:
                    self.rt.push(channel or key, m, recv)
            return n.create_subscription(typ, TOPICS[key], cb, q(key))

        def odom_conv(msg, recv):
            self.rt.odom_pose = cv.pose2d_from_pose(msg.pose.pose)
            return cv.odom_from_ros(msg, recv, t0)

        def batt_conv(msg, recv):
            return cv.battery_from_ros(msg, self.battery_cap)

        self.subs = [
            sub("lidar_points", PointCloud2,
                lambda m, r: cv.lidar_from_pointcloud2(m, r, t0, h, self.lidar_pattern), "lidar"),
            sub("camera_detections", CameraDetectionArray, lambda m, r: cv.camera_from_ros(m, r, t0), "camera"),
            sub("acoustic", AcousticFrame, lambda m, r: cv.acoustic_from_ros(m, r, t0)),
            sub("landmarks", LandmarkObservationArray, lambda m, r: cv.landmarks_from_ros(m, r, t0)),
            sub("imu", Imu, lambda m, r: cv.imu_from_ros(m, r, t0)),
            sub("odom", Odometry, odom_conv),
            sub("contact", ContactState, lambda m, r: cv.contact_from_ros(m, r, t0)),
            sub("battery", BatteryState, batt_conv),
            sub("workflow", RosWorkflowEvent, lambda m, r: cv.workflow_from_ros(m, t0)),
        ]
        if self.radar_input == "pointcloud2":
            self.subs.append(sub("radar_points", PointCloud2,
                                 lambda m, r: cv.radar_from_pointcloud2(m, r, t0, **self.radar_opts), "radar"))
        else:
            self.subs.append(sub("radar", RadarDetectionArray, lambda m, r: cv.radar_from_ros(m, r, t0)))
        self.srv = [
            n.create_service(OperatorAck, SERVICES["operator_ack"][0], self._srv_ack),
            n.create_service(VerifyClaim, SERVICES["verify_claim"][0], self._srv_verify),
            n.create_service(ExplainVerdict, SERVICES["explain"][0], self._srv_explain),
        ]
        # cmd_vel is created last: sim_bridge_node treats its appearance as "autonomy ready"
        self.pub["cmd_vel"] = n.create_publisher(Twist, TOPICS["cmd_vel"], q("cmd_vel"))
        self.timer = n.create_timer(1.0 / self.rate, self._tick)

    # ------------------------------------------------------------------
    def _due(self, key: str, now: float) -> bool:
        hz = self.pub_rates[key]
        if hz <= 0:
            return False
        if now - self._last_pub.get(key, -1e9) >= 1.0 / hz - 1e-6 or now < self._last_pub.get(key, 0.0):
            self._last_pub[key] = now
            return True
        return False

    def _tick(self) -> None:
        import time as _time

        rt, cv, t0 = self.rt, self.cv, self.t0
        if rt is None:
            return
        if self._ros_now() <= 0.0:               # use_sim_time and no /clock yet
            return
        now = self._mission_now()
        if self.finished:
            self.pub["cmd_vel"].publish(cv.cmd_to_twist(VelocityCommand()))
            return
        if not rt.asm.ready(tuple(self.required)):
            if self._first_tick_wall is None:
                self._first_tick_wall = _time.monotonic()
            if _time.monotonic() - self._first_tick_wall < self.ready_timeout:
                # hold still, and keep a lockstep simulator stepping until the required data arrives
                self.pub["cmd_vel"].publish(cv.cmd_to_twist(VelocityCommand()))
                return
            self.log.warn(f"required channels {self.required} silent after {self.ready_timeout:.0f}s; "
                          "starting anyway", once=True)
        res = rt.tick(now)
        self.pub["cmd_vel"].publish(cv.cmd_to_twist(res.cmd))
        self.pub["acoustic_probe"].publish(self._string(res.cmd.acoustic_probe_target or ""))
        self._publish_outputs(res.t, res.safety, res.verdicts, res.provenance)
        st = rt.stack
        self.pub["pose"].publish(cv.pose_cov_to_ros(st.ekf.x, st.ekf.P, now, t0, self.frames["map"]))
        if self.tf is not None and rt.odom_pose is not None:
            from medortrace_ros.frames import compose_map_to_odom
            self.tf.sendTransform(cv.transform_msg(self.frames["map"], self.frames["odom"],
                                                   compose_map_to_odom(st.ekf.x, rt.odom_pose), now, t0))
        if res.nbv is not None:
            self.pub["nbv"].publish(cv.nbv_to_ros(res.nbv, t0, self.frames["map"]))
            self.pub["path"].publish(cv.path_to_ros(res.nbv.path, now, t0, self.frames["map"]))
        if self._due("beliefs", now):
            self.pub["item_beliefs"].publish(cv.beliefs_to_ros(beliefs_out(st), t0, self.frames["map"]))
        if self._due("scene_graph", now):
            self.pub["scene_graph"].publish(cv.scene_graph_to_ros(st.scene_graph(), now, t0, self.frames["map"]))
        if self._due("maps", now):
            self.pub["uncertainty"].publish(cv.uncertainty_to_ros(uncertainty_out(st), t0, self.frames["map"]))
            self.pub["occupancy"].publish(cv.occupancy_to_ros(occupancy_out(st), t0, self.frames["map"]))
        if self._due("diagnostics", now):
            self.pub["diagnostics"].publish(self._diagnostics(now))
        if self.stop_at_end and rt.mission_over(now):
            self._finish(now)

    def _publish_outputs(self, t, safety, verdicts, provenance) -> None:
        cv, t0 = self.cv, self.t0
        if safety is not None:
            self.pub["safety"].publish(cv.safety_to_ros(safety, t0, self.frames["base"]))
        for p in provenance:
            self.pub["provenance"].publish(cv.provenance_to_ros(p, t0, self.frames["map"]))
        for v in verdicts:
            self.pub["verdicts"].publish(cv.verdict_to_ros(v, t0, self.frames["map"]))
            self.log.info(f"verdict {v.claim_id}: {v.verdict} p={v.posterior:.2f} ({v.reason})")

    def _finish(self, now: float) -> None:
        verdicts, prov = self.rt.finalize(now)
        self._publish_outputs(now, None, verdicts, prov)
        self.pub["cmd_vel"].publish(self.cv.cmd_to_twist(VelocityCommand()))
        self.finished = True
        counts = {}
        for v in self.rt.verdicts:
            counts[v.verdict] = counts.get(v.verdict, 0) + 1
        self.log.info(f"mission complete at t={now:.1f}s: verdicts {counts}; "
                      f"provenance chain valid={self.rt.stack.prov.verify_chain()}")
        self._export()

    def _export(self) -> None:
        if self.audit_dir and self.rt is not None and not getattr(self, "_exported", False):
            self._exported = True
            d = self.rt.export_audit(self.audit_dir)
            self.log.info(f"audit record written to {d}")

    @staticmethod
    def _string(s: str):
        from std_msgs.msg import String
        return String(data=s)

    def _diagnostics(self, now: float):
        from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

        arr = DiagnosticArray(header=self.cv.ros_header(now, "", self.t0))
        for ch, h in self.rt.asm.health(now).items():
            age = h["age_s"]
            lim = self.rt.asm.cfg.max_age_s.get(ch, float("inf"))
            if h["received"] == 0:
                level, msg = DiagnosticStatus.STALE, "no data"
            elif age is not None and np.isfinite(lim) and age > 3 * lim:
                level, msg = DiagnosticStatus.WARN, f"stale ({age:.2f}s)"
            else:
                level, msg = DiagnosticStatus.OK, "ok"
            arr.status.append(DiagnosticStatus(level=level, name=f"medortrace/assembler/{ch}", message=msg,
                                               hardware_id="medortrace_autonomy",
                                               values=[KeyValue(key=k, value=str(v)) for k, v in h.items()]))
        return arr

    # ---- services -----------------------------------------------------
    def _srv_ack(self, req, resp):
        from medortrace_ros.records import MODE_CODE

        hint = np.array([req.pose_hint.x, req.pose_hint.y, req.pose_hint.theta]) if req.relocalize else None
        ok, msg, mode = self.rt.operator_ack(hint, self._mission_now())
        self.log.info(f"operator {req.operator_id or '?'}: {msg} {req.note}")
        resp.accepted, resp.message, resp.mode_after = ok, msg, MODE_CODE[mode]
        return resp

    def _srv_verify(self, req, resp):
        cv = self.cv
        t_ref = None if (req.t_ref.sec == 0 and req.t_ref.nanosec == 0) else cv.to_sec(req.t_ref) - self.t0
        r = self.rt.verify_claim(req.item_id, req.slot_id, t_ref, req.deadline_s, req.claim_id or None)
        resp.accepted, resp.claim_id, resp.message = r["accepted"], r.get("claim_id", ""), r["message"]
        resp.current_posterior = float(r.get("posterior", float("nan")))
        resp.decided = bool(r.get("decided", False))
        if r.get("verdict") is not None:
            resp.verdict = cv.verdict_to_ros(r["verdict"], self.t0, self.frames["map"])
        return resp

    def _srv_explain(self, req, resp):
        r = self.rt.explain(req.claim_id, int(req.max_evidence) or 5)
        resp.found, resp.message = r["found"], r["message"]
        if r["found"]:
            resp.verdict = self.cv.verdict_to_ros(r["verdict"], self.t0, self.frames["map"])
            resp.explanation_json, resp.custody_chain_json = r["explanation_json"], r["custody_chain_json"]
        return resp

    def shutdown(self) -> None:
        if self.rt is None:
            return
        try:
            self.pub["cmd_vel"].publish(self.cv.cmd_to_twist(VelocityCommand()))
        except Exception:  # noqa: BLE001 - context may already be shut down
            pass
        self._export()


def main(args=None) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = rclpy.create_node("medortrace_autonomy")
    app = None
    try:
        app = AutonomyNode(node)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if app is not None:
            app.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


__all__ = ["AutonomyRuntime", "AutonomyNode", "TickResult", "CHANNELS", "main"]


if __name__ == "__main__":
    main()
