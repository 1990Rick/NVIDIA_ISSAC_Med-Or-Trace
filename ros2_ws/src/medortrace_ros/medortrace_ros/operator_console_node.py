"""Operator console: answers HANDOVER requests through the ``/medortrace/operator/ack`` service.

When the safety supervisor cannot resolve a persisting stop it escalates to
HANDOVER and the robot waits for a human.  This node watches
``/medortrace/safety/state`` and, per ``mode``:

``auto``         simulated remote operator, identical to the episode runner's
                 ``SimulatedOperator``: answers after a uniform random delay
                 (``delay_min_s``..``delay_max_s``) and re-localises the robot
                 from ground truth + noise (``/medortrace/sim/ground_truth/odom``,
                 standing in for the OR camera system) when available;
``rviz``         waits for an RViz "2D Pose Estimate" (``/initialpose``) and
                 acknowledges with that pose (``ack_without_pose_after_s`` > 0
                 also allows a plain acknowledgement after that long);
``interactive``  prompts on the terminal: ``y`` acknowledge, ``r x y theta``
                 acknowledge and re-localise, anything else ignores.

It also prints every ABSTAIN verdict (a request for human confirmation) with
its reason.  The decision logic (:class:`OperatorPolicy`) is ROS-free.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

HANDOVER_CODE = 4


@dataclass
class AckDecision:
    relocalize: bool
    pose: np.ndarray | None
    note: str


class OperatorPolicy:
    """When and how to acknowledge a pending HANDOVER (mirrors eval.runner.SimulatedOperator)."""

    def __init__(self, mode: str = "auto", delay_s: tuple[float, float] = (8.0, 20.0), seed: int = 0,
                 noise: tuple[float, float, float] = (0.02, 0.02, 0.01), ack_without_pose_after_s: float = 0.0):
        if mode not in ("auto", "rviz", "interactive"):
            raise ValueError(f"operator mode must be auto|rviz|interactive, got {mode!r}")
        self.mode = mode
        self.delay = delay_s
        self.rng = np.random.default_rng(seed)
        self.noise = np.asarray(noise, float)
        self.plain_after = float(ack_without_pose_after_s)
        self.pending_since: float | None = None
        self.due: float | None = None
        self.log: list[dict] = []

    def on_state(self, t: float, handover_open: bool) -> bool:
        """Feed the latest safety state; True when a new request just opened."""
        if handover_open and self.pending_since is None:
            self.pending_since = t
            self.due = t + float(self.rng.uniform(*self.delay))
            return True
        if not handover_open and self.pending_since is not None:
            self.pending_since = None                     # resolved elsewhere (another console, a restart)
            self.due = None
        return False

    def poll(self, t: float, truth_pose: np.ndarray | None = None, rviz_pose: np.ndarray | None = None,
             typed: AckDecision | None = None) -> AckDecision | None:
        if self.pending_since is None:
            return None
        dec = None
        if self.mode == "auto" and t >= self.due:
            if truth_pose is not None:
                dec = AckDecision(True, np.asarray(truth_pose, float) + self.rng.normal(0, self.noise),
                                  "auto: relocalise+resume")
            else:
                dec = AckDecision(False, None, "auto: resume")
        elif self.mode == "rviz":
            if rviz_pose is not None:
                dec = AckDecision(True, np.asarray(rviz_pose, float), "rviz: 2D pose estimate")
            elif self.plain_after > 0 and t - self.pending_since >= self.plain_after:
                dec = AckDecision(False, None, "rviz: acknowledged without pose")
        elif self.mode == "interactive" and typed is not None:
            dec = typed
        if dec is not None:
            self.log.append({"t_request": self.pending_since, "t_ack": t, "duration": t - self.pending_since,
                             "action": dec.note})
            self.pending_since = None
            self.due = None
        return dec


def parse_operator_input(line: str) -> AckDecision | None:
    """Terminal answer -> decision: ``y`` | ``r x y theta``."""
    parts = line.strip().lower().split()
    if not parts:
        return None
    if parts[0] in ("y", "yes", "ack"):
        return AckDecision(False, None, "interactive: acknowledged")
    if parts[0] == "r" and len(parts) == 4:
        try:
            return AckDecision(True, np.array([float(v) for v in parts[1:4]]), "interactive: relocalised")
        except ValueError:
            return None
    return None


# =====================================================================================================================
class OperatorConsoleNode:
    def __init__(self, node):
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from medortrace_msgs.msg import ClaimVerdict, SafetyState
        from medortrace_msgs.srv import OperatorAck
        from nav_msgs.msg import Odometry

        from medortrace_ros.qos import load_qos_config, qos_profile
        from medortrace_ros.topics import SERVICES, TOPICS

        self.node = node
        self.log = node.get_logger()

        def p(name, default):
            from medortrace_ros.node_utils import param
            return param(node, name, default)

        self.operator_id = str(p("operator_id", "operator_console"))
        delay = (float(p("delay_min_s", 8.0)), float(p("delay_max_s", 20.0)))
        self.policy = OperatorPolicy(str(p("mode", "auto")), delay, int(p("seed", 0)),
                                     ack_without_pose_after_s=float(p("ack_without_pose_after_s", 0.0)))
        self.use_truth = bool(p("relocalize_from_ground_truth", True))
        qcfg = load_qos_config(str(p("qos_file", "")) or None)
        q = lambda k: qos_profile(k, qcfg)
        node.create_subscription(SafetyState, TOPICS["safety"], self._on_safety, q("safety"))
        node.create_subscription(ClaimVerdict, TOPICS["verdicts"], self._on_verdict, q("verdicts"))
        node.create_subscription(Odometry, TOPICS["ground_truth"], self._on_truth, q("ground_truth"))
        node.create_subscription(PoseWithCovarianceStamped, TOPICS["initialpose"], self._on_initialpose,
                                 q("initialpose"))
        self.client = node.create_client(OperatorAck, SERVICES["operator_ack"][0])
        self.truth: np.ndarray | None = None
        self.rviz_pose: np.ndarray | None = None
        self.typed: AckDecision | None = None
        self._in_flight = False
        self._lock = threading.Lock()
        self.timer = node.create_timer(0.1, self._on_timer)
        self.log.info(f"operator console: mode={self.policy.mode}")

    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _pose2d(p) -> np.ndarray:
        from medortrace_ros.convert import pose2d_from_pose
        return pose2d_from_pose(p)

    def _on_truth(self, msg) -> None:
        self.truth = self._pose2d(msg.pose.pose)

    def _on_initialpose(self, msg) -> None:
        self.rviz_pose = self._pose2d(msg.pose.pose)

    def _on_verdict(self, msg) -> None:
        if msg.verdict == msg.ABSTAIN:
            self.log.warn(f"ABSTAIN {msg.claim_id}: is {msg.item_id} in {msg.slot_id}? robot cannot tell "
                          f"(p={msg.posterior:.2f}, {msg.reason}) - please confirm")

    def _on_safety(self, msg) -> None:
        t = self._now()
        opened = self.policy.on_state(t, msg.mode == HANDOVER_CODE and msg.operator_request_open)
        if opened:
            self.rviz_pose = None
            self.log.warn(f"HANDOVER requested ({'; '.join(msg.reasons)}) - mode {self.policy.mode}")
            if self.policy.mode == "interactive":
                threading.Thread(target=self._prompt, daemon=True).start()

    def _prompt(self) -> None:
        try:
            line = input("[medortrace] HANDOVER - 'y' to acknowledge, 'r x y theta' to re-localise: ")
        except EOFError:
            return
        with self._lock:
            self.typed = parse_operator_input(line)

    def _on_timer(self) -> None:
        if self._in_flight:
            return
        with self._lock:
            typed, self.typed = self.typed, None
        dec = self.policy.poll(self._now(), self.truth if self.use_truth else None, self.rviz_pose, typed)
        if dec is None:
            return
        if not self.client.service_is_ready():
            self.log.error("operator ack service unavailable; request stays open")
            self.policy.on_state(self._now(), True)
            return
        from medortrace_msgs.srv import OperatorAck

        from medortrace_ros.convert import pose2d_msg

        req = OperatorAck.Request(operator_id=self.operator_id, relocalize=dec.relocalize, note=dec.note)
        if dec.relocalize and dec.pose is not None:
            req.pose_hint = pose2d_msg(dec.pose)
        self._in_flight = True
        fut = self.client.call_async(req)
        fut.add_done_callback(self._on_ack_done)

    def _on_ack_done(self, fut) -> None:
        self._in_flight = False
        try:
            r = fut.result()
            self.log.info(f"operator ack: accepted={r.accepted} {r.message}")
        except Exception as e:  # noqa: BLE001 - report and keep the console running
            self.log.error(f"operator ack failed: {e}")


def main(args=None) -> None:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init(args=args)
    node = rclpy.create_node("medortrace_operator_console")
    try:
        OperatorConsoleNode(node)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
