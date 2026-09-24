"""Topic / service registry of the MED-OR-TRACE ROS 2 graph (no ROS imports).

Single source of truth for names, types, QoS profile keys, nominal rates and the
producing / consuming nodes.  The Isaac Sim OmniGraph bridge
(``medortrace.isaac.ros2_bridge.TOPICS``) is merged in unchanged, so the topics
published by the digital twin, by ``sim_bridge_node`` and by the physical
robot's drivers (after remapping, see ``launch/robot.launch.py``) are the same.
docs/ros2_message_graph.md renders this table.
"""

from __future__ import annotations

from dataclasses import dataclass

from medortrace.isaac.ros2_bridge import TOPICS as ISAAC_TOPICS

NS = "/medortrace"


@dataclass(frozen=True)
class TopicSpec:
    key: str
    name: str
    type: str                    # ROS interface type, e.g. "sensor_msgs/msg/PointCloud2"
    qos: str                     # profile key in config/qos.yaml
    rate: str                    # nominal rate (lite defaults)
    producer: str
    consumer: str


_SPECS = [
    # ---- time & transforms --------------------------------------------------------------------------------------
    TopicSpec("clock", ISAAC_TOPICS["clock"], "rosgraph_msgs/msg/Clock", "clock", "1/sim step",
              "sim_bridge or Isaac OmniGraph", "all nodes (use_sim_time)"),
    TopicSpec("tf", ISAAC_TOPICS["tf"], "tf2_msgs/msg/TFMessage", "tf", "10 Hz",
              "sim_bridge / base driver (odom->base_link), autonomy (map->odom)", "rviz, drivers"),
    # ---- raw sensors ----------------------------------------------------------------------------------------------
    TopicSpec("lidar_points", ISAAC_TOPICS["lidar_points"], "sensor_msgs/msg/PointCloud2", "sensor_dense", "5 Hz",
              "sim_bridge or Isaac RTX lidar or lidar driver", "autonomy"),
    TopicSpec("rgb", ISAAC_TOPICS["rgb"], "sensor_msgs/msg/Image", "sensor_dense", "5 Hz",
              "Isaac OmniGraph or camera driver", "detector (external), rviz"),
    TopicSpec("depth", ISAAC_TOPICS["depth"], "sensor_msgs/msg/Image", "sensor_dense", "5 Hz",
              "Isaac OmniGraph or camera driver", "detector (external)"),
    TopicSpec("camera_info", ISAAC_TOPICS["camera_info"], "sensor_msgs/msg/CameraInfo", "sensor_data", "5 Hz",
              "Isaac OmniGraph or camera driver", "detector (external)"),
    TopicSpec("camera_detections", f"{NS}/perception/camera/detections", "medortrace_msgs/msg/CameraDetectionArray",
              "sensor_data", "5 Hz", "sim_bridge or detector", "autonomy"),
    TopicSpec("landmarks", f"{NS}/perception/landmarks", "medortrace_msgs/msg/LandmarkObservationArray",
              "sensor_data", "5 Hz", "sim_bridge or fiducial detector", "autonomy"),
    TopicSpec("radar", f"{NS}/sensors/radar/detections", "medortrace_msgs/msg/RadarDetectionArray", "sensor_data",
              "10 Hz", "sim_bridge or radar adapter", "autonomy"),
    TopicSpec("radar_points", f"{NS}/sensors/radar/points", "sensor_msgs/msg/PointCloud2", "sensor_data", "10 Hz",
              "radar driver (physical robot)", "autonomy (radar.input=pointcloud2)"),
    TopicSpec("acoustic", f"{NS}/sensors/acoustic/echoes", "medortrace_msgs/msg/AcousticFrame", "sensor_data",
              "2 Hz while probing", "sim_bridge or acoustic probe driver", "autonomy"),
    TopicSpec("imu", f"{NS}/sensors/imu", "sensor_msgs/msg/Imu", "sensor_data", "100 Hz",
              "sim_bridge or IMU driver", "autonomy"),
    TopicSpec("odom", ISAAC_TOPICS["odom"], "nav_msgs/msg/Odometry", "sensor_data", "10 Hz",
              "sim_bridge or Isaac OmniGraph or base driver", "autonomy"),
    TopicSpec("contact", f"{NS}/sensors/contact", "medortrace_msgs/msg/ContactState", "sensor_data", "10 Hz",
              "sim_bridge or bumper / arm driver", "autonomy"),
    TopicSpec("battery", f"{NS}/battery", "sensor_msgs/msg/BatteryState", "sensor_data", "1-10 Hz",
              "sim_bridge or BMS driver", "autonomy"),
    # ---- workflow / mission ---------------------------------------------------------------------------------------
    TopicSpec("workflow", f"{NS}/workflow/events", "medortrace_msgs/msg/WorkflowEvent", "events", "sporadic",
              "sim_bridge or workflow_gateway", "autonomy"),
    TopicSpec("workflow_transcript", f"{NS}/workflow/transcript", "std_msgs/msg/String", "events", "sporadic",
              "workflow_gateway", "audit / rosbag"),
    TopicSpec("mission", f"{NS}/mission", "std_msgs/msg/String", "latched", "once",
              "sim_bridge or facility server", "autonomy (mission_source=topic)"),
    # ---- commands -------------------------------------------------------------------------------------------------
    TopicSpec("cmd_vel", ISAAC_TOPICS["cmd_vel"], "geometry_msgs/msg/Twist", "command", "10 Hz",
              "autonomy (safety-gated)", "sim_bridge or Isaac OmniGraph or base driver"),
    TopicSpec("acoustic_probe", f"{NS}/acoustic/probe_target", "std_msgs/msg/String", "command", "10 Hz",
              "autonomy", "sim_bridge or acoustic probe driver"),
    # ---- autonomy outputs -----------------------------------------------------------------------------------------
    TopicSpec("pose", f"{NS}/localization/pose", "geometry_msgs/msg/PoseWithCovarianceStamped", "state_volatile",
              "10 Hz", "autonomy", "rviz"),
    TopicSpec("item_beliefs", f"{NS}/belief/items", "medortrace_msgs/msg/ItemBeliefArray", "state", "2 Hz",
              "autonomy", "dashboards, rosbag"),
    TopicSpec("scene_graph", f"{NS}/belief/scene_graph", "medortrace_msgs/msg/SceneGraph", "state", "1 Hz",
              "autonomy", "dashboards, rosbag"),
    TopicSpec("uncertainty", f"{NS}/belief/uncertainty", "medortrace_msgs/msg/UncertaintyGrid", "state", "1 Hz",
              "autonomy", "dashboards, rosbag"),
    TopicSpec("occupancy", f"{NS}/belief/occupancy", "nav_msgs/msg/OccupancyGrid", "state", "1 Hz",
              "autonomy", "rviz"),
    TopicSpec("nbv", f"{NS}/planning/nbv", "medortrace_msgs/msg/NextBestView", "state", "on change",
              "autonomy", "dashboards, rosbag"),
    TopicSpec("path", f"{NS}/planning/path", "nav_msgs/msg/Path", "state", "on change", "autonomy", "rviz"),
    TopicSpec("verdicts", f"{NS}/verification/verdicts", "medortrace_msgs/msg/ClaimVerdict", "audit", "sporadic",
              "autonomy", "operator_console, OR information system"),
    TopicSpec("provenance", f"{NS}/provenance/events", "medortrace_msgs/msg/ProvenanceEvent", "audit",
              "~10-40 Hz", "autonomy", "audit logger, rosbag"),
    TopicSpec("safety", f"{NS}/safety/state", "medortrace_msgs/msg/SafetyState", "state", "10 Hz",
              "autonomy", "operator_console, rviz"),
    TopicSpec("diagnostics", "/diagnostics", "diagnostic_msgs/msg/DiagnosticArray", "state_volatile", "1 Hz",
              "autonomy (assembler health), sim_bridge", "rqt_robot_monitor"),
    # ---- simulation-only / operator -----------------------------------------------------------------------------
    TopicSpec("ground_truth", f"{NS}/sim/ground_truth/odom", "nav_msgs/msg/Odometry", "state_volatile", "10 Hz",
              "sim_bridge", "operator_console (auto mode), evaluation"),
    TopicSpec("initialpose", "/initialpose", "geometry_msgs/msg/PoseWithCovarianceStamped", "command", "sporadic",
              "rviz (2D Pose Estimate)", "operator_console (rviz mode)"),
]

SPECS: dict[str, TopicSpec] = {s.key: s for s in _SPECS}
TOPICS: dict[str, str] = {s.key: s.name for s in _SPECS}

SERVICES = {
    "operator_ack": (f"{NS}/operator/ack", "medortrace_msgs/srv/OperatorAck", "operator_console", "autonomy"),
    "verify_claim": (f"{NS}/verification/verify_claim", "medortrace_msgs/srv/VerifyClaim", "OR system / CLI",
                     "autonomy"),
    "explain": (f"{NS}/verification/explain", "medortrace_msgs/srv/ExplainVerdict", "OR system / CLI", "autonomy"),
}

# The digital twin's OmniGraph topics are part of this graph verbatim.
assert all(TOPICS[k] == v for k, v in ISAAC_TOPICS.items()), "topic registry diverged from ros2_bridge.TOPICS"


def topic(key: str) -> str:
    return TOPICS[key]
