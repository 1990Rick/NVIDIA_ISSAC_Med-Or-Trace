"""MED-OR-TRACE on the physical prototype: same autonomy node, real drivers via topic remapping.

    ros2 launch medortrace_ros robot.launch.py mission_file:=/data/or3/case_0412_survey.json \\
        lidar_topic:=/ouster/points odom_topic:=/odom cmd_vel_topic:=/cmd_vel operator_mode:=rviz

Only the middleware edge changes between simulation and the robot:

* the autonomy node's input topics are *remapped* onto the drivers' topics (defaults below are typical
  driver names - override per platform); nothing is republished, no extra hop;
* ``radar_input:=pointcloud2`` accepts a radar driver's point cloud (x, y, z, velocity, intensity) instead
  of ``medortrace_msgs/RadarDetectionArray``;
* driver lidar clouds carry only returns, so the scan pattern of the OR16 lidar (16 rings, +-15 deg,
  0.2 deg) is used to re-create the no-return rays needed for free-space carving;
* wall-clock time (no ``use_sim_time``); the mission origin t0 is the node start unless the mission file
  sets ``t0``;
* camera detections, fiducials and acoustic echoes come from the perception drivers that publish the
  ``medortrace_msgs`` types (the Isaac Sim detector surrogate is replaced by the trained detector);
* the OR workflow arrives through ``workflow_gateway_node`` (``mode: follow`` tails the live OR log /
  voice transcript) or directly from the OR information system on /medortrace/workflow/events;
* sensor extrinsics come from configs/robot/rig.yaml (``publish_rig_tf:=false`` if a URDF /
  robot_state_publisher already provides base_link -> *_link); driver frame ids must be the rig's
  ``*_link`` names (or add static aliases), because the stack models each sensor by its rig mount.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from medortrace_ros.launch_utils import config_path, rig_static_tf_nodes
from medortrace_ros.topics import TOPICS

# autonomy-node input / output (key in medortrace_ros.topics.TOPICS) -> launch argument with the driver topic
REMAP_ARGS = {
    "lidar_points": ("lidar_topic", "/ouster/points", "sensor_msgs/PointCloud2 (lidar driver)"),
    "camera_detections": ("camera_detections_topic", "/detector/detections",
                          "medortrace_msgs/CameraDetectionArray (on-robot detector)"),
    "landmarks": ("landmarks_topic", "/fiducials/observations",
                  "medortrace_msgs/LandmarkObservationArray (fiducial detector)"),
    "radar": ("radar_topic", "/radar/detections", "medortrace_msgs/RadarDetectionArray"),
    "radar_points": ("radar_points_topic", "/radar/points", "sensor_msgs/PointCloud2 (radar driver)"),
    "acoustic": ("acoustic_topic", "/acoustic_probe/echoes", "medortrace_msgs/AcousticFrame"),
    "acoustic_probe": ("acoustic_probe_topic", "/acoustic_probe/target", "std_msgs/String (probe aim)"),
    "imu": ("imu_topic", "/imu/data", "sensor_msgs/Imu"),
    "odom": ("odom_topic", "/odom", "nav_msgs/Odometry (base driver)"),
    "contact": ("contact_topic", "/bumper/contact", "medortrace_msgs/ContactState"),
    "battery": ("battery_topic", "/battery_state", "sensor_msgs/BatteryState (BMS)"),
    "cmd_vel": ("cmd_vel_topic", "/cmd_vel", "geometry_msgs/Twist (base driver input)"),
}


def generate_launch_description():
    L = LaunchConfiguration
    args = [
        DeclareLaunchArgument("mission_file", description="mission JSON/YAML (see medortrace_ros/mission.py)"),
        DeclareLaunchArgument("radar_input", default_value="pointcloud2", description="detections | pointcloud2"),
        DeclareLaunchArgument("operator_mode", default_value="rviz", description="rviz | interactive"),
        DeclareLaunchArgument("workflow_log", default_value="", description="live OR log / transcript JSONL"),
        DeclareLaunchArgument("audit_dir", default_value="", description="audit record at mission end"),
        DeclareLaunchArgument("publish_rig_tf", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=config_path("params.yaml")),
        DeclareLaunchArgument("rviz", default_value="false"),
    ] + [DeclareLaunchArgument(arg, default_value=default, description=desc)
         for arg, default, desc in REMAP_ARGS.values()]
    params = L("params_file")
    remaps = [(TOPICS[key], L(arg)) for key, (arg, _, _) in REMAP_ARGS.items()]
    autonomy = Node(
        package="medortrace_ros", executable="autonomy_node", name="medortrace_autonomy", output="screen",
        remappings=remaps,
        parameters=[params, {
            "use_sim_time": False, "mission_source": "file", "mission_file": L("mission_file"),
            "time_origin": "mission", "radar.input": L("radar_input"), "audit_dir": L("audit_dir"),
            "lidar.fill_no_return_rays": True, "lidar.pattern.rings": 16, "lidar.pattern.elev_min_deg": -15.0,
            "lidar.pattern.elev_max_deg": 15.0, "lidar.pattern.az_res_deg": 0.2,
            "required_channels": ["odom", "imu"], "ready_timeout_s": 10.0}])
    nodes = [
        autonomy,
        Node(package="medortrace_ros", executable="operator_console_node", name="medortrace_operator_console",
             output="screen", emulate_tty=True,
             parameters=[params, {"mode": L("operator_mode"), "relocalize_from_ground_truth": False}]),
        Node(package="medortrace_ros", executable="workflow_gateway_node", name="medortrace_workflow_gateway",
             output="screen", condition=IfCondition(PythonExpression(["'", L("workflow_log"), "' != ''"])),
             parameters=[params, {"log_file": L("workflow_log"), "mode": "follow", "time_origin": "mission"}]),
        Node(package="rviz2", executable="rviz2", name="rviz2", output="log", condition=IfCondition(L("rviz")),
             arguments=["-d", config_path("medortrace.rviz")]),
    ]
    return LaunchDescription(args + nodes + rig_static_tf_nodes(condition=IfCondition(L("publish_rig_tf"))))
