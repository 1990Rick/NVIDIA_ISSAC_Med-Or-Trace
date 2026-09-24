"""MED-OR-TRACE ROS 2 graph on the NVIDIA Isaac Sim digital twin.

    ros2 launch medortrace_ros isaac_sim.launch.py isaac_python:=$HOME/isaacsim/python.sh \\
        scenario:=scenarios/reflective.yaml seed:=7 headless:=true

``sim_bridge_node --backend isaac`` must run inside Isaac Sim's own Python (``python.sh``): it creates
the SimulationApp with the ``isaacsim.ros2.bridge`` extension, authors the USD stage from the same
``Episode`` as the lite simulator (``medortrace.isaac.backend.IsaacBackend``) and publishes the same
topics as in ``sim_lite.launch.py``.  The Isaac process therefore needs rclpy + medortrace_msgs built
for *its* Python version (Isaac Sim 4.5: 3.10 = Humble; Isaac Sim 5.x: 3.11 - build the workspace with
that interpreter or use Isaac Sim's bundled ROS 2 libraries, see the package README).

``isaac_graph:=true`` adds the OmniGraph of ``medortrace.isaac.ros2_bridge`` (via
``IsaacBackend.add_reset_hook(ros2_reset_hook(drive_from_cmd_vel=False))``): /clock from the simulation,
the RTX lidar point cloud, RGB / depth / camera_info and the sensor frames under base_link.  The bridge
node then skips /clock and the lidar topic, removes the graph's world-parented base_link TF and its
odometry nodes, and keeps publishing /medortrace/odom and odom->base_link itself, so the TF tree stays
map->odom->base_link->*_link and the odometry carries the same fault model as in lockstep mode.  The
backend keeps driving the wheels from the autonomy command (one controller), the rig's static TF is not
re-published (the graph publishes the sensor frames), and the autonomy node re-bins the RTX cloud onto the
OR16 rings x ``sensors.lidar.az_res_deg`` ray grid, like ``IsaacBackend`` does in-process.  Autonomy,
operator console and gateway are the same nodes as in simulation-lite.

Alternative (robot driven through the OmniGraph's twist subscriber instead of the lockstep bridge):
``./python.sh scripts/isaac/ros2_sim.py --bridge medortrace_ros.sim_bridge_node:create`` plus the other
nodes of this file (``isaac_python:=none`` skips starting the bridge process).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from medortrace_ros.launch_utils import config_path, medortrace_env, rig_static_tf_nodes


def generate_launch_description():
    L = LaunchConfiguration
    args = [
        DeclareLaunchArgument("isaac_python", default_value=os.environ.get(
            "ISAACSIM_PYTHON", os.path.expanduser("~/isaacsim/python.sh")), description="Isaac Sim python.sh"),
        DeclareLaunchArgument("scenario", default_value="scenarios/nominal.yaml"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("duration", default_value="0.0"),
        DeclareLaunchArgument("policy", default_value="active"),
        DeclareLaunchArgument("headless", default_value="true"),
        DeclareLaunchArgument("isaac_graph", default_value="false"),
        DeclareLaunchArgument("lockstep", default_value="true"),
        DeclareLaunchArgument("operator_mode", default_value="auto"),
        DeclareLaunchArgument("workflow_log", default_value=""),
        DeclareLaunchArgument("audit_dir", default_value=""),
        DeclareLaunchArgument("params_file", default_value=config_path("params.yaml")),
        DeclareLaunchArgument("rviz", default_value="false"),
    ]
    params = L("params_file")
    graph_on = PythonExpression(["'", L("isaac_graph"), "'.lower() == 'true'"])
    graph_off = PythonExpression(["'", L("isaac_graph"), "'.lower() != 'true'"])
    run_bridge = PythonExpression(["'", L("isaac_python"), "'.lower() != 'none'"])
    headless_flag = PythonExpression(["'--headless' if '", L("headless"), "'.lower() == 'true' else ''"])
    graph_flag = PythonExpression(["'--isaac-graph' if '", L("isaac_graph"), "'.lower() == 'true' else ''"])
    no_log = PythonExpression(["'", L("workflow_log"), "' == ''"])
    bridge = ExecuteProcess(
        cmd=[L("isaac_python"), "-m", "medortrace_ros.sim_bridge_node", "--backend", "isaac",
             "--scenario", L("scenario"), "--seed", L("seed"), "--duration", L("duration"), headless_flag, graph_flag,
             "--ros-args", "-r", "__node:=medortrace_sim_bridge", "--params-file", params,
             "-p", ["policy:=", L("policy")], "-p", ["lockstep:=", L("lockstep")],
             "-p", ["publish_workflow:=", no_log]],
        additional_env=medortrace_env(), output="screen", shell=False, condition=IfCondition(run_bridge))
    common = {"use_sim_time": True}
    nodes = [
        bridge,
        Node(package="medortrace_ros", executable="autonomy_node", name="medortrace_autonomy", output="screen",
             parameters=[params, {**common, "mission_source": "topic", "audit_dir": L("audit_dir"),
                                  "lidar.regrid_to_pattern": graph_on, "lidar.pattern.rings": 16,
                                  "lidar.pattern.elev_min_deg": -15.0, "lidar.pattern.elev_max_deg": 15.0,
                                  "lidar.pattern.az_res_deg": 0.0}]),         # 0: the stack's az_res_deg
        Node(package="medortrace_ros", executable="operator_console_node", name="medortrace_operator_console",
             output="screen", emulate_tty=True, parameters=[params, {**common, "mode": L("operator_mode")}]),
        Node(package="medortrace_ros", executable="workflow_gateway_node", name="medortrace_workflow_gateway",
             output="screen", condition=IfCondition(PythonExpression(["'", L("workflow_log"), "' != ''"])),
             parameters=[params, {**common, "log_file": L("workflow_log")}]),
        Node(package="rviz2", executable="rviz2", name="rviz2", output="log", condition=IfCondition(L("rviz")),
             arguments=["-d", config_path("medortrace.rviz")], parameters=[common]),
    ]
    return LaunchDescription(args + nodes + rig_static_tf_nodes(condition=IfCondition(graph_off), use_sim_time=True))
