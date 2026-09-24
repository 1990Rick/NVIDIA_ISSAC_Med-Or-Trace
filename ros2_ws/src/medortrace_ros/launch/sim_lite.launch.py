"""Full MED-OR-TRACE ROS 2 graph on the lite simulator (no Isaac Sim, laptop-friendly).

    ros2 launch medortrace_ros sim_lite.launch.py scenario:=scenarios/cf_a.yaml seed:=3 rviz:=true

Nodes: sim_bridge_node (--backend lite, owns /clock), autonomy_node (use_sim_time, mission from the
latched /medortrace/mission topic), operator_console_node (auto = SimulatedOperator), optional
workflow_gateway_node (replays ``workflow_log`` instead of the simulator's own workflow stream),
static rig TF (configs/robot/rig.yaml) and optional RViz.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from medortrace_ros.launch_utils import config_path, rig_static_tf_nodes


def generate_launch_description():
    L = LaunchConfiguration
    args = [
        DeclareLaunchArgument("scenario", default_value="scenarios/nominal.yaml"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("duration", default_value="0.0", description="episode length [s]; 0 = scenario"),
        DeclareLaunchArgument("policy", default_value="active", description="active | fixed_route | passive"),
        DeclareLaunchArgument("lockstep", default_value="true"),
        DeclareLaunchArgument("rate_factor", default_value="1.0", description="x real time; 0 = max speed"),
        DeclareLaunchArgument("operator_mode", default_value="auto", description="auto | rviz | interactive"),
        DeclareLaunchArgument("workflow_log", default_value="", description="JSONL to replay instead of sim log"),
        DeclareLaunchArgument("audit_dir", default_value="", description="write the audit record here at the end"),
        DeclareLaunchArgument("params_file", default_value=config_path("params.yaml")),
        DeclareLaunchArgument("rviz", default_value="false"),
    ]
    params = L("params_file")
    no_log = PythonExpression(["'", L("workflow_log"), "' == ''"])
    has_log = PythonExpression(["'", L("workflow_log"), "' != ''"])
    nodes = [
        Node(package="medortrace_ros", executable="sim_bridge_node", name="medortrace_sim_bridge", output="screen",
             arguments=["--backend", "lite"],
             parameters=[params, {"scenario": L("scenario"), "seed": L("seed"), "duration": L("duration"),
                                  "policy": L("policy"), "lockstep": L("lockstep"), "rate_factor": L("rate_factor"),
                                  "publish_workflow": no_log}]),
        Node(package="medortrace_ros", executable="autonomy_node", name="medortrace_autonomy", output="screen",
             parameters=[params, {"use_sim_time": True, "mission_source": "topic", "audit_dir": L("audit_dir")}]),
        Node(package="medortrace_ros", executable="operator_console_node", name="medortrace_operator_console",
             output="screen", emulate_tty=True,
             parameters=[params, {"use_sim_time": True, "mode": L("operator_mode"), "seed": L("seed")}]),
        Node(package="medortrace_ros", executable="workflow_gateway_node", name="medortrace_workflow_gateway",
             output="screen", condition=IfCondition(has_log),
             parameters=[params, {"use_sim_time": True, "log_file": L("workflow_log"), "time_origin": "mission"}]),
        Node(package="rviz2", executable="rviz2", name="rviz2", output="log", condition=IfCondition(L("rviz")),
             arguments=["-d", config_path("medortrace.rviz")], parameters=[{"use_sim_time": True}]),
    ]
    return LaunchDescription(args + nodes + rig_static_tf_nodes(use_sim_time=True))
