"""ROS 2 bridge for the Isaac Sim digital twin (OmniGraph, ``isaacsim.ros2.bridge``).

Builds one action graph that publishes simulation clock, TF, RTX lidar point
cloud, RGB / depth / camera_info, odometry, and subscribes ``/cmd_vel``
(wired into the articulation's wheel drives through a differential
controller).  Topic names match docs/ros2_message_graph.md so the same
``medortrace_ros`` nodes run against simulation or the physical prototype.

Radar, acoustic, workflow and verification topics are custom messages
(``medortrace_msgs``) published by ``medortrace_ros/sim_bridge_node.py``,
which runs inside the Isaac Sim Python process next to this graph.
"""

from __future__ import annotations

GRAPH_PATH = "/World/ROS2Graph"

TOPICS = {
    "clock": "/clock",
    "lidar_points": "/medortrace/sensors/lidar/points",
    "rgb": "/medortrace/sensors/camera/rgb",
    "depth": "/medortrace/sensors/camera/depth",
    "camera_info": "/medortrace/sensors/camera/camera_info",
    "odom": "/medortrace/odom",
    "cmd_vel": "/medortrace/cmd_vel",
    "tf": "/tf",
}


def build_ros2_graph(robot_prim: str = "/World/Robot", lidar_render_product: str | None = None,
                     camera_render_product: str | None = None, wheel_radius: float = 0.085,
                     wheel_track: float = 0.44) -> str:  # pragma: no cover - requires Isaac Sim
    import omni.graph.core as og

    ns_bridge = "isaacsim.ros2.bridge"
    ns_core = "isaacsim.core.nodes"
    keys = og.Controller.Keys
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", f"{ns_core}.IsaacReadSimulationTime"),
        ("Context", f"{ns_bridge}.ROS2Context"),
        ("PublishClock", f"{ns_bridge}.ROS2PublishClock"),
        ("PublishTF", f"{ns_bridge}.ROS2PublishTransformTree"),
        ("ComputeOdom", f"{ns_core}.IsaacComputeOdometry"),
        ("PublishOdom", f"{ns_bridge}.ROS2PublishOdometry"),
        ("SubscribeTwist", f"{ns_bridge}.ROS2SubscribeTwist"),
        ("BreakLin", "omni.graph.nodes.BreakVector3"),
        ("BreakAng", "omni.graph.nodes.BreakVector3"),
        ("DiffController", "isaacsim.robot.wheeled_robots.DifferentialController"),
        ("ArticulationController", f"{ns_core}.IsaacArticulationController"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ("Context.outputs:context", "PublishClock.inputs:context"),
        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
        ("Context.outputs:context", "PublishTF.inputs:context"),
        ("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"),
        ("ComputeOdom.outputs:execOut", "PublishOdom.inputs:execIn"),
        ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
        ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
        ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
        ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
        ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
        ("Context.outputs:context", "PublishOdom.inputs:context"),
        ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
        ("Context.outputs:context", "SubscribeTwist.inputs:context"),
        ("SubscribeTwist.outputs:linearVelocity", "BreakLin.inputs:tuple"),
        ("SubscribeTwist.outputs:angularVelocity", "BreakAng.inputs:tuple"),
        ("SubscribeTwist.outputs:execOut", "DiffController.inputs:execIn"),
        ("BreakLin.outputs:x", "DiffController.inputs:linearVelocity"),
        ("BreakAng.outputs:z", "DiffController.inputs:angularVelocity"),
        ("DiffController.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
    ]
    values = [
        ("PublishTF.inputs:targetPrims", [robot_prim]),
        ("ComputeOdom.inputs:chassisPrim", [robot_prim + "/base_link"]),
        ("PublishOdom.inputs:topicName", TOPICS["odom"].lstrip("/")),
        ("PublishOdom.inputs:odomFrameId", "odom"),
        ("PublishOdom.inputs:chassisFrameId", "base_link"),
        ("SubscribeTwist.inputs:topicName", TOPICS["cmd_vel"].lstrip("/")),
        ("DiffController.inputs:wheelRadius", wheel_radius),
        ("DiffController.inputs:wheelDistance", wheel_track),
        ("ArticulationController.inputs:targetPrim", [robot_prim]),
        ("ArticulationController.inputs:jointNames", ["left_wheel_joint", "right_wheel_joint"]),
    ]
    if lidar_render_product:
        nodes.append(("LidarHelper", f"{ns_bridge}.ROS2RtxLidarHelper"))
        connections += [("OnPlaybackTick.outputs:tick", "LidarHelper.inputs:execIn"),
                        ("Context.outputs:context", "LidarHelper.inputs:context")]
        values += [("LidarHelper.inputs:renderProductPath", lidar_render_product),
                   ("LidarHelper.inputs:topicName", TOPICS["lidar_points"].lstrip("/")),
                   ("LidarHelper.inputs:type", "point_cloud"), ("LidarHelper.inputs:frameId", "lidar_link")]
    if camera_render_product:
        for name, typ, topic in (("RgbHelper", "rgb", "rgb"), ("DepthHelper", "depth", "depth"),
                                 ("InfoHelper", "camera_info", "camera_info")):
            node_type = f"{ns_bridge}.ROS2CameraInfoHelper" if typ == "camera_info" else f"{ns_bridge}.ROS2CameraHelper"
            nodes.append((name, node_type))
            connections += [("OnPlaybackTick.outputs:tick", f"{name}.inputs:execIn"),
                            ("Context.outputs:context", f"{name}.inputs:context")]
            values += [(f"{name}.inputs:renderProductPath", camera_render_product),
                       (f"{name}.inputs:topicName", TOPICS[topic].lstrip("/")),
                       (f"{name}.inputs:frameId", "camera_link")]
            if typ != "camera_info":
                values.append((f"{name}.inputs:type", typ))
    og.Controller.edit({"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
                       {keys.CREATE_NODES: nodes, keys.CONNECT: connections, keys.SET_VALUES: values})
    return GRAPH_PATH
