"""ROS 2 bridge for the Isaac Sim digital twin (OmniGraph, ``isaacsim.ros2.bridge``).

Builds one action graph that publishes simulation clock, TF (base_link in
the world frame + the rig's sensor frames under base_link), RTX lidar point
cloud, RGB / depth / camera_info and odometry.  With ``drive_from_cmd_vel``
it also subscribes ``/medortrace/cmd_vel`` and wires it into the
articulation's wheel drives through a differential controller; leave it off
when the in-process autonomy stack drives the robot (``run_episode_isaac.py
--ros2``), otherwise two controllers would fight over the wheel targets.
Topic names match docs/ros2_message_graph.md so the same ``medortrace_ros``
nodes run against simulation or the physical prototype.

Radar, acoustic, workflow and verification topics are custom messages
(``medortrace_msgs``) published by ``medortrace_ros/sim_bridge_node.py``,
which runs inside the Isaac Sim Python process next to this graph.

Node-type namespaces differ between Isaac Sim 4.5+/5.x (``isaacsim.*``) and
<= 4.2 (``omni.isaac.*``) and are resolved by ``compat.ros2_node_namespaces``.
``ROS2CameraInfoHelper`` only exists in 4.5+; older releases publish
camera_info through ``ROS2CameraHelper`` with ``type="camera_info"``.
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
SENSOR_FRAMES = ("lidar_link", "camera_link", "radar_link", "acoustic_link", "imu_link")


def graph_spec(ns: dict[str, str], robot_prim: str = "/World/Robot", base_link: str | None = None,
               lidar_render_product: str | None = None, camera_render_product: str | None = None,
               wheel_radius: float = 0.085, wheel_track: float = 0.44, drive_from_cmd_vel: bool = True,
               camera_info_helper: bool = True, limits: dict | None = None) -> tuple[list, list, list]:
    """(nodes, connections, values) for ``og.Controller.edit`` (pure python, inspectable in tests)."""
    b, core, wheeled = ns["bridge"], ns["core"], ns["wheeled"]
    base = base_link or f"{robot_prim}/base_link"
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", f"{core}.IsaacReadSimulationTime"),
        ("Context", f"{b}.ROS2Context"),
        ("PublishClock", f"{b}.ROS2PublishClock"),
        ("PublishTF", f"{b}.ROS2PublishTransformTree"),
        ("PublishSensorTF", f"{b}.ROS2PublishTransformTree"),
        ("ComputeOdom", f"{core}.IsaacComputeOdometry"),
        ("PublishOdom", f"{b}.ROS2PublishOdometry"),
    ]
    connections = [
        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ("Context.outputs:context", "PublishClock.inputs:context"),
        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
        ("Context.outputs:context", "PublishTF.inputs:context"),
        ("OnPlaybackTick.outputs:tick", "PublishSensorTF.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishSensorTF.inputs:timeStamp"),
        ("Context.outputs:context", "PublishSensorTF.inputs:context"),
        ("OnPlaybackTick.outputs:tick", "ComputeOdom.inputs:execIn"),
        ("ComputeOdom.outputs:execOut", "PublishOdom.inputs:execIn"),
        ("ComputeOdom.outputs:linearVelocity", "PublishOdom.inputs:linearVelocity"),
        ("ComputeOdom.outputs:angularVelocity", "PublishOdom.inputs:angularVelocity"),
        ("ComputeOdom.outputs:position", "PublishOdom.inputs:position"),
        ("ComputeOdom.outputs:orientation", "PublishOdom.inputs:orientation"),
        ("ReadSimTime.outputs:simulationTime", "PublishOdom.inputs:timeStamp"),
        ("Context.outputs:context", "PublishOdom.inputs:context"),
    ]
    values = [
        ("PublishTF.inputs:targetPrims", [base]),
        ("PublishSensorTF.inputs:parentPrim", [base]),
        ("PublishSensorTF.inputs:targetPrims", [f"{base}/{f}" for f in SENSOR_FRAMES]),
        ("ComputeOdom.inputs:chassisPrim", [base]),
        ("PublishOdom.inputs:topicName", TOPICS["odom"].lstrip("/")),
        ("PublishOdom.inputs:odomFrameId", "odom"),
        ("PublishOdom.inputs:chassisFrameId", "base_link"),
    ]
    if drive_from_cmd_vel:
        lim = limits or {}
        nodes += [
            ("SubscribeTwist", f"{b}.ROS2SubscribeTwist"),
            ("BreakLin", "omni.graph.nodes.BreakVector3"),
            ("BreakAng", "omni.graph.nodes.BreakVector3"),
            ("DiffController", f"{wheeled}.DifferentialController"),
            ("ArticulationController", f"{core}.IsaacArticulationController"),
        ]
        connections += [
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
        values += [
            ("SubscribeTwist.inputs:topicName", TOPICS["cmd_vel"].lstrip("/")),
            ("DiffController.inputs:wheelRadius", float(wheel_radius)),
            ("DiffController.inputs:wheelDistance", float(wheel_track)),
            ("ArticulationController.inputs:targetPrim", [base]),
            ("ArticulationController.inputs:jointNames", ["left_wheel_joint", "right_wheel_joint"]),
        ]
        if "max_v" in lim:
            values.append(("DiffController.inputs:maxLinearSpeed", float(lim["max_v"])))
        if "max_omega" in lim:
            values.append(("DiffController.inputs:maxAngularSpeed", float(lim["max_omega"])))
    if lidar_render_product:
        nodes.append(("LidarHelper", f"{b}.ROS2RtxLidarHelper"))
        connections += [("OnPlaybackTick.outputs:tick", "LidarHelper.inputs:execIn"),
                        ("Context.outputs:context", "LidarHelper.inputs:context")]
        values += [("LidarHelper.inputs:renderProductPath", lidar_render_product),
                   ("LidarHelper.inputs:topicName", TOPICS["lidar_points"].lstrip("/")),
                   ("LidarHelper.inputs:type", "point_cloud"), ("LidarHelper.inputs:frameId", "lidar_link")]
    if camera_render_product:
        helpers = [("RgbHelper", "rgb", "rgb"), ("DepthHelper", "depth", "depth"),
                   ("InfoHelper", "camera_info", "camera_info")]
        for name, typ, topic in helpers:
            info_node = typ == "camera_info" and camera_info_helper
            nodes.append((name, f"{b}.ROS2CameraInfoHelper" if info_node else f"{b}.ROS2CameraHelper"))
            connections += [("OnPlaybackTick.outputs:tick", f"{name}.inputs:execIn"),
                            ("Context.outputs:context", f"{name}.inputs:context")]
            values += [(f"{name}.inputs:renderProductPath", camera_render_product),
                       (f"{name}.inputs:topicName", TOPICS[topic].lstrip("/")),
                       (f"{name}.inputs:frameId", "camera_link")]
            if not info_node:
                values.append((f"{name}.inputs:type", typ))
    return nodes, connections, values


def build_ros2_graph(robot_prim: str = "/World/Robot", lidar_render_product: str | None = None,
                     camera_render_product: str | None = None, wheel_radius: float = 0.085,
                     wheel_track: float = 0.44, drive_from_cmd_vel: bool = True, base_link: str | None = None,
                     limits: dict | None = None) -> str:  # pragma: no cover - requires Isaac Sim
    import omni.graph.core as og
    import omni.usd

    from medortrace.isaac.compat import ros2_node_namespaces

    ns = ros2_node_namespaces()
    keys = og.Controller.Keys
    last = None
    for info_helper in (True, False):
        nodes, connections, values = graph_spec(ns, robot_prim, base_link, lidar_render_product,
                                                camera_render_product, wheel_radius, wheel_track,
                                                drive_from_cmd_vel, info_helper, limits)
        try:
            og.Controller.edit({"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
                               {keys.CREATE_NODES: nodes, keys.CONNECT: connections, keys.SET_VALUES: values})
            return GRAPH_PATH
        except Exception as e:
            last = e
            stage = omni.usd.get_context().get_stage()
            if stage.GetPrimAtPath(GRAPH_PATH).IsValid():
                stage.RemovePrim(GRAPH_PATH)
            if not camera_render_product:
                break
    raise RuntimeError(f"could not build the ROS 2 graph: {last}")


def ros2_reset_hook(drive_from_cmd_vel: bool):
    """``IsaacBackend.add_reset_hook`` callback that builds the graph once the sensors exist."""

    def hook(be) -> None:  # pragma: no cover - requires Isaac Sim
        from medortrace.common.config import CONFIG_DIR, load_yaml
        from medortrace.isaac.sensors import render_product_path
        rig = load_yaml(CONFIG_DIR / "robot" / "rig.yaml")
        path = build_ros2_graph(robot_prim="/World/Robot", base_link=be.base_path,
                                lidar_render_product=render_product_path(be.lidar.rp),
                                camera_render_product=render_product_path(be.camera.rp),
                                wheel_radius=float(rig.get("wheel_radius", 0.085)),
                                wheel_track=float(rig.get("wheel_track", 0.44)),
                                drive_from_cmd_vel=drive_from_cmd_vel, limits=rig.get("limits"))
        be.ros2_graph = path
        print(f"[medortrace] ROS 2 graph built at {path} (cmd_vel drives wheels: {drive_from_cmd_vel})")

    return hook
