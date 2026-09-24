"""ROS 2 messages <-> medortrace dataclasses (lazy ROS imports).

Every ROS import happens inside the function that needs it, so this module
imports (and its pure-numpy helpers are testable) without a ROS installation.

Time
    The stack runs on *mission time* (seconds since the case started, the
    clock of ``WorkflowEvent.t``, claim deadlines and ``StackInputs.duration``).
    ROS stamps are absolute; ``t0`` (the mission origin, see ``mission.py``)
    converts between them: ``mission = ros_seconds - t0``.  Incoming headers
    keep the *sensor's* stamp in ``Header.stamp`` and get the node-clock
    receive time in ``Header.recv_stamp`` (never overwritten by the stamp, so
    the stack's time-sync monitor can detect skewed sensor clocks).

Lidar (``sensor_msgs/PointCloud2`` via ``sensor_msgs_py``)
    ``LidarScan`` holds one entry per *ray* (``directions``, ``ranges`` with
    ``inf`` for no return) so the occupancy layer can carve free space along
    rays that returned nothing.  ``sim_bridge_node`` publishes that losslessly
    as an unorganised cloud with one point per ray and the extra fields
    ``range, dir_x, dir_y, dir_z`` (x, y, z = NaN for no-return rays) and,
    in simulation, ``gt_ghost`` / ``gt_object`` supervision fields.  Clouds
    from real drivers or the Isaac RTX helper only have x, y, z [, intensity,
    ring]; directions and ranges are then derived from the points and, if a
    scan pattern is configured, the missing rays of the pattern are filled in
    as no-return rays (:func:`fill_no_return_rays`).
"""

from __future__ import annotations

import json
import math

import numpy as np

from medortrace.belief.items import CLASSES
from medortrace.common.msgs import (
    AcousticEcho,
    AcousticFrame,
    CameraDetection,
    CameraFrame,
    ContactState,
    Header,
    ImuSample,
    LandmarkFrame,
    LandmarkObservation,
    LidarScan,
    RadarDetection,
    RadarFrame,
    VelocityCommand,
    WheelOdometry,
    WorkflowEvent,
    WorkflowEventType,
)
from medortrace_ros.frames import quat_xyzw_to_yaw, yaw_to_quat_xyzw

# sensor_msgs/PointField datatype codes
PF_INT8, PF_UINT8, PF_INT16, PF_UINT16, PF_INT32, PF_UINT32, PF_FLOAT32, PF_FLOAT64 = range(1, 9)
PF_NUMPY = {PF_INT8: np.int8, PF_UINT8: np.uint8, PF_INT16: np.int16, PF_UINT16: np.uint16, PF_INT32: np.int32,
            PF_UINT32: np.uint32, PF_FLOAT32: np.float32, PF_FLOAT64: np.float64}
LIDAR_FIELDS = (("x", PF_FLOAT32), ("y", PF_FLOAT32), ("z", PF_FLOAT32), ("intensity", PF_FLOAT32),
                ("ring", PF_UINT16), ("range", PF_FLOAT32), ("dir_x", PF_FLOAT32), ("dir_y", PF_FLOAT32),
                ("dir_z", PF_FLOAT32))
LIDAR_GT_FIELDS = (("gt_ghost", PF_UINT8), ("gt_object", PF_INT32))
RING_FIELD_ALIASES = ("ring", "channel", "laser_id")


# =====================================================================================================================
# time & headers
# =====================================================================================================================
def to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def to_time(t: float):
    from builtin_interfaces.msg import Time

    t = max(0.0, float(t))
    sec = int(math.floor(t))
    nsec = int(round((t - sec) * 1e9))
    if nsec >= 1_000_000_000:
        sec, nsec = sec + 1, nsec - 1_000_000_000
    return Time(sec=sec, nanosec=nsec)


def ros_header(t_mission: float, frame_id: str, t0: float = 0.0):
    from std_msgs.msg import Header as RosHeader

    return RosHeader(stamp=to_time(t_mission + t0), frame_id=frame_id)


def header_from_ros(h, recv: float, t0: float = 0.0) -> Header:
    """ROS header -> medortrace Header (stamp: sensor clock, recv_stamp: node clock, both mission time)."""
    return Header(stamp=to_sec(h.stamp) - t0, recv_stamp=float(recv), frame_id=h.frame_id)


def clock_msg(t_ros: float):
    from rosgraph_msgs.msg import Clock

    return Clock(clock=to_time(t_ros))


# =====================================================================================================================
# lidar (pure numpy part)
# =====================================================================================================================
def lidar_to_columns(scan: LidarScan, include_gt: bool = True) -> dict[str, np.ndarray]:
    """One row per ray: the lossless PointCloud2 layout used by sim_bridge_node."""
    R = len(scan.ranges)
    fin = np.isfinite(scan.ranges)
    xyz = np.full((R, 3), np.nan, np.float32)
    xyz[fin] = scan.directions[fin] * scan.ranges[fin, None]
    inten = np.zeros(R, np.float32)
    ring = np.zeros(R, np.uint16)
    if len(scan.intensity) == fin.sum():
        inten[fin] = scan.intensity
    if len(scan.ring) == fin.sum():
        ring[fin] = np.clip(scan.ring, 0, 65535)
    cols = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "intensity": inten, "ring": ring,
            "range": scan.ranges.astype(np.float32), "dir_x": scan.directions[:, 0].astype(np.float32),
            "dir_y": scan.directions[:, 1].astype(np.float32), "dir_z": scan.directions[:, 2].astype(np.float32)}
    if include_gt and scan.gt_is_ghost is not None and len(scan.gt_is_ghost) == R:
        cols["gt_ghost"] = np.asarray(scan.gt_is_ghost).astype(np.uint8)
    if include_gt and scan.gt_object_id is not None and len(scan.gt_object_id) == R:
        cols["gt_object"] = np.asarray(scan.gt_object_id).astype(np.int32)
    return cols


def fill_no_return_rays(directions: np.ndarray, ranges: np.ndarray, pattern: dict) -> tuple[np.ndarray, np.ndarray]:
    """Append the rays of a (rings x azimuth) scan pattern that produced no point as ``inf``-range rays.

    ``pattern``: ``rings``, ``elev_min_deg``, ``elev_max_deg``, ``az_res_deg`` (same keys as the
    ``sensors.lidar`` scenario config).  Returned rays whose elevation is off-pattern are kept as-is.
    """
    rings = int(pattern.get("rings", 16))
    el0 = np.deg2rad(float(pattern.get("elev_min_deg", -15.0)))
    el1 = np.deg2rad(float(pattern.get("elev_max_deg", 15.0)))
    daz = np.deg2rad(float(pattern.get("az_res_deg", 2.0)))
    n_az = int(round(2 * np.pi / daz))
    el_grid = np.linspace(el0, el1, rings) if rings > 1 else np.array([0.5 * (el0 + el1)])
    d_el = (el1 - el0) / max(rings - 1, 1)
    occupied = np.zeros((rings, n_az), dtype=bool)
    if len(directions):
        el = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
        az = np.arctan2(directions[:, 1], directions[:, 0])
        ri = np.clip(np.round((el - el0) / d_el).astype(int), 0, rings - 1) if rings > 1 else np.zeros(len(el), int)
        on = np.abs(el - el_grid[ri]) <= 0.5 * d_el + 1e-6 if rings > 1 else np.ones(len(el), bool)
        ai = np.round((az + np.pi) / daz).astype(int) % n_az
        occupied[ri[on], ai[on]] = True
    miss_r, miss_a = np.nonzero(~occupied)
    if not len(miss_r):
        return directions, ranges
    E, A = el_grid[miss_r], -np.pi + miss_a * daz
    d_new = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], 1)
    return np.vstack([directions, d_new]), np.concatenate([ranges, np.full(len(d_new), np.inf)])


def lidar_from_columns(cols: dict[str, np.ndarray], header: Header, sensor_height: float = 0.9,
                       pattern: dict | None = None, min_range: float = 0.05) -> LidarScan:
    """Column dict (from any PointCloud2 layout) -> LidarScan (sensor frame)."""
    names = set(cols)
    ghost = cols["gt_ghost"].astype(bool) if "gt_ghost" in names else None
    obj = cols["gt_object"].astype(np.int64) if "gt_object" in names else None
    ring_name = next((n for n in RING_FIELD_ALIASES if n in names), None)
    if {"range", "dir_x", "dir_y", "dir_z"} <= names:
        dirs = np.stack([cols["dir_x"], cols["dir_y"], cols["dir_z"]], 1).astype(float)
        rng = cols["range"].astype(float)
        rng = np.where(np.isfinite(rng) & (rng >= min_range), rng, np.inf)
        inten_all = cols["intensity"].astype(float) if "intensity" in names else np.ones(len(rng))
        ring_all = cols[ring_name].astype(np.int64) if ring_name else None
    else:
        xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(float)
        dist = np.linalg.norm(xyz, axis=1)
        ok = np.isfinite(dist) & (dist >= min_range)
        xyz, dist = xyz[ok], dist[ok]
        dirs = xyz / dist[:, None]
        rng = dist
        inten_all = cols["intensity"][ok].astype(float) if "intensity" in names else np.ones(len(rng))
        ring_all = cols[ring_name][ok].astype(np.int64) if ring_name else None
        ghost = ghost[ok] if ghost is not None else None
        obj = obj[ok] if obj is not None else None
    if ring_all is None:
        el = np.arcsin(np.clip(dirs[:, 2], -1, 1))
        n_r = int((pattern or {}).get("rings", 16))
        ring_all = (np.digitize(el, np.linspace(el.min() - 1e-6, el.max() + 1e-6, n_r + 1)) - 1
                    if len(el) else np.zeros(0, np.int64))
    if pattern:
        n0 = len(rng)
        dirs, rng = fill_no_return_rays(dirs, rng, pattern)
        pad = len(rng) - n0
        if ghost is not None:
            ghost = np.concatenate([ghost, np.zeros(pad, bool)])
        if obj is not None:
            obj = np.concatenate([obj, np.full(pad, -1, np.int64)])
        inten_all = np.concatenate([inten_all, np.zeros(pad)])
        ring_all = np.concatenate([ring_all, np.zeros(pad, np.int64)])
    fin = np.isfinite(rng)
    return LidarScan(header, dirs[fin] * rng[fin, None], inten_all[fin], ring_all[fin], dirs, rng,
                     sensor_height=float(sensor_height), gt_is_ghost=ghost, gt_object_id=obj)


# =====================================================================================================================
# lidar (ROS part)
# =====================================================================================================================
def _point_fields(cols: dict[str, np.ndarray]):
    from sensor_msgs.msg import PointField

    spec = dict(LIDAR_FIELDS + LIDAR_GT_FIELDS)
    fields, off = [], 0
    for name in cols:
        dt = spec[name]
        fields.append(PointField(name=name, offset=off, datatype=dt, count=1))
        off += np.dtype(PF_NUMPY[dt]).itemsize
    return fields


def lidar_to_pointcloud2(scan: LidarScan, t0: float = 0.0, include_gt: bool = True):
    from sensor_msgs_py import point_cloud2 as pc2

    cols = lidar_to_columns(scan, include_gt)
    fields = _point_fields(cols)
    arr = np.zeros(len(scan.ranges), dtype=pc2.dtype_from_fields(fields))
    for name, v in cols.items():
        arr[name] = v
    return pc2.create_cloud(ros_header(scan.header.stamp, scan.header.frame_id or "lidar_link", t0), fields, arr)


def pointcloud2_columns(msg, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    from sensor_msgs_py import point_cloud2 as pc2

    present = [f.name for f in msg.fields]
    want = [n for n in (names or present) if n in present]
    arr = pc2.read_points(msg, field_names=want, skip_nans=False)
    return {n: np.asarray(arr[n]).reshape(-1) for n in want}


def lidar_from_pointcloud2(msg, recv: float, t0: float = 0.0, sensor_height: float = 0.9,
                           pattern: dict | None = None) -> LidarScan:
    return lidar_from_columns(pointcloud2_columns(msg), header_from_ros(msg.header, recv, t0), sensor_height,
                              pattern)


# =====================================================================================================================
# camera
# =====================================================================================================================
def _reorder_logits(logits: np.ndarray, names: list[str]) -> np.ndarray:
    if not names or list(names) == list(CLASSES):
        return logits
    out = np.full(len(CLASSES), -20.0)
    for j, c in enumerate(CLASSES):
        if c in names and names.index(c) < len(logits):
            out[j] = logits[names.index(c)]
    return out


def camera_to_ros(frame: CameraFrame, t0: float = 0.0):
    from geometry_msgs.msg import Point
    from medortrace_msgs.msg import CameraDetection as RosDet
    from medortrace_msgs.msg import CameraDetectionArray, SurfaceObservation

    m = CameraDetectionArray(header=ros_header(frame.header.stamp, frame.header.frame_id or "camera_link", t0),
                             fov_h=float(frame.fov_h), max_range=float(frame.max_range),
                             exposure_ok=bool(frame.exposure_ok), class_names=list(CLASSES))
    m.detections = [RosDet(cls=d.cls, item_id_hint=d.item_id_hint or "", bearing=float(d.bearing),
                           elevation=float(d.elevation), range=float(d.range),
                           logits=[float(v) for v in np.asarray(d.logits).ravel()],
                           visible_fraction=float(d.visible_fraction), glare=float(d.glare),
                           gt_item_id=d.gt_item_id or "") for d in frame.detections]
    m.surfaces = [SurfaceObservation(class_name=c, material=mat, position=Point(x=float(p[0]), y=float(p[1]),
                                                                                  z=float(p[2])))
                  for c, mat, p in frame.surfaces]
    return m


def camera_from_ros(msg, recv: float, t0: float = 0.0) -> CameraFrame:
    names = list(msg.class_names)
    dets = [CameraDetection(cls=d.cls, item_id_hint=d.item_id_hint or None, bearing=float(d.bearing),
                            elevation=float(d.elevation), range=float(d.range),
                            logits=_reorder_logits(np.asarray(d.logits, float), names),
                            visible_fraction=float(d.visible_fraction), glare=float(d.glare),
                            gt_item_id=d.gt_item_id or None) for d in msg.detections]
    surf = [(s.class_name, s.material, np.array([s.position.x, s.position.y, s.position.z])) for s in msg.surfaces]
    return CameraFrame(header_from_ros(msg.header, recv, t0), dets, float(msg.fov_h), float(msg.max_range),
                       bool(msg.exposure_ok), surf)


# =====================================================================================================================
# radar
# =====================================================================================================================
def radar_to_ros(frame: RadarFrame, t0: float = 0.0):
    from medortrace_msgs.msg import RadarDetection as RosDet
    from medortrace_msgs.msg import RadarDetectionArray

    return RadarDetectionArray(
        header=ros_header(frame.header.stamp, frame.header.frame_id or "radar_link", t0),
        detections=[RosDet(range=float(d.range), azimuth=float(d.azimuth), elevation=float(d.elevation),
                           radial_velocity=float(d.radial_velocity), rcs_dbsm=float(d.rcs_dbsm),
                           through_fabric=bool(d.through_fabric), gt_object_id=d.gt_object_id or "")
                    for d in frame.detections])


def radar_from_ros(msg, recv: float, t0: float = 0.0) -> RadarFrame:
    return RadarFrame(header_from_ros(msg.header, recv, t0), [
        RadarDetection(float(d.range), float(d.azimuth), float(d.elevation), float(d.radial_velocity),
                       float(d.rcs_dbsm), bool(d.through_fabric), d.gt_object_id or None) for d in msg.detections])


def radar_from_columns(cols: dict[str, np.ndarray], header: Header, velocity_field: str = "velocity",
                       rcs_field: str = "intensity", rcs_is_db: bool = False) -> RadarFrame:
    """Radar point cloud (x, y, z, doppler, rcs|snr) -> RadarFrame (physical-radar driver input)."""
    xyz = np.stack([cols["x"], cols["y"], cols["z"]], 1).astype(float)
    r = np.linalg.norm(xyz, axis=1)
    ok = np.isfinite(r) & (r > 1e-3)
    vr = cols[velocity_field].astype(float) if velocity_field in cols else np.zeros(len(r))
    if rcs_field in cols:
        raw = cols[rcs_field].astype(float)
        rcs = raw if rcs_is_db else 10.0 * np.log10(np.maximum(raw, 1e-6))
    else:
        rcs = np.zeros(len(r))
    dets = [RadarDetection(float(r[k]), float(np.arctan2(xyz[k, 1], xyz[k, 0])), float(np.arcsin(xyz[k, 2] / r[k])),
                           float(vr[k]), float(rcs[k])) for k in np.nonzero(ok)[0]]
    return RadarFrame(header, dets)


def radar_from_pointcloud2(msg, recv: float, t0: float = 0.0, velocity_field: str = "velocity",
                           rcs_field: str = "intensity", rcs_is_db: bool = False) -> RadarFrame:
    return radar_from_columns(pointcloud2_columns(msg), header_from_ros(msg.header, recv, t0), velocity_field,
                              rcs_field, rcs_is_db)


# =====================================================================================================================
# acoustic / landmarks / contact
# =====================================================================================================================
def acoustic_to_ros(frame: AcousticFrame, t0: float = 0.0):
    from medortrace_msgs.msg import AcousticEcho as RosEcho
    from medortrace_msgs.msg import AcousticFrame as RosFrame

    def gt(v):
        return -1 if v is None else int(bool(v))

    return RosFrame(header=ros_header(frame.header.stamp, frame.header.frame_id or "acoustic_link", t0),
                    echoes=[RosEcho(target_region=e.target_region, energy=float(e.energy), delay_s=float(e.delay_s),
                                    path_occluded=bool(e.path_occluded), gt_hard_reflector=gt(e.gt_hard_reflector))
                            for e in frame.echoes])


def acoustic_from_ros(msg, recv: float, t0: float = 0.0) -> AcousticFrame:
    return AcousticFrame(header_from_ros(msg.header, recv, t0), [
        AcousticEcho(e.target_region, float(e.energy), float(e.delay_s), bool(e.path_occluded),
                     None if e.gt_hard_reflector < 0 else bool(e.gt_hard_reflector)) for e in msg.echoes])


def landmarks_to_ros(frame: LandmarkFrame, t0: float = 0.0):
    from medortrace_msgs.msg import LandmarkObservation as RosObs
    from medortrace_msgs.msg import LandmarkObservationArray

    return LandmarkObservationArray(
        header=ros_header(frame.header.stamp, frame.header.frame_id or "camera_link", t0),
        observations=[RosObs(landmark_id=o.landmark_id, range=float(o.range), bearing=float(o.bearing))
                      for o in frame.observations])


def landmarks_from_ros(msg, recv: float, t0: float = 0.0) -> LandmarkFrame:
    return LandmarkFrame(header_from_ros(msg.header, recv, t0), [
        LandmarkObservation(o.landmark_id, float(o.range), float(o.bearing)) for o in msg.observations])


def contact_to_ros(c: ContactState, t0: float = 0.0):
    from medortrace_msgs.msg import ContactState as RosContact

    return RosContact(header=ros_header(c.header.stamp, c.header.frame_id or "base_link", t0),
                      in_contact=bool(c.in_contact), force_n=float(c.force_n), location=c.location or "",
                      effort=[] if c.effort is None else [float(v) for v in np.asarray(c.effort).ravel()])


def contact_from_ros(msg, recv: float, t0: float = 0.0) -> ContactState:
    return ContactState(header_from_ros(msg.header, recv, t0), bool(msg.in_contact), float(msg.force_n),
                        msg.location or "bumper", np.asarray(msg.effort, float) if len(msg.effort) else None)


# =====================================================================================================================
# proprioception: IMU, odometry, battery
# =====================================================================================================================
def imu_to_ros(s: ImuSample, t0: float = 0.0):
    from geometry_msgs.msg import Vector3
    from sensor_msgs.msg import Imu

    m = Imu(header=ros_header(s.header.stamp, s.header.frame_id or "imu_link", t0))
    m.orientation_covariance[0] = -1.0                       # orientation not provided
    m.linear_acceleration = Vector3(x=float(s.lin_acc[0]), y=float(s.lin_acc[1]), z=float(s.lin_acc[2]))
    m.angular_velocity = Vector3(x=float(s.ang_vel[0]), y=float(s.ang_vel[1]), z=float(s.ang_vel[2]))
    return m


def imu_from_ros(msg, recv: float, t0: float = 0.0) -> ImuSample:
    a, w = msg.linear_acceleration, msg.angular_velocity
    return ImuSample(header_from_ros(msg.header, recv, t0), np.array([a.x, a.y, a.z]), np.array([w.x, w.y, w.z]))


def _pose_msg(x: float, y: float, yaw: float):
    from geometry_msgs.msg import Point, Pose, Quaternion

    qx, qy, qz, qw = yaw_to_quat_xyzw(yaw)
    return Pose(position=Point(x=float(x), y=float(y), z=0.0), orientation=Quaternion(x=qx, y=qy, z=qz, w=qw))


def odom_to_ros(o: WheelOdometry, pose_odom: np.ndarray, t0: float = 0.0, odom_frame: str = "odom",
                base_frame: str = "base_link"):
    from nav_msgs.msg import Odometry

    m = Odometry(header=ros_header(o.header.stamp, odom_frame, t0), child_frame_id=base_frame)
    m.pose.pose = _pose_msg(*pose_odom)
    m.twist.twist.linear.x = float(o.v)
    m.twist.twist.angular.z = float(o.omega)
    m.twist.covariance[0] = 0.01 ** 2
    m.twist.covariance[35] = 0.005 ** 2
    return m


def odom_from_ros(msg, recv: float, t0: float = 0.0) -> WheelOdometry:
    h = header_from_ros(msg.header, recv, t0)
    h.frame_id = msg.child_frame_id or "base_link"
    return WheelOdometry(h, float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z))


def pose2d_from_pose(p) -> np.ndarray:
    q = p.orientation
    return np.array([p.position.x, p.position.y, quat_xyzw_to_yaw(q.x, q.y, q.z, q.w)])


def battery_to_ros(wh: float, t_mission: float, t0: float = 0.0, capacity_wh: float = 480.0,
                   nominal_v: float = 25.6):
    from sensor_msgs.msg import BatteryState

    return BatteryState(header=ros_header(t_mission, "base_link", t0), voltage=float(nominal_v),
                        charge=float(wh / nominal_v), capacity=float(capacity_wh / nominal_v),
                        design_capacity=float(capacity_wh / nominal_v), percentage=float(wh / capacity_wh),
                        power_supply_status=BatteryState.POWER_SUPPLY_STATUS_DISCHARGING,
                        power_supply_technology=BatteryState.POWER_SUPPLY_TECHNOLOGY_LIFE, present=True)


def battery_from_ros(msg, capacity_wh: float = 480.0) -> float | None:
    """Remaining energy [Wh]: charge x voltage when reported, else percentage x capacity."""
    if math.isfinite(msg.charge) and msg.charge > 0 and math.isfinite(msg.voltage) and msg.voltage > 0:
        return float(msg.charge * msg.voltage)
    if math.isfinite(msg.percentage):
        frac = msg.percentage / 100.0 if msg.percentage > 1.0 else msg.percentage   # some drivers send 0..100
        return float(frac * capacity_wh)
    return None


# =====================================================================================================================
# workflow
# =====================================================================================================================
def workflow_to_ros(ev: WorkflowEvent, t0: float = 0.0, source: str = "sim"):
    from medortrace_msgs.msg import WorkflowEvent as RosEvent

    return RosEvent(header=ros_header(ev.t, "map", t0), type=ev.type.value, event_id=ev.event_id or "",
                    item_id=ev.item_id or "", src=ev.src or "", dst=ev.dst or "", reporter=ev.reporter or "",
                    confidence=float(ev.confidence), source=source,
                    payload_json=json.dumps(ev.payload or {}, sort_keys=True, default=str))


def workflow_from_ros(msg, t0: float = 0.0) -> WorkflowEvent | None:
    try:
        typ = WorkflowEventType(msg.type)
    except ValueError:
        return None
    try:
        payload = json.loads(msg.payload_json) if msg.payload_json else {}
    except json.JSONDecodeError:
        payload = {"raw": msg.payload_json}
    if msg.source:
        payload.setdefault("source", msg.source)
    return WorkflowEvent(to_sec(msg.header.stamp) - t0, typ, msg.item_id or None, msg.src or None, msg.dst or None,
                         msg.reporter or "unknown", float(msg.confidence), msg.event_id or "", payload)


# =====================================================================================================================
# commands
# =====================================================================================================================
def cmd_to_twist(cmd: VelocityCommand):
    from geometry_msgs.msg import Twist

    m = Twist()
    m.linear.x = float(cmd.v)
    m.angular.z = float(cmd.omega)
    return m


def twist_to_cmd(msg, probe_target: str | None = None) -> VelocityCommand:
    return VelocityCommand(float(msg.linear.x), float(msg.angular.z), acoustic_probe_target=probe_target or None)


# =====================================================================================================================
# autonomy outputs (records -> medortrace_msgs)
# =====================================================================================================================
def pose2d_msg(p):
    from medortrace_msgs.msg import Pose2D

    return Pose2D(x=float(p[0]), y=float(p[1]), theta=float(p[2]))


def verdict_to_ros(v, t0: float = 0.0, frame: str = "map"):
    from medortrace_msgs.msg import ClaimVerdict

    return ClaimVerdict(
        header=ros_header(v.t, frame, t0), verdict=int(v.code), claim_id=v.claim_id, kind=v.kind, item_id=v.item_id,
        slot_id=v.slot_id, t_ref=to_time(v.t_ref + t0), posterior=float(v.posterior), reason=v.reason,
        direct_evidence=bool(v.direct), map_slot=v.map_slot or "",
        supporting_evidence_ids=[e for e, _ in v.supporting], supporting_weights=[float(w) for _, w in v.supporting],
        contradicting_evidence_ids=[e for e, _ in v.contradicting],
        contradicting_weights=[float(w) for _, w in v.contradicting],
        provenance_node_id=v.node_id, provenance_hash=v.node_hash)


def provenance_to_ros(p, t0: float = 0.0, frame: str = "map"):
    from medortrace_msgs.msg import ProvenanceEvent

    return ProvenanceEvent(header=ros_header(p.t, frame, t0), index=int(p.index), node_id=p.node_id, kind=p.kind,
                           t=float(p.t), sensor=p.sensor, attributed_to=p.attributed_to,
                           content_digest=p.content_digest, prev_hash=p.prev_hash, hash=p.hash,
                           json_attrs=p.json_attrs)


def safety_to_ros(s, t0: float = 0.0, frame: str = "base_link"):
    from medortrace_msgs.msg import SafetyState

    return SafetyState(header=ros_header(s.t, frame, t0), mode=int(s.code), mode_name=s.mode, reasons=list(s.reasons),
                       category=s.category, collision_prob=s.collision_prob, pred_clearance=s.pred_clearance,
                       human_clearance=s.human_clearance, loc_std=s.loc_std, nis=s.nis, lidar_age=s.lidar_age,
                       path_entropy=s.path_entropy, contact_force=s.contact_force, battery_frac=s.battery_frac,
                       operator_request_open=s.operator_request_open, handover_requests=int(s.handover_requests),
                       transitions=int(s.transitions))


def beliefs_to_ros(b, t0: float = 0.0, frame: str = "map"):
    from medortrace_msgs.msg import ItemBelief, ItemBeliefArray

    zero = to_time(0.0)
    return ItemBeliefArray(header=ros_header(b.t, frame, t0), slot_ids=list(b.slot_ids), slot_kinds=list(b.slot_kinds),
                           items=[ItemBelief(item_id=i.item_id, cls=i.cls, criticality=float(i.criticality),
                                             probabilities=[float(v) for v in i.probabilities], map_slot=i.map_slot,
                                             map_probability=float(i.map_probability),
                                             entropy_bits=float(i.entropy_bits),
                                             last_seen=zero if i.last_seen is None else to_time(i.last_seen + t0))
                                  for i in b.items])


def path_to_ros(path_xy: np.ndarray, t_mission: float, t0: float = 0.0, frame: str = "map"):
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path

    h = ros_header(t_mission, frame, t0)
    poses = []
    pts = np.asarray(path_xy, float).reshape(-1, 2)
    for k, p in enumerate(pts):
        nxt = pts[min(k + 1, len(pts) - 1)]
        yaw = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0])) if k + 1 < len(pts) else 0.0
        poses.append(PoseStamped(header=h, pose=_pose_msg(p[0], p[1], yaw)))
    return Path(header=h, poses=poses)


def nbv_to_ros(n, t0: float = 0.0, frame: str = "map"):
    from medortrace_msgs.msg import NextBestView

    keys = sorted(n.breakdown)
    return NextBestView(header=ros_header(n.t, frame, t0), pose=pose2d_msg(n.pose),
                        path=path_to_ros(n.path, n.t, t0, frame), target_slot=n.target_slot,
                        probe_region=n.probe_region, score=float(n.score), breakdown_keys=keys,
                        breakdown_values=[float(n.breakdown[k]) for k in keys])


def scene_graph_to_ros(sg: dict, t_mission: float, t0: float = 0.0, frame: str = "map"):
    from medortrace_msgs.msg import SceneGraph

    return SceneGraph(header=ros_header(t_mission, frame, t0), n_nodes=len(sg.get("nodes", [])),
                      n_edges=len(sg.get("edges", [])), json=json.dumps(sg, separators=(",", ":"), default=float))


def _map_meta(g, t0: float):
    from nav_msgs.msg import MapMetaData

    nx, ny = g.data.shape
    return MapMetaData(map_load_time=to_time(g.t + t0), resolution=float(g.res), width=int(nx), height=int(ny),
                       origin=_pose_msg(g.origin[0], g.origin[1], 0.0))


def uncertainty_to_ros(g, t0: float = 0.0, frame: str = "map"):
    from medortrace_msgs.msg import UncertaintyGrid

    data = np.asarray(g.data, np.float32).T.ravel()        # [ix, iy] -> row-major (iy rows, x fastest)
    return UncertaintyGrid(header=ros_header(g.t, frame, t0), info=_map_meta(g, t0), layer="entropy+ambiguous",
                           units="bits", max_value=float(data.max()) if len(data) else 0.0, data=data.tolist())


def occupancy_to_ros(g, t0: float = 0.0, frame: str = "map"):
    from nav_msgs.msg import OccupancyGrid

    p = np.asarray(g.data, np.float32).T.ravel()
    occ = np.where(np.isfinite(p), np.clip(np.round(p * 100.0), 0, 100), -1).astype(np.int8)
    return OccupancyGrid(header=ros_header(g.t, frame, t0), info=_map_meta(g, t0), data=occ.tolist())


def pose_cov_to_ros(x: np.ndarray, P: np.ndarray, t_mission: float, t0: float = 0.0, frame: str = "map"):
    from geometry_msgs.msg import PoseWithCovarianceStamped

    m = PoseWithCovarianceStamped(header=ros_header(t_mission, frame, t0))
    m.pose.pose = _pose_msg(*x)
    cov = np.zeros((6, 6))
    idx = [0, 1, 5]
    for a in range(3):
        for b in range(3):
            cov[idx[a], idx[b]] = P[a, b]
    m.pose.covariance = [float(v) for v in cov.ravel()]
    return m


def transform_msg(parent: str, child: str, pose: np.ndarray, t_mission: float, t0: float = 0.0):
    """Planar TransformStamped parent -> child from (x, y, yaw)."""
    from geometry_msgs.msg import TransformStamped

    m = TransformStamped(header=ros_header(t_mission, parent, t0), child_frame_id=child)
    m.transform.translation.x = float(pose[0])
    m.transform.translation.y = float(pose[1])
    qx, qy, qz, qw = yaw_to_quat_xyzw(float(pose[2]))
    m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = qx, qy, qz, qw
    return m
