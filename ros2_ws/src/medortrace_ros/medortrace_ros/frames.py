"""TF frame tree of the MED-OR-TRACE rig (no ROS imports).

    map --(autonomy_node: EKF estimate x odometry^-1)--> odom
    odom --(sim_bridge_node | base driver: wheel odometry)--> base_link
    base_link --(static, configs/robot/rig.yaml "frames")--> lidar_link, camera_link, radar_link,
                                                             acoustic_link, imu_link

``configs/robot/rig.yaml`` is the single source of truth for the sensor
extrinsics: the USD rig (``medortrace.usd.robot_rig``), the lite simulator's
mount parameters and the static transforms published by the launch files all
read it.  ``base_link`` is the floor-level footprint centre, x forward, z up
(REP 103/105).  ``rpy_deg`` is roll / pitch / yaw in degrees (fixed axes, ROS
convention), so ``camera_link`` pitched +25 deg looks *down* at the tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medortrace.common.config import load_yaml

MAP_FRAME = "map"
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
SENSOR_FRAMES = ("lidar_link", "camera_link", "radar_link", "acoustic_link", "imu_link")


@dataclass(frozen=True)
class StaticFrame:
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]           # radians
    quat_xyzw: tuple[float, float, float, float]


def rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (float(sr * cp * cy - cr * sp * sy), float(cr * sp * cy + sr * cp * sy),
            float(cr * cp * sy - sr * sp * cy), float(cr * cp * cy + sr * sp * sy))


def yaw_to_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))


def quat_xyzw_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def load_rig(path: str | Path = "robot/rig.yaml") -> dict:
    return load_yaml(path)


def rig_static_frames(rig: dict | None = None) -> list[StaticFrame]:
    """base_link -> sensor frames from the rig description."""
    rig = rig if rig is not None else load_rig()
    out = []
    for child, f in (rig.get("frames") or {}).items():
        xyz = tuple(float(v) for v in f.get("xyz", (0.0, 0.0, 0.0)))
        rpy = tuple(float(np.deg2rad(v)) for v in f.get("rpy_deg", (0.0, 0.0, 0.0)))
        out.append(StaticFrame(BASE_FRAME, child, xyz, rpy, rpy_to_quat_xyzw(*rpy)))
    return out


def lidar_mount(rig: dict | None = None) -> tuple[float, float]:
    """(mount_x, mount_height) of lidar_link; the stack models the lidar by these two numbers."""
    for f in rig_static_frames(rig):
        if f.child == "lidar_link":
            return f.xyz[0], f.xyz[2]
    return 0.1, 0.9


def compose_map_to_odom(map_base: np.ndarray, odom_base: np.ndarray) -> np.ndarray:
    """T_map_odom = T_map_base * T_odom_base^-1 for planar poses (x, y, yaw)."""
    xm, ym, tm = map_base
    xo, yo, to = odom_base
    th = tm - to
    c, s = np.cos(th), np.sin(th)
    return np.array([xm - (c * xo - s * yo), ym - (s * xo + c * yo), float(np.arctan2(np.sin(th), np.cos(th)))])
