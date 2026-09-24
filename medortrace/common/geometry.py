"""Small, dependency-light geometry helpers (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    return (float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0)))


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=float)

    @staticmethod
    def from_array(a) -> "Pose2D":
        return Pose2D(float(a[0]), float(a[1]), float(wrap_angle(a[2])))

    def transform_points(self, pts_local: np.ndarray) -> np.ndarray:
        """Robot-frame (N,2|3) points -> world frame."""
        pts = np.atleast_2d(pts_local)
        out = pts.copy().astype(float)
        out[:, :2] = pts[:, :2] @ rot2(self.theta).T + np.array([self.x, self.y])
        return out

    def inverse_transform_points(self, pts_world: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts_world)
        out = pts.copy().astype(float)
        out[:, :2] = (pts[:, :2] - np.array([self.x, self.y])) @ rot2(self.theta)
        return out

    def distance_to(self, xy) -> float:
        return float(np.hypot(xy[0] - self.x, xy[1] - self.y))


@dataclass
class OrientedBox:
    """Box with yaw-only orientation (sufficient for OR furniture on a flat floor).

    ``center`` is the 3D centre, ``half`` the half extents along the box axes.
    """

    center: np.ndarray
    half: np.ndarray
    yaw: float = 0.0

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float)
        self.half = np.asarray(self.half, dtype=float)

    @property
    def z_min(self) -> float:
        return float(self.center[2] - self.half[2])

    @property
    def z_max(self) -> float:
        return float(self.center[2] + self.half[2])

    def corners_xy(self) -> np.ndarray:
        hx, hy = self.half[0], self.half[1]
        local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        return local @ rot2(self.yaw).T + self.center[:2]

    def contains_xy(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        pts = np.atleast_2d(pts)[:, :2]
        local = (pts - self.center[:2]) @ rot2(self.yaw)
        return (np.abs(local[:, 0]) <= self.half[0] + margin) & (
            np.abs(local[:, 1]) <= self.half[1] + margin
        )

    def distance_xy(self, pts: np.ndarray) -> np.ndarray:
        """Signed-ish (>=0 outside, 0 inside) planar distance from points to the box."""
        pts = np.atleast_2d(pts)[:, :2]
        local = np.abs((pts - self.center[:2]) @ rot2(self.yaw)) - self.half[:2]
        outside = np.linalg.norm(np.maximum(local, 0.0), axis=1)
        return outside

    def top_center(self) -> np.ndarray:
        return np.array([self.center[0], self.center[1], self.z_max])


@dataclass
class Polygon2D:
    vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    def contains(self, pts: np.ndarray) -> np.ndarray:
        """Even-odd rule point in polygon, vectorised over points."""
        pts = np.atleast_2d(pts)[:, :2]
        v = np.asarray(self.vertices)
        inside = np.zeros(len(pts), dtype=bool)
        n = len(v)
        for i in range(n):
            x1, y1 = v[i]
            x2, y2 = v[(i + 1) % n]
            cond = (y1 > pts[:, 1]) != (y2 > pts[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = (x2 - x1) * (pts[:, 1] - y1) / (y2 - y1 + 1e-12) + x1
            inside ^= cond & (pts[:, 0] < xint)
        return inside


def segment_point_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Distance from points p (N,2) to segment ab."""
    ab = b - a
    denom = float(ab @ ab) + 1e-12
    t = np.clip(((p - a) @ ab) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)
