"""Backend-neutral message types.

These dataclasses are the contract between simulator backends (lite / Isaac
Sim / physical robot via ROS 2) and the autonomy stack.  Each has a 1:1 ROS 2
counterpart in ``ros2_ws/src/medortrace_msgs`` (see docs/ros2_message_graph.md)
and converters live in ``ros2_ws/src/medortrace_ros/medortrace_ros/convert.py``.

Every sensor message carries two stamps:

* ``stamp``      - the time the sensor *claims* the data was acquired (may be
                   skewed by a faulty clock);
* ``recv_stamp`` - the time the autonomy stack received it (monotonic host
                   clock).  The difference feeds the time-sync health monitor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


@dataclass
class Header:
    stamp: float
    recv_stamp: float
    frame_id: str
    seq: int = 0


@dataclass
class LidarScan:
    header: Header
    # sensor-frame points (N,3), intensity (N,), return index (N,), ring (N,)
    points: np.ndarray
    intensity: np.ndarray
    ring: np.ndarray
    # Ray origin and unit directions in the *sensor* frame for free-space carving.
    directions: np.ndarray
    ranges: np.ndarray            # (R,) range per ray, np.inf if no return
    sensor_height: float = 0.9
    # Ground-truth only, populated by simulators for supervision / evaluation.
    gt_is_ghost: np.ndarray | None = None
    gt_object_id: np.ndarray | None = None


@dataclass
class CameraDetection:
    cls: str
    item_id_hint: str | None      # decoded tag (only when the tag is readable)
    bearing: float                # rad, in camera/robot frame
    elevation: float
    range: float                  # from aligned depth
    logits: np.ndarray            # raw class logits from the (learned) front end
    visible_fraction: float
    glare: float                  # specular highlight score in [0,1]
    gt_item_id: str | None = None  # supervision only


@dataclass
class CameraFrame:
    header: Header
    detections: list[CameraDetection]
    fov_h: float
    max_range: float
    exposure_ok: bool = True
    # Semantic surface observations: (class_name, material, world-free xyz in camera frame)
    surfaces: list[tuple[str, str, np.ndarray]] = field(default_factory=list)


@dataclass
class RadarDetection:
    range: float
    azimuth: float
    elevation: float
    radial_velocity: float        # Doppler, m/s (positive = receding)
    rcs_dbsm: float
    through_fabric: bool = False  # true if the path crossed a drape/fabric
    gt_object_id: str | None = None


@dataclass
class RadarFrame:
    header: Header
    detections: list[RadarDetection]


@dataclass
class AcousticEcho:
    """Active acoustic probe aimed at a region (e.g. a draped tray).

    ``energy`` is the normalised echo energy in the range gate around the
    target; diffracted paths are allowed so line-of-sight is not required.
    """

    target_region: str
    energy: float
    delay_s: float
    path_occluded: bool
    gt_hard_reflector: bool | None = None


@dataclass
class AcousticFrame:
    header: Header
    echoes: list[AcousticEcho]


@dataclass
class ImuSample:
    header: Header
    lin_acc: np.ndarray           # (3,) m/s^2 body frame
    ang_vel: np.ndarray           # (3,) rad/s body frame


@dataclass
class WheelOdometry:
    header: Header
    v: float
    omega: float


@dataclass
class ContactState:
    header: Header
    in_contact: bool
    force_n: float
    location: str = "bumper"
    effort: np.ndarray | None = None   # joint efforts of the manipulator (Nm)


@dataclass
class LandmarkObservation:
    """Range/bearing to a surveyed fiducial (wall AprilTag / ceiling marker)."""

    landmark_id: str
    range: float
    bearing: float


@dataclass
class LandmarkFrame:
    header: Header
    observations: list[LandmarkObservation]


class WorkflowEventType(str, Enum):
    HANDOFF = "handoff"          # item passes between holders/slots
    PLACE = "place"              # item placed on a surface slot
    DISCARD = "discard"          # item into waste / kick bucket
    COUNT = "count"              # announced count checkpoint
    OPEN = "open"                # item opened onto sterile field
    SPECIMEN_OUT = "specimen_out"


@dataclass
class WorkflowEvent:
    """An event reported by the OR workflow system (voice/log/EHR integration)."""

    t: float
    type: WorkflowEventType
    item_id: str | None
    src: str | None
    dst: str | None
    reporter: str = "circulating_nurse"
    confidence: float = 0.9
    event_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorBundle:
    """Everything the stack receives in one control tick (any field may be None)."""

    t: float
    lidar: LidarScan | None = None
    camera: CameraFrame | None = None
    radar: RadarFrame | None = None
    acoustic: AcousticFrame | None = None
    imu: list[ImuSample] = field(default_factory=list)
    odom: WheelOdometry | None = None
    contact: ContactState | None = None
    landmarks: LandmarkFrame | None = None
    workflow: list[WorkflowEvent] = field(default_factory=list)
    battery_wh: float | None = None


@dataclass
class VelocityCommand:
    v: float = 0.0
    omega: float = 0.0
    acoustic_probe_target: str | None = None
    manipulation: dict[str, Any] | None = None
