"""Engine-agnostic OR scene specification.

A :class:`SceneSpec` is the single source of truth from which we author the USD
stage (``medortrace.usd``), drive the lite simulator (``medortrace.sim``) and
build the robot's *prior* map (possibly corrupted, see ``sim.faults``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from medortrace.common.geometry import OrientedBox


@dataclass
class SceneObject:
    name: str
    kind: str                     # wall, or_table, back_table, mayo_stand, cart, ...
    box: OrientedBox
    material: str
    semantic: str                 # semantic class label for segmentation
    movable: bool = False
    sterile: bool = False
    rigid_body: bool = False      # dynamic PhysX body (carts) vs static collider
    mass_kg: float = 0.0
    visible_to_lidar: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class Slot:
    """A discrete place a critical item can be (the support of item beliefs)."""

    id: str
    kind: str                     # surface | container | under_drape | hand | floor | elsewhere
    anchor: str                   # object or agent name
    position: np.ndarray          # nominal world position (hands: updated online)
    hidden_from_camera: bool = False
    needs_top_view: bool = False  # open container: only visible from close/above
    sterile: bool = False
    acoustic_region: str | None = None
    radius: float = 0.25          # spatial extent for association

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)


@dataclass
class SterileZone:
    name: str
    box: OrientedBox
    keepout_margin: float = 0.5   # robot keep-out margin beyond the sterile boundary


@dataclass
class Landmark:
    id: str
    position: np.ndarray          # (3,)


@dataclass
class StaffSpec:
    name: str
    role: str                     # surgeon, assistant, scrub_nurse, circulator, anesthetist
    home: np.ndarray              # (2,)
    sterile: bool                 # scrubbed-in staff stay inside the sterile area
    waypoints: list[np.ndarray] = field(default_factory=list)
    speed: float = 1.0
    radius: float = 0.25
    roaming: bool = False


@dataclass
class ItemSpec:
    id: str
    cls: str                      # sponge, clamp, needle_driver, specimen, implant_box
    material: str
    initial_slot: str
    criticality: float = 1.0      # weight in NBV objective & metrics
    fungible: bool = False        # visually indistinguishable within the class
    tag_readable: bool = True     # has a 2D code the camera can decode when close
    metallic: bool = False
    size: tuple[float, float, float] = (0.1, 0.05, 0.02)


@dataclass
class Light:
    name: str
    position: np.ndarray
    intensity: float
    kind: str = "surgical"        # surgical | ambient


@dataclass
class SceneSpec:
    seed: int
    room: tuple[float, float, float]
    objects: list[SceneObject]
    slots: list[Slot]
    sterile_zones: list[SterileZone]
    landmarks: list[Landmark]
    staff: list[StaffSpec]
    items: list[ItemSpec]
    lights: list[Light]
    robot_start: np.ndarray
    dock: np.ndarray
    hidden_cause: dict[str, Any] = field(default_factory=dict)
    nuisance: dict[str, Any] = field(default_factory=dict)
    family: str = "nominal"
    scenario_id: str = ""

    # ---- convenience -------------------------------------------------------
    def object(self, name: str) -> SceneObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def slot(self, sid: str) -> Slot:
        for s in self.slots:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def slot_ids(self) -> list[str]:
        return [s.id for s in self.slots]

    def item(self, iid: str) -> ItemSpec:
        for i in self.items:
            if i.id == iid:
                return i
        raise KeyError(iid)

    def in_keepout(self, xy: np.ndarray, extra: float = 0.0) -> np.ndarray:
        xy = np.atleast_2d(xy)
        mask = np.zeros(len(xy), dtype=bool)
        for z in self.sterile_zones:
            mask |= z.box.contains_xy(xy, margin=z.keepout_margin + extra)
        return mask

    def to_json(self) -> str:
        def enc(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, OrientedBox):
                return {"center": o.center.tolist(), "half": o.half.tolist(), "yaw": o.yaw}
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            raise TypeError(type(o))

        return json.dumps(asdict(self), default=enc, indent=1)
