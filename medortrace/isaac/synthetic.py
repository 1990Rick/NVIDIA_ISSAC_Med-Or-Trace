"""Planning for the Replicator synthetic-data program (pure numpy; pxr only inside the USD helpers).

``scripts/isaac/generate_synthetic_data.py`` renders; everything that decides
*what* is rendered lives here so it is deterministic and unit-testable:

* **Units.**  Registry entries are grouped into render units: a counterfactual
  pair (both arms, same ``pair_id``) or a single scenario.  The registry split
  is pair-level already; :func:`make_units` refuses a unit whose arms disagree
  (that would leak a matched pair across train/test).
* **Shared schedule.**  Per unit, one RNG keyed on ``(master seed, unit key)``
  draws the frame times and the camera viewpoints, and seeds the nuisance
  randomiser.  Both arms therefore render the same frames from the same poses
  with the same nuisance: frame ``k`` of arm A and arm B is a matched pair
  that differs only through the hidden cause (and its causal consequences).
* **Causal state at time t** (:func:`causal_states`): item truth slots from the
  workflow ground truth, item positions with the lite placement rules
  (``medortrace.isaac.truth``), staff positions from the robot-free shadow
  population (the dataset scene has no robot).
* **Viewpoints** (:func:`sample_viewpoints`): robot-like camera poses - base in
  free space (footprint + clearance clear of every object in the robot's
  height band, inside the room, outside the sterile keep-out, away from staff)
  in *every* arm, camera at the mast height/pitch of ``configs/sensors``,
  mostly aimed at a slot 0.8-3.5 m away (inspection views), sometimes random.
* **Labels / index** (:func:`frame_labels`, :func:`build_index`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medortrace.common.rng import stable_hash
from medortrace.isaac.truth import CustodyTimeline, item_offsets, item_position, item_prim_center, staff_positions
from medortrace.world.agents import StaffPopulation
from medortrace.world.scene import SceneSpec

CAMERA_PATH = "/World/DatasetCamera"
HORIZONTAL_APERTURE_MM = 20.955          # same aperture as the rig camera (medortrace.usd.robot_rig)
ROBOT_RADIUS = 0.28
ROBOT_HEIGHT = 1.55


# ---------------------------------------------------------------------------
@dataclass
class Unit:
    key: str                 # pair_id for counterfactual pairs, else scenario_id
    entries: list            # RegistryEntry, one per arm
    split: str

    @property
    def is_pair(self) -> bool:
        return len(self.entries) > 1


def make_units(entries: list) -> list[Unit]:
    groups: dict[str, list] = {}
    for e in entries:
        groups.setdefault(e.pair_id or e.scenario_id, []).append(e)
    units = []
    for key, es in groups.items():
        splits = {e.split for e in es}
        if len(splits) != 1:
            raise ValueError(f"unit {key} spans splits {sorted(splits)}: pair-level split violated")
        units.append(Unit(key, sorted(es, key=lambda e: e.scenario_id), splits.pop()))
    return units


def unit_seed(master_seed: int, unit_key: str) -> int:
    return stable_hash(f"medortrace.synthetic:{int(master_seed)}:{unit_key}") & (2**63 - 1)


# ---------------------------------------------------------------------------
@dataclass
class Viewpoint:
    position: np.ndarray       # (3,) optical centre, world
    yaw: float
    pitch: float               # negative = looking down
    target_slot: str | None = None

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cp, sp, cy, sy = np.cos(self.pitch), np.sin(self.pitch), np.cos(self.yaw), np.sin(self.yaw)
        f = np.array([cp * cy, cp * sy, sp])
        u = np.array([-sp * cy, -sp * sy, cp])
        return np.cross(f, u), u, f          # right, up, forward

    def matrix(self) -> np.ndarray:
        """USD camera-to-world transform (row-vector convention; camera looks down -Z, +Y up)."""
        r, u, f = self.axes()
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3], m[3, :3] = r, u, -f, self.position
        return m

    def quat_wxyz(self) -> np.ndarray:
        from pxr import Gf
        q = Gf.Matrix4d(*self.matrix().ravel().tolist()).ExtractRotationQuat()
        return np.array([q.GetReal(), *q.GetImaginary()])

    def as_dict(self) -> dict:
        return {"position": self.position.tolist(), "yaw": float(self.yaw), "pitch": float(self.pitch),
                "quat_wxyz": self.quat_wxyz().tolist(), "matrix_row_major": self.matrix().tolist(),
                "target_slot": self.target_slot}


def camera_intrinsics(resolution, hfov_deg: float) -> dict:
    W, H = (int(x) for x in resolution)
    hfov = np.deg2rad(hfov_deg)
    f_mm = HORIZONTAL_APERTURE_MM / (2 * np.tan(hfov / 2))
    fx = (W / 2) / np.tan(hfov / 2)
    return {"resolution": [W, H], "hfov_deg": float(hfov_deg), "focal_length_mm": float(f_mm),
            "horizontal_aperture_mm": HORIZONTAL_APERTURE_MM, "vertical_aperture_mm": HORIZONTAL_APERTURE_MM * H / W,
            "fx": float(fx), "fy": float(fx), "cx": W / 2, "cy": H / 2}


def freeze_physics(stage) -> int:
    """Disable rigid-body dynamics on a dataset stage (frames are posed, not simulated).

    If Replicator's step ever advances the timeline, PhysX would otherwise let items settle or fall,
    which would change locked transforms and trip the causal lock.  Returns the number of bodies frozen.
    """
    from pxr import Usd, UsdPhysics
    n = 0
    for p in Usd.PrimRange(stage.GetPseudoRoot()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            n += 1
    return n


def author_camera(stage, intr: dict, path: str = CAMERA_PATH):
    """Define the dataset camera prim (rig-matching intrinsics); its pose is set per frame."""
    from pxr import Gf, UsdGeom
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(float(intr["focal_length_mm"]))
    cam.CreateHorizontalApertureAttr(float(intr["horizontal_aperture_mm"]))
    cam.CreateVerticalApertureAttr(float(intr["vertical_aperture_mm"]))
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 30.0))
    return cam


def set_camera_pose(stage, vp: Viewpoint, path: str = CAMERA_PATH) -> None:
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    ops = [op for op in xf.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTransform]
    op = ops[0] if ops else None
    if op is None:
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp()
    op.Set(Gf.Matrix4d(*vp.matrix().ravel().tolist()))


# ---------------------------------------------------------------------------
def base_free(spec: SceneSpec, xy: np.ndarray, radius: float = ROBOT_RADIUS, clearance: float = 0.1,
              z_band=(0.02, ROBOT_HEIGHT)) -> bool:
    """Robot base at ``xy`` is inside the room, outside the sterile keep-out and clear of objects."""
    W, D, _ = spec.room
    x, y = float(xy[0]), float(xy[1])
    if not (radius + clearance < x < W - radius - clearance and radius + clearance < y < D - radius - clearance):
        return False
    if spec.in_keepout(np.array([[x, y]]))[0]:
        return False
    for o in spec.objects:
        if o.box.z_max < z_band[0] or o.box.z_min > z_band[1]:
            continue
        if o.box.distance_xy(np.array([[x, y]]))[0] < radius + clearance:
            return False
    return True


def _staff_clear(xy: np.ndarray, staff_xy: list[dict[str, np.ndarray]], min_dist: float) -> bool:
    for s in staff_xy:
        for p in s.values():
            if np.linalg.norm(np.asarray(p)[:2] - xy) < min_dist:
                return False
    return True


def _target_slots(spec: SceneSpec) -> list:
    return [s for s in spec.slots if s.kind not in ("hand", "elsewhere") and np.all(np.isfinite(s.position))]


def sample_viewpoints(specs: list[SceneSpec], staff_per_frame: list[list[dict]], rng: np.random.Generator,
                      mount_height: float = 1.45, pitch_deg: float = -25.0, p_inspect: float = 0.75,
                      max_tries: int = 400) -> list[Viewpoint]:
    """One viewpoint per frame, valid in every arm (``staff_per_frame[k]`` = per-arm staff xy at frame k)."""
    ref = specs[0]
    W, D, _ = ref.room
    slots = _target_slots(ref)
    out = []
    for staff in staff_per_frame:
        vp = None
        for _ in range(max_tries):
            target = None
            if slots and rng.random() < p_inspect:
                target = slots[int(rng.integers(0, len(slots)))]
                ang = rng.uniform(-np.pi, np.pi)
                xy = target.position[:2] + rng.uniform(0.8, 3.5) * np.array([np.cos(ang), np.sin(ang)])
            else:
                xy = rng.uniform([0.0, 0.0], [W, D])
            if not all(base_free(s, xy) for s in specs) or not _staff_clear(xy, staff, ROBOT_RADIUS + 0.35):
                continue
            if target is not None:
                d = target.position[:2] - xy
                yaw = float(np.arctan2(d[1], d[0]) + rng.normal(0, np.deg2rad(8.0)))
            else:
                yaw = float(rng.uniform(-np.pi, np.pi))
            h = mount_height + float(rng.uniform(-0.02, 0.02))
            pitch = np.deg2rad(pitch_deg + float(rng.uniform(-3.0, 3.0)))
            vp = Viewpoint(np.array([xy[0], xy[1], h]), yaw, float(pitch), target.id if target else None)
            break
        if vp is None:   # fall back to the robot's start pose (always free by construction)
            rs = ref.robot_start
            vp = Viewpoint(np.array([rs[0], rs[1], mount_height]), float(rs[2]), np.deg2rad(pitch_deg), None)
        out.append(vp)
    return out


def frame_times(eps: list, n: int, rng: np.random.Generator, event_frac: float = 0.5,
                t_margin: float = 1.0, t_max: float | None = None) -> np.ndarray:
    """Sorted frame times shared by all arms; ``event_frac`` of them just after hidden-cause truth moves.

    ``t_max`` restricts frames to the start of the episode (the episode itself is unchanged).
    """
    T = min(float(ep.workflow.duration) for ep in eps)
    if t_max is not None:
        T = min(T, float(t_max))
    events = sorted({float(m.t) for ep in eps for m in ep.workflow.truth if m.cause != "workflow" and m.t < T - 6.0})
    n_ev = int(round(event_frac * n)) if events else 0
    t_ev = [events[int(rng.integers(0, len(events)))] + float(rng.uniform(0.5, 6.0)) for _ in range(n_ev)]
    t_uni = rng.uniform(t_margin, max(t_margin, T - t_margin), n - n_ev).tolist()
    return np.sort(np.clip(np.array(t_ev + t_uni, float), 0.0, T - 1e-3))


# ---------------------------------------------------------------------------
@dataclass
class CausalState:
    t: float
    slots: dict[str, str]
    item_support: dict[str, np.ndarray]
    staff_xy: dict[str, np.ndarray]
    staff_heading: dict[str, float] = field(default_factory=dict)


def causal_states(ep, times: np.ndarray, dt: float | None = None) -> list[CausalState]:
    """Ground-truth state of one arm at each (sorted) frame time; staff = robot-free shadow twin."""
    dt = float(dt or ep.cfg.get("episode", {}).get("dt", 0.1))
    pop = StaffPopulation(ep.spec, ep.workflow.staff_tasks, ep.streams, False, ep.cfg.get("agents"))
    custody = CustodyTimeline(ep.workflow)
    offsets = item_offsets(ep)
    t = 0.0
    out = []
    for tf in np.sort(np.asarray(times, float)):
        while t < tf - 1e-9:
            pop.step(t, dt, None, None)
            t += dt
        custody.advance(float(tf))
        staff = staff_positions(pop)
        support = {i.id: item_position(ep.spec, i.id, custody.slots[i.id], offsets, staff) for i in ep.spec.items}
        out.append(CausalState(float(tf), dict(custody.slots), support, staff,
                               {a.spec.name: a.heading for a in pop.agents}))
    return out


def apply_causal_state(stage, spec: SceneSpec, state: CausalState) -> None:
    """Move item and staff prims to the truth state (call inside ``CausalLock.causal_edit()``)."""
    from pxr import Gf, UsdGeom

    from medortrace.usd.scene_builder import safe

    def translate_op(prim):
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                return op
        raise KeyError(f"{prim.GetPath()} has no translate op")

    for it in spec.items:
        prim = stage.GetPrimAtPath(f"/World/Items/{safe(it.id)}")
        sup = state.item_support[it.id]
        translate_op(prim).Set(Gf.Vec3d(*map(float, item_prim_center(spec, it.id, sup))))
        im = UsdGeom.Imageable(prim)
        if np.all(np.isfinite(sup)):
            im.MakeVisible()
        else:
            im.MakeInvisible()
    for name, xy in state.staff_xy.items():
        prim = stage.GetPrimAtPath(f"/World/Staff/{safe(name)}")
        if prim.IsValid():
            translate_op(prim).Set(Gf.Vec3d(float(xy[0]), float(xy[1]), 0.875))


# ---------------------------------------------------------------------------
def frame_labels(entry, ep, unit: Unit, frame: int, state: CausalState, vp: Viewpoint, intr: dict,
                 rnd_info: dict) -> dict:
    spec = ep.spec
    return {
        "schema": "medortrace.synthetic_frame/1",
        "frame_uid": f"{entry.scenario_id}/{frame:06d}",
        "pair_frame_key": f"{unit.key}/{frame:06d}",
        "scenario_id": entry.scenario_id, "family": entry.family, "seed": int(entry.seed),
        "pair_id": entry.pair_id, "split": entry.split, "cfg_hash": entry.cfg_hash,
        "hidden_factor": str(spec.hidden_cause.get("factor", "none")),
        "hidden_value": str(spec.hidden_cause.get("value", "none")),
        "hidden_cause": dict(spec.hidden_cause),
        "t": float(state.t),
        "items": {i.id: {"cls": i.cls, "slot": state.slots[i.id],
                         "slot_kind": spec.slot(state.slots[i.id]).kind if state.slots[i.id] in spec.slot_ids()
                         else state.slots[i.id],
                         "position": state.item_support[i.id].tolist(),
                         "in_room": bool(np.all(np.isfinite(state.item_support[i.id])))} for i in spec.items},
        "staff": {n: [float(p[0]), float(p[1])] for n, p in state.staff_xy.items()},
        "faults": {**ep.faults.active(float(state.t)), "specular_gain": float(ep.faults.specular_gain),
                   "floor_wet": bool(ep.faults.floor_wet)},
        "causal_signature": rnd_info.get("causal_signature"),
        "nuisance": {k: rnd_info.get(k) for k in ("nuisance_seed", "nuisance_frame", "nuisance_signature",
                                                  "randomizers", "fog")},
        "camera": {**vp.as_dict(), **intr, "prim": CAMERA_PATH},
    }


def build_index(records: list[dict], meta: dict | None = None) -> dict:
    """Dataset index with pair-level splits; raises if any matched frame pair straddles splits."""
    pairs: dict[str, list[dict]] = {}
    for r in records:
        pairs.setdefault(r["pair_frame_key"], []).append(r)
    leaks = [k for k, rs in pairs.items() if len({r["split"] for r in rs}) > 1]
    if leaks:
        raise ValueError(f"split leakage across matched frames: {leaks[:5]}")
    mismatched = [k for k, rs in pairs.items() if len(rs) > 1 and len({r["nuisance_signature"] for r in rs}) > 1]
    splits: dict[str, int] = {}
    for r in records:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    return {
        "schema": "medortrace.synthetic_index/1", "meta": meta or {},
        "n_frames": len(records), "splits": splits,
        "pairs": {k: [r["frame_uid"] for r in rs] for k, rs in pairs.items() if len(rs) > 1},
        "nuisance_mismatched_pairs": mismatched,
        "frames": records,
    }
