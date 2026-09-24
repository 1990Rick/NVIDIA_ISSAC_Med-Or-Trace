"""Isaac Sim sensor adapters -> MED-OR-TRACE messages.

Each adapter creates the Isaac sensor on the rig frame authored by
``medortrace.usd.robot_rig`` and converts its output to the backend-neutral
message types in ``medortrace.common.msgs``.

=================  =============================================  ======================================
Modality           Isaac Sim source                               Notes
=================  =============================================  ======================================
Lidar              RTX lidar (``IsaacSensorCreateRtxLidar``),     custom profile from
                   ``...CreateRTXLidarScanBuffer`` annotator      configs/sensors/rtx_lidar_or16.json
Camera             RTX render product + Replicator annotators     RGB, semantic/instance seg., depth,
                   (rgb, semantic_segmentation, bbox2d, depth)    2D boxes -> detector adapter
Radar              RTX radar (``IsaacSensorCreateRtxRadar``) +    Doppler point cloud
                   radar point-cloud annotator
Acoustic           PhysX scene-query echo model (an RTX acoustic  same AcousticEcho contract
                   prim is created when the experimental
                   extension exists, for visualisation only)
IMU / contact      ``isaacsim.sensors.physics`` IMUSensor,        ground-truth supervision
                   ContactSensor; joint efforts from articulation
Landmarks          AprilTag surrogate: GT tag poses + PhysX LOS   replace with a real tag detector on
                                                                  RGB for sim-to-real studies
=================  =============================================  ======================================

Contract details that keep the two backends interchangeable:

* **Lidar** returns are re-binned onto a fixed (ring x azimuth) ray grid
  (:func:`grid_scan`): ``directions``/``ranges`` cover *every* ray with
  ``inf`` for no return, exactly like the lite scan, so the front end's
  free-space carving works unchanged.  Rings are the profile's emitter
  elevations; the azimuth resolution is ``cfg.sensors.lidar.az_res_deg``.
* **Camera** bearings/elevations are exact pinhole rays rotated by the mast
  pitch (:func:`pixel_rays`), and ranges are Euclidean (depth-to-image-plane
  x ray norm), matching ``simulate_camera``.  The ``gt_surrogate`` detector
  (:func:`gt_surrogate_detections`) applies the lite detection/confusion/tag
  model to Replicator's tight 2D boxes (structured arrays).
* **Radar** detections are re-expressed about the base centre at the radar
  mount height (the front end's convention).
* **LOS rays** (landmarks, acoustic) start outside the robot's own colliders;
  a ray origin inside the mast would otherwise report a zero-distance hit.

Annotator and command names differ across Isaac Sim releases; each adapter
tries the known names in order and reports which one it used
(``adapter.backend_info``).  ``scripts/isaac/validate_sensor_configs.py``
prints the resolved configuration.  Everything above the adapter classes is
numpy/pxr only and unit-testable outside Isaac Sim.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medortrace.common.config import CONFIG_DIR, load_yaml
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
)
from medortrace.isaac.compat import contact_sensor_cls, imu_sensor_cls
from medortrace.sim.sensors_lite import CLASSES, SIMILARITY

LIDAR_COMMANDS = ["IsaacSensorCreateRtxLidar"]
RADAR_COMMANDS = ["IsaacSensorCreateRtxRadar"]
ACOUSTIC_COMMANDS = ["IsaacSensorCreateRtxAcoustic", "IsaacSensorCreateAcoustic"]
LIDAR_ANNOTATORS = ["IsaacCreateRTXLidarScanBuffer", "RtxSensorCpuIsaacCreateRTXLidarScanBuffer"]
RADAR_ANNOTATORS = ["IsaacComputeRTXRadarPointCloud", "RtxSensorCpuIsaacComputeRTXRadarPointCloud"]
ROBOT_PRIM = "/World/Robot"

# RTX non-visual material vocabulary (Omniverse SimReady non-visual material spec as documented for
# Isaac Sim 4.5/5.x RTX sensors).  Unknown tokens make the material fall back to a default return model.
NONVISUAL_BASE = (
    "none", "aluminum", "steel", "oxidized_steel", "iron", "oxidized_iron", "silver", "brass", "bronze",
    "oxidized_bronze_patina", "tin", "plastic", "fiberglass", "carbon_fiber", "vinyl", "plexiglass", "pvc", "nylon",
    "polyester", "clear_glass", "frosted_glass", "one_way_mirror", "mirror", "ceramic_glass", "asphalt", "concrete",
    "leaf_grass", "dead_leaf_grass", "rubber", "wood", "bark", "cardboard", "paper", "fabric", "skin", "fur_hair",
    "leather", "marble", "brick", "stone", "gravel", "dirt", "mud", "water", "salt_water", "snow", "ice",
    "calibration_lambertian",
)
NONVISUAL_COATING = ("none", "paint", "clearcoat", "paint_clearcoat")
NONVISUAL_ATTRIBUTES = ("none", "emissive", "retroreflective", "single_sided", "visually_transparent")
NONVISUAL_VOCAB = {"base": NONVISUAL_BASE, "coating": NONVISUAL_COATING, "attributes": NONVISUAL_ATTRIBUTES}
# project tokens (medortrace.world.materials) that are not in the vocabulary -> closest vocabulary token
NONVISUAL_ALIASES = {"base": {"steel_stainless": "steel", "stainless_steel": "steel", "glass": "clear_glass",
                              "acrylic": "plexiglass", "hdpe": "plastic"},
                     "coating": {}, "attributes": {}}


# ---------------------------------------------------------------------------
# pure helpers (numpy / pxr only)
# ---------------------------------------------------------------------------
def nonvisual_report(stage, fix: bool = False) -> list[dict]:
    """Check (and optionally repair via aliases) the RTX non-visual tokens on every material.

    Returns one row per material: ``{"material", "base", "coating", "attributes", "issues", "fixed"}``.
    """
    from pxr import Sdf, Usd, UsdShade
    rows = []
    for p in Usd.PrimRange(stage.GetPseudoRoot()):
        if not p.IsA(UsdShade.Material):
            continue
        row = {"material": str(p.GetPath()), "issues": [], "fixed": []}
        for key, vocab in NONVISUAL_VOCAB.items():
            a = p.GetAttribute(f"omni:simready:nonvisual:{key}")
            val = a.Get() if a and a.IsValid() else None
            row[key] = val
            if val is None:
                row["issues"].append(f"missing omni:simready:nonvisual:{key}")
                if fix:
                    p.CreateAttribute(f"omni:simready:nonvisual:{key}", Sdf.ValueTypeNames.Token).Set("none")
                    row["fixed"].append(f"{key}: <missing> -> none")
            elif str(val) not in vocab:
                alias = NONVISUAL_ALIASES[key].get(str(val))
                row["issues"].append(f"{key}={val!r} not in RTX vocabulary" + (f" (alias {alias!r})" if alias else ""))
                if fix and alias:
                    a.Set(alias)
                    row["fixed"].append(f"{key}: {val} -> {alias}")
        rows.append(row)
    return rows


def normalize_nonvisual_tokens(stage) -> list[dict]:
    """Rewrite non-vocabulary tokens to their aliases in the opened (in-memory) stage."""
    return [r for r in nonvisual_report(stage, fix=True) if r["fixed"]]


def lidar_elevations(profile_path: str | Path) -> np.ndarray:
    """Emitter elevations (deg, sorted) of an RTX lidar JSON profile."""
    prof = json.loads(Path(profile_path).read_text())["profile"]
    st = prof.get("emitterStates", [{}])[0]
    el = np.asarray(st.get("elevationDeg", []), float)
    if not len(el):
        n = int(prof.get("numberOfEmitters", 16))
        el = np.linspace(prof.get("downElevationDeg", -15.0), prof.get("upElevationDeg", 15.0), n)
    return np.sort(el)


def grid_scan(points: np.ndarray, intensity: np.ndarray | None, elev_deg: np.ndarray, az_res_deg: float = 2.0,
              max_range: float = 30.0, object_ids: np.ndarray | None = None):
    """Bin sensor-frame returns onto a fixed (ring x azimuth) grid, nearest return per cell.

    Returns ``(points (N,3), intensity (N,), ring (N,), directions (R,3), ranges (R,), obj (R,))``
    with ``R = len(elev_deg) * round(360 / az_res_deg)`` and ``ranges = inf`` for empty cells
    (same layout as ``simulate_lidar``: points/intensity/ring are the finite rays in ray order).
    """
    elev_deg = np.sort(np.asarray(elev_deg, float))
    n_ring = len(elev_deg)
    n_az = int(round(360.0 / az_res_deg))
    R = n_ring * n_az
    ring_c = np.repeat(np.arange(n_ring), n_az)
    az_c = np.deg2rad((np.tile(np.arange(n_az), n_ring) + 0.5) * az_res_deg)
    el_c = np.deg2rad(elev_deg[ring_c])
    dirs = np.stack([np.cos(el_c) * np.cos(az_c), np.cos(el_c) * np.sin(az_c), np.sin(el_c)], axis=1)
    ranges = np.full(R, np.inf)
    inten = np.zeros(R)
    obj = np.full(R, -1, dtype=np.int64)
    pts = np.asarray(points, float).reshape(-1, 3)
    if len(pts):
        dist = np.linalg.norm(pts, axis=1)
        ok = np.isfinite(dist) & (dist > 1e-3) & (dist <= max_range)
        idx_ok = np.nonzero(ok)[0]
        p, d = pts[ok], dist[ok]
        el = np.degrees(np.arcsin(np.clip(p[:, 2] / d, -1.0, 1.0)))
        az = np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0
        ring = np.argmin(np.abs(el[:, None] - elev_deg[None, :]), axis=1)
        cell = ring * n_az + (np.floor(az / az_res_deg).astype(int) % n_az)
        order = np.lexsort((d, cell))
        first = order[np.r_[True, cell[order][1:] != cell[order][:-1]]] if len(order) else order
        c = cell[first]
        ranges[c] = d[first]
        dirs[c] = p[first] / d[first, None]
        if intensity is not None and len(intensity) == len(pts):
            iv = np.asarray(intensity, float)[idx_ok][first]
            inten[c] = iv / max(1.0, float(np.nanmax(iv))) if len(iv) else iv
        else:
            inten[c] = 1.0
        if object_ids is not None and len(object_ids) == len(pts):
            obj[c] = np.asarray(object_ids).astype(np.int64)[idx_ok][first]
    fin = np.isfinite(ranges)
    return dirs[fin] * ranges[fin, None], inten[fin], ring_c[fin], dirs, ranges, obj


def pixel_rays(u, v, width: int, height: int, hfov: float, pitch: float):
    """Pixel -> (bearing, elevation, ray_norm) in the level robot frame.

    The camera looks along +X of the link pitched by ``pitch`` (negative = down); image ``u`` grows to
    the right, ``v`` downwards.  ``ray_norm = sqrt(1 + x^2 + y^2)`` converts depth-to-image-plane into
    Euclidean range.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    f = (width / 2.0) / np.tan(hfov / 2.0)
    x = (u - width / 2.0) / f
    y = (v - height / 2.0) / f
    cp, sp = np.cos(pitch), np.sin(pitch)
    # d = forward - x * left - y * up, with forward=(cp,0,sp), left=(0,1,0), up=(-sp,0,cp)
    dx = cp + y * sp
    dy = -x
    dz = sp - y * cp
    bearing = np.arctan2(dy, dx)
    elev = np.arctan2(dz, np.hypot(dx, dy))
    return bearing, elev, np.sqrt(1.0 + x * x + y * y)


def _records(data) -> list[dict]:
    """Replicator structured array / list of dicts -> list of plain dicts."""
    if data is None:
        return []
    if isinstance(data, np.ndarray):
        if not data.dtype.names:
            return []
        arr = np.atleast_1d(data)
        return [{n: (r[n].item() if hasattr(r[n], "item") else r[n]) for n in arr.dtype.names} for r in arr]
    return [dict(r) for r in data]


def _label_class(labels: dict, sid) -> str | None:
    lab = None
    for key in (str(sid), sid):
        if key in labels:
            lab = labels[key]
            break
    if lab is None:
        try:
            lab = labels.get(int(sid))
        except (TypeError, ValueError):
            lab = None
    if isinstance(lab, dict):
        lab = lab.get("class")
    if not lab:
        return None
    for c in str(lab).split(","):
        if c.strip() in CLASSES:
            return c.strip()
    return None


def _window_depth(depth: np.ndarray, u: float, v: float, half: int = 2) -> float:
    if depth is None or depth.ndim < 2:
        return float("nan")
    H, W = depth.shape[:2]
    iu, iv = int(np.clip(u, 0, W - 1)), int(np.clip(v, 0, H - 1))
    win = np.asarray(depth[max(0, iv - half):iv + half + 1, max(0, iu - half):iu + half + 1], float)
    win = win[np.isfinite(win) & (win > 0)]
    return float(np.median(win)) if len(win) else float("nan")


def gt_surrogate_detections(boxes, depth: np.ndarray, width: int, height: int, hfov: float, pitch: float,
                            rng: np.random.Generator, items: dict[str, dict] | None = None,
                            cam_cfg: dict | None = None, glare_gain: float = 1.0,
                            haze: float = 0.0) -> list[CameraDetection]:
    """Lite camera model applied to Replicator ``bounding_box_2d_tight`` output.

    ``boxes`` is the annotator's ``{"data": structured array, "info": {"idToLabels", "primPaths", ...}}``;
    ``items`` maps item id -> ``{"cls", "size", "glare", "tag_readable", "prim"}`` (prim path).
    Visibility is ``1 - occlusionRatio`` (true RTX occlusion), everything else follows ``simulate_camera``.
    """
    cc = {"max_range": 5.0, "pd0": 0.92, "fp_rate": 0.05, "tag_range": 1.6, "logit_scale": 8.0}
    cc.update({k: v for k, v in (cam_cfg or {}).items() if k in cc})
    items = items or {}
    prim_to_item = {str(v.get("prim")): k for k, v in items.items() if v.get("prim")}
    info = boxes.get("info", {}) if isinstance(boxes, dict) else {}
    labels = info.get("idToLabels", {}) or {}
    paths = list(info.get("primPaths", []) or [])
    recs = _records(boxes.get("data") if isinstance(boxes, dict) else boxes)
    dets: list[CameraDetection] = []
    for k, bx in enumerate(recs):
        cls = _label_class(labels, bx.get("semanticId", -1))
        iid = prim_to_item.get(str(paths[k])) if k < len(paths) else None
        if iid is not None:
            cls = items[iid].get("cls", cls)
        if cls not in CLASSES:
            continue
        u = 0.5 * (float(bx["x_min"]) + float(bx["x_max"]))
        v = 0.5 * (float(bx["y_min"]) + float(bx["y_max"]))
        occl = float(bx.get("occlusionRatio", 0.0))
        vis = float(np.clip(1.0 - (occl if np.isfinite(occl) and occl >= 0 else 0.0), 0.0, 1.0))
        z = _window_depth(depth, u, v)
        b, e, norm = pixel_rays(u, v, width, height, hfov, pitch)
        rng3 = float(z * norm)
        if not np.isfinite(rng3) or vis <= 0.0 or rng3 > cc["max_range"] or rng3 < 0.2:
            continue
        it = items.get(iid, {}) if iid else {}
        glare = float(np.clip(float(it.get("glare", 0.0)) * glare_gain * rng.uniform(0.5, 1.2), 0, 1))
        size = max(tuple(it.get("size", (0.1, 0.05, 0.02)))[:2])
        range_factor = float(np.clip(1.2 - rng3 / cc["max_range"], 0.1, 1.0)) * float(np.clip(size / 0.08, 0.5, 1.0))
        pd = cc["pd0"] * vis * range_factor * (1 - 0.45 * glare) * (1 - haze)
        if rng.random() > pd:
            continue
        ci = CLASSES.index(cls)
        margin = cc["logit_scale"] * vis * range_factor * (1 - 0.6 * glare)
        logits = margin * SIMILARITY[ci] + rng.normal(0, 0.8, len(CLASSES))
        hint = None
        if iid and it.get("tag_readable", True) and rng3 < cc["tag_range"] and vis > 0.7 and glare < 0.6 \
                and rng.random() < 0.85:
            hint = iid
        dets.append(CameraDetection(CLASSES[int(np.argmax(logits))], hint, float(b) + float(rng.normal(0, 0.01)),
                                    float(e) + float(rng.normal(0, 0.01)), rng3 * (1 + float(rng.normal(0, 0.02))),
                                    logits, vis, glare, iid))
    # false positives (reflections, glove/gauze look-alikes), as in the lite model
    for _ in range(rng.poisson(cc["fp_rate"] * (1 + 2 * haze + 0.5 * (glare_gain - 1)))):
        b = float(rng.uniform(-1, 1) * hfov / 2)
        r = float(rng.uniform(0.8, cc["max_range"] * 0.8))
        e = float(pitch + rng.uniform(-0.3, 0.3))
        ci = int(rng.integers(0, len(CLASSES)))
        logits = 1.5 * SIMILARITY[ci] + rng.normal(0, 0.8, len(CLASSES))
        dets.append(CameraDetection(CLASSES[int(np.argmax(logits))], None, b, e, r, logits, 0.5, 0.5, None))
    return dets


def _field(d: dict, *names):
    """First present field among names, searched at top level then in ``info``."""
    info = d.get("info", {}) if isinstance(d.get("info"), dict) else {}
    for src in (d, info):
        for n in names:
            if n in src and src[n] is not None:
                return src[n]
    return None


def spherical_points(d: dict, range_keys=("distance", "distances", "range", "radialDistance"),
                     az_keys=("azimuth", "azimuths"), el_keys=("elevation", "elevations")) -> np.ndarray | None:
    """Sensor-frame points from per-return (range, azimuth, elevation) fields, if the annotator provides them.

    These fields are sensor-frame by definition, whereas the ``data`` point cloud may be world-frame when
    the release ignores ``transformPoints=False``.  Angles in degrees are detected (|az| > 2 pi or
    |el| > pi/2) and converted.
    """
    r, az, el = _field(d, *range_keys), _field(d, *az_keys), _field(d, *el_keys)
    if r is None or az is None or el is None:
        return None
    r, az, el = (np.asarray(x, float).reshape(-1) for x in (r, az, el))
    if not (len(r) == len(az) == len(el)) or not len(r):
        return None
    if np.nanmax(np.abs(az)) > 2 * np.pi + 0.1 or np.nanmax(np.abs(el)) > np.pi / 2 + 0.1:
        az, el = np.deg2rad(az), np.deg2rad(el)
    return np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], axis=1)


def radar_detections(points: np.ndarray, radial_velocity: np.ndarray | None, rcs: np.ndarray | None,
                     mount_offset=(0.25, 0.0, 0.0)) -> list[RadarDetection]:
    """Radar-frame points -> detections about the base centre at the radar height (front-end convention)."""
    pts = np.asarray(points, float).reshape(-1, 3) + np.array([mount_offset[0], mount_offset[1], 0.0])
    n = len(pts)
    vr = np.zeros(n) if radial_velocity is None else np.asarray(radial_velocity, float).reshape(-1)
    rc = np.zeros(n) if rcs is None else np.asarray(rcs, float).reshape(-1)
    r = np.linalg.norm(pts, axis=1)
    out = []
    for k in np.nonzero(r > 1e-3)[0]:
        out.append(RadarDetection(float(r[k]), float(np.arctan2(pts[k, 1], pts[k, 0])),
                                  float(np.arcsin(np.clip(pts[k, 2] / r[k], -1, 1))),
                                  float(vr[k]) if k < len(vr) else 0.0, float(rc[k]) if k < len(rc) else 0.0))
    return out


def los_blocked(raycast, origin: np.ndarray, target: np.ndarray, start: float = 0.35, end_margin: float = 0.1,
                self_prefix: str = ROBOT_PRIM) -> bool:
    """Line-of-sight test that starts ``start`` m out of the robot and ignores the robot's own colliders.

    ``raycast(origin, dir, distance) -> dict`` is PhysX ``raycast_closest``.
    """
    d = np.asarray(target, float) - np.asarray(origin, float)
    L = float(np.linalg.norm(d))
    if L <= start + end_margin:
        return False
    u = d / L
    hit = raycast(tuple(map(float, np.asarray(origin, float) + u * start)), tuple(map(float, u)),
                  L - start - end_margin)
    if not hit or not hit.get("hit", False):
        return False
    body = str(hit.get("rigidBody") or hit.get("collision") or "")
    return not body.startswith(self_prefix)


# ---------------------------------------------------------------------------
# Isaac Sim adapters (omni imports are lazy)
# ---------------------------------------------------------------------------
def _register_profile_folder() -> None:
    """Make Isaac's RTX sensor profile search path include configs/sensors."""
    import carb
    s = carb.settings.get_settings()
    folder = str(CONFIG_DIR / "sensors") + "/"
    for key in ("/app/sensors/nv/lidar/profileBaseFolder", "/app/sensors/nv/radar/profileBaseFolder"):
        cur = s.get(key) or []
        cur = list(cur) if isinstance(cur, (list, tuple)) else [cur]
        if folder not in cur:
            s.set(key, cur + [folder])


def _get_annotator(names: list[str], init_params: dict | None = None):
    import omni.replicator.core as rep
    last = None
    for n in names:
        try:
            if init_params:
                return rep.AnnotatorRegistry.get_annotator(n, init_params=init_params), n
            return rep.AnnotatorRegistry.get_annotator(n), n
        except Exception as e:  # pragma: no cover
            last = e
    raise RuntimeError(f"no annotator among {names}: {last}")


def _create_rtx_sensor(commands: list[str], path: str, parent: str, profile: str):
    import omni.kit.commands
    from pxr import Gf
    last = None
    for cmd in commands:
        try:
            ok, prim = omni.kit.commands.execute(cmd, path=path, parent=parent, config=profile,
                                                 translation=Gf.Vec3d(0, 0, 0), orientation=Gf.Quatd(1, 0, 0, 0))
            if prim is not None:
                return prim, cmd
        except Exception as e:  # pragma: no cover
            last = e
    raise RuntimeError(f"could not create RTX sensor with {commands} (profile {profile}): {last}")


def render_product_path(rp) -> str:
    return str(getattr(rp, "path", rp))


class RtxLidarAdapter:
    def __init__(self, parent: str, profile: str = "rtx_lidar_or16", mount_height: float = 0.9,
                 az_res_deg: float = 2.0, max_range: float = 20.0):
        import omni.replicator.core as rep
        _register_profile_folder()
        self.prim, cmd = _create_rtx_sensor(LIDAR_COMMANDS, "rtx_lidar", parent, profile)
        self.rp = rep.create.render_product(str(self.prim.GetPath()), [1, 1], name="medortrace_lidar")
        self.ann, self.ann_name = _get_annotator(LIDAR_ANNOTATORS)
        try:
            self.ann.initialize(outputDistance=True, outputIntensity=True, outputObjectId=True,
                                outputAzimuth=True, outputElevation=True, transformPoints=False)
        except Exception:
            pass
        self.ann.attach([self.rp])
        self.mount_height = mount_height
        self.elev = lidar_elevations(CONFIG_DIR / "sensors" / f"{profile}.json")
        self.az_res = float(az_res_deg)
        self.max_range = float(max_range)
        self.backend_info = {"prim": str(self.prim.GetPath()), "command": cmd, "annotator": self.ann_name,
                             "profile": profile, "render_product": render_product_path(self.rp),
                             "grid": [len(self.elev), int(round(360 / self.az_res))]}
        self._seq = 0

    def read(self, t: float, stamp_fn) -> LidarScan | None:
        d = self.ann.get_data() or {}
        pts = spherical_points(d)
        if pts is None:
            raw = _field(d, "data", "points")
            pts = np.asarray(raw if raw is not None else np.zeros((0, 3)), dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return None
        inten = _field(d, "intensity", "intensities")
        obj = _field(d, "objectId", "objectIds")
        p, i, ring, dirs, ranges, oid = grid_scan(pts, None if inten is None else np.asarray(inten).reshape(-1),
                                                  self.elev, self.az_res, self.max_range,
                                                  None if obj is None else np.asarray(obj).reshape(-1))
        self._seq += 1
        return LidarScan(Header(stamp_fn("lidar", t), t, "lidar_link", self._seq), p, i, ring, dirs, ranges,
                         sensor_height=self.mount_height, gt_is_ghost=None,
                         gt_object_id=oid if obj is not None else None)


class RtxRadarAdapter:
    def __init__(self, parent: str, profile: str = "rtx_radar_or77", mount_offset=(0.25, 0.0, 0.0)):
        import omni.replicator.core as rep
        _register_profile_folder()
        self.prim, cmd = _create_rtx_sensor(RADAR_COMMANDS, "rtx_radar", parent, profile)
        self.rp = rep.create.render_product(str(self.prim.GetPath()), [1, 1], name="medortrace_radar")
        self.ann, self.ann_name = _get_annotator(RADAR_ANNOTATORS)
        try:
            self.ann.initialize(transformPoints=False)
        except Exception:
            pass
        self.ann.attach([self.rp])
        self.mount_offset = tuple(mount_offset)
        self.backend_info = {"prim": str(self.prim.GetPath()), "command": cmd, "annotator": self.ann_name,
                             "profile": profile, "render_product": render_product_path(self.rp)}
        self._seq = 0

    def read(self, t: float, stamp_fn) -> RadarFrame | None:
        d = self.ann.get_data() or {}
        pts = spherical_points(d, ("radialDistance", "radialDistances", "range", "distance"))
        if pts is None:
            raw = _field(d, "data", "points")
            pts = np.asarray(raw if raw is not None else np.zeros((0, 3)), dtype=float).reshape(-1, 3)
        vr = _field(d, "radialVelocities", "radialVelocity", "velocities")
        rcs = _field(d, "rcs", "rcss", "rcsDbsm")
        self._seq += 1
        return RadarFrame(Header(stamp_fn("radar", t), t, "radar_link", self._seq),
                          radar_detections(pts, vr, rcs, self.mount_offset))


class CameraAdapter:
    """RTX camera with Replicator annotators feeding a detector.

    ``detector`` modes:
      * ``"model:<path>"``  - a trained detector (``medortrace.isaac.detector``) run on RGB;
      * ``"gt_surrogate"``  - ground-truth tight 2D boxes + depth passed through the same
        calibrated confusion/visibility/tag model as the lite simulator (used to
        isolate planning/belief effects from detector quality).
    """

    def __init__(self, cam_prim: str, cfg: dict | None = None, detector: str = "gt_surrogate",
                 items: dict[str, dict] | None = None, rng: np.random.Generator | None = None,
                 sensor_cfg: dict | None = None, nuisance: dict | None = None):
        import omni.replicator.core as rep
        cfg = cfg or load_yaml(CONFIG_DIR / "sensors" / "rtx_camera.yaml")
        self.cfg = cfg
        self.sensor_cfg = sensor_cfg or {}
        self.rp = rep.create.render_product(cam_prim, tuple(cfg["resolution"]), name="medortrace_rgb")
        self.ann = {}
        init = {"semantic_segmentation": {"colorize": False}, "instance_id_segmentation_fast": {"colorize": False}}
        for a in cfg["annotators"]:
            try:
                an, _ = _get_annotator([a], init.get(a))
                an.attach([self.rp])
                self.ann[a] = an
            except Exception as e:  # pragma: no cover
                print(f"[medortrace] camera annotator {a} unavailable: {e}")
        self.detector = detector
        self.model = None
        if detector.startswith("model:"):
            from medortrace.isaac.detector import load_detector
            self.model = load_detector(detector.split(":", 1)[1])
        self.items = items or {}
        self.rng = rng or np.random.default_rng(0)
        self.hfov = np.deg2rad(float(cfg["hfov_deg"]))
        self.pitch = np.deg2rad(float(self.sensor_cfg.get("pitch_deg", cfg.get("pitch_deg", -25.0))))
        self.max_range = float(self.sensor_cfg.get("max_range", 5.0))
        self.W, self.H = (int(x) for x in cfg["resolution"])
        self.nuisance = nuisance or {}
        self.backend_info = {"render_product": render_product_path(self.rp), "annotators": sorted(self.ann),
                             "detector": detector, "resolution": [self.W, self.H]}
        self._seq = 0

    def read(self, t: float, stamp_fn, pitch_rad: float | None = None) -> CameraFrame | None:
        if "distance_to_image_plane" not in self.ann:
            return None
        pitch = self.pitch if pitch_rad is None else float(pitch_rad)
        depth = np.asarray(self.ann["distance_to_image_plane"].get_data(), dtype=float)
        dets: list[CameraDetection] = []
        if self.model is not None:
            if "rgb" not in self.ann:
                return None
            rgb = np.asarray(self.ann["rgb"].get_data())[..., :3]
            for (u, v, logits, frac) in self.model.predict(rgb):
                b, e, norm = pixel_rays(u, v, self.W, self.H, self.hfov, pitch)
                r = _window_depth(depth, u, v) * float(norm)
                if not np.isfinite(r) or r > self.max_range:
                    continue
                dets.append(CameraDetection(CLASSES[int(np.argmax(logits))], None, float(b), float(e), r,
                                            np.asarray(logits, float), float(frac), 0.0))
        else:
            if "bounding_box_2d_tight" not in self.ann:
                return None
            dets = gt_surrogate_detections(self.ann["bounding_box_2d_tight"].get_data(), depth, self.W, self.H,
                                           self.hfov, pitch, self.rng, self.items, self.sensor_cfg,
                                           float(self.nuisance.get("glare_gain", 1.0)),
                                           float(self.nuisance.get("haze", 0.0)))
        self._seq += 1
        return CameraFrame(Header(stamp_fn("camera", t), t, "camera_link", self._seq), dets, self.hfov,
                           self.max_range)


class AcousticAdapter:
    """Active acoustic probe: PhysX line-of-sight + the calibrated forward model of the lite simulator.

    If the experimental RTX acoustic extension provides a creation command, the sensor prim is created
    (for visualisation / future use) but measurements still come from the forward model, because the
    RTX acoustic output contract is not stable across releases.
    """

    def __init__(self, parent: str, cfg: dict | None = None):
        self.cfg = cfg or load_yaml(CONFIG_DIR / "sensors" / "rtx_acoustic.yaml")
        self.native = None
        native_cmd = None
        try:
            import omni.kit.commands
            for cmd in ACOUSTIC_COMMANDS:
                try:
                    ok, prim = omni.kit.commands.execute(cmd, path="rtx_acoustic", parent=parent)
                    if prim is not None:
                        self.native, native_cmd = prim, cmd
                        break
                except Exception:
                    continue
        except ImportError:  # pragma: no cover
            pass
        self.backend_info = {"mode": "physx_echo_model", "native_prim_command": native_cmd}
        self._seq = 0

    def read(self, t: float, stamp_fn, origin: np.ndarray, region: str, region_pos: np.ndarray,
             base_reflectivity: float, contents_reflectivity: list[float], rng) -> AcousticFrame | None:
        from omni.physx import get_physx_scene_query_interface
        d = np.asarray(region_pos, float) - np.asarray(origin, float)
        r = float(np.linalg.norm(d))
        if r > self.cfg["max_range_m"]:
            return None
        occluded = los_blocked(get_physx_scene_query_interface().raycast_closest, origin, region_pos,
                               start=0.35, end_margin=0.3)
        e = 0.15 * base_reflectivity + sum(0.6 * c for c in contents_reflectivity)
        e *= (1.0 / (1.0 + 0.3 * r * r)) * (self.cfg["diffraction_attenuation"] if occluded else 1.0)
        sigma = self.cfg["noise_sigma"] * (1.8 if occluded else 1.0)
        self._seq += 1
        echo = AcousticEcho(region, float(max(0.0, e + rng.normal(0, sigma))), 2 * r / self.cfg["speed_of_sound_mps"],
                            occluded, gt_hard_reflector=any(c > 0.8 for c in contents_reflectivity))
        return AcousticFrame(Header(stamp_fn("acoustic", t), t, "acoustic_link", self._seq), [echo])


class ImuAdapter:
    def __init__(self, prim_path: str, rate_hz: float = 100.0, filter_size: int = 4):
        IMUSensor = imu_sensor_cls()
        self.s = IMUSensor(prim_path=prim_path + "/imu", name="medortrace_imu", frequency=int(rate_hz),
                           translation=np.zeros(3), linear_acceleration_filter_size=int(filter_size))
        self._seq = 0

    def initialize(self) -> None:
        if hasattr(self.s, "initialize"):
            try:
                self.s.initialize()
            except Exception:
                pass

    def read(self, t: float, stamp_fn) -> list[ImuSample]:
        f = self.s.get_current_frame() or {}
        if "lin_acc" not in f or "ang_vel" not in f:
            return []
        self._seq += 1
        return [ImuSample(Header(stamp_fn("imu", t), t, "imu_link", self._seq),
                          np.asarray(f["lin_acc"], float).reshape(3), np.asarray(f["ang_vel"], float).reshape(3))]


class ContactAdapter:
    def __init__(self, prim_path: str, radius: float = 0.3, threshold: float = 1.0):
        ContactSensor = contact_sensor_cls()
        self.s = ContactSensor(prim_path=prim_path + "/bumper_contact", name="medortrace_bumper",
                               min_threshold=threshold, max_threshold=1e7, radius=radius)
        self._seq = 0

    def initialize(self) -> None:
        if hasattr(self.s, "initialize"):
            try:
                self.s.initialize()
            except Exception:
                pass

    def frame(self) -> dict:
        try:
            return self.s.get_current_frame() or {}
        except Exception:
            return {}

    def read(self, t: float, stamp_fn, effort: np.ndarray | None = None) -> ContactState:
        f = self.frame()
        self._seq += 1
        return ContactState(Header(stamp_fn("contact", t), t, "bumper", self._seq), bool(f.get("in_contact", False)),
                            float(f.get("force", 0.0)), effort=effort)


class LandmarkAdapter:
    """Fiducial surrogate: visible tags from GT poses with PhysX line-of-sight."""

    def __init__(self, landmarks: dict[str, np.ndarray], max_range: float = 8.0):
        self.lm = landmarks
        self.max_range = max_range
        self._seq = 0

    def read(self, t, stamp_fn, cam_pos: np.ndarray, yaw: float, rng) -> LandmarkFrame:
        from omni.physx import get_physx_scene_query_interface
        cast = get_physx_scene_query_interface().raycast_closest
        obs = []
        for k, p in self.lm.items():
            d = p - cam_pos
            r = float(np.linalg.norm(d[:2]))
            if r > self.max_range:
                continue
            # wall tags: stop 8 cm short of the wall to avoid self-occlusion (as the lite model)
            if los_blocked(cast, cam_pos, p, start=0.35, end_margin=0.08) or rng.random() > 0.9:
                continue
            b = float((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
            obs.append(LandmarkObservation(k, r + float(rng.normal(0, 0.03)), b + float(rng.normal(0, 0.01))))
        self._seq += 1
        return LandmarkFrame(Header(stamp_fn("landmarks", t), t, "camera_link", self._seq), obs)


def profile_paths() -> dict[str, Path]:
    return {"lidar": CONFIG_DIR / "sensors" / "rtx_lidar_or16.json",
            "radar": CONFIG_DIR / "sensors" / "rtx_radar_or77.json"}
