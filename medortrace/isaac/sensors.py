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
Acoustic           RTX acoustic (EXPERIMENTAL) if available,      same AcousticEcho contract
                   else PhysX scene-query echo model
IMU / contact      ``isaacsim.sensors.physics`` IMUSensor,        ground-truth supervision
                   ContactSensor; joint efforts from articulation
Landmarks          AprilTag surrogate: GT tag poses + PhysX LOS   replace with a real tag detector on
                                                                  RGB for sim-to-real studies
=================  =============================================  ======================================

Annotator and command names differ slightly across Isaac Sim releases; each
adapter tries the known names in order and reports which one it used
(``adapter.backend_info``).  ``scripts/isaac/validate_sensor_configs.py``
prints the resolved configuration.
"""

from __future__ import annotations

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

LIDAR_ANNOTATORS = ["IsaacCreateRTXLidarScanBuffer", "RtxSensorCpuIsaacCreateRTXLidarScanBuffer"]
RADAR_ANNOTATORS = ["IsaacComputeRTXRadarPointCloud", "RtxSensorCpuIsaacComputeRTXRadarPointCloud"]
ACOUSTIC_COMMANDS = ["IsaacSensorCreateRtxAcoustic", "IsaacSensorCreateAcoustic"]


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


def _get_annotator(names: list[str]):
    import omni.replicator.core as rep
    last = None
    for n in names:
        try:
            return rep.AnnotatorRegistry.get_annotator(n), n
        except Exception as e:  # pragma: no cover
            last = e
    raise RuntimeError(f"no annotator among {names}: {last}")


class RtxLidarAdapter:
    def __init__(self, parent: str, profile: str = "rtx_lidar_or16", mount_height: float = 0.9):
        import omni.kit.commands
        import omni.replicator.core as rep
        from pxr import Gf
        _register_profile_folder()
        _, self.prim = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", path="rtx_lidar", parent=parent,
                                                 config=profile, translation=(0, 0, 0),
                                                 orientation=Gf.Quatd(1, 0, 0, 0))
        self.rp = rep.create.render_product(self.prim.GetPath(), [1, 1], name="medortrace_lidar")
        self.ann, self.ann_name = _get_annotator(LIDAR_ANNOTATORS)
        try:
            self.ann.initialize(outputDistance=True, outputIntensity=True, outputObjectId=True,
                                outputAzimuth=True, outputElevation=True, transformPoints=False)
        except Exception:
            pass
        self.ann.attach([self.rp])
        self.mount_height = mount_height
        self.backend_info = {"prim": str(self.prim.GetPath()), "annotator": self.ann_name, "profile": profile}
        self._seq = 0

    def read(self, t: float, stamp_fn) -> LidarScan | None:
        d = self.ann.get_data()
        pts = np.asarray(d.get("data", d.get("points", np.zeros((0, 3)))), dtype=float).reshape(-1, 3)
        if len(pts) == 0:
            return None
        dist = np.asarray(d.get("distance", np.linalg.norm(pts, axis=1)), dtype=float)
        inten = np.asarray(d.get("intensity", np.ones(len(pts))), dtype=float)
        el = np.asarray(d.get("elevation", np.arcsin(pts[:, 2] / np.maximum(dist, 1e-6))), dtype=float)
        dirs = pts / np.maximum(dist[:, None], 1e-6)
        rings = np.digitize(el, np.linspace(el.min() - 1e-6, el.max() + 1e-6, 17)) - 1
        self._seq += 1
        obj = d.get("objectId")
        return LidarScan(Header(stamp_fn("lidar", t), t, "lidar_link", self._seq), pts, inten, rings, dirs, dist,
                         sensor_height=self.mount_height, gt_is_ghost=None,
                         gt_object_id=None if obj is None else np.asarray(obj))


class RtxRadarAdapter:
    def __init__(self, parent: str, profile: str = "rtx_radar_or77"):
        import omni.kit.commands
        import omni.replicator.core as rep
        from pxr import Gf
        _register_profile_folder()
        _, self.prim = omni.kit.commands.execute("IsaacSensorCreateRtxRadar", path="rtx_radar", parent=parent,
                                                 config=profile, translation=(0, 0, 0),
                                                 orientation=Gf.Quatd(1, 0, 0, 0))
        self.rp = rep.create.render_product(self.prim.GetPath(), [1, 1], name="medortrace_radar")
        self.ann, self.ann_name = _get_annotator(RADAR_ANNOTATORS)
        self.ann.attach([self.rp])
        self.backend_info = {"prim": str(self.prim.GetPath()), "annotator": self.ann_name, "profile": profile}
        self._seq = 0

    def read(self, t: float, stamp_fn) -> RadarFrame | None:
        d = self.ann.get_data()
        pts = np.asarray(d.get("data", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
        info = d.get("info", {}) or {}
        vr = np.asarray(info.get("radialVelocities", d.get("radialVelocity", np.zeros(len(pts)))), dtype=float)
        rcs = np.asarray(info.get("rcs", d.get("rcs", np.zeros(len(pts)))), dtype=float)
        dets = []
        for k, p in enumerate(pts):
            r = float(np.linalg.norm(p))
            if r < 1e-3:
                continue
            dets.append(RadarDetection(r, float(np.arctan2(p[1], p[0])), float(np.arcsin(p[2] / r)),
                                       float(vr[k]) if k < len(vr) else 0.0, float(rcs[k]) if k < len(rcs) else 0.0))
        self._seq += 1
        return RadarFrame(Header(stamp_fn("radar", t), t, "radar_link", self._seq), dets)


class CameraAdapter:
    """RTX camera with Replicator annotators feeding a detector.

    ``detector`` modes:
      * ``"model:<path>"``  - a trained detector (scripts/train_detector.py) run on RGB;
      * ``"gt_surrogate"``  - ground-truth 2D boxes + depth passed through the same
        calibrated confusion/visibility model as the lite simulator (used to
        isolate planning/belief effects from detector quality).
    """

    def __init__(self, cam_prim: str, cfg: dict | None = None, detector: str = "gt_surrogate",
                 item_classes: dict[str, str] | None = None, rng: np.random.Generator | None = None):
        import omni.replicator.core as rep
        cfg = cfg or load_yaml(CONFIG_DIR / "sensors" / "rtx_camera.yaml")
        self.cfg = cfg
        self.rp = rep.create.render_product(cam_prim, tuple(cfg["resolution"]), name="medortrace_rgb")
        self.ann = {}
        for a in cfg["annotators"]:
            try:
                an = rep.AnnotatorRegistry.get_annotator(a)
                if a == "semantic_segmentation":
                    an = rep.AnnotatorRegistry.get_annotator(a, init_params={"colorize": False})
                an.attach([self.rp])
                self.ann[a] = an
            except Exception as e:  # pragma: no cover
                print(f"[medortrace] camera annotator {a} unavailable: {e}")
        self.detector = detector
        self.model = None
        if detector.startswith("model:"):
            from medortrace.isaac.detector import load_detector
            self.model = load_detector(detector.split(":", 1)[1])
        self.item_classes = item_classes or {}
        self.rng = rng or np.random.default_rng(0)
        self.hfov = np.deg2rad(cfg["hfov_deg"])
        self.W, self.H = cfg["resolution"]
        self._seq = 0

    def _pixel_to_ray(self, u, v):
        f = (self.W / 2) / np.tan(self.hfov / 2)
        x = (u - self.W / 2) / f
        y = (v - self.H / 2) / f
        bearing = -np.arctan(x)
        elev = -np.arctan(y / np.sqrt(1 + x * x))
        return bearing, elev

    def read(self, t: float, stamp_fn, pitch_rad: float) -> CameraFrame | None:
        if "bounding_box_2d_tight" not in self.ann or "distance_to_image_plane" not in self.ann:
            return None
        boxes = self.ann["bounding_box_2d_tight"].get_data()
        depth = np.asarray(self.ann["distance_to_image_plane"].get_data())
        dets = []
        if self.model is not None and "rgb" in self.ann:
            rgb = np.asarray(self.ann["rgb"].get_data())[..., :3]
            for (u, v, logits, frac) in self.model.predict(rgb):
                z = float(np.nanmedian(depth[max(0, int(v) - 2):int(v) + 3, max(0, int(u) - 2):int(u) + 3]))
                b, e = self._pixel_to_ray(u, v)
                dets.append(CameraDetection(_argmax_cls(logits), None, b, e + pitch_rad, z, logits, frac, 0.0))
        else:
            from medortrace.sim.sensors_lite import CLASSES, SIMILARITY
            info = boxes.get("info", {}) if isinstance(boxes, dict) else {}
            labels = info.get("idToLabels", {})
            data = boxes.get("data", []) if isinstance(boxes, dict) else boxes
            for bx in data:
                lab = labels.get(str(int(bx["semanticId"])), {})
                cls = lab.get("class") if isinstance(lab, dict) else None
                if cls not in CLASSES:
                    continue
                u = 0.5 * (bx["x_min"] + bx["x_max"])
                v = 0.5 * (bx["y_min"] + bx["y_max"])
                occl = float(bx["occlusionRatio"]) if "occlusionRatio" in bx.dtype.names else 0.0
                vis = 1.0 - occl
                z = float(np.nanmedian(depth[max(0, int(v) - 2):int(v) + 3, max(0, int(u) - 2):int(u) + 3]))
                if not np.isfinite(z) or vis <= 0 or self.rng.random() > 0.92 * vis:
                    continue
                ci = CLASSES.index(cls)
                logits = 8.0 * vis * np.clip(1.2 - z / 5.0, 0.1, 1.0) * SIMILARITY[ci] + self.rng.normal(0, 0.8, len(CLASSES))
                b, e = self._pixel_to_ray(u, v)
                dets.append(CameraDetection(CLASSES[int(np.argmax(logits))], None, b, e + pitch_rad, z, logits, vis, 0.0))
        self._seq += 1
        return CameraFrame(Header(stamp_fn("camera", t), t, "camera_link", self._seq), dets, self.hfov, 5.0)


def _argmax_cls(logits):
    from medortrace.sim.sensors_lite import CLASSES
    return CLASSES[int(np.argmax(logits))]


class AcousticAdapter:
    """RTX acoustic (experimental) with a PhysX scene-query fallback."""

    def __init__(self, parent: str, cfg: dict | None = None):
        self.cfg = cfg or load_yaml(CONFIG_DIR / "sensors" / "rtx_acoustic.yaml")
        self.native = None
        try:
            import omni.kit.commands
            for cmd in ACOUSTIC_COMMANDS:
                try:
                    _, self.native = omni.kit.commands.execute(cmd, path="rtx_acoustic", parent=parent)
                    self.backend_info = {"mode": "rtx_acoustic", "command": cmd}
                    break
                except Exception:
                    continue
        except ImportError:  # pragma: no cover
            pass
        if self.native is None:
            self.backend_info = {"mode": "physx_fallback"}
        self._seq = 0

    def read(self, t: float, stamp_fn, origin: np.ndarray, region: str, region_pos: np.ndarray,
             base_reflectivity: float, contents_reflectivity: list[float], rng) -> AcousticFrame | None:
        """Fallback echo model: PhysX LOS test + the calibrated forward model."""
        from omni.physx import get_physx_scene_query_interface
        d = region_pos - origin
        r = float(np.linalg.norm(d))
        if r > self.cfg["max_range_m"]:
            return None
        hit = get_physx_scene_query_interface().raycast_closest(tuple(origin), tuple(d / r), r - 0.3)
        occluded = bool(hit.get("hit", False))
        e = 0.15 * base_reflectivity + sum(0.6 * c for c in contents_reflectivity)
        e *= (1.0 / (1.0 + 0.3 * r * r)) * (self.cfg["diffraction_attenuation"] if occluded else 1.0)
        sigma = self.cfg["noise_sigma"] * (1.8 if occluded else 1.0)
        self._seq += 1
        echo = AcousticEcho(region, float(max(0.0, e + rng.normal(0, sigma))), 2 * r / self.cfg["speed_of_sound_mps"],
                            occluded, gt_hard_reflector=any(c > 0.8 for c in contents_reflectivity))
        return AcousticFrame(Header(stamp_fn("acoustic", t), t, "acoustic_link", self._seq), [echo])


class ImuAdapter:
    def __init__(self, prim_path: str, rate_hz: float = 100.0):
        IMUSensor = imu_sensor_cls()
        self.s = IMUSensor(prim_path=prim_path + "/imu", name="medortrace_imu", frequency=int(rate_hz),
                           translation=np.zeros(3), linear_acceleration_filter_size=4)
        self._seq = 0

    def read(self, t: float, stamp_fn) -> list[ImuSample]:
        f = self.s.get_current_frame()
        self._seq += 1
        return [ImuSample(Header(stamp_fn("imu", t), t, "imu_link", self._seq), np.asarray(f["lin_acc"], float),
                          np.asarray(f["ang_vel"], float))]


class ContactAdapter:
    def __init__(self, prim_path: str, radius: float = 0.3, threshold: float = 1.0):
        ContactSensor = contact_sensor_cls()
        self.s = ContactSensor(prim_path=prim_path + "/bumper_contact", name="medortrace_bumper",
                               min_threshold=threshold, max_threshold=1e7, radius=radius)
        self._seq = 0

    def read(self, t: float, stamp_fn, effort: np.ndarray | None = None) -> ContactState:
        f = self.s.get_current_frame()
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
        sqi = get_physx_scene_query_interface()
        obs = []
        for k, p in self.lm.items():
            d = p - cam_pos
            r = float(np.linalg.norm(d[:2]))
            if r > self.max_range:
                continue
            L = float(np.linalg.norm(d))
            hit = sqi.raycast_closest(tuple(cam_pos), tuple(d / L), L - 0.1)
            if hit.get("hit", False) or rng.random() > 0.9:
                continue
            b = float((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
            obs.append(LandmarkObservation(k, r + float(rng.normal(0, 0.03)), b + float(rng.normal(0, 0.01))))
        self._seq += 1
        return LandmarkFrame(Header(stamp_fn("landmarks", t), t, "camera_link", self._seq), obs)


def profile_paths() -> dict[str, Path]:
    return {"lidar": CONFIG_DIR / "sensors" / "rtx_lidar_or16.json",
            "radar": CONFIG_DIR / "sensors" / "rtx_radar_or77.json"}
