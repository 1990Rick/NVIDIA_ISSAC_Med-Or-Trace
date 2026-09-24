"""Physics-inspired sensor models for the lite backend.

They are deliberately *qualitatively* faithful to the RTX sensors they stand
in for (see docs/sim_to_real_checklist.md):

* Lidar: per-ray analytic intersection, Lambertian intensity with
  inverse-square fall-off and a detection threshold, material transmissivity
  (acrylic), specular mirror bounce producing multipath ghosts (polished
  steel, monitor glass, wet floor), range noise, haze back-scatter.
* Camera: frustum + occlusion sampling (visible fraction), size/range/glare
  dependent detection probability, class-confusion logits from a learned
  front-end surrogate, tag decoding when close, false positives.
* Radar: Doppler range-rate, fabric penetration, RCS-dependent detection,
  steel multipath ghosts, coarse angular resolution.
* Acoustic (40 kHz active probe): echo energy of a target region, including
  diffracted (non line-of-sight) paths with attenuation.
* IMU / odometry / landmarks / bumper contact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from medortrace.common.msgs import (
    AcousticEcho,
    CameraDetection,
    LandmarkObservation,
    RadarDetection,
)
from medortrace.sim.raycast import CEIL_ID, FLOOR_ID, NO_HIT, RayScene, cast, segment_occluded
from medortrace.world.materials import Material

CLASSES = ["sponge", "clamp", "needle_driver", "specimen", "implant_box"]
# Visual similarity used to build confusion structure of the surrogate detector.
SIMILARITY = np.array([
    # sp   cl   nd   spec imp
    [1.0, 0.05, 0.05, 0.2, 0.1],
    [0.05, 1.0, 0.7, 0.05, 0.05],
    [0.05, 0.7, 1.0, 0.05, 0.05],
    [0.2, 0.05, 0.05, 1.0, 0.3],
    [0.1, 0.05, 0.05, 0.3, 1.0],
])


@dataclass
class LidarConfig:
    rings: int = 12
    elev_min_deg: float = -15.0
    elev_max_deg: float = 15.0
    az_res_deg: float = 2.0
    min_range: float = 0.2
    max_range: float = 20.0
    range_sigma: float = 0.02
    intensity_threshold: float = 0.004
    mount_height: float = 0.9
    mount_x: float = 0.1


def lidar_directions(cfg: LidarConfig) -> np.ndarray:
    el = np.deg2rad(np.linspace(cfg.elev_min_deg, cfg.elev_max_deg, cfg.rings))
    az = np.deg2rad(np.arange(-180.0, 180.0, cfg.az_res_deg))
    E, A = np.meshgrid(el, az, indexing="ij")
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    ring = np.repeat(np.arange(cfg.rings), len(az))
    return d, ring


def simulate_lidar(scene: RayScene, obj_materials: list[Material], obj_tags: list[list[str]],
                   origin: np.ndarray, yaw: float, cfg: LidarConfig, rng: np.random.Generator,
                   specular_gain: float = 1.0, floor_wet: bool = False, haze: float = 0.0):
    """Returns (ranges(R,), dirs_sensor(R,3), ring(R,), intensity(R,), is_ghost(R,), obj(R,))."""
    d_s, ring = lidar_directions(cfg)
    c, s = np.cos(yaw), np.sin(yaw)
    d_w = np.stack([c * d_s[:, 0] - s * d_s[:, 1], s * d_s[:, 0] + c * d_s[:, 1], d_s[:, 2]], 1)
    O = np.repeat(origin[None], len(d_w), 0)
    hits = cast(scene, O, d_w, t_max=cfg.max_range)
    R = len(d_w)
    ranges = hits.t.copy()
    obj = hits.obj.copy()
    ghost = np.zeros(R, dtype=bool)
    n_obj = scene.n_boxes + len(scene.cyl_xy)

    # material property lookup tables: [objects..., floor, ceiling, no-hit]
    refl_tab = np.array([m.lidar_reflectance for m in obj_materials] + [0.0])
    spec_tab = np.array([m.specularity for m in obj_materials] + [0.0]) * specular_gain
    trans_tab = np.array([m.transmissivity for m in obj_materials] + [0.0])
    strong_tab = np.array(["strong_specular" in t for t in obj_tags] + [False, False, False])
    if floor_wet:
        spec_tab[n_obj] = max(spec_tab[n_obj], 0.6)

    def lut(o):
        o = np.asarray(o)
        return np.where(o >= 0, o, np.where(o == FLOOR_ID, n_obj, np.where(o == CEIL_ID, n_obj + 1, n_obj + 2)))

    li = lut(obj)
    refl = refl_tab[li]
    spec = spec_tab[li]
    trans = trans_tab[li]
    strong = strong_tab[li]
    cos_inc = np.abs((d_w * hits.normal).sum(1))
    spec = np.where(strong, 0.97, np.clip(spec, 0, 0.98))
    u = rng.random(R)
    valid = np.isfinite(ranges)
    # --- transmission (e.g. acrylic): continue through the object ---------
    tr = valid & (u < trans)
    if tr.any():
        idx = np.where(tr)[0]
        excl = np.zeros(n_obj, dtype=bool)
        for i in idx:
            if 0 <= obj[i] < n_obj:
                excl[:] = False
                excl[obj[i]] = True
                h2 = cast(scene, O[i:i + 1], d_w[i:i + 1], t_max=cfg.max_range, exclude=excl)
                ranges[i] = h2.t[0]
                obj[i] = h2.obj[0]
                refl[i] = refl_tab[lut(obj[i])] * 0.7
                cos_inc[i] = abs(float(d_w[i] @ h2.normal[0]))
    # --- specular bounce: mirror-like path unless near-normal incidence ---
    sp = valid & ~tr & (rng.random(R) < spec) & (cos_inc < 0.97)
    if sp.any():
        idx = np.where(sp)[0]
        P = O[idx] + ranges[idx, None] * d_w[idx]
        n = hits.normal[idx]
        rdir = d_w[idx] - 2 * (d_w[idx] * n).sum(1, keepdims=True) * n
        h2 = cast(scene, P + 1e-3 * rdir, rdir, t_max=cfg.max_range)
        ok = np.isfinite(h2.t) & (h2.obj != NO_HIT)
        total = ranges[idx] + h2.t
        # ghost: reported along the *original* ray at the total path length
        ranges[idx] = np.where(ok & (total < cfg.max_range), total, np.inf)
        ghost[idx] = ok
        obj[idx] = np.where(ok, h2.obj, NO_HIT)
        refl[idx] = np.where(ok, 0.5 * refl_tab[lut(h2.obj)], 0.0)
        cos_inc[idx] = np.abs((rdir * h2.normal).sum(1))
    # --- intensity & detection threshold ---------------------------------
    rr = np.where(np.isfinite(ranges), ranges, 1e9)
    intensity = refl * np.maximum(cos_inc, 0.05) / np.maximum(rr, 0.5) ** 2
    detected = np.isfinite(ranges) & (intensity > cfg.intensity_threshold) & (ranges > cfg.min_range)
    ranges = np.where(detected, ranges + rng.normal(0, cfg.range_sigma, R), np.inf)
    # --- haze back-scatter: spurious short returns -------------------------
    if haze > 0:
        hz = rng.random(R) < haze * 0.02
        ranges = np.where(hz, rng.uniform(0.3, 2.0, R), ranges)
        ghost = ghost | hz
        obj = np.where(hz, NO_HIT, obj)
    intensity = np.where(np.isfinite(ranges), np.clip(intensity * 50, 0, 1), 0.0)
    return ranges, d_s, ring, intensity, ghost, obj


# ---------------------------------------------------------------------------
@dataclass
class CameraConfig:
    mount_height: float = 1.45
    pitch_deg: float = -25.0
    hfov_deg: float = 90.0
    vfov_deg: float = 65.0
    max_range: float = 5.0
    pd0: float = 0.92
    fp_rate: float = 0.05
    tag_range: float = 1.6
    logit_scale: float = 8.0


def simulate_camera(scene: RayScene, cam_pos: np.ndarray, yaw: float, cfg: CameraConfig,
                    items: list[dict], rng: np.random.Generator, glare_gain: float = 1.0,
                    haze: float = 0.0, exclude_map: dict | None = None) -> list[CameraDetection]:
    """``items``: dicts with id, cls, pos(3), size, material(Material), slot(Slot), tag_readable."""
    dets: list[CameraDetection] = []
    pitch = np.deg2rad(cfg.pitch_deg)
    for it in items:
        p = it["pos"]
        if not np.all(np.isfinite(p)):
            continue
        slot = it["slot"]
        if slot is not None and slot.hidden_from_camera:
            continue
        d = p - cam_pos
        rng_xy = np.hypot(d[0], d[1])
        rng3 = float(np.linalg.norm(d))
        if rng3 > cfg.max_range or rng3 < 0.2:
            continue
        bearing = float(np.arctan2(d[1], d[0]) - yaw)
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        elev = float(np.arctan2(d[2], rng_xy))
        if abs(bearing) > np.deg2rad(cfg.hfov_deg) / 2 or abs(elev - pitch) > np.deg2rad(cfg.vfov_deg) / 2:
            continue
        if slot is not None and slot.needs_top_view and elev > np.deg2rad(-38.0):
            continue  # cannot see into an open container from a shallow angle
        # visible fraction by occlusion sampling around the item
        offs = np.array([[0, 0, 0.03], [0.04, 0, 0.03], [-0.04, 0, 0.03], [0, 0.04, 0.03], [0, -0.04, 0.03]])
        pts = p + offs
        excl = exclude_map.get(it["id"]) if exclude_map else None
        occ = segment_occluded(scene, np.repeat(cam_pos[None], len(pts), 0), pts, exclude=excl)
        vis = float(1.0 - occ.mean())
        if vis <= 0.0:
            continue
        m: Material = it["material"]
        glare = float(np.clip(m.glare * glare_gain * rng.uniform(0.5, 1.2), 0, 1))
        size = max(it["size"][:2])
        range_factor = float(np.clip(1.2 - rng3 / cfg.max_range, 0.1, 1.0)) * float(np.clip(size / 0.08, 0.5, 1.0))
        pd = cfg.pd0 * vis * range_factor * (1 - 0.45 * glare) * (1 - haze)
        if rng.random() > pd:
            continue
        ci = CLASSES.index(it["cls"])
        margin = cfg.logit_scale * vis * range_factor * (1 - 0.6 * glare)
        logits = margin * SIMILARITY[ci] + rng.normal(0, 0.8, len(CLASSES))
        hint = None
        if it.get("tag_readable") and rng3 < cfg.tag_range and vis > 0.7 and glare < 0.6 and rng.random() < 0.85:
            hint = it["id"]
        dets.append(CameraDetection(
            cls=CLASSES[int(np.argmax(logits))], item_id_hint=hint,
            bearing=bearing + float(rng.normal(0, 0.01)), elevation=elev + float(rng.normal(0, 0.01)),
            range=rng3 * (1 + float(rng.normal(0, 0.02))), logits=logits, visible_fraction=vis,
            glare=glare, gt_item_id=it["id"]))
    # false positives (reflections, glove/gauze look-alikes)
    for _ in range(rng.poisson(cfg.fp_rate * (1 + 2 * haze + 0.5 * (glare_gain - 1)))):
        b = float(rng.uniform(-1, 1) * np.deg2rad(cfg.hfov_deg) / 2)
        r = float(rng.uniform(0.8, cfg.max_range * 0.8))
        e = float(pitch + rng.uniform(-0.3, 0.3))
        ci = int(rng.integers(0, len(CLASSES)))
        logits = 1.5 * SIMILARITY[ci] + rng.normal(0, 0.8, len(CLASSES))
        dets.append(CameraDetection(CLASSES[int(np.argmax(logits))], None, b, e, r, logits, 0.5, 0.5, None))
    return dets


# ---------------------------------------------------------------------------
@dataclass
class RadarConfig:
    mount_height: float = 0.6
    fov_deg: float = 120.0
    max_range: float = 15.0
    range_sigma: float = 0.06
    az_sigma: float = 0.03
    vel_sigma: float = 0.05
    min_speed_static_clutter: float = 0.0


def simulate_radar(scene: RayScene, obj_materials: list[Material], origin: np.ndarray, yaw: float,
                   robot_vel_world: np.ndarray, targets: list[dict], cfg: RadarConfig,
                   rng: np.random.Generator) -> list[RadarDetection]:
    """``targets``: dicts with pos(3), vel(2), rcs_dbsm, id.  Fabric-penetrating occlusion."""
    dets = []
    n_obj = scene.n_boxes + len(scene.cyl_xy)
    penetrable = np.array([obj_materials[i].radar_penetrable for i in range(scene.n_boxes)] +
                          [False] * len(scene.cyl_xy)) if n_obj else np.zeros(0, bool)
    for tg in targets:
        d = tg["pos"] - origin
        r = float(np.linalg.norm(d))
        if r > cfg.max_range or r < 0.3:
            continue
        az = float((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        if abs(az) > np.deg2rad(cfg.fov_deg) / 2:
            continue
        excl = penetrable.copy()
        if tg.get("self_index") is not None:
            excl[tg["self_index"]] = True
        through_fabric = False
        blocked = segment_occluded(scene, origin[None], tg["pos"][None], exclude=excl, tol=0.15)[0]
        if blocked:
            continue
        if tg.get("behind_fabric"):
            through_fabric = True
        # radar equation: SNR ~ RCS / r^4
        snr_db = tg["rcs_dbsm"] - 40 * np.log10(max(r, 0.5)) + 35.0 - (6.0 if through_fabric else 0.0)
        p_det = 1 / (1 + np.exp(-(snr_db - 3.0) / 2.0))
        if rng.random() > p_det:
            continue
        u = d / r
        rel_v = np.array([*tg["vel"], 0.0]) - np.array([*robot_vel_world, 0.0])
        vr = float(rel_v @ u)
        dets.append(RadarDetection(r + float(rng.normal(0, cfg.range_sigma)), az + float(rng.normal(0, cfg.az_sigma)),
                                   float(np.arcsin(np.clip(d[2] / r, -1, 1))), vr + float(rng.normal(0, cfg.vel_sigma)),
                                   float(tg["rcs_dbsm"] + rng.normal(0, 2.0)), through_fabric, tg.get("id")))
    # sporadic steel multipath ghosts
    if rng.random() < 0.08:
        dets.append(RadarDetection(float(rng.uniform(1, cfg.max_range * 0.6)), float(rng.uniform(-1, 1)),
                                   0.0, float(rng.normal(0, 0.5)), float(rng.uniform(-15, 0)), False, None))
    return dets


# ---------------------------------------------------------------------------
@dataclass
class AcousticConfig:
    max_range: float = 3.5
    noise_sigma: float = 0.05
    speed_of_sound: float = 343.0
    nlos_attenuation: float = 0.45


def simulate_acoustic(scene: RayScene, origin: np.ndarray, region: str, region_pos: np.ndarray,
                      base_reflectivity: float, contents: list[Material], cfg: AcousticConfig,
                      rng: np.random.Generator) -> AcousticEcho | None:
    d = region_pos - origin
    r = float(np.linalg.norm(d))
    if r > cfg.max_range:
        return None
    occluded = bool(segment_occluded(scene, origin[None], region_pos[None], tol=0.3)[0])
    energy = 0.15 * base_reflectivity + sum(0.6 * m.acoustic_reflectivity for m in contents)
    energy *= (1.0 / (1.0 + 0.3 * r * r)) * (cfg.nlos_attenuation if occluded else 1.0)
    sigma = cfg.noise_sigma * (1.8 if occluded else 1.0)
    return AcousticEcho(region, float(max(0.0, energy + rng.normal(0, sigma))), 2 * r / cfg.speed_of_sound,
                        occluded, gt_hard_reflector=any(m.acoustic_reflectivity > 0.8 for m in contents))


# ---------------------------------------------------------------------------
def simulate_landmarks(scene: RayScene, origin: np.ndarray, yaw: float, landmarks: list, rng: np.random.Generator,
                       max_range: float = 8.0, r_sigma: float = 0.03, b_sigma: float = 0.01) -> list[LandmarkObservation]:
    obs = []
    for lm in landmarks:
        d = lm.position - origin
        r = float(np.linalg.norm(d[:2]))
        if r > max_range:
            continue
        # wall tags: pull the endpoint 5 cm off the wall to avoid self-occlusion
        tgt = lm.position - 0.08 * d / (np.linalg.norm(d) + 1e-9)
        if segment_occluded(scene, origin[None], tgt[None], tol=0.05)[0]:
            continue
        if rng.random() > 0.9:
            continue
        b = float((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi)
        obs.append(LandmarkObservation(lm.id, r + float(rng.normal(0, r_sigma)), b + float(rng.normal(0, b_sigma))))
    return obs
