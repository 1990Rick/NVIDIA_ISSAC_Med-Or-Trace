"""Lite sensor models: physically checkable behaviour on hand-built scenes.

* lidar: a ``strong_specular`` mirror produces multipath ghosts that lie at the
  *mirror image* of the real reflector; matte surfaces produce none; clear
  acrylic lets most rays through to the wall behind;
* camera: the open-container ``needs_top_view`` rule (steeper than 38 deg);
* radar: Doppler sign convention (positive = receding) incl. ego-motion;
* acoustic / landmarks: non-line-of-sight attenuation and occlusion.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from medortrace.sim.raycast import FLOOR_ID, cast
from medortrace.sim.sensors_lite import (
    AcousticConfig,
    CameraConfig,
    LidarConfig,
    RadarConfig,
    lidar_directions,
    simulate_acoustic,
    simulate_camera,
    simulate_landmarks,
    simulate_lidar,
    simulate_radar,
)
from medortrace.world.materials import MATERIALS
from medortrace.world.scene import Landmark, Slot

MATTE_FLOOR = replace(MATERIALS["floor_vinyl"], specularity=0.0)
CEIL = MATERIALS["painted_wall"]
ORIGIN = np.array([0.0, 0.0, 0.9])


def _lidar(scene, mats, tags, seed=0, **kw):
    cfg = LidarConfig(**{"rings": 12, "az_res_deg": 1.0, "range_sigma": 0.0, **kw})
    return simulate_lidar(scene, list(mats) + [MATTE_FLOOR, CEIL], tags, ORIGIN, 0.0, cfg, np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# lidar
# ---------------------------------------------------------------------------
@pytest.fixture
def mirror_scene(make_scene, box):
    mirror = box((2.0, 0.0, 0.8), (0.02, 1.5, 0.8))          # reflecting face at x = 1.98
    target = box((1.0, 1.5, 0.4), (0.2, 0.2, 0.4))           # lambertian reflector beside the robot
    return make_scene([mirror, target]), mirror, target


def test_lidar_directions_grid():
    cfg = LidarConfig(rings=4, az_res_deg=90.0, elev_min_deg=-10, elev_max_deg=10)
    d, ring = lidar_directions(cfg)
    assert d.shape == (16, 3) and np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert ring.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert np.allclose(np.rad2deg(np.arcsin(d[::4, 2])), np.linspace(-10, 10, 4))


def test_strong_specular_mirror_ghost_at_mirrored_distance(mirror_scene):
    scene, mirror, target = mirror_scene
    mats = [MATERIALS["painted_wall"], MATERIALS["painted_wall"]]     # matte mirror material: the TAG makes it a mirror
    rng_, d_s, ring, inten, ghost, obj = _lidar(scene, mats, [["strong_specular"], []])
    g = ghost & np.isfinite(rng_) & (obj == 1)
    assert g.sum() >= 20
    P = ORIGIN + rng_[g, None] * d_s[g]
    face_x = mirror.center[0] - mirror.half[0]
    # ghosts appear *behind* the mirror, beyond the first surface along the ray
    assert np.all(P[:, 0] > face_x)
    first = cast(scene, np.repeat(ORIGIN[None], g.sum(), 0), d_s[g])
    assert np.all(first.obj == 0) and np.all(rng_[g] > first.t + 0.3)
    # ... exactly on the mirror image of the real reflector (to 1 mm: the bounce offset)
    Pm = P.copy()
    Pm[:, 0] = 2 * face_x - P[:, 0]
    outside = np.maximum(np.abs(Pm - target.center) - target.half, 0.0)
    assert np.linalg.norm(outside, axis=1).max() < 2e-3
    # total path = mirror distance + mirror->target distance
    back = cast(scene, ORIGIN + first.t[:, None] * d_s[g], d_s[g] * np.array([-1.0, 1.0, 1.0]))
    assert np.allclose(rng_[g], first.t + back.t, atol=2e-3)
    # floor ghosts via the mirror stay on the (mirrored) floor plane
    gf = ghost & np.isfinite(rng_) & (obj == FLOOR_ID)
    if gf.any():
        assert np.abs((ORIGIN + rng_[gf, None] * d_s[gf])[:, 2]).max() < 2e-3


def test_matte_surfaces_produce_no_ghosts(mirror_scene):
    scene, _, _ = mirror_scene
    rng_, d_s, ring, inten, ghost, obj = _lidar(scene, [MATERIALS["painted_wall"]] * 2, [[], []])
    assert not ghost.any()
    fin = np.isfinite(rng_)
    h = cast(scene, np.repeat(ORIGIN[None], fin.sum(), 0), d_s[fin], t_max=20.0)
    assert np.allclose(rng_[fin], h.t, atol=1e-9)           # every return is the first surface (sigma = 0)
    assert np.all((inten[fin] > 0) & (inten[fin] <= 1)) and np.all(inten[~fin] == 0)


def test_polished_steel_is_specular_without_tag(mirror_scene):
    scene, _, _ = mirror_scene
    _, _, _, _, ghost, obj = _lidar(scene, [MATERIALS["instrument_steel_polished"], MATERIALS["painted_wall"]],
                                    [[], []])
    assert (ghost & (obj == 1)).sum() > 10


def test_transmissive_acrylic_passes_most_rays(make_scene, box):
    panel = box((2.0, 0.0, 0.8), (0.01, 0.8, 0.8))
    wall = box((4.0, 0.0, 1.5), (0.05, 3.0, 1.5))
    scene = make_scene([panel, wall])
    d, _ = lidar_directions(LidarConfig(rings=12, az_res_deg=1.0))
    through_panel = cast(scene, np.repeat(ORIGIN[None], len(d), 0), d).obj == 0
    assert through_panel.sum() > 100
    acrylic = MATERIALS["acrylic_clear"]
    rng_, d_s, _, _, ghost, obj = _lidar(scene, [acrylic, MATERIALS["painted_wall"]], [[], []])
    on_wall = (obj == 1) & through_panel & np.isfinite(rng_)
    frac = on_wall.sum() / through_panel.sum()
    assert 0.72 < frac < 0.97, frac                        # transmissivity 0.85 (binomial spread)
    wall_t = cast(make_scene([wall]), np.repeat(ORIGIN[None], on_wall.sum(), 0), d_s[on_wall]).t
    assert np.allclose(rng_[on_wall], wall_t, atol=1e-9)
    # an opaque panel of the same shape stops every ray
    rng2, _, _, _, _, obj2 = _lidar(scene, [MATERIALS["plastic_hdpe"], MATERIALS["painted_wall"]], [[], []])
    assert not np.any((obj2 == 1) & through_panel)


def test_lidar_range_noise_and_threshold(make_scene, box):
    scene = make_scene([box((3.0, 0.0, 1.0), (0.05, 3.0, 1.0))])
    exact, *_ = _lidar(scene, [MATERIALS["painted_wall"]], [[]], range_sigma=0.0)
    noisy, *_ = _lidar(scene, [MATERIALS["painted_wall"]], [[]], range_sigma=0.05, seed=1)
    both = np.isfinite(exact) & np.isfinite(noisy)
    err = noisy[both] - exact[both]
    assert abs(err.mean()) < 0.01 and 0.035 < err.std() < 0.065
    # intensity ~ reflectance * cos / r^2 against a detection threshold: a black
    # surface at 8 m is invisible where a painted one is not
    dark = replace(MATERIALS["rubber_black"], lidar_reflectance=0.01)
    far = make_scene([box((8.0, 0.0, 1.0), (0.05, 2.0, 1.0))])
    d, _ = lidar_directions(LidarConfig(rings=12, az_res_deg=1.0))
    aimed = cast(far, np.repeat(ORIGIN[None], len(d), 0), d).obj == 0
    assert aimed.sum() > 100
    r, *_ = _lidar(far, [dark], [[]])
    assert np.isinf(r[aimed]).all()
    r, *_ = _lidar(far, [MATERIALS["painted_wall"]], [[]])          # same wall, bright paint: detected
    assert np.isfinite(r[aimed]).mean() > 0.9


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------
def _cam_item(slot, pos):
    return {"id": "sponge_1", "cls": "sponge", "pos": np.asarray(pos, float), "size": (0.1, 0.1, 0.01),
            "material": MATERIALS["cotton_sponge"], "slot": slot, "tag_readable": True}


def _detect_count(scene, slot, item_xy, n=60, **cam_kw):
    cfg = CameraConfig(pd0=1.0, fp_rate=0.0, **cam_kw)
    rng = np.random.default_rng(3)
    cam = np.array([0.0, 0.0, cfg.mount_height])
    hits, tags = 0, 0
    for _ in range(n):
        dets = simulate_camera(scene, cam, 0.0, cfg, [_cam_item(slot, [*item_xy, 0.3])], rng)
        hits += len(dets)
        tags += sum(d.item_id_hint == "sponge_1" for d in dets)
        for d in dets:
            assert d.gt_item_id == "sponge_1" and 0 < d.visible_fraction <= 1
    return hits, tags


def test_camera_needs_top_view_rule(make_scene):
    scene = make_scene()
    open_bin = Slot("bin:inside", "container", "bin", (0, 0, 0.3), needs_top_view=True)
    shelf = Slot("shelf:top", "surface", "shelf", (0, 0, 0.3))
    # elevation from 1.45 m down to 0.3 m: 1.0 m away -> -49 deg (steep), 2.0 m away -> -30 deg (shallow)
    assert np.rad2deg(np.arctan2(0.3 - 1.45, 1.0)) < -38 < np.rad2deg(np.arctan2(0.3 - 1.45, 2.0))
    steep, _ = _detect_count(scene, open_bin, (1.0, 0.0))
    shallow, _ = _detect_count(scene, open_bin, (2.0, 0.0))
    shallow_surface, _ = _detect_count(scene, shelf, (2.0, 0.0))
    assert steep > 40                      # pd ~ 0.9 when looking down into the container
    assert shallow == 0                    # cannot see into an open container from a shallow angle
    assert shallow_surface > 30            # same geometry on an open surface is visible


def test_camera_hidden_slot_occlusion_and_tags(make_scene, box):
    hidden = Slot("drape:under", "under_drape", "drape", (0, 0, 0.3), hidden_from_camera=True)
    shelf = Slot("shelf:top", "surface", "shelf", (0, 0, 0.3))
    assert _detect_count(make_scene(), hidden, (1.0, 0.0))[0] == 0
    # a wall between camera and item removes all detections
    blocked = make_scene([box((0.6, 0.0, 1.0), (0.05, 1.0, 1.0))])
    assert _detect_count(blocked, shelf, (1.0, 0.0))[0] == 0
    # tags decode only within tag_range
    near_hits, near_tags = _detect_count(make_scene(), shelf, (1.0, 0.0))
    far_hits, far_tags = _detect_count(make_scene(), shelf, (2.5, 0.0))
    assert near_tags > 0.6 * near_hits and far_hits > 0 and far_tags == 0


# ---------------------------------------------------------------------------
# radar
# ---------------------------------------------------------------------------
def _radar_vr(target_vel, robot_vel=(0.0, 0.0), pos=(5.0, 0.0, 1.0), frames=40):
    cfg = RadarConfig()
    rng = np.random.default_rng(8)
    origin = np.array([0.0, 0.0, cfg.mount_height])
    vr = []
    for _ in range(frames):
        dets = simulate_radar(scene_empty(), [], origin, 0.0, np.array(robot_vel, float),
                              [{"pos": np.array(pos, float), "vel": np.array(target_vel, float), "rcs_dbsm": 0.0,
                                "id": "p"}], cfg, rng)
        vr += [d.radial_velocity for d in dets if d.gt_object_id == "p"]
    assert len(vr) > frames // 2
    return float(np.mean(vr))


def scene_empty():
    from medortrace.sim.raycast import RayScene
    return RayScene(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros((0, 2)), np.zeros(0), np.zeros(0))


def test_radar_doppler_sign_convention():
    los = np.array([5.0, 0.0, 0.4]) / np.linalg.norm([5.0, 0.0, 0.4])
    assert _radar_vr((1.0, 0.0)) == pytest.approx(los[0] * 1.0, abs=0.05)      # receding -> positive
    assert _radar_vr((-1.0, 0.0)) == pytest.approx(-los[0], abs=0.05)          # approaching -> negative
    assert _radar_vr((0.0, 1.0)) == pytest.approx(0.0, abs=0.05)               # tangential -> ~0
    # ego-motion: robot driving at a static target sees it approach
    assert _radar_vr((0.0, 0.0), robot_vel=(0.5, 0.0)) == pytest.approx(-0.5 * los[0], abs=0.05)


def test_radar_penetrates_fabric_but_not_steel(make_scene, box):
    cfg = RadarConfig()
    origin = np.array([0.0, 0.0, cfg.mount_height])
    scene = make_scene([box((2.5, 0.0, 1.0), (0.02, 1.0, 1.0))])
    tgt = [{"pos": np.array([5.0, 0.0, 1.0]), "vel": np.array([1.0, 0.0]), "rcs_dbsm": 5.0, "id": "p"}]

    def n_hits(mat):
        rng = np.random.default_rng(2)
        return sum(any(d.gt_object_id == "p" for d in simulate_radar(scene, [mat], origin, 0.0, np.zeros(2), tgt,
                                                                      cfg, rng)) for _ in range(30))

    assert n_hits(MATERIALS["surgical_drape"]) > 20
    assert n_hits(MATERIALS["stainless_steel_brushed"]) == 0


# ---------------------------------------------------------------------------
# acoustic & landmarks
# ---------------------------------------------------------------------------
def test_acoustic_nlos_attenuation_and_hard_reflector(make_scene, box):
    cfg = AcousticConfig(noise_sigma=0.0)
    region = np.array([2.0, 0.0, 0.8])
    clamp = MATERIALS["instrument_steel_polished"]
    rng = np.random.default_rng(0)
    los = simulate_acoustic(make_scene(), np.array([0.0, 0.0, 1.2]), "r", region, 0.5, [clamp], cfg, rng)
    nlos = simulate_acoustic(make_scene([box((1.0, 0.0, 1.0), (0.05, 1.0, 1.0))]), np.array([0.0, 0.0, 1.2]), "r",
                             region, 0.5, [clamp], cfg, rng)
    empty = simulate_acoustic(make_scene(), np.array([0.0, 0.0, 1.2]), "r", region, 0.5, [], cfg, rng)
    assert not los.path_occluded and nlos.path_occluded
    assert nlos.energy == pytest.approx(los.energy * cfg.nlos_attenuation)
    assert los.gt_hard_reflector and not empty.gt_hard_reflector and empty.energy < los.energy
    assert los.delay_s == pytest.approx(2 * np.linalg.norm(region - [0, 0, 1.2]) / cfg.speed_of_sound)
    assert simulate_acoustic(make_scene(), np.array([10.0, 0, 1.2]), "r", region, 0.5, [], cfg, rng) is None


def test_landmarks_occlusion_and_range_bearing(make_scene, box):
    lms = [Landmark("a", np.array([4.0, 0.0, 1.6])), Landmark("b", np.array([0.0, 4.0, 1.6])),
           Landmark("c", np.array([20.0, 0.0, 1.6]))]
    scene = make_scene([box((0.0, 2.0, 1.0), (1.0, 0.05, 1.0))])       # blocks the view to "b"
    seen = {}
    rng = np.random.default_rng(1)
    for _ in range(50):
        for o in simulate_landmarks(scene, np.array([0.0, 0.0, 1.45]), np.pi / 2, lms, rng):
            seen.setdefault(o.landmark_id, []).append((o.range, o.bearing))
    assert set(seen) == {"a"}                                          # b occluded, c out of range
    r, b = np.array(seen["a"]).mean(0)
    assert r == pytest.approx(4.0, abs=0.02) and b == pytest.approx(-np.pi / 2, abs=0.01)
