"""Geometry helpers and the analytic ray caster vs closed-form answers.

Every expected value below is derived by hand (slab / quadratic / plane
intersection), so a regression in the vectorised caster shows up as a numeric
mismatch rather than a changed-but-plausible number.
"""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.common.geometry import OrientedBox, Polygon2D, Pose2D, rot2, segment_point_distance, wrap_angle
from medortrace.sim.raycast import CEIL_ID, FLOOR_ID, NO_HIT, cast, segment_occluded, segments_blocked


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# geometry.py
# ---------------------------------------------------------------------------
def test_wrap_angle_range_and_equivalence():
    a = np.linspace(-20.0, 20.0, 401)
    w = wrap_angle(a)
    assert np.all(w >= -np.pi - 1e-12) and np.all(w < np.pi + 1e-12)
    # same direction on the unit circle
    assert np.allclose(np.cos(w), np.cos(a)) and np.allclose(np.sin(w), np.sin(a))
    assert wrap_angle(3 * np.pi / 2) == pytest.approx(-np.pi / 2)


def test_pose2d_transform_roundtrip_and_known_point():
    p = Pose2D(1.0, 2.0, np.pi / 2)
    local = np.array([[1.0, 0.0, 0.5], [0.0, 2.0, 0.0]])
    world = p.transform_points(local)
    # x-forward of a robot facing +y at (1,2) is (1,3); z untouched
    assert np.allclose(world[0], [1.0, 3.0, 0.5])
    assert np.allclose(world[1], [-1.0, 2.0, 0.0])
    assert np.allclose(p.inverse_transform_points(world), local)
    assert Pose2D.from_array([0, 0, 7.0]).theta == pytest.approx(wrap_angle(7.0))
    assert p.distance_to((4.0, 6.0)) == pytest.approx(np.hypot(3.0, 4.0))


def test_oriented_box_contains_and_distance(box):
    b = box((2.0, 1.0, 0.5), (1.0, 0.5, 0.5), np.pi / 2)   # rotated: spans x in [1.5,2.5], y in [0,2]
    assert b.contains_xy(np.array([[2.4, 1.9], [1.6, 0.1]])).all()
    assert not b.contains_xy(np.array([[2.9, 1.0]]))[0]
    assert b.contains_xy(np.array([[2.9, 1.0]]), margin=0.41)[0]
    assert b.distance_xy(np.array([[3.5, 1.0]]))[0] == pytest.approx(1.0)
    assert b.distance_xy(np.array([[3.5, 3.0]]))[0] == pytest.approx(np.hypot(1.0, 1.0))
    assert b.distance_xy(np.array([[2.0, 1.0]]))[0] == 0.0
    c = b.corners_xy()
    assert np.allclose(np.sort(c[:, 0]), [1.5, 1.5, 2.5, 2.5]) and np.allclose(np.sort(c[:, 1]), [0, 0, 2, 2])
    assert (b.z_min, b.z_max) == (0.0, 1.0)


def test_polygon_contains_even_odd_and_segment_distance():
    square = Polygon2D(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float))
    pts = np.array([[1, 1], [3, 1], [-0.1, 1], [1.9, 1.9]], float)
    assert square.contains(pts).tolist() == [True, False, False, True]
    pts = np.array([[1.0, 1.0], [3.0, 0.0], [-1.0, -1.0]])
    d = segment_point_distance(np.array([0.0, 0.0]), np.array([2.0, 0.0]), pts)
    assert np.allclose(d, [1.0, 1.0, np.sqrt(2.0)])
    assert np.allclose(rot2(0.3) @ rot2(-0.3), np.eye(2))


# ---------------------------------------------------------------------------
# raycast.cast
# ---------------------------------------------------------------------------
def test_axis_aligned_box_hit_distance_and_normal(make_scene, box):
    sc = make_scene([box((5.0, 0.0, 1.0), (1.0, 1.0, 1.0))])
    h = cast(sc, np.array([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]]), np.array([[1.0, 0, 0], [-1.0, 0, 0]]))
    assert np.allclose(h.t, [4.0, 4.0])
    assert h.obj.tolist() == [0, 0]
    assert np.allclose(h.normal, [[-1, 0, 0], [1, 0, 0]])


def test_yawed_box_hit_and_world_normal(make_scene, box):
    # 4 x 1 box rotated 90 deg: along world x it is only 1 m deep (x in [4.5, 5.5])
    sc = make_scene([box((5.0, 0.0, 1.0), (2.0, 0.5, 1.0), np.pi / 2)])
    h = cast(sc, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    assert h.t[0] == pytest.approx(4.5)
    assert np.allclose(h.normal[0], [-1.0, 0.0, 0.0], atol=1e-9)
    # top face from above
    h = cast(sc, np.array([[5.0, 1.0, 3.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert h.t[0] == pytest.approx(1.0) and np.allclose(h.normal[0], [0, 0, 1])


def test_box_hits_lie_on_surface_with_outward_normals(make_scene, box, rng):
    """Property test on an arbitrarily yawed box: every hit point is on the box
    surface, the normal is the rotated face axis and faces the incoming ray."""
    b = box((3.0, 2.0, 0.8), (0.7, 0.3, 0.6), 0.61)
    sc = make_scene([b], ceiling=50.0)
    orig = np.repeat(np.array([[0.0, 0.0, 0.9]]), 4000, 0)
    target = b.center + rng.uniform(-1, 1, (4000, 3)) * b.half * 1.3
    D = target - orig
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    h = cast(sc, orig, D, t_max=100.0)
    on_box = h.obj == 0
    assert on_box.sum() > 1000
    P = orig[on_box] + h.t[on_box, None] * D[on_box]
    local = P - b.center
    local[:, :2] = local[:, :2] @ rot2(b.yaw)
    ratio = np.abs(local) / b.half
    assert np.allclose(ratio.max(axis=1), 1.0, atol=1e-6)            # on the surface
    assert np.all(ratio <= 1.0 + 1e-6)
    n = h.normal[on_box]
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    assert np.all((n * D[on_box]).sum(1) < 0)                          # faces the ray
    face = ratio.argmax(axis=1)
    n_local = n.copy()
    n_local[:, :2] = n[:, :2] @ rot2(b.yaw)
    expect = np.zeros_like(n_local)
    expect[np.arange(len(face)), face] = np.sign(local[np.arange(len(face)), face])
    assert np.allclose(n_local, expect, atol=1e-9)
    # rays aimed outside the (1.3x) box that miss it hit floor/ceiling instead
    assert set(np.unique(h.obj[~on_box])) <= {FLOOR_ID, CEIL_ID, NO_HIT}


def test_cylinder_hits_and_normals(make_scene):
    sc = make_scene(cylinders=[(3.0, 0.0, 0.5, 1.75)])
    orig = np.array([[0.0, 0.0, 1.0], [0.0, 0.3, 1.0], [0.0, 0.0, 2.0], [0.0, 0.6, 1.0]])
    D = np.array([[1.0, 0, 0]] * 4)
    h = cast(sc, orig, D)
    # head-on
    assert h.t[0] == pytest.approx(2.5) and h.obj[0] == 0
    assert np.allclose(h.normal[0], [-1, 0, 0])
    # offset chord: x = 3 - sqrt(r^2 - y^2) = 2.6, normal (-0.4, 0.3)/0.5
    assert h.t[1] == pytest.approx(2.6) and np.allclose(h.normal[1], [-0.8, 0.6, 0.0])
    # above the cylinder top (2.0 > 1.75) and passing beside it: no hit at all
    assert h.obj[2] == NO_HIT and np.isinf(h.t[2])
    assert h.obj[3] == NO_HIT
    # cylinder indices are offset by the number of boxes
    sc2 = make_scene([OrientedBox(np.array([0.0, 5.0, 1.0]), np.array([0.2, 0.2, 1.0]))], [(3.0, 0.0, 0.5, 1.75)])
    assert cast(sc2, orig[:1], D[:1]).obj[0] == 1


def test_floor_and_ceiling_planes(make_scene):
    sc = make_scene(ceiling=3.0)
    orig = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    D = np.array([_unit([1, 0, -1]), _unit([0, 1, 1]), [1.0, 0.0, 0.0]])
    h = cast(sc, orig, D)
    assert h.obj.tolist() == [FLOOR_ID, CEIL_ID, NO_HIT]
    assert h.t[0] == pytest.approx(np.sqrt(2.0)) and h.t[1] == pytest.approx(2 * np.sqrt(2.0))
    assert np.allclose(h.normal[0], [0, 0, 1]) and np.allclose(h.normal[1], [0, 0, -1])


def test_nearest_hit_exclude_mask_and_t_max(make_scene, box):
    sc = make_scene([box((6.0, 0, 1), (0.5, 0.5, 1)), box((3.0, 0, 1), (0.5, 0.5, 1))], [(9.0, 0.0, 0.3, 1.75)])
    orig, D = np.array([[0.0, 0, 1.0]]), np.array([[1.0, 0, 0]])
    h = cast(sc, orig, D)
    assert h.obj[0] == 1 and h.t[0] == pytest.approx(2.5)
    h = cast(sc, orig, D, exclude=np.array([False, True, False]))
    assert h.obj[0] == 0 and h.t[0] == pytest.approx(5.5)
    h = cast(sc, orig, D, exclude=np.array([True, True, False]))
    assert h.obj[0] == 2 and h.t[0] == pytest.approx(8.7)
    h = cast(sc, orig, D, t_max=2.0)
    assert h.obj[0] == NO_HIT and np.isinf(h.t[0])
    assert len(cast(sc, np.zeros((0, 3)), np.zeros((0, 3))).t) == 0


# ---------------------------------------------------------------------------
# occlusion queries
# ---------------------------------------------------------------------------
def test_segment_occluded_ignores_floor(make_scene, box):
    sc = make_scene([box((2.0, 0, 0.5), (0.2, 1.0, 0.5))])
    a = np.array([[0.0, 0, 0.8], [0.0, 0, 1.5], [0.0, 3.0, 1.0]])
    b = np.array([[4.0, 0, 0.8], [4.0, 0, 1.5], [4.0, 3.0, 0.0]])   # third ends *on* the floor
    assert segment_occluded(sc, a, b).tolist() == [True, False, False]


def test_segments_blocked_own_object_logic(make_scene, box):
    """An item resting in a recess of its own table is visible; the same geometry
    without the ``own`` hint, or with a person in between, is occluded."""
    table = box((2.0, 0.0, 0.45), (0.5, 0.5, 0.45))            # top at z = 0.9
    sc = make_scene([table], [(1.0, 0.0, 0.25, 1.75)])           # person at x = 1
    sc_free = make_scene([table])
    cam = np.array([[0.0, 0.0, 1.5]])
    item = np.array([[2.3, 0.0, 0.85]])                          # 5 cm below the table top
    # geometric sanity: the ray enters the table top ~0.18 m before the item
    L = np.linalg.norm(item - cam)
    h = cast(sc_free, cam, (item - cam) / L)
    assert h.obj[0] == 0 and 0.1 < L - h.t[0] < 0.3
    assert segments_blocked(sc_free, cam, item).tolist() == [True]
    assert segments_blocked(sc_free, cam, item, own=np.array([0])).tolist() == [False]
    # own hint does not excuse a *different* object (the person) in between
    assert segments_blocked(sc, cam, item, own=np.array([0])).tolist() == [True]
    # own object hit far before the end point (> own_tol) still blocks
    far_side = np.array([[4.0, 0.0, 0.5]])                       # behind the table, low
    assert segments_blocked(sc_free, cam, far_side, own=np.array([0])).tolist() == [True]
    # batched with per-segment own indices, global exclude makes the person transparent
    a = np.repeat(cam, 2, 0)
    b = np.vstack([item, item])
    own = np.array([0, -99])
    assert segments_blocked(sc_free, a, b, own=own).tolist() == [False, True]
    assert segments_blocked(sc, a, b, own=own, exclude=np.array([False, True])).tolist() == [False, True]
    assert segments_blocked(sc, np.zeros((0, 3)), np.zeros((0, 3))).shape == (0,)
