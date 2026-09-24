"""Regression tests for the mechanisms added after the counterfactual checks and the
adversarial review: path geometry for MPPI, the conservative costmap EDT, the
map-change evidence gates (observed absence + outline registration), the NBV
anticipation and traffic terms, the staff-traffic map and the retreat guard."""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.autonomy.stack import AutonomyStack, stack_inputs_from_episode
from medortrace.belief.occupancy import OccupancyBelief
from medortrace.belief.world_model import ChangeDiagnoser
from medortrace.common.config import load_config
from medortrace.common.geometry import OrientedBox
from medortrace.planning.costmap import Costmap
from medortrace.planning.mpc import lookahead_point, polyline_projection
from medortrace.sim.episode import build_episode
from medortrace.world.scene import SceneObject, SterileZone


# ---------------------------------------------------------------------------- path geometry
def test_polyline_projection_and_lookahead():
    path = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    d, rem = polyline_projection(np.array([[1.0, 0.5], [2.5, 1.0], [3.0, 3.0], [-1.0, 0.0]]), path)
    assert np.allclose(d, [0.5, 0.5, np.sqrt(2), 1.0])
    assert np.allclose(rem, [3.0, 1.0, 0.0, 4.0])                      # arc length left from the projection
    assert np.allclose(lookahead_point(np.array([1.0, 0.2]), path, 0.8), [1.8, 0.0])
    assert np.allclose(lookahead_point(np.array([1.9, 0.0]), path, 0.8), [2.0, 0.7])   # around the corner
    assert np.allclose(lookahead_point(np.array([2.0, 1.9]), path, 0.8), [2.0, 2.0])   # clamps at the goal


# ---------------------------------------------------------------------------- costmap
def test_costmap_edt_is_conservative_and_interpolated():
    occ = OccupancyBelief((4.0, 4.0, 3.0), res=0.1)
    occ.set_prior_from_boxes([OrientedBox(np.array([2.05, 2.05, 0.5]), np.array([0.04, 0.04, 0.5]))])
    cm = Costmap(occ, [], robot_radius=0.28)
    # a single occupied cell [2.0, 2.1)^2: the EDT is measured to its boundary, never beyond
    for x in (2.3, 2.37, 2.45, 2.62):
        true_clearance = x - 2.1
        assert cm.lookup(np.array([[x, 2.05]]), "edt")[0] <= true_clearance + 1e-9
        assert cm.lookup(np.array([[x, 2.05]]), "edt")[0] >= true_clearance - 0.051
    # bilinear: continuous between cell centres
    e = cm.lookup(np.array([[2.35, 2.05], [2.40, 2.05], [2.45, 2.05]]), "edt")
    assert e[0] < e[1] < e[2]


# ---------------------------------------------------------------------------- map change evidence
def _cart():
    return SceneObject("cart_1", "cart", OrientedBox(np.array([2.0, 2.5, 0.5]), np.array([0.35, 0.25, 0.5])),
                       "stainless_steel_brushed", "instrument_cart", movable=True)


def test_outline_registration_recovers_a_move_and_rejects_scatter(rng):
    c = np.array([2.55, 2.15])                                           # true shift (0.55, -0.35)
    face1 = np.c_[np.linspace(c[0] - 0.35, c[0] + 0.35, 30), np.full(30, c[1] - 0.25)]
    face2 = np.c_[np.full(20, c[0] + 0.35), np.linspace(c[1] - 0.25, c[1] + 0.25, 20)]
    d = ChangeDiagnoser([_cart()])
    d.res_buf.append(np.vstack([face1, face2]) + rng.normal(0, 0.02, (50, 2)))
    shift, support, n_on = d.fit_object_shift("cart_1")
    assert np.allclose(shift, [0.55, -0.35], atol=0.06) and support > 0.8 and n_on >= 40
    d2 = ChangeDiagnoser([_cart()])
    d2.res_buf.append(rng.uniform([1.2, 1.7], [2.8, 3.3], (50, 2)))    # ghosts / clutter near the cart
    _, support2, _ = d2.fit_object_shift("cart_1")
    assert support2 < d2.min_outline_support


def test_vacated_fraction_needs_rays_through_the_footprint():
    cart = _cart()
    occ = OccupancyBelief((5.0, 5.0, 3.0), res=0.1)
    occ.set_prior_from_boxes([cart.box])
    d = ChangeDiagnoser([cart])
    assert d.vacated_fraction("cart_1", occ) == 0.0                     # never looked through it
    # rays at 0.5 m height passing straight through the surveyed footprint (the cart has gone)
    ends = np.array([[3.5, y, 0.5] for y in np.linspace(2.3, 2.7, 9)])
    for _ in range(6):
        occ.integrate_scan(np.array([0.2, 2.5, 0.5]), ends, np.ones(len(ends)), np.zeros(len(ends)),
                           np.zeros(len(ends), bool), step=0.05)
    assert d.vacated_fraction("cart_1", occ) >= d.min_vacated


# ---------------------------------------------------------------------------- stack-level mechanisms
@pytest.fixture(scope="module")
def stack():
    cfg = load_config("scenarios/nominal.yaml")
    ep = build_episode(cfg, 7)
    st = AutonomyStack(stack_inputs_from_episode(ep), cfg)
    st.cm = Costmap(st.occ, st.inp.zones, st.radius)
    return st


def test_traffic_map_accumulates_and_decays(stack):
    stack.traffic[:] = 0.0
    p = np.array([[1.0, 1.0]])
    for _ in range(600):                                                 # a person standing for 60 s
        stack._update_traffic(list(p), 0.1)
    c = stack.occ.grid2d.world_to_cell(p)[0]
    assert 0.3 < stack.traffic[c[0], c[1]] <= 1.0
    far = stack.occ.grid2d.world_to_cell(np.array([[3.0, 3.0]]))[0]
    assert stack.traffic[far[0], far[1]] == 0.0
    before = stack.traffic[c[0], c[1]]
    for _ in range(600):
        stack._update_traffic([], 0.1)
    assert stack.traffic[c[0], c[1]] < 0.7 * before                     # decays with traffic_tau


def test_activity_prior_and_log_events(stack):
    rates = stack.activity_rates()
    field = [s.id for s in stack.inp.slots if s.sterile and s.kind == "surface"]
    assert field and all(rates.get(sid, 0.0) >= 0.5 for sid in field)
    assert all(not sid.startswith("hand:") for sid in rates)


def test_nbv_watch_term_prefers_views_of_active_slots(stack):
    st = stack
    pose = np.array([*st.inp.start_pose[:2], st.inp.start_pose[2]])
    rng = np.random.default_rng(0)
    g0 = st.nbv.plan(pose, st.items, st.occ, st.cm, np.zeros((0, 2)), {}, {}, rng, activity={})
    rng = np.random.default_rng(0)
    g1 = st.nbv.plan(pose, st.items, st.occ, st.cm, np.zeros((0, 2)), {}, {}, rng,
                     activity=st.activity_rates())
    assert g1 is not None and g1.breakdown["watch"] > 0.0
    assert g0 is None or g0.breakdown["watch"] == 0.0


def test_nbv_traffic_penalty_avoids_walkways(stack):
    st = stack
    pose = np.array([*st.inp.start_pose[:2], st.inp.start_pose[2]])
    rng = np.random.default_rng(1)
    g = st.nbv.plan(pose, st.items, st.occ, st.cm, np.zeros((0, 2)), {}, {}, rng, activity=st.activity_rates())
    traffic = np.zeros_like(st.traffic)
    c = st.occ.grid2d.world_to_cell(g.pose[None, :2])[0]
    traffic[max(0, c[0] - 6):c[0] + 7, max(0, c[1] - 6):c[1] + 7] = 1.0   # the chosen spot is a walkway
    rng = np.random.default_rng(1)
    g2 = st.nbv.plan(pose, st.items, st.occ, st.cm, np.zeros((0, 2)), {}, {}, rng, activity=st.activity_rates(),
                     traffic=traffic)
    assert g2.breakdown["traffic"] < 1.0 and np.linalg.norm(g2.pose[:2] - g.pose[:2]) > 0.3


def test_retreat_never_enters_the_sterile_keepout(stack):
    st = stack
    z = st.inp.zones[0]
    # robot just outside the keep-out, facing it, person right behind it
    edge = z.box.center[:2] + np.array([z.box.half[0] + z.keepout_margin + 0.2, 0.0])
    pose = np.array([edge[0], edge[1], np.pi])                          # facing the sterile field

    class _T:                                                            # a person-like track behind
        person_like = True
        x = np.array([edge[0] + 0.6, edge[1], 0.0, 0.0])

    v, w = st._retreat_cmd(pose, [_T()])
    ahead = pose[:2] + 0.3 * np.array([np.cos(pose[2]), np.sin(pose[2])])
    assert bool(st.cm.lookup(ahead[None], "keepout")[0])
    assert v <= 0.0                                                      # does not drive into the keep-out


def test_sterile_zone_helper_is_consistent():
    z = SterileZone("f", OrientedBox(np.array([2.0, 2.0, 0.5]), np.array([1.0, 0.5, 0.5])), keepout_margin=0.3)
    assert z.box.contains_xy(np.array([[3.2, 2.0]]), margin=0.3)[0]
    assert not z.box.contains_xy(np.array([[3.4, 2.0]]), margin=0.3)[0]
