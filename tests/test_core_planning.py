"""Planning: risk-aware MPPI (closed loop on a unicycle), grid planners and the
next-best-view information gain."""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.autonomy.stack import AutonomyStack, stack_inputs_from_episode
from medortrace.belief.occupancy import OccupancyBelief
from medortrace.common.config import load_config
from medortrace.planning.costmap import Costmap
from medortrace.planning.grid import GridSpec, astar, dijkstra_field, extract_path, nearest_free, rasterize_boxes
from medortrace.planning.mpc import MppiController
from medortrace.planning.nbv import _entropy, expected_info_gain
from medortrace.sim.episode import build_episode
from medortrace.world.scene import SterileZone

R_ROBOT = 0.28


def _costmap(boxes=(), zones=(), room=(6.0, 4.0, 3.0)):
    occ = OccupancyBelief(room)
    occ.set_prior_from_boxes(list(boxes))
    return Costmap(occ, list(zones), R_ROBOT)


def _closed_loop(cm, x0, path, humans=None, steps=120, seed=0):
    """Integrate the MPC output on the nominal unicycle model."""
    mpc = MppiController()
    rng = np.random.default_rng(seed)
    x = np.array(x0, float)
    hs = np.zeros((0, 12, mpc.H, 2))
    if humans is not None:
        hs = np.broadcast_to(np.asarray(humans, float)[:, None, None, :], (len(humans), 12, mpc.H, 2)).copy()
    traj, res = [x.copy()], []
    for _ in range(steps):
        r = mpc.solve(x, path, cm, hs, R_ROBOT, rng)
        res.append(r)
        x = x + np.array([r.v * np.cos(x[2]) * mpc.dt, r.v * np.sin(x[2]) * mpc.dt, r.omega * mpc.dt])
        traj.append(x.copy())
        if np.linalg.norm(x[:2] - path[-1]) < 0.15:
            break
    return np.array(traj), res


STRAIGHT = np.linspace([1.0, 2.0], [4.5, 2.0], 15)


# ---------------------------------------------------------------------------
# MPPI
# ---------------------------------------------------------------------------
def test_mpc_reaches_goal_in_free_space():
    cm = _costmap()
    traj, res = _closed_loop(_costmap(), [1.0, 2.0, 0.0], STRAIGHT)
    assert np.linalg.norm(traj[-1, :2] - STRAIGHT[-1]) < 0.2
    assert len(traj) < 110                                        # 3.5 m at <= 0.7 m/s plus acceleration
    assert all(r.feasible for r in res) and max(r.v for r in res) <= 0.7 + 1e-9
    assert np.all(cm.lookup(traj[:, :2], "edt") > R_ROBOT)
    assert np.max(np.abs(traj[:, 1] - 2.0)) < 0.3                 # tracks the reference line


def test_mpc_never_enters_keepout():
    zone = SterileZone("field", _box((2.75, 2.0, 0.0), (0.4, 0.4, 0.0)), 0.3)
    cm = _costmap(zones=[zone])
    # (a) reference path straight through the sterile field: the robot must stop short of it
    traj, _ = _closed_loop(cm, [1.0, 2.0, 0.0], STRAIGHT, steps=100)
    assert not cm.lookup(traj[:, :2], "keepout").any()
    assert traj[:, 0].max() < 2.75 - 0.4 - 0.3
    # (b) with a planned detour it reaches the goal, still never entering the keep-out
    s = nearest_free(cm.lethal, tuple(cm.grid.world_to_cell(np.array([[1.0, 2.0]]))[0]))
    g = nearest_free(cm.lethal, tuple(cm.grid.world_to_cell(np.array([[4.5, 2.0]]))[0]))
    path = cm.grid.cell_to_world(np.array(astar(cm.soft, cm.lethal, s, g)))
    traj, _ = _closed_loop(cm, [1.0, 2.0, 0.0], path, steps=250)
    assert not cm.lookup(traj[:, :2], "keepout").any()
    assert np.linalg.norm(traj[-1, :2] - path[-1]) < 0.2
    # keep-out is enforced on the 0.1 m costmap grid: continuous intrusion into the 0.3 m
    # margin is bounded by half a cell diagonal and the sterile zone itself is never touched
    assert not zone.box.contains_xy(traj[:, :2]).any()
    assert np.max(zone.keepout_margin - zone.box.distance_xy(traj[:, :2])) < 0.5 * np.sqrt(2) * cm.grid.res


def test_mpc_stops_when_a_human_blocks_the_corridor():
    walls = [_box((3.0, 0.7, 0.5), (3.0, 0.7, 0.5)), _box((3.0, 3.3, 0.5), (3.0, 0.7, 0.5))]   # 1.2 m corridor
    cm = _costmap(walls)
    free, _ = _closed_loop(cm, [1.0, 2.0, 0.0], STRAIGHT, steps=120)
    assert np.linalg.norm(free[-1, :2] - STRAIGHT[-1]) < 0.2       # corridor itself is passable
    human = np.array([3.0, 2.0])
    traj, res = _closed_loop(cm, [1.0, 2.0, 0.0], STRAIGHT, humans=human[None], steps=100)
    d = np.linalg.norm(traj[:, :2] - human, axis=1)
    assert traj[:, 0].max() < human[0] - 1.0                      # never squeezes past or up to the person
    assert d.min() - R_ROBOT - 0.3 > 0.5                          # keeps a clear surface-to-surface gap
    assert np.mean(np.abs([r.v for r in res[-20:]])) < 0.1        # it has come to a stop
    assert all(r.feasible for r in res)


def test_mpc_reports_predicted_collision_risk():
    cm = _costmap()
    mpc = MppiController()
    x0 = np.array([1.0, 2.0, 0.0])
    near = np.broadcast_to(np.array([1.5, 2.0])[None, None, None], (1, 12, mpc.H, 2)).copy()
    far = near + np.array([3.0, 1.5])
    r_near = mpc.solve(x0, STRAIGHT, cm, near, R_ROBOT, np.random.default_rng(0))
    mpc.reset()
    r_far = mpc.solve(x0, STRAIGHT, cm, far, R_ROBOT, np.random.default_rng(0))
    assert r_near.min_pred_clearance < 0.2 and r_near.cvar > r_far.cvar
    assert r_far.min_pred_clearance > 1.0 and r_far.collision_prob == 0.0
    r_none = MppiController().solve(x0, STRAIGHT, cm, np.zeros((0, 12, 20, 2)), R_ROBOT, np.random.default_rng(0))
    assert r_none.min_pred_clearance == 10.0 and r_none.cvar == 0.0


def _box(c, h, yaw=0.0):
    from medortrace.common.geometry import OrientedBox
    return OrientedBox(np.array(c, float), np.array(h, float), yaw)


# ---------------------------------------------------------------------------
# grid planners / costmap
# ---------------------------------------------------------------------------
def test_astar_and_dijkstra_agree_and_avoid_lethal():
    g = GridSpec(np.zeros(2), 0.1, (40, 30))
    lethal = rasterize_boxes(g, [_box((2.0, 1.2, 0.5), (0.1, 1.2, 0.5))], inflate=0.1)    # wall with a gap on top
    cost = np.zeros(lethal.shape)
    s, t = (5, 10), (35, 10)
    path = astar(cost, lethal, s, t)
    assert path[0] == s and path[-1] == t and not any(lethal[c] for c in path)
    dist, parent = dijkstra_field(cost, lethal, s)
    assert extract_path(parent, t)[0] == s
    steps = np.diff(np.array(path), axis=0)
    length = np.sum(np.where(np.abs(steps).sum(1) == 2, 1.4142, 1.0))
    assert length == pytest.approx(dist[t], rel=1e-6)            # both optimal on the same graph
    assert astar(cost, lethal, s, (20, 5)) is None               # goal inside the wall
    assert nearest_free(lethal, (20, 5)) is not None and not lethal[nearest_free(lethal, (20, 5))]


def test_costmap_layers():
    zone = SterileZone("field", _box((3.0, 2.0, 0.0), (0.5, 0.5, 0.0)), 0.3)
    cm = _costmap([_box((1.0, 1.0, 0.5), (0.3, 0.3, 0.5))], [zone])
    assert cm.lookup(np.array([[1.0, 1.0]]), "lethal")[0] and cm.lookup(np.array([[1.0, 1.0]]), "edt")[0] == 0
    assert cm.lookup(np.array([[3.75, 2.0]]), "keepout")[0] and not cm.lookup(np.array([[3.9, 2.0]]), "keepout")[0]
    assert cm.lookup(np.array([[1.75, 1.0]]), "edt")[0] == pytest.approx(0.4, abs=0.11)
    assert cm.lookup(np.array([[-1.0, 1.0]]), "lethal")[0]      # outside the map is lethal
    assert cm.lookup(np.array([[-1.0, 1.0]]), "edt")[0] == 0.0


# ---------------------------------------------------------------------------
# NBV
# ---------------------------------------------------------------------------
def test_eig_properties(rng):
    B = rng.dirichlet(np.ones(6), size=5)
    pd = rng.uniform(0, 1, (5, 6)) * (rng.random((5, 6)) < 0.5)
    g = expected_info_gain(B, pd)
    assert g.shape == (5,) and np.all(g >= 0)
    n_vis = ((pd > 0.03).any(axis=0)).sum()
    assert np.all(g <= _entropy(B) * n_vis + 1e-9)               # each slot term is at most H(b)
    assert np.all(expected_info_gain(B, np.zeros_like(pd)) == 0)  # nothing visible: no information
    assert np.all(expected_info_gain(B, np.full_like(pd, 0.02)) == 0)
    # perfect binary sensor on a 50/50 belief resolves exactly one bit
    assert expected_info_gain(np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]]), fp=0.0)[0] == pytest.approx(1.0)
    # uncertain beliefs gain more than confident ones under the same view
    pd1 = np.zeros((2, 4))
    pd1[:, 0] = 0.8
    confident = np.array([0.97, 0.01, 0.01, 0.01])
    uncertain = np.full(4, 0.25)
    gc, gu = expected_info_gain(np.vstack([confident, uncertain]), pd1)
    assert gu > gc > 0
    # better detection -> more information
    gains = [expected_info_gain(uncertain[None], np.array([[p, 0, 0, 0]]))[0] for p in (0.2, 0.5, 0.9)]
    assert gains[0] < gains[1] < gains[2]


def test_nbv_plan_returns_admissible_view():
    cfg = load_config()
    ep = build_episode(cfg, 17)
    stack = AutonomyStack(stack_inputs_from_episode(ep), cfg)
    stack.cm = Costmap(stack.occ, stack.inp.zones, stack.radius)
    pose = stack.inp.start_pose
    goal = stack.nbv.plan(pose, stack.items, stack.occ, stack.cm, np.zeros((0, 2)), {}, {},
                          np.random.default_rng(0))
    assert goal is not None
    p = goal.pose[None, :2]
    assert not stack.cm.lookup(p, "lethal")[0] and not stack.cm.lookup(p, "keepout")[0]
    assert np.allclose(goal.path[0], pose[:2]) and np.allclose(goal.path[-1], goal.pose[:2])
    cells = stack.cm.grid.world_to_cell(goal.path)
    assert not stack.cm.keepout[cells[:, 0], cells[:, 1]].any()
    assert set(goal.breakdown) >= {"eig", "modal", "unc", "path", "risk", "sterile"}
    assert goal.breakdown["path"] >= 0 and np.isfinite(goal.score)
