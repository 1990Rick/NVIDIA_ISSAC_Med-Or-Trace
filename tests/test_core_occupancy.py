"""3D/4D occupancy belief: carving, floor handling, ghost routing, temporal decay."""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.belief.occupancy import OccupancyBelief, binary_entropy, logit

RES = 0.1


def _occ(**kw):
    return OccupancyBelief((4.0, 4.0, 3.0), res=RES, **kw)


def _vox(p):
    return tuple(np.floor(np.asarray(p) / RES).astype(int))


def _scan(occ, origin, ends, ghost=None, dynamic=None, step=RES / 2, **kw):
    """Integrate one scan; the sampling step defaults to half a voxel so that every
    traversed voxel is visited (the stack's default 0.15 m step is tested separately)."""
    ends = np.atleast_2d(np.asarray(ends, float))
    n = len(ends)
    ghost = np.zeros(n) if ghost is None else np.asarray(ghost, float)
    dyn = np.zeros(n, bool) if dynamic is None else np.asarray(dynamic, bool)
    occ.integrate_scan(np.asarray(origin, float), ends, 1.0 - ghost, ghost, dyn, step=step, **kw)


ORIGIN = np.array([0.55, 2.05, 0.95])


def test_free_space_carving_and_endpoint_occupied():
    occ = _occ()
    end = np.array([3.05, 2.05, 0.95])
    _scan(occ, ORIGIN, end)
    ray_cells = [_vox([x, 2.05, 0.95]) for x in np.arange(0.65, 2.9, 0.1)]
    for c in ray_cells:
        assert occ.static[c] < 0 and occ.observed[c], c
    assert occ.static[_vox(end)] == pytest.approx(OccupancyBelief.L_OCC)
    # cells off the ray are untouched (unknown, p = 0.5)
    assert occ.static[_vox([1.55, 3.05, 0.95])] == 0 and not occ.observed[_vox([1.55, 3.05, 0.95])]
    assert occ.prob()[_vox(end)] > 0.6 and occ.prob()[ray_cells[3]] < 0.45
    # repeated evidence saturates at the clamp
    for _ in range(20):
        _scan(occ, ORIGIN, end)
    assert occ.static[_vox(end)] == pytest.approx(OccupancyBelief.L_MAX)
    assert occ.static[ray_cells[3]] == pytest.approx(OccupancyBelief.L_MIN)


def test_default_step_carves_only_cells_on_the_ray():
    """With the stack's default sampling step (0.15 m > voxel) carving is sparse but
    never touches voxels off the ray and never passes the end point."""
    occ = _occ()
    end = np.array([3.05, 2.05, 0.95])
    occ.integrate_scan(ORIGIN, end[None], np.ones(1), np.zeros(1), np.zeros(1, bool))
    carved = np.argwhere(occ.static < 0)
    assert len(carved) >= 0.6 * 24                        # 24 voxels strictly between origin and end
    assert np.all(carved[:, 1] == 20) and np.all(carved[:, 2] == 9)
    assert carved[:, 0].min() >= 5 and carved[:, 0].max() < 30


def test_no_return_rays_carve_up_to_carve_max():
    occ = _occ()
    occ.integrate_scan(ORIGIN, np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0, bool),
                       free_rays=np.array([[1.0, 0.0, 0.0]]), carve_max=2.0, step=RES / 2)
    assert occ.static[_vox([1.95, 2.05, 0.95])] < 0
    assert occ.static[_vox([2.85, 2.05, 0.95])] == 0      # beyond carve_max: unknown
    assert not (occ.static > 0).any()                      # no-return rays never add occupancy


def test_floor_returns_are_not_occupied():
    occ = _occ()
    floor_hit = np.array([2.55, 2.05, 0.03])
    _scan(occ, ORIGIN, floor_hit)
    assert occ.static[_vox(floor_hit)] <= 0
    assert (occ.static < 0).sum() > 5                      # but the ray still carved free space
    assert occ.column_occupancy()[_vox(floor_hit)[:2]] <= 0.5


def test_ghost_points_go_to_ambiguous_layer_and_carve_only_to_mirror():
    occ = _occ()
    ghost_pt = np.array([3.55, 2.05, 0.95])
    mirror_dist = 1.5                                       # suspected mirror at x = 2.05
    _scan(occ, ORIGIN, ghost_pt, ghost=[0.9], carve_limit=np.array([mirror_dist]))
    ij = _vox(ghost_pt)[:2]
    assert occ.ambiguous[ij] == pytest.approx(0.45)         # 0.5 * ghost probability
    assert occ.static[_vox(ghost_pt)] == pytest.approx(0.1 * OccupancyBelief.L_OCC, rel=1e-5)
    assert occ.column_occupancy()[ij] < 0.55                # weak static evidence only
    assert occ.static[_vox([1.35, 2.05, 0.95])] < 0         # before the mirror: carved
    for x in (2.25, 2.65, 3.05, 3.35):                     # behind the mirror: never carved
        assert occ.static[_vox([x, 2.05, 0.95])] == 0, x
    # the ambiguity is exposed in the uncertainty field (entropy + ambiguous mass)
    unc = occ.uncertainty_field()
    assert unc[ij] > binary_entropy(occ.column_occupancy()[ij]) + 0.4
    # a later ray looking *through* the ghost location resolves it (ambiguous mass decays)
    before = occ.ambiguous[ij]
    _scan(occ, np.array([3.55, 0.25, 0.95]), np.array([3.55, 3.85, 0.95]))
    assert occ.ambiguous[ij] == pytest.approx(0.8 * before)
    # low ghost probability does not touch the ambiguous layer
    occ2 = _occ()
    _scan(occ2, ORIGIN, ghost_pt, ghost=[0.1])
    assert occ2.ambiguous.sum() == 0


def test_dynamic_points_bypass_static_layer():
    occ = _occ()
    person = np.array([2.05, 2.05, 1.05])
    _scan(occ, ORIGIN, person, dynamic=[True])
    assert occ.static[_vox(person)] <= 0 and occ.dynamic[_vox(person)[:2]] == pytest.approx(0.6)
    occ.decay(occ.tau_d)
    assert occ.dynamic[_vox(person)[:2]] == pytest.approx(0.6 * np.exp(-1.0), rel=1e-5)


def test_temporal_decay_relaxes_to_prior(box):
    occ = _occ(tau_static=10.0)
    cart = box((2.0, 2.0, 0.5), (0.3, 0.3, 0.5))
    occ.set_prior_from_boxes([cart])
    c_in, c_out = _vox([2.05, 2.05, 0.55]), _vox([1.05, 1.05, 0.55])
    assert occ.prior[c_in] == pytest.approx(1.2) and occ.prior[c_out] == pytest.approx(-1.5)
    # the cart was moved: repeated rays carve its old location free
    for _ in range(10):
        _scan(occ, np.array([0.55, 2.05, 0.55]), np.array([3.55, 2.05, 0.55]))
    dev0 = occ.static[c_in] - occ.prior[c_in]
    assert occ.static[c_in] < 0 and dev0 < -1.0
    occ.decay(10.0)                                           # one time constant
    assert occ.static[c_in] - occ.prior[c_in] == pytest.approx(dev0 * np.exp(-1.0), rel=1e-4)
    occ.decay(200.0)
    assert np.allclose(occ.static, occ.prior, atol=1e-6)      # unobserved evidence ages back to the map
    occ.ambiguous[:] = 1.0
    occ.decay(occ.tau_a)
    assert np.allclose(occ.ambiguous, np.exp(-1.0))


def test_blocked_segments_and_column_occupancy(box):
    occ = _occ()
    occ.set_prior_from_boxes([box((2.0, 2.0, 0.5), (0.3, 0.3, 0.5))])
    a = np.array([[0.5, 2.0, 0.5], [0.5, 2.0, 1.5], [0.5, 1.0, 0.5]])
    b = np.array([[3.5, 2.0, 0.5], [3.5, 2.0, 1.5], [3.5, 1.0, 0.5]])
    assert occ.blocked_segments(a, b).tolist() == [True, False, False]
    col = occ.column_occupancy()
    assert col[_vox([2.0, 2.0, 0])[:2]] > 0.7 and col[_vox([1.0, 1.0, 0])[:2]] < 0.2
    # deterministic mode: hard 0/1 map and zero uncertainty (ablation)
    det = _occ(deterministic=True)
    det.set_prior_from_boxes([box((2.0, 2.0, 0.5), (0.3, 0.3, 0.5))])
    assert set(np.unique(det.prob())) <= {0.0, 1.0} and not det.uncertainty_field().any()


def test_uncertainty_field_and_unknown_fraction():
    occ = _occ()
    assert np.allclose(occ.uncertainty_field(), 1.0)          # p = 0.5 everywhere: 1 bit
    assert occ.unknown_fraction() == 1.0
    # one horizontal ray lowers entropy of its voxels, but the column (max over the
    # robot's height band) stays uncertain until the whole band has been seen free
    _scan(occ, ORIGIN, np.array([3.05, 2.05, 0.95]))
    ij = _vox([1.55, 2.05, 0])[:2]
    assert occ.uncertainty_field()[ij] == pytest.approx(1.0)
    for z in np.arange(0.15, 1.65, 0.1):
        _scan(occ, [0.55, 2.05, z], np.array([3.05, 2.05, z]))
    u = occ.uncertainty_field()
    assert u[ij] < 0.95 and u[_vox([1.55, 0.55, 0])[:2]] == pytest.approx(1.0)
    assert occ.unknown_fraction() < 1.0
    assert logit(0.5) == 0 and binary_entropy(np.array([0.5]))[0] == pytest.approx(1.0)
