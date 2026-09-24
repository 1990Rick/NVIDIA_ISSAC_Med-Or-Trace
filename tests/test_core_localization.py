"""EKF localisation consistency and rigid scan-to-map alignment (CF-D machinery)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from medortrace.belief.localization import EkfLocalizer
from medortrace.belief.world_model import ChangeDiagnoser, scan_match
from medortrace.common.geometry import OrientedBox, rot2, wrap_angle
from medortrace.common.msgs import Header, ImuSample, LandmarkFrame, LandmarkObservation, WheelOdometry
from medortrace.planning.grid import GridSpec, rasterize_boxes
from medortrace.world.scene import SceneObject, Slot

LANDMARKS = {f"tag_{i}": np.array([*p, 1.6]) for i, p in
             enumerate([(0, 1), (0, 5), (4, 6), (8, 5), (8, 1), (4, 0)])}
CHI2_3DOF_99 = 11.34


def _straight_run(seed: int, landmarks: bool = True, T: float = 20.0, v: float = 0.3):
    rng = np.random.default_rng(seed)
    dt = 0.1
    x = np.array([1.0, 3.0, 0.0])
    ekf = EkfLocalizer(x, LANDMARKS)
    nees, std = [], []
    for k in range(int(T / dt)):
        t = (k + 1) * dt
        x = x + np.array([v * dt * np.cos(x[2]), v * dt * np.sin(x[2]), 0.0])
        ekf.predict(WheelOdometry(Header(t, t, "base_link"), v + rng.normal(0, 0.01), rng.normal(0, 0.005)), [], dt)
        if landmarks and k % 2 == 0:
            obs = []
            for lid, p in LANDMARKS.items():
                d = p[:2] - x[:2]
                obs.append(LandmarkObservation(lid, float(np.hypot(*d) + rng.normal(0, 0.03)),
                                               float(wrap_angle(np.arctan2(d[1], d[0]) - x[2]) + rng.normal(0, 0.01))))
            ekf.update_landmarks(LandmarkFrame(Header(t, t, "camera_link"), obs), t)
        e = ekf.x - x
        e[2] = wrap_angle(e[2])
        nees.append(float(e @ np.linalg.solve(ekf.P, e)))
        std.append(ekf.pos_std)
    return ekf, x, np.array(nees), np.array(std)


@pytest.mark.parametrize("seed", [0, 1])
def test_ekf_consistent_on_straight_run_with_landmarks(seed):
    ekf, x, nees, std = _straight_run(seed)
    assert np.linalg.norm(ekf.x[:2] - x[:2]) < 0.05
    assert np.linalg.norm(ekf.x[:2] - x[:2]) < 3 * ekf.pos_std + 0.01
    # consistent (or conservative): mean NEES <= dim, and rarely above the 99% chi-square bound
    assert nees.mean() < 3.0 and (nees > CHI2_3DOF_99).mean() < 0.05
    # innovation statistics: E[NIS] = 2 for a matched filter; the filter's R is conservative
    assert 0.2 < ekf.nis_avg < 3.0
    assert ekf.rejection_rate() == 0.0 and ekf.accepted > 100
    assert std[-1] < 0.03 and ekf.heading_std < 0.02


def test_ekf_without_landmarks_grows_uncertainty():
    ekf_lm, *_ = _straight_run(0, landmarks=True)
    ekf_dr, x, nees, std = _straight_run(0, landmarks=False)
    assert np.all(np.diff(std) > -1e-12)                          # dead reckoning: monotone growth
    assert ekf_dr.pos_std > 5 * ekf_lm.pos_std
    assert nees.mean() < 3.0                                      # still consistent, just less certain


def test_ekf_rejects_gross_outliers_and_accepts_pose_fixes():
    ekf = EkfLocalizer(np.array([1.0, 3.0, 0.0]), LANDMARKS)
    x0 = ekf.x.copy()
    bad = LandmarkFrame(Header(1.0, 1.0, "cam"), [LandmarkObservation("tag_3", 20.0, 0.1),
                                                  LandmarkObservation("unknown_tag", 1.0, 0.0)])
    ekf.update_landmarks(bad, 1.0)
    assert ekf.rejected == 1 and ekf.accepted == 0 and np.allclose(ekf.x, x0)
    assert ekf.nis_avg > 4 * EkfLocalizer.CHI2_2DOF_99
    # a pose pseudo-measurement pulls the estimate toward it proportionally to the covariances
    ekf2 = EkfLocalizer(np.array([1.0, 3.0, 0.0]), LANDMARKS, init_std=(0.1, 0.1, 0.05))
    assert ekf2.update_pose(np.array([1.1, 3.0, 0.0]), np.diag([0.1, 0.1, 0.05]) ** 2)
    assert ekf2.x[0] == pytest.approx(1.05) and ekf2.pos_std < 0.1
    assert not ekf2.update_pose(np.array([5.0, 3.0, 0.0]), np.diag([0.01, 0.01, 0.01]) ** 2)   # gated
    # the gyro is fused with wheel odometry for yaw rate, and alone during odometry dropout
    ekf3 = EkfLocalizer(np.zeros(3), {})
    imu = [ImuSample(Header(0.1, 0.1, "imu"), np.zeros(3), np.array([0, 0, 0.2]))]
    ekf3.predict(None, imu, 1.0)
    assert ekf3.x[2] == pytest.approx(0.2) and ekf3.pos_std > 0.1


# ---------------------------------------------------------------------------
# scan matching
# ---------------------------------------------------------------------------
W, D = 8.0, 6.0
BOXES = [OrientedBox((W / 2, -0.05, 1), (W / 2, 0.05, 1)), OrientedBox((W / 2, D + 0.05, 1), (W / 2, 0.05, 1)),
         OrientedBox((-0.05, D / 2, 1), (0.05, D / 2, 1)), OrientedBox((W + 0.05, D / 2, 1), (0.05, D / 2, 1)),
         OrientedBox((2.0, 4.5, 0.5), (0.4, 0.3, 0.5)), OrientedBox((6.0, 1.5, 0.5), (0.3, 0.6, 0.5), 0.4),
         OrientedBox((5.0, 4.0, 0.5), (0.25, 0.25, 0.5))]


@pytest.fixture(scope="module")
def map_edt():
    g = GridSpec(np.zeros(2), 0.05, (int(np.ceil(W / 0.05)), int(np.ceil(D / 0.05))))
    occ = rasterize_boxes(g, BOXES, inflate=g.res, z_band=(0.05, 1.8))
    return distance_transform_edt(~occ) * g.res, g


def _surface_points(rng, n_per_edge=40):
    pts = []
    for b in BOXES:
        c = b.corners_xy()
        for i in range(4):
            s = rng.uniform(0, 1, n_per_edge)
            pts.append(c[i] + s[:, None] * (c[(i + 1) % 4] - c[i]))
    P = np.concatenate(pts)
    P = P[(P[:, 0] > 0) & (P[:, 0] < W) & (P[:, 1] > 0) & (P[:, 1] < D)]
    return np.c_[P, np.full(len(P), 0.9)]


def _misalign(P, pivot, t0, th0):
    """Inverse of the rigid motion scan_match estimates (rotate about pivot, then translate)."""
    Q = P.copy()
    Q[:, :2] = (P[:, :2] - pivot - t0) @ rot2(-th0).T + pivot
    return Q


@pytest.mark.parametrize("t0,th_deg", [((0.14, -0.21), 3.0), ((-0.07, 0.28), -2.0), ((0.1, -0.17), 2.4)])
def test_scan_match_recovers_known_rigid_offset(map_edt, rng, t0, th_deg):
    edt, g = map_edt
    P = _surface_points(rng)
    pivot = np.array([4.0, 3.0])
    m = scan_match(_misalign(P, pivot, np.array(t0), np.deg2rad(th_deg)), pivot, edt, g)
    step_xy, step_th = 0.07, 1.0                                  # search grid resolution
    assert m.dx == pytest.approx(t0[0], abs=step_xy / 2 + 1e-6)
    assert m.dy == pytest.approx(t0[1], abs=step_xy / 2 + 1e-6)
    assert np.rad2deg(m.dth) == pytest.approx(th_deg, abs=step_th / 2 + 1e-6)
    assert m.cost < 0.3 * m.cost0 and m.gain > 0.7 and m.magnitude > 0.1


def test_scan_match_identity_when_aligned(map_edt, rng):
    edt, g = map_edt
    m = scan_match(_surface_points(rng), np.array([4.0, 3.0]), edt, g)
    assert abs(m.dx) < 1e-9 and abs(m.dy) < 1e-9 and abs(m.dth) < 1e-9
    assert m.cost0 == pytest.approx(0.0) and m.gain == pytest.approx(0.0)


def test_change_diagnoser_distinguishes_drift_from_moved_cart(map_edt, rng):
    edt, g = map_edt
    prior = [SceneObject(f"b{i}", "wall" if i < 4 else "cart", b, "painted_wall", "x", movable=i >= 4)
             for i, b in enumerate(BOXES)]
    pose = np.array([4.0, 3.0, 0.0])
    P = _surface_points(rng)
    # drift: the whole scan is rigidly displaced; the residual points are spread everywhere
    drift = ChangeDiagnoser(prior)
    Q = _misalign(P, pose[:2], np.array([0.21, -0.14]), np.deg2rad(2.0))
    res = np.ones(len(Q), bool)
    for k in range(6):
        dg = drift.observe(float(k), Q, res, pose, edt, g, nis_avg=12.0, rng=rng)
    assert dg is not None and dg.cause == "loc_drift"
    corr = drift.pose_correction(pose)
    assert corr is not None and corr[:2] == pytest.approx(pose[:2] + [0.21, -0.14], abs=0.04)
    # map change: the scan fits except for points of one moved movable object
    change = ChangeDiagnoser(prior)
    moved = P.copy()
    cart = BOXES[6]
    on_cart = np.linalg.norm(P[:, :2] - cart.center[:2], axis=1) < 0.4
    moved[on_cart, :2] += np.array([0.6, -0.5])
    res = on_cart.copy()
    for k in range(6):
        dg = change.observe(float(k), moved, res, pose, edt, g, nis_avg=2.0, rng=rng)
    assert dg is not None and dg.cause == "map_change" and dg.object == "b6"
    assert dg.shift == pytest.approx([0.6, -0.5], abs=0.15)
    slots = [Slot("b6:top", "surface", "b6", (5.0, 4.0, 1.0))]
    change.reanchor(dg, slots)
    assert slots[0].position[:2] == pytest.approx(np.array([5.0, 4.0]) + dg.shift)
    assert change.summary()["cause"] == "map_change"
