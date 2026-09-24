"""Fault injection: sampled schedules, how the lite backend applies them, and the
CF-D / rare-geometry / map-corruption world modifications."""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.common.config import load_config
from medortrace.common.msgs import VelocityCommand
from medortrace.perception.sync import PoseHistory, TimeSyncMonitor
from medortrace.sim.episode import build_episode
from medortrace.sim.faults import SENSORS, FaultModel, sample_faults
from medortrace.sim.lite_backend import LiteBackend


def _run_backend(ep, n_steps, cmd=VelocityCommand()):
    be = LiteBackend(ep.cfg)
    bundles = [be.reset(ep)]
    for _ in range(n_steps):
        bundles.append(be.step(cmd))
    return be, bundles


# ---------------------------------------------------------------------------
# schedules
# ---------------------------------------------------------------------------
def test_dropout_windows_are_poisson_disjoint_and_queried_correctly():
    cfg = {"faults": {"dropout_rate_per_min": {"lidar": 3.0, "camera": 0.0}, "dropout_mean_s": 4.0}}
    T = 3600.0
    fm = sample_faults(cfg, T, np.random.default_rng(0), {"factor": "none"})
    wins = fm.dropouts["lidar"]
    assert "camera" not in fm.dropouts and "radar" not in fm.dropouts
    starts = np.array([a for a, b in wins])
    ends = np.array([b for a, b in wins])
    assert np.all(ends > starts) and np.all(starts[1:] >= ends[:-1]) and ends[-1] <= T
    # ~3 windows/min arrivals (gaps exclude the dropout durations), exponential durations
    assert 120 < len(wins) < 200
    assert np.mean(ends - starts) == pytest.approx(4.0, rel=0.25)
    a, b = wins[3]
    mid = 0.5 * (a + b)
    assert fm.dropped("lidar", a) and fm.dropped("lidar", mid) and not fm.dropped("lidar", b)
    assert not fm.dropped("camera", mid)
    gap = 0.5 * (b + wins[4][0])
    assert not fm.dropped("lidar", gap)
    assert fm.active(mid)["dropout_lidar"] and not fm.active(gap)["dropout_lidar"]
    assert fm.any_sensor_fault(mid) and not fm.any_sensor_fault(gap)
    assert set(fm.active(0.0)) >= {f"dropout_{s}" for s in SENSORS} | {"skew", "odom_drift"}
    # labels mirror the schedule (ground truth for fault-stratified metrics)
    assert fm.labels["dropouts"]["lidar"] == wins


def test_skew_sampling_and_stamp():
    cfg = {"faults": {"timestamp_skew": {"camera": {"offset_s": [0.15, 0.45], "drift_ppm": 800}}}}
    fm = sample_faults(cfg, 100.0, np.random.default_rng(1), {})
    off, drift = fm.skew["camera"]
    assert 0.15 <= off <= 0.45 and drift == pytest.approx(800e-6)
    assert fm.stamp("camera", 50.0) == pytest.approx(50.0 + off + 50.0 * 800e-6)
    assert fm.stamp("lidar", 50.0) == 50.0
    assert fm.active(10.0)["skew"]
    assert not FaultModel(skew={"camera": (0.01, 0.0)}).active(10.0)["skew"]   # within 50 ms tolerance


# ---------------------------------------------------------------------------
# lite backend applies the schedule
# ---------------------------------------------------------------------------
def test_backend_honours_dropout_windows(default_cfg):
    ep = build_episode(load_config(override={"episode": {"duration_s": 20.0}}), 3)
    ep.faults.dropouts = {"lidar": [(0.45, 1.45)], "odom": [(0.25, 0.55)]}
    _, bundles = _run_backend(ep, 20)
    lidar_t = [b.t for b in bundles if b.lidar is not None]
    assert lidar_t and all(not (0.45 <= t < 1.45) for t in lidar_t)
    assert any(t >= 1.45 for t in lidar_t) and any(t < 0.45 for t in lidar_t)
    no_odom = [round(b.t, 2) for b in bundles if b.odom is None]
    assert no_odom == [0.3, 0.4, 0.5]
    assert all(b.camera is not None for b in bundles if 0.45 <= b.t < 1.45 and round(b.t * 10) % 2 == 0)


def test_backend_stamps_skewed_sensor_and_sync_monitor_retimes(default_cfg):
    ep = build_episode(load_config(override={"episode": {"duration_s": 20.0}}), 4)
    ep.faults.skew = {"camera": (0.3, 1e-3)}
    _, bundles = _run_backend(ep, 30)
    cams = [b.camera.header for b in bundles if b.camera is not None]
    lid = [b.lidar.header for b in bundles if b.lidar is not None]
    assert len(cams) >= 5
    for h in cams:
        assert h.stamp - h.recv_stamp == pytest.approx(0.3 + 1e-3 * h.recv_stamp)
    assert all(h.stamp == h.recv_stamp for h in lid)
    # the sync monitor detects the offset and falls back to the host receive time
    sync = TimeSyncMonitor()
    used = [sync.correct("camera", h.stamp, h.recv_stamp) for h in cams]
    assert sync.skewed["camera"] and used[-1] == pytest.approx(cams[-1].recv_stamp)
    assert [sync.correct("lidar", h.stamp, h.recv_stamp) for h in lid] == [h.stamp for h in lid]
    assert not sync.skewed["lidar"]
    # a one-off late message on a healthy clock is rejected as stale (> max_age) ...
    for k in range(10):
        assert sync.correct("radar", 0.1 * k, 0.1 * k) == pytest.approx(0.1 * k)
    assert sync.correct("radar", 0.0, 2.0) is None and sync.rejected["radar"] == 1
    assert not sync.skewed["radar"]
    # ... while a persistent offset is re-timed; a disabled monitor trusts stamps blindly
    assert TimeSyncMonitor(enabled=False).correct("camera", 5.3, 5.0) == 5.3


def test_pose_history_interpolation():
    ph = PoseHistory()
    ph.add(0.0, np.array([0.0, 0.0, 3.0]))
    ph.add(1.0, np.array([1.0, 2.0, -3.0]))                    # heading wraps through +-pi
    p = ph.at(0.5)
    assert p[:2] == pytest.approx([0.5, 1.0])
    assert abs(np.cos(p[2]) + 1.0) < 1e-3                      # halfway between 3.0 and -3.0 is ~pi
    assert ph.at(-1.0)[0] == 0.0
    assert ph.at(1.5)[:2] == pytest.approx([1.5, 3.0])        # bounded linear extrapolation


def test_backend_applies_odometry_bias_after_start(default_cfg):
    ep = build_episode(load_config(override={"episode": {"duration_s": 20.0}}), 5)
    ep.faults.odom_bias = (0.1, 0.05)
    ep.faults.odom_bias_start = 1.0
    be, bundles = _run_backend(ep, 25, VelocityCommand(0.3, 0.0))
    before = [b.odom for b in bundles if b.odom is not None and 0.5 < b.t < 1.0]
    after = [b.odom for b in bundles if b.odom is not None and b.t >= 1.5]
    assert np.mean([o.omega for o in before]) == pytest.approx(0.0, abs=0.01)
    assert np.mean([o.omega for o in after]) == pytest.approx(0.05, abs=0.01)
    # the true robot velocity is unbiased; the reported one is scaled by (1 + 0.1)
    assert after[-1].v == pytest.approx(be.vel[0] * 1.1, abs=0.04)
    assert ep.faults.active(2.0)["odom_drift"] and not ep.faults.active(0.5)["odom_drift"]


# ---------------------------------------------------------------------------
# world modifications
# ---------------------------------------------------------------------------
def test_cf_d_loc_drift_vs_cart_moved(cf_episode):
    drift, moved = cf_episode("CF-D", "loc_drift", 21), cf_episode("CF-D", "cart_moved", 21)
    for ep in (drift, moved):
        assert (20.0, 110.0) in ep.faults.dropouts["landmarks"]            # both arms: fiducials blinded
    assert drift.faults.odom_bias == (0.08, 0.03) and drift.faults.odom_bias_start == 20.0
    assert moved.faults.odom_bias == (0.0, 0.0)
    assert not any(e["kind"] == "cf_d_cart_moved" for e in drift.faults.map_edits)
    # cart_moved: the TRUE cart and its slot moved by the scripted shift; survey + prior map are stale
    shift = np.array([0.55, -0.35, 0.0])
    c_true, c_survey = moved.spec.object("cart_1"), moved.survey_spec.object("cart_1")
    assert np.allclose(c_true.box.center - c_survey.box.center, shift)
    assert "moved_since_survey" in c_true.tags and "moved_since_survey" not in c_survey.tags
    assert np.allclose(moved.spec.slot("cart_1:top").position - moved.survey_spec.slot("cart_1:top").position, shift)
    prior_cart = next(o for o in moved.prior_map if o.name == "cart_1")
    assert np.allclose(prior_cart.box.center, c_survey.box.center)
    # loc_drift: the world is unchanged
    assert np.allclose(drift.spec.object("cart_1").box.center, drift.survey_spec.object("cart_1").box.center)
    assert np.allclose(drift.spec.object("cart_1").box.center, c_survey.box.center)


def test_rare_geometry_is_real_but_absent_from_prior_map():
    ep = build_episode(load_config("scenarios/rare_geometry.yaml",
                                   {"faults": {"rare_geometry": ["iv_pole_fallen", "cart_tipped"]}}), 8)
    names = {o.name for o in ep.spec.objects}
    assert "fallen_iv_pole" in names
    assert "fallen_iv_pole" not in {o.name for o in ep.prior_map}
    pole = ep.spec.object("fallen_iv_pole")
    assert pole.box.z_max <= 0.06                      # below the lowest lidar ring's reach near the robot
    cart = ep.spec.object("cart_2")
    assert "rare" in cart.tags and cart.box.z_max == pytest.approx(0.5)
    assert next(o for o in ep.prior_map if o.name == "cart_2").box.z_max == pytest.approx(1.0)


def test_map_corruption_only_touches_the_prior():
    cfg = load_config(override={"faults": {"map_corruption": {"shift_n": 2, "max_shift": 0.5, "phantom_n": 1,
                                                              "drop_n": 1}}})
    ep = build_episode(cfg, 9)
    prior = {o.name: o for o in ep.prior_map}
    survey = {o.name: o for o in ep.survey_spec.objects}
    phantoms = [n for n in prior if n.startswith("phantom_")]
    assert len(phantoms) == 1 and phantoms[0] not in {o.name for o in ep.spec.objects}
    corrupted = [n for n, o in prior.items() if "corrupted" in o.tags and not n.startswith("phantom_")]
    assert len(corrupted) == 2
    for n in corrupted:
        d = prior[n].box.center[:2] - survey[n].box.center[:2]
        assert 0 < np.abs(d).max() <= 0.5 and survey[n].movable
    dropped = [n for n, o in survey.items() if o.kind in ("cart", "kick_bucket", "waste_bin") and n not in prior]
    assert len(dropped) == 1
    # the true world keeps every surveyed object where it was
    for n, o in survey.items():
        assert np.allclose(ep.spec.object(n).box.center, o.box.center)
