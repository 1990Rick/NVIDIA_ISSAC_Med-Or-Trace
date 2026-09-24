"""People tracker: constant-velocity tracking, radar range-rate updates, staff identities."""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.belief.tracker import PeopleTracker


def _track_two_walkers(rng, T=5.0, dt=0.2, sigma=0.05):
    """Returns the tracker, final true positions, true velocities and the per-step
    velocity estimates (second half of the run) of the track nearest to each walker."""
    tr = PeopleTracker()
    p = np.array([[1.0, 1.0], [1.0, 4.0]])
    v = np.array([[0.8, 0.3], [0.5, -0.1]])
    v_est = [[], []]
    n = int(T / dt) + 1
    for k in range(n):
        t = k * dt
        tr.predict(t)
        truth = p + v * t
        tr.update_positions(truth + rng.normal(0, sigma, (2, 2)), t)
        tr.prune(t)
        if k >= n // 2:
            for j in range(2):
                tk = min(tr.tracks, key=lambda x: np.linalg.norm(x.x[:2] - truth[j]))
                v_est[j].append(tk.x[2:].copy())
    return tr, p + v * T, v, np.array(v_est)


def test_constant_velocity_targets_tracked(rng):
    tr, p_end, v, v_est = _track_two_walkers(rng)
    assert len(tr.tracks) == 2 and len(tr.confirmed()) == 2 and len(tr.people()) == 2
    for tk in tr.tracks:
        j = int(np.argmin(np.linalg.norm(p_end - tk.x[:2], axis=1)))
        assert np.linalg.norm(tk.x[:2] - p_end[j]) < 0.15              # measurement noise is 5 cm
        assert np.linalg.norm(tk.x[2:] - v[j]) < 0.25                  # single-epoch estimate (3 sigma)
        assert tk.person_like and tk.hits == 26
    # time-averaged velocity estimate is unbiased (no identity swap between the walkers)
    for j in range(2):
        assert np.linalg.norm(v_est[j].mean(0) - v[j]) < 0.08
    # future samples are centred on the constant-velocity extrapolation and spread with time
    S = tr.predict_samples(10, 0.1, 400, rng)
    assert S.shape == (2, 400, 10, 2)
    for k, tk in enumerate(tr.people()):
        expect = tk.x[:2] + tk.x[2:] * 1.0
        assert np.linalg.norm(S[k, :, -1].mean(0) - expect) < 0.1
        assert S[k, :, -1].std(0).mean() > S[k, :, 0].std(0).mean()
    # tracks die when unobserved for longer than max_age
    tr.prune(5.0 + tr.max_age + 0.1)
    assert tr.tracks == []
    assert PeopleTracker().predict_samples(10, 0.1, 5, rng).shape == (0, 5, 10, 2)


def test_static_clutter_is_not_person_like(rng):
    tr = PeopleTracker()
    for k in range(20):
        tr.predict(k * 0.2)
        tr.update_positions(np.array([[2.0, 2.0]]) + rng.normal(0, 0.03, (1, 2)), k * 0.2)
    assert len(tr.confirmed()) == 1 and tr.people() == []


def test_radar_range_rate_initialises_and_corrects_velocity():
    tr = PeopleTracker()
    u = np.array([1.0, 0.0])
    # a moving radar return with no track spawns one with the line-of-sight velocity
    tr.update_radar([(np.array([4.0, 1.0]), 0.9, u)], 0.0, robot_vel_world=np.array([0.3, 0.0]))
    assert len(tr.tracks) == 1 and tr.tracks[0].x[2:] == pytest.approx([1.2, 0.0])   # ego-motion compensated
    # a static return does not
    tr.update_radar([(np.array([0.0, 4.0]), -0.3, np.array([0.0, 1.0]))], 0.0, robot_vel_world=np.array([0.0, 0.3]))
    assert len(tr.tracks) == 1
    # an existing position-only track learns its radial velocity from one Doppler update
    tr2 = PeopleTracker()
    tr2.update_positions(np.array([[4.0, 1.0]]), 0.0)
    tr2.update_radar([(np.array([4.0, 1.0]), 1.0, u)], 0.0, robot_vel_world=np.zeros(2))
    assert tr2.tracks[0].x[2] == pytest.approx(1.0, abs=0.1) and abs(tr2.tracks[0].x[3]) < 0.05


def test_staff_identities_anchor_sterile_staff_and_follow_roaming_staff():
    homes = {"surgeon": np.array([2.0, 2.0]), "circulator": np.array([5.0, 5.0])}
    tr = PeopleTracker(staff_homes={k: v.copy() for k, v in homes.items()})
    tr.sterile_names = {"surgeon"}
    ids = []
    for k in range(12):
        t = 0.2 * k
        tr.predict(t)
        # the surgeon track drifts away from the station; the circulator walks 1.65 m away from home
        tr.update_positions(np.array([[2.0 + 0.08 * k, 2.0], [5.0 + 0.15 * k, 5.0]]), t)
        tr.prune(t)
        ids.append({tk.identity: tk.x[:2].copy() for tk in tr.tracks})
    assert "surgeon" in ids[0] and "circulator" in ids[0]
    # the sterile identity is anchored to its station (0.8 m gate), never dragged along
    assert np.allclose(tr.staff_homes["surgeon"], homes["surgeon"])
    last_surgeon = max(k for k, d in enumerate(ids) if "surgeon" in d)
    assert np.linalg.norm(ids[last_surgeon]["surgeon"] - homes["surgeon"]) < 0.8
    assert "surgeon" not in ids[-1]
    # the roaming identity follows its track far beyond its initial home
    assert "circulator" in ids[-1] and ids[-1]["circulator"][0] > 6.5
    assert np.allclose(tr.staff_homes["circulator"], ids[-1]["circulator"])
    # a stranger appearing at the surgeon's station inherits the anchored identity
    tr.update_positions(np.array([[2.1, 2.0]]), 2.4)
    tr.prune(2.4)
    assert any(tk.identity == "surgeon" and np.linalg.norm(tk.x[:2] - [2.1, 2.0]) < 0.1 for tk in tr.tracks)
