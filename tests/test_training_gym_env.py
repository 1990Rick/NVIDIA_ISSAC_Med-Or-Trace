"""Smoke tests of the gym-style NBV environment (short lite episodes)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medortrace.autonomy.gym_env import (  # noqa: E402
    CANDIDATE_FEATURES,
    GLOBAL_FEATURES,
    BoxSpace,
    DiscreteSpace,
    NbvEnv,
    _PlannerHook,
    register,
)


def test_weights_mode_reset_steps_and_episode_end():
    env = NbvEnv(duration_s=6.0, decision_period_s=3.0)
    obs = env.reset(seed=11)
    assert isinstance(env.action_space, BoxSpace) and env.action_space.shape == (len(env.weight_keys),)
    assert obs.shape == env.observation_space.shape == (len(GLOBAL_FEATURES) + 4 * len(CANDIDATE_FEATURES),)
    assert len(env.observation_names) == obs.shape[0] and obs.dtype == np.float32 and np.all(np.isfinite(obs))
    a = np.zeros(len(env.weight_keys))
    a[env.weight_keys.index("w_risk")] = 1.0
    obs, r, done, info = env.step(a)
    assert not done and np.isfinite(r) and info["t"] == pytest.approx(3.0, abs=0.11)
    assert info["weights"]["w_risk"] == pytest.approx(env.base_weights["w_risk"] * 4.0)
    assert env.stack.nbv.inner.w == info["weights"]            # the stack replans with the agent's weights
    assert set(info["reward_terms"]) >= {"correct", "wrong", "abstain", "entropy", "near_collision", "sterile_s"}
    obs, r, done, info = env.step(env.action_space.sample(np.random.default_rng(0)))
    assert done and np.all(np.isfinite(obs)) and np.isfinite(r)
    assert info["t"] == pytest.approx(6.0, abs=0.11)
    m = info["metrics"]
    assert m["claims_answered"] == info["n_verdicts"] and "handoff_success" in m
    assert len(env.stack.telemetry) == len(env.truth_log.t)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(len(env.weight_keys)))


def test_select_mode_commits_the_chosen_goal():
    env = NbvEnv(duration_s=6.0, action_mode="select")
    obs = env.reset(seed=11)
    assert isinstance(env.action_space, DiscreteSpace) and env.option_names[-1] == "hold"
    assert env.action_space.n == len(env.option_names) == len(env.preset_names) + 1
    i = next((k for k, c in enumerate(env._candidates) if c is not None), None)
    assert i is not None, "no admissible NBV candidate at the start pose"
    cand = env._candidates[i]
    obs, r, done, info = env.step(i)
    assert info["option"] == env.option_names[i] and info["option_valid"]
    assert env.stack.goal is cand or env.stack.goal.target_slot == "__dock__"
    hold_pose = env.stack.ekf.x.copy()
    obs, r, done, info = env.step(len(env.option_names) - 1)
    assert info["option"] == "hold" and done
    np.testing.assert_allclose(env.hook.committed.pose, hold_pose)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(99)


def test_determinism_and_hook_forwarding():
    def rollout():
        env = NbvEnv(duration_s=3.0, decision_period_s=3.0)
        o0 = env.reset(seed=7)
        o1, r, done, _ = env.step(np.full(len(env.weight_keys), 0.5))
        return o0, o1, r, done

    a, b = rollout(), rollout()
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2] and a[3] and b[3]

    class Inner:
        vis = "old"
        w = {}

        def plan(self, *args, **kw):
            return "planned"

    h = _PlannerHook(Inner())
    h.vis = "new"                                              # the stack sets nbv.vis after a map change
    assert h.inner.vis == "new" and h.plan() == "planned"
    h.committed = "goal"
    assert h.plan() == "goal" and h.inner.vis == "new"


def test_config_validation_and_optional_gymnasium():
    with pytest.raises(TypeError):
        NbvEnv(not_a_field=1)
    with pytest.raises(ValueError):
        NbvEnv(action_mode="bogus")
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        assert register() is False
    else:
        assert register() is True
