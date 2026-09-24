"""Gym-style environment for learning the active-perception (next-best-view) policy.

Dependency-free (numpy only): ``reset(seed) -> obs`` and
``step(action) -> (obs, reward, done, info)``.  If ``gymnasium`` is installed
the environment is also registered as ``MedOrTrace-NBV-v0`` (5-tuple API via
:func:`make_gymnasium_env`).

What the agent controls
    The full :class:`~medortrace.autonomy.stack.AutonomyStack` runs underneath
    (localisation, beliefs, verifier, MPPI, safety supervisor) on the lite
    simulator.  The agent only acts at *decision points* every
    ``decision_period_s`` of sim time (default 3 s = 30 control ticks), on the
    goal-selection layer: the stack's NBV planner is replaced by a thin hook
    that either re-weights it or answers with the agent's chosen goal.  The
    safety supervisor keeps the final say over velocity in both modes.

Action space (``env.action_space``)
    * ``action_mode="weights"`` - :class:`BoxSpace` ``[-1, 1]^K`` over the NBV
      weight keys ``env.weight_keys`` (the keys of the base policy file):
      ``w_k = base_k * exp(weight_scale * a_k)`` (default scale ln 4, i.e.
      x0.25 .. x4).  The stack keeps replanning on its own schedule with these
      weights until the next decision.
    * ``action_mode="select"`` - :class:`DiscreteSpace` over
      ``env.option_names``: one goal per weight preset (``presets``; the goal
      the NBV planner picks under that preset, all presets scored on the same
      sampled viewpoint set) plus ``"hold"`` (stay at the current estimate).
      The chosen goal is committed until the next decision point.

Observation (``env.observation_names``; float32 vector, ``G + P * F`` entries)
    ``G = 24`` global features - pose estimate (x/W, y/D, cos, sin yaw), pose
    std-dev, NIS/25, supervisor mode one-hot (5), battery fraction, time
    fraction, tracked people/5, estimated human clearance/5, mean and max item
    entropy (normalised by log2 #slots), open claims/10, claim urgency/10,
    min claim slack/60 s, fraction of open claims with direct evidence, path
    entropy ahead, has-goal flag, distance to goal/5 - followed by one block of
    ``F = 7`` features per preset candidate (valid flag, custody EIG, modal
    EIG, log1p map uncertainty in view, path length/5, human risk, sterile
    proximity; the NBV breakdown terms).  The candidate blocks are computed in
    both action modes.  Everything is the robot's own belief - no truth.

Reward (per decision step; coefficients in :class:`RewardConfig`)
    ``verified-claim progress - risk/safety penalties``::

        + correct * (#VERIFIED/REFUTED verdicts that are right)
        + wrong   * (#wrong assertions)                      (negative)
        + abstain * (#ABSTAIN)                               (small negative: costs staff attention)
        + entropy * (drop of criticality-weighted item entropy, bits)   (potential-based shaping)
        + near_collision * (#entries into < 0.3 m human clearance)
        + collision_agent * (#robot-person contacts) + collision_static * (#static contacts)
        + sterile_s * (s inside the sterile keep-out) + proximity * integral exp(-(d-0.3)/0.3) dt
        + safe_stop * (#uncertainty-triggered STOPs) + handover * (#HANDOVER requests)

    Verdict correctness and the safety terms come from simulator truth (as in
    ``medortrace.eval.metrics``); at episode end the open claims are closed
    (``AutonomyStack.finalize``) and ``info["metrics"]`` holds the full
    benchmark metrics of the episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from medortrace.autonomy.stack import AutonomyStack, stack_inputs_from_episode
from medortrace.common.config import deep_merge, load_config, load_yaml
from medortrace.eval.metrics import HUMAN_R, NEAR_COLLISION_M, ROBOT_R, TruthLog
from medortrace.planning.nbv import DEFAULT_WEIGHTS, ViewGoal
from medortrace.safety.supervisor import Mode
from medortrace.sim.episode import build_episode

GYMNASIUM_ID = "MedOrTrace-NBV-v0"
MODES = [m.value for m in Mode]
CANDIDATE_FEATURES = ["valid", "eig", "modal", "log1p_unc", "path_5m", "risk", "sterile"]
GLOBAL_FEATURES = ["x_frac", "y_frac", "cos_yaw", "sin_yaw", "pose_std_m", "nis_25"] + [f"mode_{m}" for m in MODES] + [
    "battery_frac", "time_frac", "tracks_5", "human_clear_5m", "item_entropy_mean", "item_entropy_max",
    "open_claims_10", "urgency_10", "min_slack_60s", "direct_frac", "path_entropy", "has_goal", "goal_dist_5m"]

# log-space multipliers applied to the base weights (select mode options / observation candidates)
DEFAULT_PRESETS: dict[str, dict[str, float]] = {
    "default": {},
    "info_greedy": {"w_eig": 3.0, "w_modal": 3.0, "w_path": 0.5, "w_urgent": 2.0},
    "cautious": {"w_risk": 4.0, "w_sterile": 4.0, "w_path": 2.0},
    "explore": {"w_unc": 5.0, "w_eig": 0.5},
}


# ---------------------------------------------------------------------------
@dataclass
class BoxSpace:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: type = np.float32

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return rng.uniform(lo, hi, self.shape).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))


@dataclass
class DiscreteSpace:
    n: int

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, self.n))

    def contains(self, x) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n


@dataclass
class RewardConfig:
    correct: float = 1.0
    wrong: float = -2.0
    abstain: float = -0.25
    entropy: float = 0.05
    near_collision: float = -0.5
    collision_agent: float = -5.0
    collision_static: float = -2.0
    sterile_s: float = -5.0
    proximity: float = -0.1
    safe_stop: float = -0.2
    handover: float = -1.0


@dataclass
class EnvConfig:
    scenario: str = "scenarios/nominal.yaml"
    overrides: dict = field(default_factory=dict)      # deep-merged into the scenario config
    sim_overrides: dict = field(default_factory=dict)  # world/simulator only (sim-to-real randomisation; the
                                                       # stack keeps the unperturbed config, as in eval.runner)
    entries: list | None = None                        # RegistryEntry list: reset(seed) samples one of them
    duration_s: float | None = 60.0                    # episode length override (None: scenario default)
    decision_period_s: float = 3.0
    action_mode: str = "weights"                       # weights | select
    weight_scale: float = math.log(4.0)
    base_weights: str | dict = "policies/nbv_default.yaml"
    presets: dict = field(default_factory=lambda: {k: dict(v) for k, v in DEFAULT_PRESETS.items()})
    reward: RewardConfig = field(default_factory=RewardConfig)
    operator: bool = True                              # simulated operator answers HANDOVER requests


class _PlannerHook:
    """Stands in for ``stack.nbv``; the stack keeps calling ``plan()`` on its own replanning schedule.

    ``committed`` (select mode) answers every call with the agent's goal until the next decision;
    otherwise the call goes to the wrapped planner (whose weights the env sets in weights mode).
    Attribute writes other than the hook's own (e.g. ``stack.nbv.vis`` after a map change) go to the planner.
    """

    _own = ("inner", "committed")

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "committed", None)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name in self._own:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def plan(self, *args, **kwargs):
        if self.committed is not None:
            return self.committed
        return self.inner.plan(*args, **kwargs)


# ---------------------------------------------------------------------------
class NbvEnv:
    """See the module docstring for the action / observation / reward definitions."""

    def __init__(self, config: EnvConfig | None = None, **kwargs):
        cfg = replace(config or EnvConfig(), **kwargs)     # TypeError on unknown fields; caller's config untouched
        if cfg.action_mode not in ("weights", "select"):
            raise ValueError("action_mode must be 'weights' or 'select'")
        self.cfg = cfg
        bw = cfg.base_weights if isinstance(cfg.base_weights, dict) else load_yaml(cfg.base_weights).get("weights", {})
        self.base_weights = {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in bw.items()}}
        self.weight_keys = list(self.base_weights)
        self.preset_names = list(cfg.presets)
        self.preset_weights = [{k: w * float(cfg.presets[p].get(k, 1.0)) for k, w in self.base_weights.items()}
                               for p in self.preset_names]
        self.option_names = self.preset_names + ["hold"]
        self.observation_names = GLOBAL_FEATURES + [f"{p}.{f}" for p in self.preset_names for f in CANDIDATE_FEATURES]
        n_obs = len(self.observation_names)
        self.observation_space = BoxSpace(np.full(n_obs, -np.inf, np.float32), np.full(n_obs, np.inf, np.float32),
                                          (n_obs,))
        if cfg.action_mode == "weights":
            k = len(self.weight_keys)
            self.action_space = BoxSpace(-np.ones(k, np.float32), np.ones(k, np.float32), (k,))
        else:
            self.action_space = DiscreteSpace(len(self.option_names))
        self._seed_rng = np.random.default_rng()
        self.stack: AutonomyStack | None = None

    # ------------------------------------------------------------------
    def _episode_cfg(self, seed: int | None) -> tuple[dict, int]:
        if seed is None:
            seed = int(self._seed_rng.integers(1, 2**31 - 1))
        if self.cfg.entries:
            e = self.cfg.entries[int(np.random.default_rng(seed).integers(0, len(self.cfg.entries)))]
            cfg, ep_seed = e.resolve(), int(e.seed)
        else:
            cfg, ep_seed = load_config(self.cfg.scenario), int(seed)
        cfg = deep_merge(cfg, self.cfg.overrides)
        if self.cfg.duration_s:
            cfg = deep_merge(cfg, {"episode": {"duration_s": float(self.cfg.duration_s)}})
        cfg = deep_merge(cfg, {"autonomy": {"policy": "active"}, "_policy_seed": ep_seed * 7919 + 17})
        return cfg, ep_seed

    def reset(self, seed: int | None = None) -> np.ndarray:
        from medortrace.eval.runner import SimulatedOperator, make_backend
        cfg, ep_seed = self._episode_cfg(seed)
        sim_cfg = deep_merge(cfg, self.cfg.sim_overrides) if self.cfg.sim_overrides else cfg
        self.ep = ep = build_episode(sim_cfg, ep_seed)
        self.seed = ep_seed
        self.be = make_backend("lite", sim_cfg)
        self.bundle = self.be.reset(ep)
        self.stack = AutonomyStack(stack_inputs_from_episode(ep), cfg)
        self.hook = _PlannerHook(self.stack.nbv)
        self.stack.nbv = self.hook
        self.hook.inner.w = dict(self.base_weights)
        self.op = SimulatedOperator(ep.streams.fork("robot", "operator")) if self.cfg.operator else None
        self.claims = {c.id: c for c in ep.workflow.claims}
        self.truth_log = TruthLog()
        self.verdicts: list = []
        self.T = float(ep.workflow.duration)
        self.decision = 0
        self.done = False
        self._near = False
        self._coll_agent = False
        self._coll_static = False
        self._n_events = 0
        self._tick()                           # one control tick so beliefs, costmap and trackers exist
        self._H = self._weighted_entropy()
        self._candidates = self._make_candidates()
        return self._observe()

    # ------------------------------------------------------------------
    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        if self.stack is None or self.done:
            raise RuntimeError("call reset() before step() (and after an episode is done)")
        info: dict = {"decision": self.decision}
        if self.cfg.action_mode == "weights":
            a = np.clip(np.asarray(action, float).reshape(-1), -1.0, 1.0)
            if a.shape != (len(self.weight_keys),):
                raise ValueError(f"weights action needs shape ({len(self.weight_keys)},), got {a.shape}")
            mult = np.exp(self.cfg.weight_scale * a)
            w = {k: self.base_weights[k] * float(m) for k, m in zip(self.weight_keys, mult)}
            self.hook.committed = None
            self.hook.inner.w = w
            info["weights"] = w
        else:
            i = int(action)
            if not 0 <= i < len(self.option_names):
                raise ValueError(f"select action must be in [0, {len(self.option_names)}), got {action!r}")
            goal = self._hold_goal() if self.option_names[i] == "hold" else self._candidates[i]
            self.hook.committed = goal
            if goal is not None:
                st = self.stack
                st.goal, st.goal_t, st._arrived_t = goal, st.t, None
            info["option"] = self.option_names[i]
            info["option_valid"] = goal is not None
        terms = {k: 0.0 for k in RewardConfig.__dataclass_fields__}
        n_ticks = max(1, int(round(self.cfg.decision_period_s / self.be.dt)))
        for _ in range(n_ticks):
            if self.be.t >= self.T - 1e-9:
                break
            self._tick(terms)
        if self.be.t >= self.T - 1e-9:
            self._finish(terms)
        H = self._weighted_entropy()
        terms["entropy"] += self._H - H
        self._H = H
        rc = self.cfg.reward
        reward = float(sum(getattr(rc, k) * v for k, v in terms.items()))
        self.decision += 1
        if not self.done:
            self._candidates = self._make_candidates()
        info.update({"t": float(self.be.t), "reward_terms": terms, "mode": self.stack.sup.mode.value,
                     "n_verdicts": len(self.verdicts), "seed": self.seed})
        if self.done:
            info["metrics"] = self.metrics
        return self._observe(), reward, self.done, info

    # ------------------------------------------------------------------
    def _tick(self, terms: dict | None = None) -> None:
        st, be = self.stack, self.be
        cmd = st.step(self.bundle, be.dt)
        self._score_verdicts(st.telemetry[-1].verdicts, terms)
        self.bundle = be.step(cmd)
        tr = be.truth()
        if self.op is not None:
            self.op.step(be.t, st, tr.robot_pose)
        self._log_truth(tr)
        if terms is None:
            return
        d = np.linalg.norm(tr.agent_pos - tr.robot_pose[:2], axis=1) - ROBOT_R - HUMAN_R if len(tr.agent_pos) else []
        dmin = float(np.min(d)) if len(d) else np.inf
        near = dmin < NEAR_COLLISION_M
        terms["near_collision"] += float(near and not self._near)
        terms["collision_agent"] += float(tr.collision_agent and not self._coll_agent)
        terms["collision_static"] += float(tr.collision_static and not self._coll_static)
        self._near, self._coll_agent, self._coll_static = near, tr.collision_agent, tr.collision_static
        terms["sterile_s"] += be.dt * float(tr.in_keepout)
        if np.isfinite(dmin):
            terms["proximity"] += be.dt * float(np.exp(-(dmin - NEAR_COLLISION_M) / 0.3))
        ev = st.sup.events[self._n_events:]
        self._n_events = len(st.sup.events)
        terms["safe_stop"] += sum(1 for e in ev if e.mode_to == "STOP" and e.category == "uncertainty")
        terms["handover"] += sum(1 for e in ev if e.mode_to == "HANDOVER")

    def _score_verdicts(self, verdicts: list, terms: dict | None) -> None:
        self.verdicts += verdicts
        if terms is None:
            return
        for v in verdicts:
            c = self.claims.get(v.claim_id)
            if c is None:
                continue
            if v.verdict.value == "ABSTAIN":
                terms["abstain"] += 1
                continue
            truth_ok = self.ep.workflow.truth_slot(c.item_id, c.t_ref) == c.slot_id
            terms["correct" if (v.verdict.value == "VERIFIED") == truth_ok else "wrong"] += 1

    def _log_truth(self, tr) -> None:
        tl = self.truth_log
        tl.t.append(tr.t)
        tl.robot.append(tr.robot_pose)
        tl.agents.append(tr.agent_pos)
        tl.shadow.append(tr.shadow_pos)
        tl.agent_names = tr.agent_names
        tl.collision_agent.append(tr.collision_agent)
        tl.collision_static.append(tr.collision_static)
        tl.in_keepout.append(tr.in_keepout)
        tl.in_keepout_margin.append(bool(self.ep.spec.in_keepout(tr.robot_pose[None, :2])[0]))
        tl.battery.append(tr.battery_wh)
        tl.energy.append(tr.energy_used_wh)
        tl.item_slots.append(tr.item_slots)
        tl.fault_any.append(self.ep.faults.any_sensor_fault(tr.t))

    def _finish(self, terms: dict) -> None:
        """Mirror ``eval.runner.run_episode``: last stack step, close open claims, compute metrics."""
        from medortrace.eval.metrics import compute_metrics
        st = self.stack
        st.step(self.bundle, self.be.dt)
        self._score_verdicts(st.telemetry[-1].verdicts, terms)
        grace = float(self.ep.cfg.get("workflow", {}).get("claim_grace_s", 25.0))
        self._score_verdicts(st.finalize(self.T + grace), terms)
        st.telemetry.pop()                     # keep telemetry aligned with the truth log (as the runner does)
        oplog = self.op.log if self.op is not None else []
        self.metrics = compute_metrics(self.ep, self.truth_log, st, self.verdicts, self.be, oplog,
                                       self.be.rp.battery_wh)
        self.metrics["provenance_chain_valid"] = float(st.prov.verify_chain())
        self.be.close()
        self.done = True

    # ------------------------------------------------------------------
    def _weighted_entropy(self) -> float:
        it = self.stack.items
        return float(sum(st.spec.criticality * it.entropy(i) for i, st in it.items.items()))

    def _plan_inputs(self):
        st = self.stack
        pose = st.ekf.x.copy()
        people = st._occluding_people(st.tracker.confirmed())
        return pose, people, st.verifier.urgency(st.t), st.verifier.urgent_slots(st.t)

    def _make_candidates(self) -> list[ViewGoal | None]:
        """Best goal of the NBV planner under each preset, all scored on the same sampled viewpoints."""
        st, inner = self.stack, self.hook.inner
        if st.cm is None:
            return [None] * len(self.preset_names)
        pose, people, urg, urg_slots = self._plan_inputs()
        seed = [int(self.seed) & (2**63 - 1), int(self.decision), 0xB5]
        saved = inner.w
        out = []
        try:
            for w in self.preset_weights:
                inner.w = dict(w)
                out.append(inner.plan(pose, st.items, st.occ, st.cm, people, urg, urg_slots,
                                      np.random.default_rng(seed)))
        finally:
            inner.w = saved
        return out

    def _hold_goal(self) -> ViewGoal:
        p = self.stack.ekf.x.copy()
        return ViewGoal(p, p[None, :2].copy(), 0.0, None, None, {"hold": 1.0})

    def _observe(self) -> np.ndarray:
        st = self.stack
        W, D, _ = st.inp.room
        x, y, th = st.ekf.x
        mode = st.sup.mode.value
        items = st.items
        n_slots = max(len(items.slot_ids), 2)
        H = np.array([items.entropy(i) for i in items.items]) / np.log2(n_slots)
        t = st.t
        open_c = list(st.verifier.open.values())
        slack = min([oc.claim.t_due - t for oc in open_c if t >= oc.claim.t_ref], default=60.0)
        urg = sum(st.verifier.urgency(t).values())
        goal = st.goal
        batt = (self.bundle.battery_wh or st.battery_cap) / st.battery_cap
        g = [x / W, y / D, np.cos(th), np.sin(th), st.ekf.pos_std, min(st.ekf.nis_avg / 25.0, 4.0)]
        g += [float(mode == m) for m in MODES]
        g += [batt, t / self.T, len(st.tracker.confirmed()) / 5.0,
              min(st._human_clearance(st.ekf.x, st.tracker.confirmed()), 5.0) / 5.0,
              float(H.mean()) if len(H) else 0.0, float(H.max()) if len(H) else 0.0, len(open_c) / 10.0,
              min(urg, 50.0) / 10.0, float(np.clip(slack, 0.0, 60.0)) / 60.0,
              float(np.mean([oc.direct for oc in open_c])) if open_c else 0.0, float(st._path_entropy),
              float(goal is not None),
              float(np.linalg.norm(goal.pose[:2] - st.ekf.x[:2])) / 5.0 if goal is not None else 0.0]
        c = []
        for cand in self._candidates:
            if cand is None:
                c += [0.0] * len(CANDIDATE_FEATURES)
                continue
            b = cand.breakdown
            c += [1.0, b.get("eig", 0.0), b.get("modal", 0.0), float(np.log1p(max(b.get("unc", 0.0), 0.0))),
                  b.get("path", 0.0) / 5.0, b.get("risk", 0.0), b.get("sterile", 0.0)]
        obs = np.asarray(g + c, np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

    def close(self) -> None:
        if getattr(self, "be", None) is not None and not self.done:
            self.be.close()


# ---------------------------------------------------------------------------
def make_gymnasium_env(**kwargs):
    """``gymnasium.Env`` adapter (reset -> (obs, info), step -> 5-tuple); needs ``gymnasium``."""
    import gymnasium
    from gymnasium import spaces

    class GymnasiumNbvEnv(gymnasium.Env):
        metadata = {"render_modes": []}

        def __init__(self, **kw):
            self.env = NbvEnv(**kw)
            self.observation_space = spaces.Box(-np.inf, np.inf, self.env.observation_space.shape, np.float32)
            if isinstance(self.env.action_space, DiscreteSpace):
                self.action_space = spaces.Discrete(self.env.action_space.n)
            else:
                self.action_space = spaces.Box(-1.0, 1.0, self.env.action_space.shape, np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            if seed is None:
                seed = int(self.np_random.integers(1, 2**31 - 1))
            return self.env.reset(seed), {"seed": self.env.seed}

        def step(self, action):
            obs, r, done, info = self.env.step(action)
            return obs, r, done, False, info

        def close(self):
            self.env.close()

    return GymnasiumNbvEnv(**kwargs)


def register(env_id: str = GYMNASIUM_ID) -> bool:
    """Register with gymnasium if installed (idempotent); returns True when registered."""
    try:
        import gymnasium
    except ImportError:
        return False
    if env_id not in gymnasium.registry:
        gymnasium.register(id=env_id, entry_point="medortrace.autonomy.gym_env:make_gymnasium_env")
    return True


register()
