"""Episode runner: backend <-> autonomy stack loop, simulated operator,
truth logging, metrics and dataset export.

``sim_overrides`` (sim-to-real sensitivity) is deep-merged into the config
seen by ``build_episode`` and the backend *only*: it perturbs the world (sensor
physics, staff behaviour, robot dynamics, workflow realism), while the autonomy
stack keeps the unperturbed config because its sensor / motion models encode
what the robot *believes* the world to be.  The episode's workflow protocol
(claim deadlines) is part of the world and is read from the episode.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medortrace.autonomy.stack import AutonomyStack, stack_inputs_from_episode
from medortrace.common.config import deep_merge
from medortrace.common.msgs import VelocityCommand
from medortrace.data.writer import TrajectoryWriter
from medortrace.eval.metrics import TruthLog, compute_metrics
from medortrace.sim.episode import build_episode


@dataclass
class EpisodeResult:
    scenario_id: str
    seed: int
    policy: str
    metrics: dict
    wall_time_s: float
    out_dir: str | None = None


class SimulatedOperator:
    """Remote operator answering HANDOVER requests after a response delay.

    On acknowledgement the operator re-localises the robot from the OR camera
    system (truth + small noise), then control returns to the stack.
    """

    def __init__(self, rng: np.random.Generator, delay_s=(8.0, 20.0)):
        self.rng = rng
        self.delay = delay_s
        self.pending_since = None
        self.due = None
        self.log: list[dict] = []

    def step(self, t: float, stack: AutonomyStack, truth_pose: np.ndarray) -> None:
        if stack.operator_request_open and self.pending_since is None:
            self.pending_since = t
            self.due = t + float(self.rng.uniform(*self.delay))
        if self.pending_since is not None and t >= self.due:
            hint = truth_pose + self.rng.normal(0, [0.02, 0.02, 0.01])
            stack.operator_intervention(hint, t)
            self.log.append({"t_request": self.pending_since, "t_ack": t, "duration": t - self.pending_since,
                             "action": "relocalise+resume"})
            self.pending_since = None
            self.due = None


def make_backend(name: str, cfg: dict):
    if name == "lite":
        from medortrace.sim.lite_backend import LiteBackend
        return LiteBackend(cfg)
    if name == "isaac":
        from medortrace.isaac.backend import IsaacBackend  # requires Isaac Sim python
        return IsaacBackend(cfg)
    raise ValueError(name)


def run_episode(cfg: dict, seed: int, backend: str = "lite", out_dir: str | Path | None = None,
                policy: str | None = None, duration: float | None = None, save_raw: bool = False,
                autonomy_override: dict | None = None, verbose: bool = False,
                sim_overrides: dict | None = None) -> EpisodeResult:
    t_wall = time.time()
    if policy:
        cfg = deep_merge(cfg, {"autonomy": {"policy": policy}})
    if autonomy_override:
        cfg = deep_merge(cfg, {"autonomy": autonomy_override})
    if duration:
        cfg = deep_merge(cfg, {"episode": {"duration_s": float(duration)}})
    cfg = deep_merge(cfg, {"_policy_seed": int(seed) * 7919 + 17})
    sim_cfg = deep_merge(cfg, sim_overrides) if sim_overrides else cfg   # world only; the stack keeps ``cfg``
    ep = build_episode(sim_cfg, seed)
    be = make_backend(backend, sim_cfg)
    bundle = be.reset(ep)
    stack = AutonomyStack(stack_inputs_from_episode(ep), cfg)
    op = SimulatedOperator(ep.streams.fork("robot", "operator"))
    writer = TrajectoryWriter(out_dir, save_raw) if out_dir else None
    tl = TruthLog()
    verdicts = []
    dt = be.dt
    T = ep.workflow.duration
    cmd = VelocityCommand()
    while be.t < T - 1e-9:
        cmd = stack.step(bundle, dt)
        verdicts += stack.telemetry[-1].verdicts
        if writer:
            writer.add_raw(bundle.t, bundle)
        bundle = be.step(cmd)
        tr = be.truth()
        op.step(be.t, stack, tr.robot_pose)
        tl.t.append(tr.t)
        tl.robot.append(tr.robot_pose)
        tl.agents.append(tr.agent_pos)
        tl.shadow.append(tr.shadow_pos)
        tl.agent_names = tr.agent_names
        tl.collision_agent.append(tr.collision_agent)
        tl.collision_static.append(tr.collision_static)
        tl.in_keepout.append(tr.in_keepout)
        tl.in_keepout_margin.append(bool(ep.spec.in_keepout(tr.robot_pose[None, :2])[0]))
        tl.battery.append(tr.battery_wh)
        tl.energy.append(tr.energy_used_wh)
        tl.item_slots.append(tr.item_slots)
        tl.fault_any.append(ep.faults.any_sensor_fault(tr.t))
        if verbose and int(be.t / dt) % 100 == 0:
            s = stack.telemetry[-1]
            print(f"t={be.t:6.1f} mode={s.mode:8s} pose_err={np.linalg.norm(s.pose_est[:2]-tr.robot_pose[:2]):.2f} "
                  f"goal={s.target_slot} v={cmd.v:.2f} tracks={s.n_tracks} verdicts={len(verdicts)}")
    # one last stack step consumes the final bundle, then close open claims
    stack.step(bundle, dt)
    verdicts += stack.telemetry[-1].verdicts
    verdicts += stack.finalize(T + ep.cfg.get("workflow", {}).get("claim_grace_s", 25.0))
    stack.telemetry.pop()  # keep telemetry aligned with truth log
    metrics = compute_metrics(ep, tl, stack, verdicts, be, op.log, be.rp.battery_wh if hasattr(be, "rp") else 480.0)
    metrics["provenance_chain_valid"] = float(stack.prov.verify_chain())
    metrics["wall_time_s"] = time.time() - t_wall
    pol = cfg.get("autonomy", {}).get("policy", "active")
    out = None
    if writer:
        name = f"{ep.spec.scenario_id or 'scenario'}__s{seed}__{pol}"
        out = str(writer.write(name, ep, stack, tl, metrics, verdicts, op.log, backend, pol))
    be.close()
    return EpisodeResult(ep.spec.scenario_id, seed, pol, metrics, time.time() - t_wall, out)
