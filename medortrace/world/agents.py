"""Staff agents: task-driven social-force pedestrians with robot-yielding.

Each episode simulates two copies of the staff population in lockstep:

* ``actual`` - interacts with the robot (repulsion + yielding);
* ``shadow`` - identical tasks and identical noise draws, but the robot is
  absent.

The difference between the two is the robot's *imposed* effect on people:
task delay (arrival-time differences at task goals) and path disruption
(extra distance travelled / lateral deviation).  This counterfactual shadow is
cheap in simulation and impossible to measure in a real OR.

In Isaac Sim the same controller drives kinematic capsule proxies or
``omni.anim.people`` characters via ``medortrace.isaac.staff``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medortrace.planning.grid import GridSpec, astar, nearest_free, rasterize_boxes, smooth_path
from medortrace.world.scene import SceneSpec, StaffSpec
from medortrace.world.workflow import StaffTask


@dataclass
class AgentState:
    spec: StaffSpec
    pos: np.ndarray
    vel: np.ndarray
    tasks: list[StaffTask]
    rng: np.random.Generator
    task_idx: int = 0
    path: list[np.ndarray] = field(default_factory=list)
    dwell_until: float = -1.0
    active_goal: np.ndarray | None = None
    arrivals: dict[int, float] = field(default_factory=dict)
    distance: float = 0.0
    sway: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @property
    def heading(self) -> float:
        return float(np.arctan2(self.vel[1], self.vel[0])) if np.linalg.norm(self.vel) > 1e-3 else 0.0


class StaffPopulation:
    def __init__(self, spec: SceneSpec, tasks: dict[str, list[StaffTask]], streams, robot_aware: bool,
                 params: dict | None = None):
        self.spec = spec
        self.robot_aware = robot_aware
        p = params or {}
        self.A = p.get("A_agent", 2.0)
        self.B = p.get("B_agent", 0.3)
        self.A_robot = p.get("A_robot", 4.0)
        self.B_robot = p.get("B_robot", 0.45)
        self.tau = p.get("tau", 0.5)
        self.yield_dist = p.get("yield_distance", 1.3)
        W, D, _ = spec.room
        self.grid = GridSpec(np.array([0.0, 0.0]), 0.1, (int(np.ceil(W / 0.1)), int(np.ceil(D / 0.1))))
        boxes = [o.box for o in spec.objects if o.kind not in ("light_head", "monitor")]
        self._boxes = [o.box for o in spec.objects if o.kind not in ("wall", "light_head", "monitor")]
        base = rasterize_boxes(self.grid, boxes, inflate=0.25, z_band=(0.02, 1.8))
        pts = self.grid.centers().reshape(-1, 2)
        keep = np.zeros(len(pts), dtype=bool)
        for z in spec.sterile_zones:
            keep |= z.box.contains_xy(pts, margin=0.25)
        self.lethal_nonsterile = base | keep.reshape(self.grid.shape)
        self.lethal_sterile = base
        self.agents: list[AgentState] = []
        for s in spec.staff:
            # identical noise stream for actual/shadow copies of the same agent
            rng = streams.fork("agents", f"agent:{s.name}")
            self.agents.append(AgentState(s, s.home.astype(float).copy(), np.zeros(2),
                                          list(tasks.get(s.name, [])), rng))

    # ------------------------------------------------------------------
    def _plan(self, a: AgentState, goal: np.ndarray) -> list[np.ndarray]:
        lethal = self.lethal_sterile if a.spec.sterile else self.lethal_nonsterile
        s = nearest_free(lethal, tuple(self.grid.world_to_cell(a.pos)[0]))
        g = nearest_free(lethal, tuple(self.grid.world_to_cell(goal)[0]))
        if s is None or g is None:
            return [goal]
        cells = astar(np.zeros(lethal.shape), lethal, s, g)
        if cells is None:
            return [goal]
        xy = self.grid.cell_to_world(np.array(cells))
        xy = smooth_path(xy, lethal, self.grid)
        return [p for p in xy[1:]] + [goal]

    def step(self, t: float, dt: float, robot_xy: np.ndarray | None, robot_vel: np.ndarray | None) -> None:
        forces = []
        for a in self.agents:
            noise = a.rng.normal(0.0, 1.0, 2)  # always drawn -> identical streams in shadow
            if a.spec.roaming:
                # task scheduling
                if a.task_idx < len(a.tasks) and t >= a.tasks[a.task_idx].t_start and a.active_goal is None:
                    a.active_goal = a.tasks[a.task_idx].goal.copy()
                    a.path = self._plan(a, a.active_goal)
                desired = np.zeros(2)
                if a.active_goal is not None and t >= a.dwell_until:
                    while a.path and np.linalg.norm(a.path[0] - a.pos) < 0.25:
                        a.path.pop(0)
                    if a.path:
                        e = a.path[0] - a.pos
                        desired = e / (np.linalg.norm(e) + 1e-9) * a.spec.speed
                    else:
                        a.arrivals[a.task_idx] = t
                        a.dwell_until = t + a.tasks[a.task_idx].dwell
                        a.task_idx += 1
                        a.active_goal = None
                f = (desired - a.vel) / self.tau + 0.15 * noise
            else:
                # scrubbed-in staff: small Ornstein-Uhlenbeck sway around home
                a.sway += dt * (-0.5 * a.sway) + np.sqrt(dt) * 0.05 * noise
                target = a.spec.home + a.sway
                f = (target - a.pos) * 2.0 - a.vel * 2.0
            # agent-agent repulsion
            for b in self.agents:
                if b is a:
                    continue
                d = a.pos - b.pos
                dist = np.linalg.norm(d) + 1e-9
                f = f + self.A * np.exp((a.spec.radius + b.spec.radius - dist) / self.B) * d / dist * (0.3 if not a.spec.roaming else 1.0)
            # obstacle repulsion (roaming only; sterile staff hold position)
            if a.spec.roaming:
                for bx in self._boxes:
                    dd = float(bx.distance_xy(a.pos[None])[0])
                    if dd < 0.6:
                        loc = a.pos - bx.center[:2]
                        f = f + 1.5 * np.exp(-dd / 0.15) * loc / (np.linalg.norm(loc) + 1e-9)
            # robot interaction
            if self.robot_aware and robot_xy is not None:
                d = a.pos - robot_xy
                dist = np.linalg.norm(d) + 1e-9
                gain = 1.0 if a.spec.roaming else 0.25
                f = f + gain * self.A_robot * np.exp((a.spec.radius + 0.35 - dist) / self.B_robot) * d / dist
            forces.append(f)
        for a, f in zip(self.agents, forces):
            a.vel = a.vel + dt * f
            vmax = a.spec.speed * 1.3
            # yielding: slow down when the robot is ahead within the yield distance
            if self.robot_aware and robot_xy is not None and a.spec.roaming:
                d = robot_xy - a.pos
                dist = np.linalg.norm(d)
                sp = np.linalg.norm(a.vel)
                if dist < self.yield_dist and sp > 1e-3 and (d @ a.vel) / (dist * sp + 1e-9) > 0.5:
                    vmax *= max(0.25, (dist - 0.5) / (self.yield_dist - 0.5))
            sp = np.linalg.norm(a.vel)
            if sp > vmax:
                a.vel *= vmax / sp
            step = a.vel * dt
            a.pos = a.pos + step
            a.distance += float(np.linalg.norm(step))

    # ------------------------------------------------------------------
    def positions(self) -> np.ndarray:
        return np.array([a.pos for a in self.agents])

    def velocities(self) -> np.ndarray:
        return np.array([a.vel for a in self.agents])

    def names(self) -> list[str]:
        return [a.spec.name for a in self.agents]

    def get(self, name: str) -> AgentState:
        for a in self.agents:
            if a.spec.name == name:
                return a
        raise KeyError(name)
