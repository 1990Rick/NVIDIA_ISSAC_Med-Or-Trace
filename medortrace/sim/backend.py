"""Simulator backend interface shared by the lite simulator and Isaac Sim."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from medortrace.common.msgs import SensorBundle, VelocityCommand
from medortrace.sim.episode import Episode


@dataclass
class TruthSnapshot:
    t: float
    robot_pose: np.ndarray              # (3,) x, y, theta
    robot_vel: np.ndarray               # (2,) v, omega
    agent_names: list[str]
    agent_pos: np.ndarray               # (N,2) with robot
    agent_vel: np.ndarray
    shadow_pos: np.ndarray              # (N,2) robot-free counterfactual
    item_slots: dict[str, str]
    item_pos: dict[str, np.ndarray]
    collision_agent: bool = False
    collision_static: bool = False
    contact_force: float = 0.0
    battery_wh: float = 0.0
    energy_used_wh: float = 0.0
    in_keepout: bool = False
    fault_active: dict = field(default_factory=dict)


class SimBackend(abc.ABC):
    episode: Episode

    @abc.abstractmethod
    def reset(self, episode: Episode) -> SensorBundle: ...

    @abc.abstractmethod
    def step(self, cmd: VelocityCommand) -> SensorBundle: ...

    @abc.abstractmethod
    def truth(self) -> TruthSnapshot: ...

    @property
    @abc.abstractmethod
    def t(self) -> float: ...

    @property
    @abc.abstractmethod
    def dt(self) -> float: ...

    def close(self) -> None:  # pragma: no cover - optional
        pass
