"""Robot articulation control in Isaac Sim (PhysX).

* Differential drive: (v, omega) -> wheel angular velocity targets
  (inverse kinematics of a diff-drive with wheel radius r and track b):
      w_left  = (v - omega * b / 2) / r
      w_right = (v + omega * b / 2) / r
  Commands are rate-limited with the same acceleration limits as the lite
  simulator so both backends share controller tuning.
* Retrieval arm: shoulder position target + gripper jaw target, measured joint
  efforts returned for the contact gate (``medortrace.manipulation``).
* Energy: electrical power estimated from wheel torque x speed / efficiency
  + idle + sensor load, integrated into the battery state.
"""

from __future__ import annotations

import numpy as np

from medortrace.isaac.compat import articulation_cls


class RobotController:
    def __init__(self, prim_path: str = "/World/Robot", rig: dict | None = None, cfg: dict | None = None):
        rig = rig or {}
        r = (cfg or {}).get("robot", {})
        self.r = float(rig.get("wheel_radius", 0.085))
        self.b = float(rig.get("wheel_track", 0.44))
        self.max_v = float(r.get("max_v", 0.7))
        self.max_w = float(r.get("max_omega", 1.2))
        self.acc = float(r.get("max_acc", 0.6))
        self.alpha = float(r.get("max_alpha", 1.8))
        self.p_idle = float(r.get("power_idle_w", 38.0)) + float(r.get("power_sensors_w", 27.0))
        self.eta = float(r.get("drivetrain_efficiency", 0.75))
        self.battery_wh = float(r.get("battery_capacity_wh", 480.0))
        self.energy_used = 0.0
        Art = articulation_cls()
        self.art = Art(prim_path)
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self._dof = None

    def initialize(self) -> None:
        if hasattr(self.art, "initialize"):
            try:
                self.art.initialize()
            except TypeError:
                pass
        names = list(getattr(self.art, "dof_names", []) or getattr(self.art, "joint_names", []))
        self._dof = {n: i for i, n in enumerate(names)}

    def _set_velocity_targets(self, idx: list[int], vel: np.ndarray) -> None:
        if hasattr(self.art, "set_dof_velocity_targets"):          # experimental API (5.x)
            self.art.set_dof_velocity_targets(vel[None], dof_indices=idx)
        else:                                                         # SingleArticulation (4.x)
            from isaacsim.core.utils.types import ArticulationAction  # type: ignore
            self.art.apply_action(ArticulationAction(joint_velocities=vel, joint_indices=np.array(idx)))

    def command(self, v: float, omega: float, dt: float) -> None:
        v = float(np.clip(v, -0.3, self.max_v))
        omega = float(np.clip(omega, -self.max_w, self.max_w))
        self.v_cmd += float(np.clip(v - self.v_cmd, -self.acc * dt, self.acc * dt))
        self.w_cmd += float(np.clip(omega - self.w_cmd, -self.alpha * dt, self.alpha * dt))
        wl = (self.v_cmd - self.w_cmd * self.b / 2) / self.r
        wr = (self.v_cmd + self.w_cmd * self.b / 2) / self.r
        idx = [self._dof.get("left_wheel_joint", 0), self._dof.get("right_wheel_joint", 1)]
        self._set_velocity_targets(idx, np.array([wl, wr]))

    def efforts(self) -> np.ndarray:
        for fn in ("get_dof_projected_joint_forces", "get_measured_joint_efforts", "get_applied_joint_efforts"):
            if hasattr(self.art, fn):
                try:
                    return np.asarray(getattr(self.art, fn)()).reshape(-1)
                except Exception:
                    continue
        return np.zeros(len(self._dof or {}))

    def update_energy(self, dt: float, extra_w: float = 0.0) -> float:
        e = self.efforts()
        try:
            vel = np.asarray(self.art.get_joint_velocities()).reshape(-1)
        except Exception:
            vel = np.zeros_like(e)
        wheels = [self._dof.get("left_wheel_joint", 0), self._dof.get("right_wheel_joint", 1)]
        p_mech = float(np.sum(np.abs(e[wheels] * vel[wheels]))) if len(e) > max(wheels) and len(vel) > max(wheels) else 0.0
        p = self.p_idle + extra_w + p_mech / self.eta
        self.energy_used += p * dt / 3600.0
        return max(0.0, self.battery_wh - self.energy_used)

    def arm(self, shoulder_deg: float, jaw_m: float) -> None:
        idx = [self._dof.get("arm_shoulder", 2), self._dof.get("gripper_jaw", 3)]
        tgt = np.array([np.deg2rad(shoulder_deg), jaw_m])
        if hasattr(self.art, "set_dof_position_targets"):
            self.art.set_dof_position_targets(tgt[None], dof_indices=idx)
        else:
            from isaacsim.core.utils.types import ArticulationAction  # type: ignore
            self.art.apply_action(ArticulationAction(joint_positions=tgt, joint_indices=np.array(idx)))
