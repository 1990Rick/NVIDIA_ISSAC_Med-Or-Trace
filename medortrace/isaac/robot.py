"""Robot articulation control in Isaac Sim (PhysX).

* Differential drive: (v, omega) -> wheel angular velocity targets
  (inverse kinematics of a diff-drive with wheel radius r and track b):
      w_left  = (v - omega * b / 2) / r
      w_right = (v + omega * b / 2) / r
  Commands are rate-limited with the same acceleration limits as the lite
  simulator so both backends share controller tuning.  Both wheel joints
  rotate about +Y, so positive targets drive the base forward (+X).
* Retrieval arm: shoulder position target + gripper jaw target, measured joint
  efforts returned for the contact gate (``medortrace.manipulation``).
* Energy: electrical power estimated from wheel torque x speed / efficiency
  + idle + sensor load, integrated into the battery state.

Works with the stable ``SingleArticulation`` (4.5/5.x) and the 5.x
experimental ``Articulation`` (see ``compat.PRIMS_API``).  The articulation
root is the prim carrying ``UsdPhysics.ArticulationRootAPI`` (``base_link`` in
the rig authored by ``medortrace.usd.robot_rig``), resolved with
:func:`find_articulation_root`.
"""

from __future__ import annotations

import numpy as np

from medortrace.isaac.compat import articulation_action_cls, articulation_cls, is_experimental, to_numpy

WHEELS = ("left_wheel_joint", "right_wheel_joint")
ARM = ("arm_shoulder", "gripper_jaw")


def find_articulation_root(stage, prim_path: str) -> str:
    """Path of the prim with ``ArticulationRootAPI`` at or below ``prim_path`` (pxr only)."""
    from pxr import Usd, UsdPhysics
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise KeyError(f"no prim at {prim_path}")
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(p.GetPath())
    return prim_path


def yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class RobotController:
    def __init__(self, prim_path: str = "/World/Robot/base_link", rig: dict | None = None, cfg: dict | None = None):
        rig = rig or {}
        r = (cfg or {}).get("robot", {})
        self.prim_path = prim_path
        self.r = float(rig.get("wheel_radius", 0.085))
        self.b = float(rig.get("wheel_track", 0.44))
        self.max_v = float(r.get("max_v", 0.7))
        self.max_w = float(r.get("max_omega", 1.2))
        self.acc = float(r.get("max_acc", 0.6))
        self.alpha = float(r.get("max_alpha", 1.8))
        self.p_idle = float(r.get("power_idle_w", 38.0)) + float(r.get("power_sensors_w", 27.0))
        self.eta = float(r.get("drivetrain_efficiency", 0.75))
        self.battery_wh = float(r.get("battery_capacity_wh", 480.0))   # backend sets the episode's start charge
        self.energy_used = 0.0
        Art = articulation_cls()
        self.art = Art(prim_path)
        self.experimental = is_experimental(self.art)
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self._dof: dict[str, int] = {}

    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Call after ``World.reset()`` (physics views exist only while simulating)."""
        if not self.experimental and hasattr(self.art, "initialize"):
            try:
                self.art.initialize()
            except TypeError:
                pass
        names: list[str] = []
        for attr in ("dof_names", "joint_names"):
            v = getattr(self.art, attr, None)
            if v:
                names = list(v)
                break
        self._dof = {n: i for i, n in enumerate(names)}
        missing = [j for j in WHEELS if j not in self._dof]
        if missing:
            print(f"[medortrace] WARNING: wheel joints {missing} not found in articulation DOFs {names}")

    def _idx(self, names: tuple[str, ...], defaults: tuple[int, ...]) -> list[int]:
        return [self._dof.get(n, d) for n, d in zip(names, defaults)]

    def _set_velocity_targets(self, idx: list[int], vel: np.ndarray) -> None:
        if self.experimental:
            self.art.set_dof_velocity_targets(vel[None], dof_indices=idx)
        else:
            Action = articulation_action_cls()
            self.art.apply_action(Action(joint_velocities=vel, joint_indices=np.array(idx)))

    # ------------------------------------------------------------------
    def command(self, v: float, omega: float, dt: float) -> None:
        v = float(np.clip(v, -0.3, self.max_v))
        omega = float(np.clip(omega, -self.max_w, self.max_w))
        self.v_cmd += float(np.clip(v - self.v_cmd, -self.acc * dt, self.acc * dt))
        self.w_cmd += float(np.clip(omega - self.w_cmd, -self.alpha * dt, self.alpha * dt))
        wl = (self.v_cmd - self.w_cmd * self.b / 2) / self.r
        wr = (self.v_cmd + self.w_cmd * self.b / 2) / self.r
        self._set_velocity_targets(self._idx(WHEELS, (0, 1)), np.array([wl, wr]))

    def joint_velocities(self) -> np.ndarray:
        for fn in ("get_dof_velocities", "get_joint_velocities"):
            if hasattr(self.art, fn):
                try:
                    return to_numpy(getattr(self.art, fn)()).reshape(-1)
                except Exception:
                    continue
        return np.zeros(len(self._dof))

    def measured_twist(self) -> tuple[float, float]:
        """(v, omega) from wheel joint velocities; falls back to the commanded twist."""
        jv = self.joint_velocities()
        il, ir = self._idx(WHEELS, (0, 1))
        if len(jv) > max(il, ir) and np.all(np.isfinite(jv[[il, ir]])):
            wl, wr = float(jv[il]), float(jv[ir])
            return self.r * (wl + wr) / 2, self.r * (wr - wl) / self.b
        return self.v_cmd, self.w_cmd

    def world_pose(self) -> np.ndarray:
        """(x, y, yaw) of the articulation root from the physics view."""
        if self.experimental:
            p, q = self.art.get_world_poses()
            p, q = to_numpy(p).reshape(-1, 3)[0], to_numpy(q).reshape(-1, 4)[0]
        else:
            p, q = self.art.get_world_pose()
            p, q = to_numpy(p).reshape(3), to_numpy(q).reshape(4)
        return np.array([p[0], p[1], yaw_from_quat_wxyz(q)])

    def efforts(self) -> np.ndarray:
        for fn in ("get_dof_projected_joint_forces", "get_measured_joint_efforts", "get_applied_joint_efforts"):
            if hasattr(self.art, fn):
                try:
                    return to_numpy(getattr(self.art, fn)()).reshape(-1)
                except Exception:
                    continue
        return np.zeros(len(self._dof))

    def arm_efforts(self) -> np.ndarray:
        e = self.efforts()
        idx = self._idx(ARM, (2, 3))
        return e[idx] if len(e) > max(idx) else np.zeros(len(idx))

    def update_energy(self, dt: float, extra_w: float = 0.0) -> float:
        e = self.efforts()
        vel = self.joint_velocities()
        wheels = self._idx(WHEELS, (0, 1))
        ok = len(e) > max(wheels) and len(vel) > max(wheels)
        p_mech = float(np.sum(np.abs(e[wheels] * vel[wheels]))) if ok else 0.0
        p = self.p_idle + extra_w + p_mech / self.eta
        self.energy_used += p * dt / 3600.0
        return max(0.0, self.battery_wh - self.energy_used)

    def arm(self, shoulder_deg: float, jaw_m: float) -> None:
        idx = self._idx(ARM, (2, 3))
        tgt = np.array([np.deg2rad(shoulder_deg), jaw_m])
        if self.experimental:
            self.art.set_dof_position_targets(tgt[None], dof_indices=idx)
        else:
            Action = articulation_action_cls()
            self.art.apply_action(Action(joint_positions=tgt, joint_indices=np.array(idx)))
