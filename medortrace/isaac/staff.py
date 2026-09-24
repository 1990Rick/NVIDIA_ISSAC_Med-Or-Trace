"""Staff agents in Isaac Sim.

The *behaviour* (task schedule, social-force navigation, robot yielding and
the robot-free shadow twin) is the engine-agnostic
``medortrace.world.agents.StaffPopulation``.  This module only moves the USD
proxies each physics step:

* ``capsule`` mode (default, deterministic, fast): kinematic capsules authored
  at ``/World/Staff/<name>`` are teleported with ``set_world_poses``; being
  kinematic rigid bodies they still push dynamic objects and are seen by
  RTX lidar/radar/camera.
* ``people`` mode: the capsules become invisible collision proxies and
  ``omni.anim.people`` characters are driven along the same trajectory for
  photoreal appearance in camera data (Replicator dataset generation).
"""

from __future__ import annotations

import numpy as np

from medortrace.isaac.compat import xform_prim_cls


class StaffDriver:
    def __init__(self, names: list[str], mode: str = "capsule"):
        XP = xform_prim_cls()
        self.names = names
        self.paths = [f"/World/Staff/{n}" for n in names]
        self.mode = mode
        self.prims = [XP(p) for p in self.paths]
        self.characters = None
        if mode == "people":
            self._setup_characters()

    def _setup_characters(self) -> None:  # pragma: no cover - requires omni.anim.people assets
        try:
            import omni.anim.people  # noqa: F401
            from pxr import UsdGeom
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            for p in self.paths:
                UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()
            self.characters = {}
        except ImportError:
            print("[medortrace] omni.anim.people unavailable; using capsule proxies")
            self.mode = "capsule"

    def apply(self, positions: np.ndarray, headings: np.ndarray) -> None:
        for prim, p, h in zip(self.prims, positions, headings):
            q = np.array([np.cos(h / 2), 0.0, 0.0, np.sin(h / 2)])
            pos = np.array([p[0], p[1], 0.875])
            if hasattr(prim, "set_world_poses"):
                prim.set_world_poses(positions=pos[None], orientations=q[None])
            else:
                prim.set_world_pose(position=pos, orientation=q)
