"""Staff agents in Isaac Sim.

The *behaviour* (task schedule, social-force navigation, robot yielding and
the robot-free shadow twin) is the engine-agnostic
``medortrace.world.agents.StaffPopulation``.  This module only moves the USD
proxies each control tick:

* ``capsule`` mode (default, deterministic, fast): kinematic capsules authored
  at ``/World/Staff/<safe(name)>`` by ``medortrace.usd.scene_builder`` are
  teleported with world-pose writes; being kinematic rigid bodies they still
  push dynamic objects and are seen by RTX lidar/radar/camera.
* ``people`` mode (EXPERIMENTAL): the capsules become invisible collision
  proxies and animated characters (Isaac ``People`` assets, referenced under
  ``/World/StaffCharacters``) follow the same trajectory for photoreal camera
  data.  The walk cycle is driven through ``omni.anim.graph.core`` variables
  when that API is available; asset names and graph variables are release
  dependent, so any failure falls back to capsules with a warning.
"""

from __future__ import annotations

import numpy as np

from medortrace.isaac.compat import XformGroup, assets_root_path
from medortrace.usd.scene_builder import safe

CAPSULE_Z = 0.875   # capsule centre height authored by the scene builder (1.75 m tall)
# role -> Isaac People character (``<assets>/Isaac/People/Characters/<name>/<name>.usd``)
CHARACTERS = {"surgeon": "M_Medical_01", "assistant": "F_Medical_01", "scrub_nurse": "F_Medical_01",
              "circulator": "M_Medical_01", "anesthetist": "F_Medical_01"}


def yaw_quats(headings: np.ndarray) -> np.ndarray:
    h = np.asarray(headings, float).reshape(-1)
    return np.stack([np.cos(h / 2), np.zeros_like(h), np.zeros_like(h), np.sin(h / 2)], axis=1)


class StaffDriver:
    def __init__(self, names: list[str], mode: str = "capsule", roles: dict[str, str] | None = None):
        self.names = list(names)
        self.roles = roles or {}
        self.paths = [f"/World/Staff/{safe(n)}" for n in self.names]
        self.mode = mode
        self.proxies = XformGroup(self.paths)
        self.characters: XformGroup | None = None
        self._anim = {}
        if mode == "people":
            self._setup_characters()

    def _setup_characters(self) -> None:  # pragma: no cover - requires Isaac People assets
        try:
            import omni.usd
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            root = assets_root_path()
            if root is None:
                raise RuntimeError("Isaac asset root not reachable")
            paths = []
            for n in self.names:
                ch = CHARACTERS.get(self.roles.get(n, ""), "M_Medical_01")
                p = f"/World/StaffCharacters/{safe(n)}"
                prim = stage.DefinePrim(p, "Xform")
                prim.GetReferences().AddReference(f"{root}/Isaac/People/Characters/{ch}/{ch}.usd")
                paths.append(p)
            for p in self.paths:
                UsdGeom.Imageable(stage.GetPrimAtPath(p)).MakeInvisible()
            self.characters = XformGroup(paths)
            try:
                import omni.anim.graph.core as ag
                self._anim = {p: ag.get_character(p) for p in paths}
            except Exception as e:
                print(f"[medortrace] animation graph unavailable ({e}); characters will glide without a walk cycle")
        except Exception as e:
            print(f"[medortrace] people mode unavailable ({e}); using capsule proxies")
            self.mode = "capsule"
            self.characters = None

    def apply(self, positions: np.ndarray, headings: np.ndarray, speeds: np.ndarray | None = None) -> None:
        pos = np.asarray(positions, float).reshape(-1, 2)
        if not len(pos):
            return
        q = yaw_quats(headings)
        self.proxies.set_world_poses(np.column_stack([pos, np.full(len(pos), CAPSULE_Z)]), q)
        if self.characters is not None:  # pragma: no cover - people mode
            self.characters.set_world_poses(np.column_stack([pos, np.zeros(len(pos))]), q)
            for (p, ch), s in zip(self._anim.items(), speeds if speeds is not None else np.zeros(len(pos))):
                try:
                    ch.set_variable("Action", "Walk" if s > 0.1 else "Idle")
                    ch.set_variable("Walk", float(s))
                except Exception:
                    pass
