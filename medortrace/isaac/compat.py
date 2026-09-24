"""Version shims for Isaac Sim APIs.

Isaac Sim 4.5 renamed ``omni.isaac.*`` extensions to ``isaacsim.*`` and 5.x
added the engine-agnostic ``isaacsim.core.experimental`` API.  Every symbol
the project uses is resolved here once, so the rest of ``medortrace.isaac``
is version-independent.

Prim API preference
    ``MEDORTRACE_ISAAC_PRIMS`` selects the prim wrappers used for the robot
    articulation and for teleporting staff/item proxies:

    * ``stable`` (default) - ``isaacsim.core.prims.Single*`` (4.5 and 5.x), which
      follow the ``World``/``SimulationContext`` life cycle the backend uses;
    * ``experimental`` - ``isaacsim.core.experimental.prims`` (5.x), batched and
      returning warp arrays (converted with :func:`to_numpy`).

    The legacy ``omni.isaac.core`` classes (<= 4.2) are the last fallback either way.

Nothing in this module imports ``omni``/``isaacsim`` at import time, so it is
safe to import before ``SimulationApp`` exists (and outside Isaac Sim).
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

PRIMS_API = os.environ.get("MEDORTRACE_ISAAC_PRIMS", "stable").strip().lower()

# new extension name -> older names (tried in order when the new one is missing)
EXTENSION_ALIASES: dict[str, list[str]] = {
    "isaacsim.sensors.rtx": ["omni.isaac.sensor"],
    "isaacsim.sensors.physics": ["omni.isaac.sensor"],
    "isaacsim.sensors.camera": ["omni.isaac.sensor"],
    "isaacsim.robot.wheeled_robots": ["omni.isaac.wheeled_robots"],
    "isaacsim.ros2.bridge": ["omni.isaac.ros2_bridge"],
    "isaacsim.core.nodes": ["omni.isaac.core_nodes"],
    "isaacsim.sensors.rtx.acoustic": ["omni.sensors.nv.acoustic"],
    "omni.anim.people": ["isaacsim.anim.people"],
}


def first_import(*candidates: str) -> Any:
    """Return the first importable ``module[:attr]`` from candidates."""
    errors = []
    for c in candidates:
        mod, _, attr = c.partition(":")
        try:
            m = importlib.import_module(mod)
            return getattr(m, attr) if attr else m
        except (ImportError, AttributeError) as e:  # pragma: no cover - depends on Isaac version
            errors.append(f"{c}: {e}")
    raise ImportError("none of the candidates are available:\n  " + "\n  ".join(errors))


def _ordered(stable: list[str], experimental: list[str], legacy: list[str]) -> list[str]:
    return (experimental + stable if PRIMS_API == "experimental" else stable + experimental) + legacy


def simulation_app_cls():
    return first_import("isaacsim:SimulationApp", "isaacsim.simulation_app:SimulationApp",
                        "omni.isaac.kit:SimulationApp")


def world_cls():
    return first_import("isaacsim.core.api:World", "omni.isaac.core:World")


def articulation_cls():
    return first_import(*_ordered(["isaacsim.core.prims:SingleArticulation"],
                                  ["isaacsim.core.experimental.prims:Articulation"],
                                  ["omni.isaac.core.articulations:Articulation"]))


def xform_prim_cls():
    return first_import(*_ordered(["isaacsim.core.prims:SingleXFormPrim"],
                                  ["isaacsim.core.experimental.prims:XformPrim"],
                                  ["omni.isaac.core.prims:XFormPrim"]))


def articulation_action_cls():
    return first_import("isaacsim.core.utils.types:ArticulationAction",
                        "omni.isaac.core.utils.types:ArticulationAction")


def imu_sensor_cls():
    return first_import("isaacsim.sensors.physics:IMUSensor", "omni.isaac.sensor:IMUSensor")


def contact_sensor_cls():
    return first_import("isaacsim.sensors.physics:ContactSensor", "omni.isaac.sensor:ContactSensor")


def camera_cls():
    return first_import("isaacsim.sensors.camera:Camera", "omni.isaac.sensor:Camera")


def lidar_rtx_cls():
    return first_import("isaacsim.sensors.rtx:LidarRtx", "omni.isaac.sensor:LidarRtx")


def differential_controller_cls():
    return first_import("isaacsim.robot.wheeled_robots.controllers.differential_controller:DifferentialController",
                        "omni.isaac.wheeled_robots.controllers.differential_controller:DifferentialController")


def is_experimental(obj_or_cls) -> bool:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    return cls.__module__.startswith("isaacsim.core.experimental")


def to_numpy(x) -> np.ndarray:
    """warp / torch / list -> numpy (the experimental API returns warp arrays)."""
    if x is None:
        return np.zeros(0)
    if hasattr(x, "numpy"):
        try:
            return np.asarray(x.numpy())
        except Exception:  # torch CUDA tensors
            return np.asarray(x.cpu().numpy())
    return np.asarray(x)


class XformGroup:
    """Uniform pose get/set over a list of prims for either prim API.

    ``set_world_poses(positions (N,3), orientations (N,4) wxyz, indices=None)``
    and ``get_world_poses() -> (positions (N,3), orientations (N,4))``.
    """

    def __init__(self, paths: list[str]):
        self.paths = list(paths)
        cls = xform_prim_cls()
        self.batched = is_experimental(cls)
        if self.batched:
            self._view = cls(self.paths) if self.paths else None
            self._prims = []
        else:
            self._view = None
            self._prims = [cls(p) for p in self.paths]

    def __len__(self) -> int:
        return len(self.paths)

    def set_world_poses(self, positions: np.ndarray, orientations: np.ndarray, indices=None) -> None:
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        orientations = np.asarray(orientations, dtype=float).reshape(-1, 4)
        idx = list(range(len(self.paths))) if indices is None else [int(i) for i in indices]
        if not idx:
            return
        if self.batched:
            self._view.set_world_poses(positions=positions, orientations=orientations, indices=idx)
        else:
            for k, i in enumerate(idx):
                self._prims[i].set_world_pose(position=positions[k], orientation=orientations[k])

    def get_world_poses(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.paths:
            return np.zeros((0, 3)), np.zeros((0, 4))
        if self.batched:
            p, q = self._view.get_world_poses()
            return to_numpy(p).reshape(-1, 3), to_numpy(q).reshape(-1, 4)
        ps, qs = zip(*(pr.get_world_pose() for pr in self._prims))
        return np.array([to_numpy(p) for p in ps]).reshape(-1, 3), np.array([to_numpy(q) for q in qs]).reshape(-1, 4)


def open_stage(path: str, app=None, max_updates: int = 600) -> None:
    """Open a USD stage in Kit and wait until it has finished loading."""
    try:
        from isaacsim.core.utils.stage import is_stage_loading  # type: ignore
        from isaacsim.core.utils.stage import open_stage as _open
    except ImportError:  # pragma: no cover - legacy
        from omni.isaac.core.utils.stage import is_stage_loading  # type: ignore
        from omni.isaac.core.utils.stage import open_stage as _open
    if not _open(str(path)):
        raise RuntimeError(f"could not open USD stage {path}")
    n = 0
    while is_stage_loading() and n < max_updates:
        if app is not None:
            app.update()
        else:
            import omni.kit.app
            omni.kit.app.get_app().update()
        n += 1


def usd_stage():
    import omni.usd
    return omni.usd.get_context().get_stage()


def assets_root_path() -> str | None:
    """Isaac Sim asset root (Nucleus / S3 mirror); None when unreachable."""
    try:
        fn = first_import("isaacsim.storage.native:get_assets_root_path",
                          "omni.isaac.nucleus:get_assets_root_path",
                          "omni.isaac.core.utils.nucleus:get_assets_root_path")
        return fn()
    except Exception:  # pragma: no cover
        return None


def ros2_node_namespaces() -> dict[str, str]:
    """OmniGraph node-type namespaces for the ROS 2 bridge (4.5+/5.x vs <= 4.2)."""
    try:
        importlib.import_module("isaacsim.ros2.bridge")
        return {"bridge": "isaacsim.ros2.bridge", "core": "isaacsim.core.nodes",
                "wheeled": "isaacsim.robot.wheeled_robots"}
    except ImportError:  # pragma: no cover - legacy
        return {"bridge": "omni.isaac.ros2_bridge", "core": "omni.isaac.core_nodes",
                "wheeled": "omni.isaac.wheeled_robots"}


def enable_extensions(names: list[str]) -> dict[str, bool]:
    """Enable Kit extensions (trying legacy aliases), returning which succeeded."""
    try:
        from isaacsim.core.utils.extensions import enable_extension  # type: ignore
    except ImportError:  # pragma: no cover
        from omni.isaac.core.utils.extensions import enable_extension  # type: ignore
    ok = {}
    for n in names:
        ok[n] = False
        for cand in [n] + EXTENSION_ALIASES.get(n, []):
            try:
                if enable_extension(cand):
                    ok[n] = True
                    break
            except Exception:  # pragma: no cover
                continue
    return ok
