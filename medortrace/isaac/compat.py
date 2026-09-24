"""Version shims for Isaac Sim APIs.

Isaac Sim 4.5 renamed ``omni.isaac.*`` extensions to ``isaacsim.*`` and 5.x
added the engine-agnostic ``isaacsim.core.experimental`` API.  Every symbol
the project uses is resolved here once, with the newest API first, so the
rest of ``medortrace.isaac`` is version-independent.
"""

from __future__ import annotations

import importlib
from typing import Any


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


def simulation_app_cls():
    return first_import("isaacsim:SimulationApp", "isaacsim.simulation_app:SimulationApp",
                        "omni.isaac.kit:SimulationApp")


def world_cls():
    return first_import("isaacsim.core.api:World", "omni.isaac.core:World")


def articulation_cls():
    """Prefer the engine-agnostic experimental Articulation (Isaac Sim 5.x)."""
    return first_import("isaacsim.core.experimental.prims:Articulation",
                        "isaacsim.core.prims:SingleArticulation",
                        "omni.isaac.core.articulations:Articulation")


def xform_prim_cls():
    return first_import("isaacsim.core.experimental.prims:XformPrim",
                        "isaacsim.core.prims:SingleXFormPrim",
                        "omni.isaac.core.prims:XFormPrim")


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


def enable_extensions(names: list[str]) -> dict[str, bool]:
    """Enable Kit extensions, returning which succeeded (optional ones may be missing)."""
    ok = {}
    try:
        from isaacsim.core.utils.extensions import enable_extension  # type: ignore
    except ImportError:  # pragma: no cover
        from omni.isaac.core.utils.extensions import enable_extension  # type: ignore
    for n in names:
        try:
            ok[n] = bool(enable_extension(n))
        except Exception:  # pragma: no cover
            ok[n] = False
    return ok
