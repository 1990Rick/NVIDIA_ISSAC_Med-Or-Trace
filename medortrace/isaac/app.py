"""SimulationApp bootstrap.

Must be called *before* importing any ``omni``/``isaacsim`` module other
than ``isaacsim`` itself (Kit loads extensions during SimulationApp init)::

    from medortrace.isaac.app import launch
    app = launch(headless=True)
    ...  # now import omni.* / isaacsim.* modules

Extension names are the Isaac Sim 4.5+/5.x ones; ``compat.enable_extensions``
falls back to the legacy ``omni.isaac.*`` names.  The returned app carries
``app._medortrace_ext_status`` ({extension: enabled}) for diagnostics.
"""

from __future__ import annotations

from medortrace.isaac.compat import enable_extensions, simulation_app_cls

REQUIRED_EXTENSIONS = [
    "isaacsim.sensors.rtx",          # RTX lidar / radar
    "isaacsim.sensors.physics",      # IMU / contact / effort
    "isaacsim.sensors.camera",
    "omni.replicator.core",          # domain randomisation + annotators + writers
    "isaacsim.robot.wheeled_robots",
]
ROS2_EXTENSIONS = ["isaacsim.ros2.bridge"]
PEOPLE_EXTENSIONS = ["omni.anim.people", "omni.anim.graph.core"]
OPTIONAL_EXTENSIONS = [
    "isaacsim.sensors.rtx.acoustic",     # RTX acoustic (experimental; name varies by release)
    "isaacsim.core.experimental.prims",  # engine-agnostic prims API (5.x)
]


def launch(headless: bool = True, width: int = 1280, height: int = 720, ros2: bool = False, people: bool = False,
           renderer: str = "RayTracedLighting", extra_extensions: list[str] | None = None):
    SimulationApp = simulation_app_cls()
    app = SimulationApp({"headless": headless, "width": width, "height": height, "renderer": renderer,
                         "anti_aliasing": 3, "multi_gpu": False})
    req = enable_extensions(REQUIRED_EXTENSIONS + (ROS2_EXTENSIONS if ros2 else []))
    missing = [k for k, v in req.items() if not v]
    if missing:
        print(f"[medortrace] WARNING: required extensions not enabled: {missing}")
    opt = enable_extensions(OPTIONAL_EXTENSIONS + (PEOPLE_EXTENSIONS if people else []) + list(extra_extensions or []))
    app.update()
    app._medortrace_ext_status = {**req, **opt}
    return app
