"""SimulationApp bootstrap.

Must be called *before* importing any ``omni``/``isaacsim`` module other
than ``isaacsim`` itself (Kit loads extensions during SimulationApp init).

    from medortrace.isaac.app import launch
    app = launch(headless=True)
    ...  # now import omni.* / isaacsim.* modules
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
OPTIONAL_EXTENSIONS = [
    "isaacsim.ros2.bridge",          # ROS 2 interface (only with --ros2)
    "isaacsim.sensors.rtx.acoustic",  # RTX acoustic (experimental; name varies by release)
    "omni.anim.people",              # animated staff characters
    "isaacsim.core.experimental",    # engine-agnostic prims API (5.x)
]


def launch(headless: bool = True, width: int = 1280, height: int = 720, ros2: bool = False,
           renderer: str = "RayTracedLighting", extra_extensions: list[str] | None = None):
    SimulationApp = simulation_app_cls()
    app = SimulationApp({"headless": headless, "width": width, "height": height, "renderer": renderer,
                         "anti_aliasing": 3, "multi_gpu": False})
    req = enable_extensions(REQUIRED_EXTENSIONS)
    missing = [k for k, v in req.items() if not v]
    if missing:
        print(f"[medortrace] WARNING: required extensions not enabled: {missing}")
    opt = [e for e in OPTIONAL_EXTENSIONS if ros2 or e != "isaacsim.ros2.bridge"]
    status = enable_extensions(opt + list(extra_extensions or []))
    app.update()
    app._medortrace_ext_status = {**req, **status}
    return app
