"""Helpers shared by the launch files (``launch`` / ``launch_ros`` imported lazily)."""

from __future__ import annotations

import os
from pathlib import Path


def share_dir() -> Path:
    """Installed share directory of medortrace_ros (source tree when not installed)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("medortrace_ros"))
    except Exception:  # noqa: BLE001 - not installed / no ament index
        return Path(__file__).resolve().parents[1]


def config_path(name: str) -> str:
    return str(share_dir() / "config" / name)


def rig_static_tf_nodes(condition=None, use_sim_time: bool | None = None) -> list:
    """One ``tf2_ros static_transform_publisher`` per sensor frame of configs/robot/rig.yaml."""
    from launch_ros.actions import Node

    try:
        from medortrace_ros.frames import rig_static_frames
        frames = rig_static_frames()
    except Exception as e:  # noqa: BLE001 - medortrace not importable in the launch process
        print(f"[medortrace_ros] WARNING: no rig TF ({e}); set MEDORTRACE_ROOT or pip install -e the repo")
        return []
    nodes = []
    for f in frames:
        args = ["--x", f"{f.xyz[0]:.4f}", "--y", f"{f.xyz[1]:.4f}", "--z", f"{f.xyz[2]:.4f}",
                "--roll", f"{f.rpy[0]:.6f}", "--pitch", f"{f.rpy[1]:.6f}", "--yaw", f"{f.rpy[2]:.6f}",
                "--frame-id", f.parent, "--child-frame-id", f.child]
        params = [] if use_sim_time is None else [{"use_sim_time": use_sim_time}]
        nodes.append(Node(package="tf2_ros", executable="static_transform_publisher", name=f"rig_tf_{f.child}",
                          arguments=args, parameters=params, output="log", condition=condition))
    return nodes


def medortrace_env() -> dict[str, str]:
    """Environment additions so child processes (e.g. Isaac Sim's python.sh) find medortrace."""
    env = {}
    try:
        import medortrace
        root = str(Path(medortrace.__file__).resolve().parents[1])
        env["MEDORTRACE_ROOT"] = root
        pp = os.environ.get("PYTHONPATH", "")
        pkg = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(p for p in (root, pkg, pp) if p)
    except Exception:  # noqa: BLE001
        pass
    return env
