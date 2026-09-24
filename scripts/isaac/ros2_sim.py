#!/usr/bin/env python3
"""Isaac Sim + ROS 2: the OR digital twin driven by an external ``/medortrace/cmd_vel``.

    # ROS 2 Humble/Jazzy sourced (or Isaac Sim's bundled ROS 2 libraries, see the Isaac docs)
    ./python.sh scripts/isaac/ros2_sim.py --scenario-id nominal__0000 --realtime
    ros2 topic pub -r 10 /medortrace/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}, angular: {z: 0.2}}"
    ros2 topic echo /medortrace/odom

What runs where:

* the scene, staff behaviour, ground-truth custody moves (workflow truth, hidden causes), fault
  model and RTX/physics sensors come from ``IsaacBackend`` exactly as in ``run_episode_isaac.py``,
  but with ``drive="external"``: the backend never commands the wheels;
* the OmniGraph built by ``medortrace.isaac.ros2_bridge`` publishes ``/clock``, ``/tf``, lidar
  points, RGB/depth/camera_info and odometry, and its ``ROS2SubscribeTwist`` ->
  ``DifferentialController`` -> ``IsaacArticulationController`` chain turns ``/medortrace/cmd_vel``
  into wheel velocity targets (one graph tick per control period);
* custom-message topics (radar, acoustic, workflow events, verification) are published by
  ``medortrace_ros``'s sim bridge node; pass ``--bridge module:factory`` to hook it (or anything
  else) in: ``factory(backend)`` must return a callable that receives every ``SensorBundle``.

The loop steps one control period (``episode.dt``) per iteration; ``--realtime`` paces it to wall
clock (otherwise it runs as fast as rendering allows and ROS consumers should use ``/clock``).
``--duration 0`` keeps running after the scripted workflow ends until the app window is closed.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

import numpy as np

import _bootstrap  # noqa: F401
from _common import add_isaac_args, add_scenario_args, detector_arg, resolve_scenario, set_prims_api


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_args(ap)
    add_isaac_args(ap)
    ap.add_argument("--duration", type=float, default=None,
                    help="sim seconds to run (default: the workflow duration; 0 = until the app closes)")
    ap.add_argument("--realtime", action="store_true", help="pace the loop to wall clock")
    ap.add_argument("--staff-mode", choices=["capsule", "people"], default="capsule")
    ap.add_argument("--detector", type=detector_arg, default="gt_surrogate")
    ap.add_argument("--work-dir", default=None, help="where the episode USD is authored (default: temp)")
    ap.add_argument("--bridge", default=None, metavar="MODULE:FACTORY",
                    help="factory(backend) -> callable(bundle), e.g. a medortrace_ros sim bridge")
    ap.add_argument("--log-every", type=float, default=5.0, help="status print period (sim s, 0 = off)")
    return ap.parse_args(argv)


def load_bridge(spec: str | None):
    if not spec:
        return None
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "create")


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg, seed, _ = resolve_scenario(a)
    set_prims_api(a.prims_api)
    from medortrace.isaac.app import launch
    app = launch(headless=a.headless, ros2=True, people=a.staff_mode == "people")
    be = None
    try:
        from medortrace.common.msgs import VelocityCommand
        from medortrace.isaac.backend import IsaacBackend
        from medortrace.isaac.ros2_bridge import TOPICS, ros2_reset_hook
        from medortrace.sim.episode import build_episode

        IsaacBackend.configure(staff_mode=a.staff_mode, detector=a.detector, physics_hz=a.physics_hz,
                               work_dir=a.work_dir, drive="external")
        IsaacBackend.add_reset_hook(ros2_reset_hook(drive_from_cmd_vel=True))
        ep = build_episode(cfg, seed)
        be = IsaacBackend(cfg)
        bundle = be.reset(ep)
        factory = load_bridge(a.bridge)
        sink = factory(be) if factory else None
        T = ep.workflow.duration if a.duration is None else (np.inf if a.duration <= 0 else float(a.duration))
        print(f"[medortrace] {ep.spec.scenario_id} seed={seed}: listening on {TOPICS['cmd_vel']}, "
              f"running {T if np.isfinite(T) else 'until closed'} s")
        wall0, next_log = time.time(), 0.0
        idle = VelocityCommand()
        while app.is_running() and be.t < T - 1e-9:
            bundle = be.step(idle)
            if sink is not None:
                sink(bundle)
            if a.realtime:
                lag = be.t - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
            if a.log_every and be.t >= next_log:
                x, y, th = be.truth().robot_pose
                v, w = be.robot.measured_twist()
                print(f"t={be.t:7.1f}  pose=({x:5.2f},{y:5.2f},{np.degrees(th):6.1f}deg)  v={v:+.2f} w={w:+.2f}  "
                      f"lidar={0 if bundle.lidar is None else len(bundle.lidar.points)}  "
                      f"cam_dets={0 if bundle.camera is None else len(bundle.camera.detections)}  "
                      f"wf_events={len(bundle.workflow)}")
                next_log += a.log_every
        return 0
    finally:
        if be is not None:
            be.close()
        app.close()


if __name__ == "__main__":
    sys.exit(main())
