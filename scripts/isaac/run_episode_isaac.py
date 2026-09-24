#!/usr/bin/env python3
"""Run one MED-OR-TRACE episode in Isaac Sim (RTX sensors + PhysX) with the full autonomy stack.

    # from the Isaac Sim install directory (python.sh = Isaac's bundled interpreter)
    ./python.sh /path/to/repo/scripts/isaac/run_episode_isaac.py --scenario-id cf_b__p0000__real_obstacle \\
        --export runs/isaac/cf_b_p0000 --duration 60
    ./python.sh scripts/isaac/run_episode_isaac.py --config scenarios/nominal.yaml --seed 7 --ros2 --no-headless
    ./python.sh scripts/isaac/run_episode_isaac.py --scenario-id nominal__0003 --detector model:det.pt \\
        --staff-mode people

The episode is exactly ``medortrace.eval.runner.run_episode(..., backend="isaac")``: the same
``build_episode`` (scene, workflow, faults, prior map) as the lite simulator, the same stack,
simulated operator, metrics and dataset writer.  Backend options are process-wide
``IsaacBackend.configure(...)`` settings so the scenario config (and its ``cfg_hash``) is identical to
the lite run of the same registry entry.

``--ros2`` additionally builds the OmniGraph ROS 2 bridge (clock, TF, lidar, camera, odometry) for
monitoring with RViz / rosbag; the wheels stay driven by the in-process stack (no cmd_vel
subscriber).  Use ``ros2_sim.py`` for an externally driven robot.

With ``--export DIR``: ``DIR/episodes/<scenario>__s<seed>__<policy>/`` (dataset schema of
``medortrace.data.writer``, ``meta.json:backend = "isaac"``), ``DIR/usd/`` (authored stage + rig) and
``DIR/isaac_run.json`` (resolved extensions, sensor annotators/commands, requested vs resolved RTX profiles,
sensor warnings, reflective-fault material edits, metrics).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from _common import add_isaac_args, add_scenario_args, detector_arg, resolve_scenario, set_prims_api


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_args(ap)
    add_isaac_args(ap)
    ap.add_argument("--policy", default=None, choices=["active", "fixed_route", "passive"])
    ap.add_argument("--duration", type=float, default=None, help="override episode duration (s)")
    ap.add_argument("--ros2", action="store_true", help="build the ROS 2 OmniGraph bridge (monitoring)")
    ap.add_argument("--export", default=None, metavar="DIR", help="write the episode dataset + USD here")
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--staff-mode", choices=["capsule", "people"], default="capsule")
    ap.add_argument("--detector", type=detector_arg, default="gt_surrogate", help="gt_surrogate | model:PATH")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    cfg, seed, entry = resolve_scenario(a)            # fail fast, before Kit boots
    set_prims_api(a.prims_api)
    from medortrace.isaac.app import launch
    app = launch(headless=a.headless, ros2=a.ros2, people=a.staff_mode == "people")
    t0 = time.time()
    try:
        # omni-dependent imports only after SimulationApp exists
        from medortrace.eval.runner import run_episode
        from medortrace.isaac.backend import IsaacBackend
        from medortrace.isaac.ros2_bridge import ros2_reset_hook

        export = Path(a.export) if a.export else None
        IsaacBackend.configure(staff_mode=a.staff_mode, detector=a.detector, physics_hz=a.physics_hz,
                               work_dir=str(export / "usd") if export else None, drive="internal")
        info: dict = {}

        def record(be) -> None:
            # backend_info dicts are live: per-frame counters (e.g. radar doppler_frames) are final at export
            info["sensors"] = {k: getattr(getattr(be, k, None), "backend_info", None)
                               for k in ("lidar", "radar", "camera", "acoustic")}
            info["sensor_warnings"] = list(be.sensor_warnings)
            info["reflective_faults"] = be.reflective_faults
            info["nonvisual_fixes"] = be.nonvisual_fixes
            info["usd"] = str(be.scene_path)
            info["articulation_root"] = be.base_path

        IsaacBackend.add_reset_hook(record)
        if a.ros2:
            IsaacBackend.add_reset_hook(ros2_reset_hook(drive_from_cmd_vel=False))
        res = run_episode(cfg, seed, backend="isaac", out_dir=str(export / "episodes") if export else None,
                          policy=a.policy, duration=a.duration, save_raw=a.save_raw, verbose=a.verbose)
        keys = [k for k in ("decision_accuracy", "brier", "calibration_ece", "abstention_rate", "wrong_assertion_rate",
                            "collisions_agent", "collisions_static", "sterile_breach_s", "loc_error_mean_m",
                            "energy_used_wh") if k in res.metrics]
        print(json.dumps({"scenario_id": res.scenario_id, "seed": res.seed, "policy": res.policy,
                          "wall_time_s": round(res.wall_time_s, 1), **{k: res.metrics[k] for k in keys}}, indent=1))
        if export:
            export.mkdir(parents=True, exist_ok=True)
            doc = {"scenario_id": res.scenario_id, "seed": seed, "registry_entry": entry.__dict__ if entry else None,
                   "policy": res.policy, "episode_dir": res.out_dir,
                   "extensions": getattr(app, "_medortrace_ext_status", {}), "options": dict(IsaacBackend.options),
                   "wall_time_s": time.time() - t0, **info, "metrics": res.metrics}
            (export / "isaac_run.json").write_text(json.dumps(doc, indent=1, default=str))
            print(f"[medortrace] wrote {export}")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
