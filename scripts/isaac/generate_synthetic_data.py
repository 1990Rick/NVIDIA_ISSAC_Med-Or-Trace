#!/usr/bin/env python3
"""Replicator synthetic-data program with causal labels and matched counterfactual frames.

    ./python.sh scripts/isaac/generate_synthetic_data.py --family counterfactual --factor CF-B --limit 5 \\
        --frames 40 --out datasets/cfb
    ./python.sh scripts/isaac/generate_synthetic_data.py --scenario-id nominal__0001 --frames 10 --out datasets/dbg
    python scripts/isaac/generate_synthetic_data.py --factor CF-B --limit 1 --frames 2 --dry-run --out /tmp/x
        # --dry-run: plan + author + randomise + label with pxr only (no Isaac Sim, no images)

Per render unit (a counterfactual *pair* or a single scenario):

1. ``build_episode`` + ``build_stage`` for every arm (no robot in the dataset scene; rigid bodies
   frozen, frames are posed rather than simulated); the scenario's reflective faults (``specular_gain``,
   ``floor_wet``) are applied as material edits, as in ``IsaacBackend``, before the causal lock is taken;
2. one shared schedule (frame times, robot-like camera viewpoints valid in every arm, nuisance
   seed) from ``medortrace.isaac.synthetic``;
3. per arm and frame: move items/staff to the ground-truth state at the frame time inside
   ``CausalLock.causal_edit()``, set the camera, ``NuisanceRandomizer.apply(frame=k)`` (causal lock
   enforced; identical nuisance in both arms), RTX fog from the same keyed RNG, one Replicator
   ``orchestrator.step`` and ``CausalLabelWriter.write`` with the frame's labels (scenario, family,
   seed, hidden factor/value, item truth slots at t, causal + nuisance signatures, camera pose and
   intrinsics).

Output::

    OUT/<scenario_id>/{rgb,semantic,instance,depth,bbox2d,camera,labels}/<frame>.*
    OUT/usd/<scenario_id>.usda        (authored stage, t=0 state)
    OUT/index.json                    frames with the registry split (pair-level: both arms of a
                                      pair and all their frames share one split), matched-pair map,
                                      nuisance-mismatch report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from _common import DEFAULT_REGISTRY

from medortrace.common.config import CONFIG_DIR, load_yaml
from medortrace.eval.registry import load_registry, select
from medortrace.isaac.synthetic import (
    CAMERA_PATH,
    apply_causal_state,
    author_camera,
    build_index,
    camera_intrinsics,
    causal_states,
    frame_labels,
    frame_times,
    freeze_physics,
    make_units,
    sample_viewpoints,
    set_camera_pose,
    unit_seed,
)
from medortrace.sim.episode import build_episode

# pxr-dependent modules (medortrace.usd, replicator_randomizers) are imported after SimulationApp starts:
# inside Isaac Sim, pxr comes from Kit's USD extension.

ANNOTATORS = ["rgb", "semantic_segmentation", "instance_id_segmentation_fast", "bounding_box_2d_tight",
              "distance_to_image_plane", "camera_params"]
ANNOTATOR_INIT = {"semantic_segmentation": {"colorize": False}, "instance_id_segmentation_fast": {"colorize": False},
                  "bounding_box_2d_tight": {"semanticTypes": ["class"]}}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--scenario-id", nargs="*", default=None, help="explicit entries (a pair's arms are added)")
    ap.add_argument("--family", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--factor", default=None, help="CF-A|CF-B|CF-C|CF-D")
    ap.add_argument("--limit", type=int, default=None, help="max render units (pairs count once)")
    ap.add_argument("--frames", type=int, default=20, help="frames per scenario (shared by a pair's arms)")
    ap.add_argument("--event-frac", type=float, default=0.5, help="fraction of frames just after hidden-cause moves")
    ap.add_argument("--t-max", type=float, default=None, help="only sample frame times in [0, t-max] s")
    ap.add_argument("--seed", type=int, default=20260924, help="master nuisance/schedule seed")
    ap.add_argument("--strength", type=float, default=1.0, help="nuisance randomisation strength")
    ap.add_argument("--resolution", type=int, nargs=2, default=None, help="W H (default: rtx_camera.yaml)")
    ap.add_argument("--rt-subframes", type=int, default=8, help="RTX subframes per captured frame")
    ap.add_argument("--warmup", type=int, default=3, help="discarded Replicator steps after opening a stage")
    ap.add_argument("--no-fog", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true", help="pxr only: no Isaac Sim, no images")
    return ap.parse_args(argv)


def select_entries(a) -> list:
    reg = load_registry(a.registry)
    if a.scenario_id:
        want = set(a.scenario_id)
        pairs = {e.pair_id for e in reg if e.scenario_id in want and e.pair_id}
        sel = [e for e in reg if e.scenario_id in want or (e.pair_id and e.pair_id in pairs)]
        missing = want - {e.scenario_id for e in sel}
        if missing:
            raise SystemExit(f"unknown scenario ids: {sorted(missing)}")
        return sel
    return select(reg, a.family, a.split, a.factor)


class ReplicatorCapture:  # pragma: no cover - requires Isaac Sim
    """Render product + annotators on the dataset camera; one ``capture()`` per frame."""

    def __init__(self, app, resolution, rt_subframes: int, warmup: int):
        import carb
        import omni.replicator.core as rep
        self.app, self.rep, self.settings = app, rep, carb.settings.get_settings()
        self.resolution = tuple(int(x) for x in resolution)
        self.rt_subframes = int(rt_subframes)
        self.warmup = int(warmup)
        self.rp = None
        self.ann: dict = {}
        try:
            rep.orchestrator.set_capture_on_play(False)
        except Exception:
            pass

    def open(self, usd: Path):
        from medortrace.isaac.compat import open_stage, usd_stage
        self.detach()
        open_stage(str(usd), self.app)
        return usd_stage()

    def attach(self, cam_path: str) -> None:
        self.rp = self.rep.create.render_product(cam_path, self.resolution, name="medortrace_dataset")
        for name in ANNOTATORS:
            try:
                init = ANNOTATOR_INIT.get(name)
                ann = self.rep.AnnotatorRegistry.get_annotator(name, init_params=init) if init else \
                    self.rep.AnnotatorRegistry.get_annotator(name)
                ann.attach([self.rp])
                self.ann[name] = ann
            except Exception as e:
                print(f"[medortrace] annotator {name} unavailable: {e}")
        for _ in range(self.warmup):
            self._step()

    def _step(self) -> None:
        try:
            self.rep.orchestrator.step(rt_subframes=self.rt_subframes, delta_time=0.0, pause_timeline=True)
        except TypeError:   # older Replicator signature
            self.rep.orchestrator.step(rt_subframes=self.rt_subframes)

    def set_fog(self, fog: dict) -> None:
        for k, v in fog.items():
            self.settings.set(k, v)

    def capture(self) -> dict:
        self._step()
        return {k: a.get_data() for k, a in self.ann.items()}

    def detach(self) -> None:
        for a in self.ann.values():
            try:
                a.detach()
            except Exception:
                pass
        self.ann = {}
        if self.rp is not None:
            try:
                self.rp.destroy()
            except Exception:
                pass
            self.rp = None


def render_unit(unit, a, out: Path, cam_cfg: dict, intr: dict, capture) -> list[dict]:
    from pxr import Usd

    from medortrace.isaac.replicator_randomizers import CausalLabelWriter, CausalLock, NuisanceRandomizer
    from medortrace.isaac.sensors import apply_reflective_faults
    from medortrace.usd.scene_builder import build_stage
    eps = [build_episode(e.resolve(), e.seed) for e in unit.entries]
    seed = unit_seed(a.seed, unit.key)
    plan = np.random.default_rng(seed)
    times = frame_times(eps, a.frames, plan, a.event_frac, t_max=a.t_max)
    states = [causal_states(ep, times) for ep in eps]
    staff_per_frame = [[st[k].staff_xy for st in states] for k in range(len(times))]
    vps = sample_viewpoints([ep.spec for ep in eps], staff_per_frame, plan,
                            float(cam_cfg.get("mount_height", 1.45)), float(cam_cfg.get("pitch_deg", -25.0)))
    records = []
    for e, ep, st_list in zip(unit.entries, eps, states):
        usd = out / "usd" / f"{e.scenario_id}.usda"
        build_stage(ep.spec, ep.materials, usd, robot_rig=None)
        stage = capture.open(usd) if capture else Usd.Stage.Open(str(usd))
        author_camera(stage, intr, CAMERA_PATH)
        freeze_physics(stage)           # posed frames: no dynamics may move locked prims between frames
        # scenario condition (same in both arms of a pair), applied before the lock takes its reference
        reflective = apply_reflective_faults(stage, ep.materials, ep.faults.specular_gain, ep.faults.floor_wet)
        lock = CausalLock(stage)
        rnd = NuisanceRandomizer(stage, seed=seed, lock=lock, strength=a.strength)
        if capture:
            capture.attach(CAMERA_PATH)
        writer = CausalLabelWriter(out / e.scenario_id)
        for k, (state, vp) in enumerate(zip(st_list, vps)):
            with lock.causal_edit():
                apply_causal_state(stage, ep.spec, state)
            set_camera_pose(stage, vp, CAMERA_PATH)
            info = rnd.apply(frame=k, haze=float(ep.spec.nuisance.get("haze", 0.0)))
            if a.no_fog:
                info["fog"] = {"/rtx/fog/enabled": False}
            data = {}
            if capture:
                capture.set_fog(info["fog"])
                data = capture.capture()
            labels = frame_labels(e, ep, unit, k, state, vp, intr, info)
            res = writer.write(data, frame_id=f"{k:06d}", labels=labels)
            records.append({"frame_uid": labels["frame_uid"], "pair_frame_key": labels["pair_frame_key"],
                            "scenario_id": e.scenario_id, "pair_id": e.pair_id, "family": e.family,
                            "split": e.split, "arm": labels["hidden_value"], "frame": k, "t": labels["t"],
                            "causal_signature": info["causal_signature"],
                            "nuisance_signature": info["nuisance_signature"],
                            "files": {kk: f"{e.scenario_id}/{v}" for kk, v in res["files"].items()}})
        print(f"[medortrace] {e.scenario_id}: {len(st_list)} frames ({'dry-run' if not capture else 'rendered'})"
              + (f", {len(reflective)} reflective-fault material edits" if reflective else ""))
    return records


def main(argv=None) -> int:
    a = parse_args(argv)
    units = make_units(select_entries(a))
    if a.limit:
        units = units[: a.limit]
    if not units:
        raise SystemExit("no registry entries selected")
    cam_yaml = load_yaml(CONFIG_DIR / "sensors" / "rtx_camera.yaml")
    cam_cfg = {**{"mount_height": 1.45, "pitch_deg": cam_yaml.get("pitch_deg", -25.0)},
               **units[0].entries[0].resolve().get("sensors", {}).get("camera", {})}
    intr = camera_intrinsics(a.resolution or cam_yaml["resolution"], float(cam_yaml.get("hfov_deg", 90.0)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    app = capture = None
    if not a.dry_run:
        from medortrace.isaac.app import launch
        app = launch(headless=a.headless)
        capture = ReplicatorCapture(app, intr["resolution"], a.rt_subframes, a.warmup)
    t0 = time.time()
    records: list[dict] = []
    try:
        for n, unit in enumerate(units):
            print(f"[medortrace] unit {n + 1}/{len(units)}: {unit.key} "
                  f"({len(unit.entries)} arm(s), split={unit.split})")
            records += render_unit(unit, a, out, cam_cfg, intr, capture)
        if capture:
            capture.detach()
        meta = {"registry": str(a.registry), "master_seed": a.seed, "frames_per_scenario": a.frames,
                "event_frac": a.event_frac, "t_max": a.t_max, "strength": a.strength, "dry_run": a.dry_run,
                "camera": intr,
                "annotators": [] if a.dry_run else ANNOTATORS, "units": [u.key for u in units],
                "wall_time_s": time.time() - t0}
        index = build_index(records, meta)
        (out / "index.json").write_text(json.dumps(index, indent=1))
        print(f"[medortrace] {index['n_frames']} frames, splits {index['splits']}, "
              f"{len(index['pairs'])} matched frame pairs, "
              f"nuisance mismatches: {len(index['nuisance_mismatched_pairs'])}")
    finally:
        if app is not None:
            app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
