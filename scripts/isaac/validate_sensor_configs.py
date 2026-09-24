#!/usr/bin/env python3
"""Validate the RTX / physics sensor configuration against the rig, the scenario config and Isaac Sim.

    python scripts/isaac/validate_sensor_configs.py --static-only           # no Isaac Sim (pxr only)
    ./python.sh scripts/isaac/validate_sensor_configs.py --frames 10        # + create sensors in Isaac Sim
    ./python.sh scripts/isaac/validate_sensor_configs.py --scenario-id reflective__0002 --no-headless

Static checks (always):
  * RTX lidar/radar JSON profiles: required fields, emitter table consistency, agreement with
    configs/sensors/sensor_rig.yaml (the human-editable source of the profiles);
  * rtx_camera.yaml: resolution aspect vs the rig camera aperture, hfov, pitch vs the rig frame,
    known annotator names;
  * rig.yaml sensor frames vs the scenario config mount heights (lite/Isaac agreement);
  * authored USD (rig + a registry scene): sensor frames, camera prim, articulation root, wheel drives;
  * RTX non-visual material tokens (``omni:simready:nonvisual:{base,coating,attributes}``) of every
    material against the vocabulary in ``medortrace.isaac.sensors`` (+ the alias the backend applies)
    and physical consistency with the medortrace:* parameters (metals, radar-penetrable fabrics);
  * the scenario's reflective faults (``specular_gain``, ``floor_wet``): the material edits the backend applies
    (``medortrace.isaac.sensors.apply_reflective_faults``) and their non-visual tokens.

Isaac Sim checks (unless ``--static-only``): open the scene with the rig, create every adapter of
``medortrace.isaac.sensors`` on the rig frames, step ``--frames`` rendered frames and print the resolved
command / annotator names and the output shapes of every sensor.  A custom RTX profile that the release did not
apply (5.x ``OmniLidar``/``OmniRadar`` prims) and radar frames without Doppler are reported as warnings.

Exit code 1 if any check reports an ERROR (``--strict``: also on warnings).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from _common import DEFAULT_REGISTRY

from medortrace.common.config import CONFIG_DIR, load_config, load_yaml
from medortrace.eval.registry import load_registry

KNOWN_ANNOTATORS = {"rgb", "LdrColor", "semantic_segmentation", "instance_segmentation", "instance_segmentation_fast",
                    "instance_id_segmentation", "instance_id_segmentation_fast", "bounding_box_2d_tight",
                    "bounding_box_2d_tight_fast", "bounding_box_2d_loose", "bounding_box_3d",
                    "distance_to_image_plane", "distance_to_camera", "normals", "motion_vectors", "camera_params",
                    "pointcloud", "occlusion"}


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, level: str, check: str, msg: str) -> None:
        self.rows.append((level, check, msg))
        print(f"[{level:5s}] {check}: {msg}")

    def ok(self, c, m):
        self.add("OK", c, m)

    def warn(self, c, m):
        self.add("WARN", c, m)

    def err(self, c, m):
        self.add("ERROR", c, m)

    def count(self, level: str) -> int:
        return sum(1 for r in self.rows if r[0] == level)


# ---------------------------------------------------------------------------
def check_profiles(rep: Report) -> None:
    rig_src = load_yaml(CONFIG_DIR / "sensors" / "sensor_rig.yaml")
    lp = json.loads((CONFIG_DIR / "sensors" / "rtx_lidar_or16.json").read_text())
    prof = lp.get("profile", {})
    need = ["scanType", "nearRangeM", "farRangeM", "numberOfEmitters", "emitterStates", "scanRateBaseHz",
            "maxReturns", "upElevationDeg", "downElevationDeg"]
    miss = [k for k in need if k not in prof]
    (rep.err if miss else rep.ok)("lidar.profile", f"missing fields {miss}" if miss else "required fields present")
    n = int(prof.get("numberOfEmitters", 0))
    st = (prof.get("emitterStates") or [{}])[0]
    lens = {k: len(st.get(k, [])) for k in ("azimuthDeg", "elevationDeg", "fireTimeNs")}
    bad = {k: v for k, v in lens.items() if v != n}
    (rep.err if bad else rep.ok)("lidar.emitters", f"table lengths {lens} != numberOfEmitters {n}" if bad
                                 else f"{n} emitters, tables consistent")
    el = np.asarray(st.get("elevationDeg", []), float)
    if len(el):
        inside = (el >= prof.get("downElevationDeg", -90) - 1e-6) & (el <= prof.get("upElevationDeg", 90) + 1e-6)
        (rep.ok if inside.all() else rep.err)("lidar.elevation", f"{el.min():.1f}..{el.max():.1f} deg within limits"
                                              if inside.all() else "emitter elevations outside up/down limits")
    src = rig_src.get("lidar", {})
    pairs = [("channels", n), ("far_range_m", prof.get("farRangeM")), ("near_range_m", prof.get("nearRangeM")),
             ("max_returns", prof.get("maxReturns")), ("scan_rate_hz", prof.get("scanRateBaseHz"))]
    diff = [f"{k}: yaml={src.get(k)} json={v}" for k, v in pairs if k in src and v is not None and
            abs(float(src[k]) - float(v)) > 1e-6]
    (rep.warn if diff else rep.ok)("lidar.source", f"JSON differs from sensor_rig.yaml ({diff}); rerun "
                                   "scripts/generate_rtx_configs.py" if diff else "JSON matches sensor_rig.yaml")
    rp = json.loads((CONFIG_DIR / "sensors" / "rtx_radar_or77.json").read_text()).get("profile", {})
    rsrc = rig_src.get("radar", {})
    rpairs = [("max_range_m", rp.get("maxRangeM")), ("fov_azimuth_deg", rp.get("azimuthFovDeg")),
              ("carrier_ghz", rp.get("carrierFrequencyGHz")), ("frame_rate_hz", rp.get("frameRateHz"))]
    rdiff = [f"{k}: yaml={rsrc.get(k)} json={v}" for k, v in rpairs if k in rsrc and v is not None and
             abs(float(rsrc[k]) - float(v)) > 1e-6]
    missing_out = {"range", "azimuth", "radialVelocity", "rcs"} - set(rp.get("outputs", []))
    (rep.warn if rdiff else rep.ok)("radar.source", f"JSON differs from sensor_rig.yaml ({rdiff})" if rdiff
                                    else "JSON matches sensor_rig.yaml")
    (rep.err if missing_out else rep.ok)("radar.outputs", f"missing outputs {sorted(missing_out)}" if missing_out
                                         else "range/azimuth/radialVelocity/rcs output")


def check_camera_and_rig(rep: Report, cfg: dict) -> None:
    cam = load_yaml(CONFIG_DIR / "sensors" / "rtx_camera.yaml")
    rig = load_yaml(CONFIG_DIR / "robot" / "rig.yaml")
    W, H = cam["resolution"]
    aspect = H / W
    (rep.ok if abs(aspect - 0.75) < 1e-6 else rep.warn)(
        "camera.aspect", f"resolution {W}x{H} aspect {aspect:.3f} vs rig vertical/horizontal aperture 0.75")
    hf = float(rig.get("camera_hfov_deg", 90.0))
    (rep.ok if abs(hf - float(cam["hfov_deg"])) < 1e-6 else rep.err)(
        "camera.hfov", f"rtx_camera.yaml {cam['hfov_deg']} deg vs rig {hf} deg")
    rig_pitch = -float(rig.get("frames", {}).get("camera_link", {}).get("rpy_deg", [0, 25, 0])[1])
    scen_pitch = cfg.get("sensors", {}).get("camera", {}).get("pitch_deg")
    for name, v in (("rtx_camera.yaml", cam.get("pitch_deg")), ("scenario sensors.camera", scen_pitch)):
        if v is not None:
            (rep.ok if abs(float(v) - rig_pitch) < 1e-6 else rep.err)(
                "camera.pitch", f"{name} pitch {v} deg vs rig camera_link pitch {rig_pitch} deg")
    unknown = [a for a in cam.get("annotators", []) if a not in KNOWN_ANNOTATORS]
    (rep.warn if unknown else rep.ok)("camera.annotators", f"unknown annotator names {unknown}" if unknown else
                                      f"{cam.get('annotators')}")
    frames = rig.get("frames", {})
    sc = cfg.get("sensors", {})
    for link, key in (("lidar_link", "lidar"), ("camera_link", "camera"), ("radar_link", "radar")):
        z = float(frames.get(link, {}).get("xyz", [0, 0, np.nan])[2])
        mh = sc.get(key, {}).get("mount_height")
        if mh is not None:
            (rep.ok if abs(z - float(mh)) < 1e-6 else rep.err)(
                f"rig.{link}", f"rig z={z} vs scenario sensors.{key}.mount_height={mh}")
        c = frames.get(link, {}).get("config")
        if c:
            (rep.ok if (CONFIG_DIR / c).exists() else rep.err)(f"rig.{link}.config", f"{c}")


def check_usd(rep: Report, scenario_id: str, registry: str, work: Path) -> Path:
    from pxr import Usd, UsdGeom, UsdPhysics

    from medortrace.isaac.robot import find_articulation_root
    from medortrace.isaac.sensors import NONVISUAL_ALIASES, nonvisual_report
    from medortrace.sim.episode import build_episode
    from medortrace.usd.robot_rig import build_rig
    from medortrace.usd.scene_builder import build_stage
    from medortrace.world.materials import MATERIALS

    e = next((x for x in load_registry(registry) if x.scenario_id == scenario_id), None)
    if e is None:
        raise SystemExit(f"unknown scenario {scenario_id}")
    ep = build_episode(e.resolve(), e.seed)
    rig = load_yaml(CONFIG_DIR / "robot" / "rig.yaml")
    rig_path = work / "robot" / "medortrace_rig.usda"
    build_rig(rig_path, rig)
    scene = work / "scenes" / f"{scenario_id}.usda"
    build_stage(ep.spec, ep.materials, scene, robot_rig=f"../robot/{rig_path.name}")
    st = Usd.Stage.Open(str(scene))
    base = find_articulation_root(st, "/World/Robot")
    (rep.ok if base.endswith("base_link") else rep.err)("usd.articulation_root", base)
    for link in rig.get("frames", {}):
        p = st.GetPrimAtPath(f"{base}/{link}")
        (rep.ok if p.IsValid() else rep.err)(f"usd.frame.{link}", str(p.GetPath()) if p.IsValid() else "missing")
    cam = st.GetPrimAtPath(f"{base}/camera_link/rgb")
    if cam.IsValid() and cam.IsA(UsdGeom.Camera):
        m = np.array(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)
        fwd = -m[2, :3] / np.linalg.norm(m[2, :3])
        yaw = float(ep.spec.robot_start[2])
        pitch = float(np.degrees(np.arcsin(fwd[2])))
        heading_err = float(np.degrees(np.arctan2(fwd[1], fwd[0]) - yaw) + 180) % 360 - 180
        ok = abs(pitch + 25.0) < 0.5 and abs(heading_err) < 0.5
        (rep.ok if ok else rep.err)("usd.camera_axis", f"optical axis pitch {pitch:.1f} deg, heading error "
                                    f"{heading_err:.2f} deg (expect -25 / 0)")
    else:
        rep.err("usd.camera", "rig camera prim missing")
    for j in ("left_wheel_joint", "right_wheel_joint"):
        p = st.GetPrimAtPath(f"/World/Robot/joints/{j}")
        ok = p.IsValid() and p.HasAPI(UsdPhysics.DriveAPI, "angular")
        (rep.ok if ok else rep.err)(f"usd.{j}", "revolute joint with angular velocity drive" if ok else "missing drive")
    rows = nonvisual_report(st, fix=False)
    bad = [r for r in rows if r["issues"]]
    for r in bad:
        fixable = all("alias" in i for i in r["issues"])
        (rep.warn if fixable else rep.err)("nonvisual." + r["material"].split("/")[-1], "; ".join(r["issues"]))
    if not bad:
        rep.ok("nonvisual", f"{len(rows)} materials, all tokens in the RTX vocabulary")
    for r in rows:
        name = r["material"].split("/")[-1]
        m = MATERIALS.get(name)
        if m is None:
            continue
        base_tok = NONVISUAL_ALIASES["base"].get(str(r["base"]), str(r["base"]))
        metal = base_tok in {"aluminum", "steel", "oxidized_steel", "iron", "oxidized_iron", "silver", "brass",
                             "bronze", "oxidized_bronze_patina", "tin"}
        if metal != (m.metallic >= 0.5):
            rep.warn(f"nonvisual.consistency.{name}", f"base token {r['base']!r} vs visual metallic={m.metallic}")
        if m.radar_penetrable and metal:
            rep.warn(f"nonvisual.consistency.{name}", f"radar_penetrable but metallic base token {r['base']!r}")
    check_reflective_faults(rep, st, ep)
    return scene


def check_reflective_faults(rep: Report, st, ep) -> None:
    """The material edits IsaacBackend.reset applies for the scenario's reflective faults (in memory, discarded)."""
    from medortrace.isaac.sensors import apply_reflective_faults, nonvisual_report
    fm = ep.faults
    if fm.specular_gain == 1.0 and not fm.floor_wet:
        rep.ok("faults.reflective", "no reflective faults in this scenario")
        return
    try:
        rows = apply_reflective_faults(st, ep.materials, fm.specular_gain, fm.floor_wet)
        wet_ok = not fm.floor_wet or any(r["fault"] == "floor_wet" for r in rows)
        desc = "; ".join(f"{r['material'].split('/')[-1]}: roughness {r['roughness'][0]:.2f}->{r['roughness'][1]:.2f}"
                         f", coating {r['coating'][1]}" for r in rows)
        (rep.ok if rows and wet_ok else rep.err)(
            "faults.reflective", f"specular_gain={fm.specular_gain}, floor_wet={fm.floor_wet}: "
                                 f"{len(rows)} material edits ({desc})")
        off = [r["material"] for r in nonvisual_report(st) if any("coating" in i for i in r["issues"])]
        (rep.err if off else rep.ok)("faults.reflective.tokens", f"coating tokens off-vocabulary: {off}" if off
                                     else "edited coating tokens are in the RTX vocabulary")
    finally:
        st.GetRootLayer().Reload()     # the authored scene stays as built (the backend edits its opened copy)


# ---------------------------------------------------------------------------
def isaac_checks(rep: Report, app, scene: Path, frames: int) -> None:  # pragma: no cover - Isaac only
    from medortrace.isaac.compat import open_stage, usd_stage, world_cls
    from medortrace.isaac.robot import find_articulation_root
    from medortrace.isaac.sensors import (
        AcousticAdapter,
        CameraAdapter,
        ContactAdapter,
        ImuAdapter,
        RtxLidarAdapter,
        RtxRadarAdapter,
        nonvisual_report,
        normalize_nonvisual_tokens,
    )
    print(f"[medortrace] extensions: {json.dumps(getattr(app, '_medortrace_ext_status', {}))}")
    open_stage(str(scene), app)
    stage = usd_stage()
    fixes = normalize_nonvisual_tokens(stage)
    rep.ok("isaac.nonvisual_aliases", f"applied {sum(len(f['fixed']) for f in fixes)} alias rewrites")
    left = [r for r in nonvisual_report(stage) if r["issues"]]
    (rep.warn if left else rep.ok)("isaac.nonvisual_after", f"{len(left)} materials still off-vocabulary")
    base = find_articulation_root(stage, "/World/Robot")
    World = world_cls()
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=0.1,
                  physics_prim_path="/World/PhysicsScene")
    imu = ImuAdapter(f"{base}/imu_link")
    contact = ContactAdapter(base)
    world.reset()
    imu.initialize()
    contact.initialize()
    adapters = {}
    for name, mk in (("lidar", lambda: RtxLidarAdapter(f"{base}/lidar_link")),
                     ("radar", lambda: RtxRadarAdapter(f"{base}/radar_link")),
                     ("camera", lambda: CameraAdapter(f"{base}/camera_link/rgb")),
                     ("acoustic", lambda: AcousticAdapter(f"{base}/acoustic_link"))):
        try:
            adapters[name] = mk()
            info = adapters[name].backend_info
            rep.ok(f"isaac.{name}", json.dumps(info, default=str))
            if info.get("profile_applied") is False:     # 5.x OmniSensor prim: the custom JSON profile was dropped
                rep.warn(f"isaac.{name}.profile", f"requested {info['profile_requested']!r}, sensor model is "
                         f"{info['profile_resolved']} ({info['prim_type']} prim)")
            for w in getattr(adapters[name], "warnings", []):
                rep.warn(f"isaac.{name}.warning", w)
        except Exception as ex:
            (rep.warn if name == "radar" else rep.err)(f"isaac.{name}", f"creation failed: {ex}"
                                                       + (" (the backend runs without radar)" if name == "radar"
                                                          else ""))
    for _ in range(max(1, frames)):
        world.step(render=True)

    def stamp(_sensor, t):
        return t

    t = frames * 0.1
    if "lidar" in adapters:
        raw = adapters["lidar"].ann.get_data() or {}
        shapes = {k: np.shape(v) for k, v in raw.items() if k != "info" and hasattr(v, "__len__")}
        scan = adapters["lidar"].read(t, stamp)
        msg = f"raw {shapes}; scan " + (f"{len(scan.points)} returns on {len(scan.ranges)} rays" if scan else "EMPTY")
        (rep.ok if scan else rep.warn)("isaac.lidar.output", msg)
    if "radar" in adapters:
        raw = adapters["radar"].ann.get_data() or {}
        fr = adapters["radar"].read(t, stamp)
        src = adapters["radar"].doppler_frames
        msg = f"raw keys {sorted(raw)}; {len(fr.detections)} detections; radial velocity source {src}"
        (rep.warn if src["zeros"] else rep.ok)("isaac.radar.output", msg)
    if "camera" in adapters:
        shapes = {}
        for k, an in adapters["camera"].ann.items():
            d = an.get_data()
            if isinstance(d, dict) and "data" in d:
                shapes[k] = np.shape(d["data"])
            else:
                shapes[k] = np.shape(d) if hasattr(d, "shape") else type(d).__name__
        fr = adapters["camera"].read(t, stamp)
        rep.ok("isaac.camera.output", f"{shapes}; frame with {len(fr.detections) if fr else 0} detections")
    imu_s = imu.read(t, stamp)
    (rep.ok if imu_s else rep.warn)("isaac.imu.output", f"{[(s.lin_acc.tolist(), s.ang_vel.tolist()) for s in imu_s]}")
    rep.ok("isaac.contact.output", f"{contact.frame()}")
    world.stop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--scenario-id", default="reflective__0000", help="test scene (a registry entry)")
    ap.add_argument("--work", default=None, help="where to author the test USD (default: a temp dir)")
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    rep = Report()
    check_profiles(rep)
    check_camera_and_rig(rep, load_config("scenarios/default.yaml"))
    work = Path(a.work or tempfile.mkdtemp(prefix="medortrace_validate_"))
    app = None
    if not a.static_only:  # pragma: no cover - Isaac only
        from medortrace.isaac.app import launch
        app = launch(headless=a.headless)       # inside Isaac Sim pxr comes from Kit: boot before authoring USD
    try:
        scene = check_usd(rep, a.scenario_id, a.registry, work)
        if app is not None:  # pragma: no cover
            isaac_checks(rep, app, scene, a.frames)
    finally:
        if app is not None:  # pragma: no cover
            app.close()
    n_err, n_warn = rep.count("ERROR"), rep.count("WARN")
    print(f"\n{n_err} error(s), {n_warn} warning(s), {rep.count('OK')} ok (test scene: {work})")
    return 1 if n_err or (a.strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
