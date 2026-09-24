"""Author the MED-OR-TRACE robot rig as a PhysX articulation in OpenUSD.

Kinematic layout (base_link frame, x forward, z up), matching configs/robot/rig.yaml:

    base_link          cylinder r=0.28 h=0.30, 42 kg, articulation root
    left/right_wheel   r=0.085, revolute about Y, velocity drives (diff drive)
    caster_front/rear  frictionless spheres (fixed)
    mast               0.12 x 0.12 x 1.15 box, fixed
    arm_shoulder       revolute (pitch) - floor-retrieval arm, effort-sensed
    arm_gripper        prismatic jaw with contact sensing
    bumper             ring collider with contact reporting
    sensor frames      lidar_link, camera_link (+ UsdGeom.Camera), radar_link,
                       acoustic_link, imu_link - RTX sensors are attached to these
                       frames at runtime by medortrace.isaac.sensors
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from medortrace.usd.scene_builder import set_xform


def _rigid(stage, path, geom, mass):
    prim = geom.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    return prim


def _joint(stage, cls, path, b0, b1, pos0, pos1=(0, 0, 0), axis="Y"):
    j = cls.Define(stage, path)
    j.CreateBody0Rel().SetTargets([b0])
    j.CreateBody1Rel().SetTargets([b1])
    j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    if hasattr(j, "CreateAxisAttr"):
        j.CreateAxisAttr(axis)
    return j


def build_rig(out_path: str | Path, rig: dict | None = None) -> Usd.Stage:
    rig = rig or {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    root = UsdGeom.Xform.Define(st, "/Robot")
    st.SetDefaultPrim(root.GetPrim())
    R = float(rig.get("footprint_radius", 0.28))
    wr = float(rig.get("wheel_radius", 0.085))
    track = float(rig.get("wheel_track", 0.44))
    base_h = 0.30
    z_base = wr + 0.02 + base_h / 2

    base = UsdGeom.Cylinder.Define(st, "/Robot/base_link")
    base.CreateRadiusAttr(R)
    base.CreateHeightAttr(base_h)
    base.CreateAxisAttr("Z")
    set_xform(base.GetPrim(), (0, 0, z_base))
    bp = _rigid(st, "/Robot/base_link", base, rig.get("base_mass_kg", 42.0))
    UsdPhysics.ArticulationRootAPI.Apply(bp)
    bp.CreateAttribute("medortrace:link", Sdf.ValueTypeNames.String).Set("base_link")
    # contact reporting on the chassis (bumper ring) for the physics contact sensor
    bp.AddAppliedSchema("PhysxContactReportAPI")
    bp.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(1.0)

    for side, y in (("left", track / 2), ("right", -track / 2)):
        w = UsdGeom.Cylinder.Define(st, f"/Robot/{side}_wheel")
        w.CreateRadiusAttr(wr)
        w.CreateHeightAttr(0.04)
        w.CreateAxisAttr("Y")
        set_xform(w.GetPrim(), (0, y, wr))
        _rigid(st, f"/Robot/{side}_wheel", w, 1.5)
        j = _joint(st, UsdPhysics.RevoluteJoint, f"/Robot/joints/{side}_wheel_joint", "/Robot/base_link",
                   f"/Robot/{side}_wheel", (0, y, wr - z_base), axis="Y")
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(0.0)
        d.CreateDampingAttr(1.0e4)          # velocity drive
        d.CreateMaxForceAttr(float(rig.get("wheel_max_torque_nm", 40.0)))
        d.CreateTargetVelocityAttr(0.0)
    for name, x in (("caster_front", R - 0.06), ("caster_rear", -R + 0.06)):
        c = UsdGeom.Sphere.Define(st, f"/Robot/{name}")
        c.CreateRadiusAttr(0.035)
        set_xform(c.GetPrim(), (x, 0, 0.035))
        _rigid(st, f"/Robot/{name}", c, 0.3)
        _joint(st, UsdPhysics.FixedJoint, f"/Robot/joints/{name}_joint", "/Robot/base_link", f"/Robot/{name}",
               (x, 0, 0.035 - z_base))
    mast = UsdGeom.Cube.Define(st, "/Robot/mast")
    mast.CreateSizeAttr(1.0)
    mh = float(rig.get("height_m", 1.55)) - (z_base + base_h / 2)
    set_xform(mast.GetPrim(), (-0.05, 0, z_base + base_h / 2 + mh / 2), 0.0, (0.12, 0.12, mh))
    _rigid(st, "/Robot/mast", mast, 6.0)
    _joint(st, UsdPhysics.FixedJoint, "/Robot/joints/mast_joint", "/Robot/base_link", "/Robot/mast",
           (-0.05, 0, base_h / 2 + mh / 2))

    # retrieval arm: shoulder pitch + prismatic gripper
    arm = UsdGeom.Cube.Define(st, "/Robot/arm_link")
    arm.CreateSizeAttr(1.0)
    set_xform(arm.GetPrim(), (R + 0.2, 0, 0.45), 0.0, (0.4, 0.05, 0.05))
    _rigid(st, "/Robot/arm_link", arm, 1.2)
    sj = _joint(st, UsdPhysics.RevoluteJoint, "/Robot/joints/arm_shoulder", "/Robot/base_link", "/Robot/arm_link",
                (R, 0, 0.45 - z_base), (-0.2, 0, 0), axis="Y")
    sj.CreateLowerLimitAttr(-10.0)
    sj.CreateUpperLimitAttr(80.0)
    sd = UsdPhysics.DriveAPI.Apply(sj.GetPrim(), "angular")
    sd.CreateStiffnessAttr(400.0)
    sd.CreateDampingAttr(40.0)
    sd.CreateMaxForceAttr(float(rig.get("arm_max_torque_nm", 15.0)))
    grip = UsdGeom.Cube.Define(st, "/Robot/gripper")
    grip.CreateSizeAttr(1.0)
    set_xform(grip.GetPrim(), (R + 0.42, 0, 0.45), 0.0, (0.04, 0.08, 0.03))
    gp = _rigid(st, "/Robot/gripper", grip, 0.2)
    gp.AddAppliedSchema("PhysxContactReportAPI")
    gj = _joint(st, UsdPhysics.PrismaticJoint, "/Robot/joints/gripper_jaw", "/Robot/arm_link", "/Robot/gripper",
                (0.2, 0, 0), (-0.02, 0, 0), axis="X")
    gj.CreateLowerLimitAttr(0.0)
    gj.CreateUpperLimitAttr(0.04)
    gd = UsdPhysics.DriveAPI.Apply(gj.GetPrim(), "linear")
    gd.CreateStiffnessAttr(2000.0)
    gd.CreateDampingAttr(100.0)
    gd.CreateMaxForceAttr(float(rig.get("gripper_max_force_n", 25.0)))

    # sensor frames on the mast / base (the RTX sensors attach here)
    frames = rig.get("frames", {
        "lidar_link": {"xyz": [0.1, 0, 0.9], "rpy_deg": [0, 0, 0], "sensor": "rtx_lidar"},
        "camera_link": {"xyz": [0.0, 0, 1.45], "rpy_deg": [0, 25, 0], "sensor": "rtx_camera"},
        "radar_link": {"xyz": [0.25, 0, 0.6], "rpy_deg": [0, 0, 0], "sensor": "rtx_radar"},
        "acoustic_link": {"xyz": [0.1, 0, 1.2], "rpy_deg": [0, 0, 0], "sensor": "rtx_acoustic"},
        "imu_link": {"xyz": [0, 0, 0.2], "rpy_deg": [0, 0, 0], "sensor": "physics_imu"},
    })
    for name, f in frames.items():
        # frames are children of the (moving) base link
        x = UsdGeom.Xform.Define(st, f"/Robot/base_link/{name}")
        p = x.GetPrim()
        xyz = np.array(f["xyz"], float) - np.array([0, 0, z_base])
        xf = UsdGeom.Xformable(p)
        xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, f.get("rpy_deg", [0, 0, 0]))))
        p.CreateAttribute("medortrace:sensor", Sdf.ValueTypeNames.String).Set(f["sensor"])
        if f["sensor"] == "rtx_camera":
            cam = UsdGeom.Camera.Define(st, f"/Robot/base_link/{name}/rgb")
            # USD cameras look down -Z with +Y up: rotate so the optical axis is +X (forward)
            # rotation with columns (X_cam, Y_cam, Z_cam) -> (-Y, +Z, -X) of the link frame
            q = Gf.Matrix3d(0, -1, 0, 0, 0, 1, -1, 0, 0).ExtractRotation().GetQuat()
            UsdGeom.Xformable(cam.GetPrim()).AddOrientOp().Set(Gf.Quatf(q))
            hfov = np.deg2rad(float(rig.get("camera_hfov_deg", 90.0)))
            aperture = 20.955
            cam.CreateHorizontalApertureAttr(aperture)
            cam.CreateVerticalApertureAttr(aperture * 0.75)
            cam.CreateFocalLengthAttr(float(aperture / (2 * np.tan(hfov / 2))))
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 30.0))
    st.GetRootLayer().customLayerData = {"medortrace:schema": "medortrace.robot_rig/1"}
    st.GetRootLayer().Save()
    return st
