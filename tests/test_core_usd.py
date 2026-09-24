"""OpenUSD authoring (usd-core): the scene + robot rig open cleanly with physics,
semantics, materials and the robot reference resolved."""

from __future__ import annotations

import numpy as np
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from medortrace.common.config import load_config, load_yaml  # noqa: E402
from medortrace.sim.episode import build_episode  # noqa: E402
from medortrace.usd.robot_rig import build_rig  # noqa: E402
from medortrace.usd.scene_builder import build_stage, safe  # noqa: E402
from medortrace.world.materials import MATERIALS  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("usd")
    ep = build_episode(load_config("scenarios/cf_b.yaml", {"hidden_cause": {"factor": "CF-B", "value": "real_obstacle"},
                                                           "scenario_id": "cf_b__test"}), 5)
    rig_path = root / "robot" / "medortrace_rig.usda"
    build_rig(rig_path, load_yaml("robot/rig.yaml"))
    scene_path = root / "scenes" / "scene.usda"
    build_stage(ep.spec, ep.materials, scene_path)          # default reference: ../robot/medortrace_rig.usda
    stage = Usd.Stage.Open(str(scene_path))
    assert stage is not None
    return ep, stage, Usd.Stage.Open(str(rig_path))


def _api_schemas(prim):
    """All authored apiSchemas, including ones not registered with usd-core (legacy Isaac)."""
    lo = prim.GetMetadata("apiSchemas")
    return set(lo.GetAddedOrExplicitItems()) if lo else set()


def test_stage_opens_with_units_metadata_and_no_composition_errors(built):
    ep, stage, _ = built
    assert stage.GetCompositionErrors() == []
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1.0
    assert stage.GetDefaultPrim().GetPath() == "/World"
    meta = stage.GetRootLayer().customLayerData
    assert meta["medortrace:hidden_factor"] == "CF-B" and meta["medortrace:hidden_value"] == "real_obstacle"
    assert meta["medortrace:seed"] == 5 and meta["medortrace:scenario_id"] == "cf_b__test"
    hc = stage.GetPrimAtPath("/World/Annotations/HiddenCause")
    assert hc.GetAttribute("medortrace:value").Get() == "real_obstacle"
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"))
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)
    assert tuple(scene.GetGravityDirectionAttr().Get()) == (0, 0, -1)


def test_robot_reference_resolves_to_articulation(built):
    ep, stage, rig = built
    robot = stage.GetPrimAtPath("/World/Robot")
    assert robot.HasAuthoredReferences() and len(robot.GetPrimStack()) == 2
    base = stage.GetPrimAtPath("/World/Robot/base_link")
    assert base.IsValid() and base.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert base.HasAPI(UsdPhysics.RigidBodyAPI) and base.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MassAPI(base).GetMassAttr().Get() == pytest.approx(42.0)
    xf = UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert np.allclose(list(xf.ExtractTranslation())[:2], ep.spec.robot_start[:2])
    # exactly one articulation root in the whole scene
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert [str(p.GetPath()) for p in roots] == ["/World/Robot/base_link"]
    assert rig.GetDefaultPrim().GetPath() == "/Robot"


def test_rig_joints_drives_and_sensor_frames(built):
    _, stage, _ = built
    joints = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")) if p.IsA(UsdPhysics.Joint)}
    assert set(joints) == {"left_wheel_joint", "right_wheel_joint", "caster_front_joint", "caster_rear_joint",
                           "mast_joint", "arm_shoulder", "gripper_jaw"}
    for name, p in joints.items():
        j = UsdPhysics.Joint(p)
        for rel in (j.GetBody0Rel(), j.GetBody1Rel()):
            (target,) = rel.GetTargets()
            assert stage.GetPrimAtPath(target).IsValid(), (name, target)          # remapped into /World/Robot
    for side in ("left", "right"):
        wj = joints[f"{side}_wheel_joint"]
        assert wj.IsA(UsdPhysics.RevoluteJoint) and UsdPhysics.RevoluteJoint(wj).GetAxisAttr().Get() == "Y"
        drive = UsdPhysics.DriveAPI(wj, "angular")
        assert drive.GetDampingAttr().Get() > 0 and drive.GetStiffnessAttr().Get() == 0   # velocity drive
        assert drive.GetMaxForceAttr().Get() == pytest.approx(40.0)
    sh = UsdPhysics.RevoluteJoint(joints["arm_shoulder"])
    assert (sh.GetLowerLimitAttr().Get(), sh.GetUpperLimitAttr().Get()) == (-10.0, 80.0)
    assert joints["gripper_jaw"].IsA(UsdPhysics.PrismaticJoint)
    assert joints["mast_joint"].IsA(UsdPhysics.FixedJoint)
    frames = {p.GetName(): p.GetAttribute("medortrace:sensor").Get()
              for p in stage.GetPrimAtPath("/World/Robot/base_link").GetChildren()}
    assert frames == {"lidar_link": "rtx_lidar", "camera_link": "rtx_camera", "radar_link": "rtx_radar",
                      "acoustic_link": "rtx_acoustic", "imu_link": "physics_imu"}
    cam = UsdGeom.Camera(stage.GetPrimAtPath("/World/Robot/base_link/camera_link/rgb"))
    assert cam.GetCamera().GetFieldOfView(Gf.Camera.FOVHorizontal) == pytest.approx(90.0, abs=1e-3)
    # optical axis (camera -Z) points along the base's +X (forward), before the link's 25 deg pitch
    rot = UsdGeom.Xformable(cam.GetPrim()).GetLocalTransformation().ExtractRotationMatrix()
    assert np.allclose(np.array(rot.GetTranspose()) @ np.array([0, 0, -1.0]), [1, 0, 0], atol=1e-6)


def test_scene_colliders_rigid_bodies_and_semantics(built):
    ep, stage, _ = built
    for o in ep.spec.objects:
        parent = "/World/Room" if o.kind == "wall" else "/World/Furniture"
        p = stage.GetPrimAtPath(f"{parent}/{safe(o.name)}")
        assert p.IsValid() and p.HasAPI(UsdPhysics.CollisionAPI), o.name
        assert p.HasAPI(UsdPhysics.RigidBodyAPI) == o.rigid_body, o.name
        assert p.GetAttribute("semantics:labels:class").Get() == [o.semantic]
        assert "SemanticsAPI:Semantics" in _api_schemas(p)                      # legacy Isaac form too
        assert p.GetAttribute("semantic:Semantics:params:semanticData").Get() == o.semantic
        ext = UsdGeom.Cube(p).ComputeLocalBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_).ComputeAlignedRange()
        assert np.allclose(np.array(ext.GetMax()) - np.array(ext.GetMin()),
                           _rotated_extent(o.box.half, o.box.yaw), atol=1e-4), o.name
    obstacle = stage.GetPrimAtPath("/World/Furniture/aisle_obstacle")
    assert obstacle.GetAttribute("medortrace:tags").Get() == "hidden_cause"
    cart = stage.GetPrimAtPath("/World/Furniture/cart_1")
    assert UsdPhysics.MassAPI(cart).GetMassAttr().Get() == pytest.approx(45.0)
    assert stage.GetPrimAtPath("/World/Room/Floor").HasAPI(UsdPhysics.CollisionAPI)
    # items: rigid, CCD, class + instance labels, mass from size x density
    for it in ep.spec.items:
        p = stage.GetPrimAtPath(f"/World/Items/{safe(it.id)}")
        assert p.GetAttribute("semantics:labels:class").Get() == [it.cls]
        assert p.GetAttribute("semantics:labels:instance_id").Get() == [it.id]
        assert p.GetAttribute("physxRigidBody:enableCCD").Get() is True
        expect_m = max(0.01, float(np.prod(it.size) * ep.materials[it.material].density))
        assert UsdPhysics.MassAPI(p).GetMassAttr().Get() == pytest.approx(expect_m, rel=1e-5)
    # staff: kinematic capsules labelled person + role
    for s in ep.spec.staff:
        p = stage.GetPrimAtPath(f"/World/Staff/{safe(s.name)}")
        assert UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() is True
        assert p.GetAttribute("semantics:labels:class").Get() == ["person"]
        assert p.GetAttribute("semantics:labels:role").Get() == [s.role]
    assert len([p for p in stage.GetPrimAtPath("/World/Lights").GetChildren() if p.IsA(UsdLux.DiskLight)]) == 2
    zone = stage.GetPrimAtPath("/World/Annotations/SterileZones/field")
    assert UsdGeom.Imageable(zone).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
    assert not zone.HasAPI(UsdPhysics.CollisionAPI)


def _rotated_extent(half, yaw):
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return np.array([2 * (half[0] * c + half[1] * s), 2 * (half[0] * s + half[1] * c), 2 * half[2]])


def test_physics_materials_nonvisual_tokens_and_bindings(built):
    ep, stage, _ = built
    for name, m in ep.materials.items():
        p = stage.GetPrimAtPath(f"/World/Looks/{safe(name)}")
        assert p.HasAPI(UsdPhysics.MaterialAPI), name
        pm = UsdPhysics.MaterialAPI(p)
        assert pm.GetStaticFrictionAttr().Get() == pytest.approx(m.static_friction, rel=1e-6)
        assert pm.GetDynamicFrictionAttr().Get() == pytest.approx(m.dynamic_friction, rel=1e-6)
        assert pm.GetRestitutionAttr().Get() == pytest.approx(m.restitution, rel=1e-6)
        assert pm.GetDensityAttr().Get() == pytest.approx(m.density, rel=1e-6)
        assert p.GetAttribute("omni:simready:nonvisual:base").Get() == m.nonvisual_base
        assert p.GetAttribute("omni:simready:nonvisual:coating").Get() == m.nonvisual_coating
        assert p.GetAttribute("medortrace:specularity").Get() == pytest.approx(m.specularity, abs=1e-6)
        assert p.GetAttribute("medortrace:radar_penetrable").Get() == m.radar_penetrable
        surf = UsdShade.Material(p).GetSurfaceOutput().GetConnectedSources()[0][0].source.GetPrim()
        assert UsdShade.Shader(surf).GetIdAttr().Get() == "UsdPreviewSurface"
    # the episode's nuisance-perturbed values are authored, not the library defaults
    assert any(ep.materials[k] != MATERIALS[k] for k in MATERIALS)
    screen = stage.GetPrimAtPath("/World/Furniture/steel_screen")
    api = UsdShade.MaterialBindingAPI(screen)
    assert api.ComputeBoundMaterial()[0].GetPath() == "/World/Looks/instrument_steel_polished"
    assert api.ComputeBoundMaterial("physics")[0].GetPath() == "/World/Looks/instrument_steel_polished"
    assert "strong_specular" in screen.GetAttribute("medortrace:tags").Get()
