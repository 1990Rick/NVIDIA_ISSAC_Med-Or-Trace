"""Author the OR digital twin as OpenUSD (runs with ``usd-core`` or inside Isaac Sim).

Everything is authored from a :class:`SceneSpec`, so the USD stage and the
lite simulator describe the same world.  The stage contains:

* ``/World/PhysicsScene``                      PhysX scene (gravity, solver settings)
* ``/World/Looks/<material>``                  UsdPreviewSurface + OmniPBR/OmniGlass MDL
                                               shader, PhysicsMaterialAPI (friction,
                                               restitution, density) and RTX non-visual
                                               material tokens for lidar/radar
* ``/World/Room/*``, ``/World/Furniture/*``    colliders; movable objects are rigid bodies
* ``/World/Items/*``                           critical items (rigid bodies, semantic labels)
* ``/World/Staff/*``                           kinematic capsule proxies (replaced by
                                               animated characters when enabled)
* ``/World/Lights/*``                          surgical DiskLights + ambient RectLight
* ``/World/Annotations/{Slots,SterileZones,Landmarks}``  task metadata (non-rendered)
* ``/World/Robot``                             reference to the robot rig USD

Semantic labels are written in both the legacy Isaac ``SemanticsAPI`` form
and the newer ``SemanticsLabelsAPI`` form so Replicator annotators work on
Isaac Sim 4.x and 5.x.  Causal labels (hidden cause, scenario id, seed) are
stored in the root layer's ``customLayerData`` and on the relevant prims so
Replicator writers can export them per frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from medortrace.world.materials import Material
from medortrace.world.scene import SceneSpec

SAFE = str.maketrans({":": "_", "-": "_", " ": "_", ".": "_"})


def safe(name: str) -> str:
    n = name.translate(SAFE)
    return n if n[0].isalpha() else "_" + n


def add_semantics(prim: Usd.Prim, label: str, sem_type: str = "class") -> None:
    """Legacy Isaac SemanticsAPI + UsdSemantics labels (Isaac Sim 5.x)."""
    inst = "Semantics" if sem_type == "class" else f"Semantics_{sem_type}"
    prim.AddAppliedSchema(f"SemanticsAPI:{inst}")
    prim.CreateAttribute(f"semantic:{inst}:params:semanticType", Sdf.ValueTypeNames.String).Set(sem_type)
    prim.CreateAttribute(f"semantic:{inst}:params:semanticData", Sdf.ValueTypeNames.String).Set(label)
    prim.AddAppliedSchema(f"SemanticsLabelsAPI:{sem_type}")
    prim.CreateAttribute(f"semantics:labels:{sem_type}", Sdf.ValueTypeNames.TokenArray).Set([label])


def set_xform(prim: Usd.Prim, t, yaw: float = 0.0, scale=None) -> None:
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*map(float, t)))
    if yaw:
        x.AddRotateZOp().Set(float(np.degrees(yaw)))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def make_material(stage: Usd.Stage, path: str, m: Material) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    # portable preview surface (Storm, usdview, any renderer)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*m.diffuse))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(m.metallic)
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(m.roughness)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(m.opacity)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    # MDL for RTX (OmniPBR / OmniGlass / OmniSurface)
    mdl = UsdShade.Shader.Define(stage, f"{path}/MDL")
    mdl.SetSourceAsset(Sdf.AssetPath(m.mdl.split("/")[-1]), "mdl")
    mdl.SetSourceAssetSubIdentifier(Path(m.mdl).stem, "mdl")
    if "Glass" in m.mdl:
        mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*m.diffuse))
        mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(m.roughness)
        mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.49)
    else:
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*m.diffuse))
        mdl.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(m.metallic)
        mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(m.roughness)
        mdl.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(m.opacity < 1.0)
        mdl.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(m.opacity)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(mdl.ConnectableAPI(), "out")
    # physics material
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr(m.static_friction)
    pm.CreateDynamicFrictionAttr(m.dynamic_friction)
    pm.CreateRestitutionAttr(m.restitution)
    pm.CreateDensityAttr(m.density)
    # RTX non-visual material attributes (lidar / radar returns)
    p = mat.GetPrim()
    p.CreateAttribute("omni:simready:nonvisual:base", Sdf.ValueTypeNames.Token).Set(m.nonvisual_base)
    p.CreateAttribute("omni:simready:nonvisual:coating", Sdf.ValueTypeNames.Token).Set(m.nonvisual_coating)
    p.CreateAttribute("omni:simready:nonvisual:attributes", Sdf.ValueTypeNames.Token).Set(m.nonvisual_attributes)
    # explicit parameters mirrored from the lite simulator (for analysis / cross-checks)
    for k in ("lidar_reflectance", "specularity", "transmissivity", "radar_rcs_gain", "acoustic_reflectivity", "glare"):
        p.CreateAttribute(f"medortrace:{k}", Sdf.ValueTypeNames.Float).Set(float(getattr(m, k)))
    p.CreateAttribute("medortrace:radar_penetrable", Sdf.ValueTypeNames.Bool).Set(bool(m.radar_penetrable))
    return mat


def bind(prim: Usd.Prim, mat: UsdShade.Material) -> None:
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(mat)
    api.Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def add_box(stage, path, box, mat, semantic, rigid=False, mass=0.0, kinematic=False, collision=True, extra=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    set_xform(cube.GetPrim(), box.center, box.yaw, 2.0 * box.half)
    prim = cube.GetPrim()
    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)
    if rigid:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr(bool(kinematic))
        if mass > 0:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    bind(prim, mat)
    add_semantics(prim, semantic)
    for k, v in (extra or {}).items():
        typ = Sdf.ValueTypeNames.Bool if isinstance(v, bool) else Sdf.ValueTypeNames.String if isinstance(v, str) \
            else Sdf.ValueTypeNames.Float
        prim.CreateAttribute(f"medortrace:{k}", typ).Set(v)
    return prim


def build_stage(
    spec: SceneSpec,
    materials: dict[str, Material],
    out_path: str | Path,
    robot_rig: str | None = "../robot/medortrace_rig.usda",
    item_positions: dict | None = None,
) -> Usd.Stage:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.GetRootLayer().customLayerData = {
        "medortrace:scenario_id": spec.scenario_id, "medortrace:family": spec.family,
        "medortrace:seed": int(spec.seed), "medortrace:hidden_factor": str(spec.hidden_cause.get("factor", "none")),
        "medortrace:hidden_value": str(spec.hidden_cause.get("value", "none")),
        "medortrace:room": Gf.Vec3d(*spec.room), "medortrace:schema": "medortrace.usd_scene/1",
    }
    ps = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr(9.81)
    # PhysX scene settings (PhysxSceneAPI) as raw attributes: TGS solver, CCD for small items
    ps.GetPrim().AddAppliedSchema("PhysxSceneAPI")
    ps.GetPrim().CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")
    ps.GetPrim().CreateAttribute("physxScene:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
    ps.GetPrim().CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt).Set(120)

    looks = {}
    UsdGeom.Scope.Define(stage, "/World/Looks")
    for name, m in materials.items():
        looks[name] = make_material(stage, f"/World/Looks/{safe(name)}", m)

    # floor
    W, D, H = spec.room
    fl = UsdGeom.Mesh.Define(stage, "/World/Room/Floor")
    fl.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(W, 0, 0), Gf.Vec3f(W, D, 0), Gf.Vec3f(0, D, 0)])
    fl.CreateFaceVertexCountsAttr([4])
    fl.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    fl.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    fl.CreateExtentAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(W, D, 0)])
    UsdPhysics.CollisionAPI.Apply(fl.GetPrim())
    bind(fl.GetPrim(), looks["floor_vinyl"])
    add_semantics(fl.GetPrim(), "floor")
    UsdGeom.Xform.Define(stage, "/World/Furniture")
    for o in spec.objects:
        parent = "/World/Room" if o.kind == "wall" else "/World/Furniture"
        add_box(stage, f"{parent}/{safe(o.name)}", o.box, looks[o.material], o.semantic,
                rigid=o.rigid_body, mass=o.mass_kg,
                extra={"kind": o.kind, "sterile": bool(o.sterile), "movable": bool(o.movable),
                       "tags": ",".join(o.tags)})

    # items
    UsdGeom.Xform.Define(stage, "/World/Items")
    for it in spec.items:
        sid = it.initial_slot
        pos = None
        if item_positions and it.id in item_positions:
            pos = item_positions[it.id]
        elif sid in spec.slot_ids() and spec.slot(sid).kind not in ("hand", "elsewhere"):
            pos = spec.slot(sid).position
        from medortrace.common.geometry import OrientedBox
        half = np.array(it.size) / 2
        visible = pos is not None and np.all(np.isfinite(pos))
        c = np.array(pos) + np.array([0, 0, half[2]]) if visible else np.array([0.0, 0.0, -5.0])
        prim = add_box(stage, f"/World/Items/{safe(it.id)}", OrientedBox(c, half), looks[it.material], it.cls,
                       rigid=True, mass=float(max(0.01, np.prod(it.size) * materials[it.material].density)),
                       extra={"item_id": it.id, "initial_slot": sid, "criticality": float(it.criticality),
                              "fungible": bool(it.fungible), "tag_readable": bool(it.tag_readable)})
        add_semantics(prim, it.id, "instance_id")
        # thin items need CCD to avoid tunnelling through tray surfaces
        prim.AddAppliedSchema("PhysxRigidBodyAPI")
        prim.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()

    # staff (kinematic capsule proxies driven by medortrace.world.agents)
    UsdGeom.Xform.Define(stage, "/World/Staff")
    for s in spec.staff:
        cap = UsdGeom.Capsule.Define(stage, f"/World/Staff/{safe(s.name)}")
        cap.CreateRadiusAttr(float(s.radius))
        cap.CreateHeightAttr(1.75 - 2 * s.radius)
        cap.CreateAxisAttr("Z")
        set_xform(cap.GetPrim(), (s.home[0], s.home[1], 0.875))
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Apply(cap.GetPrim())
        rb.CreateKinematicEnabledAttr(True)
        bind(cap.GetPrim(), looks["gown_fabric"] if not s.sterile else looks["surgical_drape"])
        add_semantics(cap.GetPrim(), "person")
        add_semantics(cap.GetPrim(), s.role, "role")
        cap.GetPrim().CreateAttribute("medortrace:sterile", Sdf.ValueTypeNames.Bool).Set(bool(s.sterile))
        cap.GetPrim().CreateAttribute("medortrace:roaming", Sdf.ValueTypeNames.Bool).Set(bool(s.roaming))

    # lights (OR surgical lights are ~40-160 klux at the field)
    UsdGeom.Xform.Define(stage, "/World/Lights")
    for li in spec.lights:
        if li.kind == "surgical":
            lt = UsdLux.DiskLight.Define(stage, f"/World/Lights/{safe(li.name)}")
            lt.CreateRadiusAttr(0.3)
            lt.CreateIntensityAttr(float(li.intensity) / 10.0)
            lt.CreateColorTemperatureAttr(4300.0)
            lt.CreateEnableColorTemperatureAttr(True)
            set_xform(lt.GetPrim(), li.position)   # DiskLight emits along -Z: pointing at the table
        else:
            lt = UsdLux.RectLight.Define(stage, f"/World/Lights/{safe(li.name)}")
            lt.CreateWidthAttr(float(W) * 0.8)
            lt.CreateHeightAttr(float(D) * 0.8)
            lt.CreateIntensityAttr(float(li.intensity))
            set_xform(lt.GetPrim(), li.position)

    # annotations
    UsdGeom.Scope.Define(stage, "/World/Annotations")
    for s in spec.slots:
        x = UsdGeom.Xform.Define(stage, f"/World/Annotations/Slots/{safe(s.id)}")
        p = s.position if np.all(np.isfinite(s.position)) else np.array([0, 0, -10.0])
        set_xform(x.GetPrim(), p)
        pr = x.GetPrim()
        for k, v in {
            "slot_id": s.id,
            "kind": s.kind,
            "anchor": s.anchor,
            "acoustic_region": s.acoustic_region or "",
        }.items():
            pr.CreateAttribute(f"medortrace:{k}", Sdf.ValueTypeNames.String).Set(v)
        for k in ("hidden_from_camera", "needs_top_view", "sterile"):
            pr.CreateAttribute(f"medortrace:{k}", Sdf.ValueTypeNames.Bool).Set(bool(getattr(s, k)))
        pr.CreateAttribute("medortrace:radius", Sdf.ValueTypeNames.Float).Set(float(s.radius))
    for z in spec.sterile_zones:
        # visual-only floor decal (no collision) + metadata used by the safety layer
        g = UsdGeom.Cube.Define(stage, f"/World/Annotations/SterileZones/{safe(z.name)}")
        g.CreateSizeAttr(1.0)
        set_xform(g.GetPrim(), (z.box.center[0], z.box.center[1], 0.001), z.box.yaw,
                  (2 * z.box.half[0], 2 * z.box.half[1], 0.002))
        UsdGeom.Imageable(g.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.guide)
        g.GetPrim().CreateAttribute("medortrace:keepout_margin", Sdf.ValueTypeNames.Float).Set(float(z.keepout_margin))
    for lm in spec.landmarks:
        x = UsdGeom.Xform.Define(stage, f"/World/Annotations/Landmarks/{safe(lm.id)}")
        set_xform(x.GetPrim(), lm.position)
        x.GetPrim().CreateAttribute("medortrace:fiducial_family", Sdf.ValueTypeNames.String).Set("tag36h11")

    # hidden cause marker (causal label preserved through randomisation)
    hc = UsdGeom.Scope.Define(stage, "/World/Annotations/HiddenCause").GetPrim()
    for k, v in spec.hidden_cause.items():
        hc.CreateAttribute(f"medortrace:{safe(k)}", Sdf.ValueTypeNames.String).Set(str(v))

    # robot
    if robot_rig:
        r = stage.DefinePrim("/World/Robot", "Xform")
        r.GetReferences().AddReference(robot_rig)
        set_xform(r, (spec.robot_start[0], spec.robot_start[1], 0.0), float(spec.robot_start[2]))
    stage.GetRootLayer().Save()
    return stage
