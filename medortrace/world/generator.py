"""Procedural OR scene family generator.

``generate_scene(cfg, streams)`` produces a :class:`SceneSpec` whose *causal*
content (the hidden condition under study) is controlled exclusively by
``cfg["hidden_cause"]`` and the ``hidden`` RNG stream, while everything else
(room size, furniture jitter, materials, clutter, lighting) is nuisance drawn
from the ``layout``/``materials``/``clutter``/``visibility`` streams.  Two
scenarios that share a seed but differ only in ``hidden_cause.value`` are a
*matched counterfactual pair*: identical nuisance, identical staff
trajectories, different hidden physical cause.
"""

from __future__ import annotations

import numpy as np

from medortrace.common.geometry import OrientedBox
from medortrace.common.rng import RngStreams
from medortrace.world.materials import MATERIALS
from medortrace.world.scene import (
    ItemSpec,
    Landmark,
    Light,
    SceneObject,
    SceneSpec,
    Slot,
    StaffSpec,
    SterileZone,
)

HIDDEN_FACTORS = {
    # factor: (description, allowed values)
    "none": ("nominal episode, no hidden-cause manipulation", ["none"]),
    "CF-A": ("retained sponge: under drape vs. correctly in kick bucket (log identical)",
             ["under_drape", "kick_bucket"]),
    "CF-B": ("lidar return in aisle: specular multipath ghost vs. real low obstacle",
             ["specular_ghost", "real_obstacle"]),
    "CF-C": ("instrument leaves mayo stand unlogged: dropped to floor vs. handed to assistant",
             ["dropped_floor", "handed_off"]),
    "CF-D": ("scan/map mismatch: odometry drift vs. cart physically moved",
             ["loc_drift", "cart_moved"]),
}


def _box(c, h, yaw=0.0):
    return OrientedBox(np.array(c, float), np.array(h, float), yaw)


def generate_scene(cfg: dict, streams: RngStreams) -> SceneSpec:
    L = streams["layout"]
    C = streams["clutter"]
    V = streams["visibility"]
    lay = cfg.get("layout", {})

    W = float(L.uniform(*lay.get("room_width", [8.0, 9.5])))
    D = float(L.uniform(*lay.get("room_depth", [7.5, 8.5])))
    H = 3.0
    tx = W * 0.5 + float(L.uniform(-0.3, 0.3))
    ty = D * 0.5 + float(L.uniform(-0.2, 0.2))
    tyaw = float(L.uniform(-0.06, 0.06))

    objs: list[SceneObject] = []
    wt = 0.1
    for name, c, h in [
        ("wall_south", (W / 2, -wt / 2, H / 2), (W / 2 + wt, wt / 2, H / 2)),
        ("wall_north", (W / 2, D + wt / 2, H / 2), (W / 2 + wt, wt / 2, H / 2)),
        ("wall_west", (-wt / 2, D / 2, H / 2), (wt / 2, D / 2, H / 2)),
        ("wall_east", (W + wt / 2, D / 2, H / 2), (wt / 2, D / 2, H / 2)),
    ]:
        objs.append(SceneObject(name, "wall", _box(c, h), "painted_wall", "wall"))

    # --- sterile core ------------------------------------------------------
    objs.append(SceneObject("or_table", "or_table", _box((tx, ty, 0.45), (1.0, 0.3, 0.45), tyaw),
                            "stainless_steel_brushed", "or_table", sterile=True, mass_kg=250))
    objs.append(SceneObject("patient_drape", "drape", _box((tx, ty, 0.68), (1.3, 0.75, 0.28), tyaw),
                            "surgical_drape", "drape", sterile=True))
    bt = (tx + 0.5 + float(L.uniform(-0.2, 0.2)), ty + 1.6)
    objs.append(SceneObject("back_table", "back_table", _box((bt[0], bt[1], 0.45), (0.75, 0.38, 0.45)),
                            "stainless_steel_brushed", "back_table", sterile=True, mass_kg=60))
    objs.append(SceneObject("back_table_drape", "drape", _box((bt[0], bt[1], 0.7), (0.8, 0.42, 0.22)),
                            "surgical_drape", "drape", sterile=True))
    mayo = (tx + 1.25, ty - 0.05)
    objs.append(SceneObject("mayo_stand", "mayo_stand", _box((mayo[0], mayo[1], 1.1), (0.25, 0.18, 0.02)),
                            "stainless_steel_brushed", "mayo_stand", sterile=True, mass_kg=12))
    objs.append(SceneObject("anesthesia_machine", "anesthesia", _box((tx - 1.85, ty, 0.7), (0.4, 0.35, 0.7)),
                            "plastic_hdpe", "anesthesia_machine", mass_kg=150))

    # --- carts, bins, counter ---------------------------------------------
    c1 = (W * 0.2 + float(L.uniform(-0.3, 0.3)), D - 0.4)
    c2 = (W * 0.82 + float(L.uniform(-0.2, 0.2)), D - 0.4)
    for name, c in (("cart_1", c1), ("cart_2", c2)):
        objs.append(SceneObject(name, "cart", _box((c[0], c[1], 0.5), (0.35, 0.25, 0.5)),
                                "stainless_steel_brushed", "instrument_cart", movable=True,
                                rigid_body=True, mass_kg=45))
    kb1 = (tx + 0.55, ty - 1.62)
    kb2 = (tx - 0.55, ty - 1.62)
    for name, c in (("kick_bucket_1", kb1), ("kick_bucket_2", kb2)):
        objs.append(SceneObject(name, "kick_bucket", _box((c[0], c[1], 0.22), (0.2, 0.2, 0.22)),
                                "stainless_steel_brushed", "kick_bucket", movable=True,
                                rigid_body=True, mass_kg=4))
    wb = (0.55, 0.55)
    objs.append(SceneObject("waste_bin", "waste_bin", _box((wb[0], wb[1], 0.35), (0.25, 0.25, 0.35)),
                            "plastic_hdpe", "waste_bin", movable=True, rigid_body=True, mass_kg=6))
    sc = (W - 1.0, 0.35)
    objs.append(SceneObject("specimen_counter", "counter", _box((sc[0], sc[1], 0.45), (0.6, 0.3, 0.45)),
                            "stainless_steel_brushed", "counter"))

    # --- overhead equipment (hits upper lidar rings; reflective glass) ------
    objs.append(SceneObject("surgical_light_1", "light_head", _box((tx - 0.3, ty, 2.0), (0.35, 0.35, 0.08)),
                            "stainless_steel_brushed", "surgical_light"))
    objs.append(SceneObject("surgical_light_2", "light_head", _box((tx + 0.5, ty + 0.2, 2.05), (0.3, 0.3, 0.08)),
                            "stainless_steel_brushed", "surgical_light"))
    objs.append(SceneObject("monitor_boom", "monitor", _box((tx + 0.3, ty - 1.25, 1.75), (0.32, 0.03, 0.2)),
                            "monitor_glass", "monitor"))

    # Freestanding mirror-finish steel screen (warming-cabinet side) near the
    # south aisle, plus a linen hamper east of it.  The hamper's specular
    # multipath image appears *behind* the screen (see CF-B).
    px = float(L.uniform(W * 0.17, W * 0.24))
    py = 1.3
    objs.append(SceneObject("steel_screen", "panel", _box((px, py, 0.8), (0.02, 0.4, 0.8)),
                            "instrument_steel_polished", "cabinet", tags=["specular"]))
    objs.append(SceneObject("linen_hamper", "clutter", _box((px + 0.9, py + 0.15, 0.4), (0.2, 0.2, 0.4)),
                            "plastic_hdpe", "hamper", movable=True, rigid_body=True, mass_kg=10))

    # --- clutter (nuisance) -------------------------------------------------
    n_clutter = int(C.integers(*cfg.get("nuisance", {}).get("clutter_count", [2, 6])))
    clutter_mats = ["plastic_hdpe", "surgical_drape", "acrylic_clear", "stainless_steel_brushed"]
    placed = 0
    tries = 0
    while placed < n_clutter and tries < 200:
        tries += 1
        # clutter hugs the walls, leaving the ring aisle around the sterile zone open
        side = int(C.integers(0, 3))
        if side == 0:
            p = (float(C.uniform(1.4, W - 1.4)), float(C.uniform(0.25, 0.45)))
        elif side == 1:
            p = (float(C.uniform(0.25, 0.45)), float(C.uniform(1.4, D - 1.4)))
        else:
            p = (float(C.uniform(W - 0.45, W - 0.25)), float(C.uniform(1.4, D - 1.4)))
        hsz = (float(C.uniform(0.1, 0.25)), float(C.uniform(0.1, 0.25)), float(C.uniform(0.2, 0.7)))
        b = _box((p[0], p[1], hsz[2]), hsz, float(C.uniform(-0.5, 0.5)))
        if any(o.box.distance_xy(np.array([p]))[0] < 0.35 for o in objs if o.kind not in ("wall", "light_head", "monitor")):
            continue
        if np.hypot(p[0] - (W - 0.7), p[1] - D * 0.5) < 1.0:
            continue  # keep the charging dock / entry clear
        objs.append(SceneObject(f"clutter_{placed}", "clutter", b, str(C.choice(clutter_mats)), "clutter",
                                movable=True, rigid_body=True, mass_kg=float(C.uniform(2, 15))))
        placed += 1

    # --- sterile zones --------------------------------------------------------
    keepout = float(cfg.get("safety", {}).get("sterile_keepout_margin", 0.3))
    zones = [
        SterileZone("field", _box((tx + 0.1, ty, 0.0), (1.45, 1.05, 0.0), tyaw), keepout),
        SterileZone("back_table", _box((bt[0], bt[1], 0.0), (0.9, 0.5, 0.0)), keepout),
    ]

    # --- slots -------------------------------------------------------------
    slots = [
        Slot("back_table:tray", "surface", "back_table", (bt[0] - 0.3, bt[1], 0.92), sterile=True, radius=0.3),
        Slot("back_table:specimen_cup", "surface", "back_table", (bt[0] + 0.45, bt[1] + 0.1, 0.95), sterile=True, radius=0.15),
        Slot("back_table:under_towel", "under_drape", "back_table", (bt[0] + 0.1, bt[1] - 0.15, 0.92),
             hidden_from_camera=True, sterile=True, acoustic_region="back_table_towel"),
        Slot("mayo:top", "surface", "mayo_stand", (mayo[0], mayo[1], 1.12), sterile=True, radius=0.2),
        Slot("field:top", "surface", "patient_drape", (tx + 0.3, ty, 0.97), sterile=True, radius=0.4),
        Slot("field:under_drape", "under_drape", "patient_drape", (tx + 0.4, ty - 0.35, 0.8),
             hidden_from_camera=True, sterile=True, acoustic_region="patient_drape"),
        Slot("kick_bucket_1:inside", "container", "kick_bucket_1", (kb1[0], kb1[1], 0.3), needs_top_view=True, radius=0.2),
        Slot("kick_bucket_2:inside", "container", "kick_bucket_2", (kb2[0], kb2[1], 0.3), needs_top_view=True, radius=0.2),
        Slot("waste_bin:inside", "container", "waste_bin", (wb[0], wb[1], 0.5), hidden_from_camera=True,
             acoustic_region="waste_bin", radius=0.25),
        Slot("specimen_counter:top", "surface", "specimen_counter", (sc[0], sc[1], 0.92), radius=0.3),
        Slot("cart_1:top", "surface", "cart_1", (c1[0], c1[1], 1.01), radius=0.3),
        Slot("cart_2:top", "surface", "cart_2", (c2[0], c2[1], 1.01), radius=0.3),
        Slot("floor:foot_of_table", "floor", "or_table", (tx + 1.75, ty - 0.9, 0.02), radius=0.3),
        Slot("floor:behind_anesthesia", "floor", "anesthesia_machine", (tx - 1.85, ty + 0.65, 0.02), radius=0.3),
        Slot("floor:behind_cart_2", "floor", "cart_2", (c2[0] + 0.55, c2[1] - 0.15, 0.02), radius=0.3),
    ]

    # --- staff -------------------------------------------------------------
    field_y_lo = ty - 0.75
    field_y_hi = ty + 0.75
    staff = [
        StaffSpec("surgeon", "surgeon", np.array([tx + 0.2, field_y_lo - 0.3]), True, speed=0.3),
        StaffSpec("assistant", "assistant", np.array([tx - 0.1, field_y_hi + 0.3]), True, speed=0.3),
        StaffSpec("scrub_nurse", "scrub_nurse", np.array([tx + 1.0, field_y_hi + 0.45]), True, speed=0.4),
        StaffSpec("anesthetist", "anesthetist", np.array([tx - 1.9, ty - 0.7]), False, speed=0.6,
                  waypoints=[np.array([tx - 1.9, ty - 0.7]), np.array([c1[0], c1[1] - 0.6]),
                             np.array([tx - 1.9, ty - 0.7])], roaming=True),
        StaffSpec("circulator", "circulating_nurse", np.array([W - 1.2, 1.1]), False, speed=1.0,
                  waypoints=[np.array([sc[0], sc[1] + 0.6]), np.array([c2[0], c2[1] - 0.65]),
                             np.array([bt[0] + 1.35, bt[1] - 0.2]), np.array([c1[0], c1[1] - 0.65]),
                             np.array([wb[0] + 0.6, wb[1] + 0.6]), np.array([tx + 0.3, ty - 2.35])],
                  roaming=True),
    ]
    for s in staff:
        s.speed *= float(streams["agents"].uniform(0.85, 1.15))

    # --- items -------------------------------------------------------------
    n_sponges = int(cfg.get("items", {}).get("sponges", 4))
    items = [ItemSpec(f"sponge_{i+1}", "sponge", "cotton_sponge", "back_table:tray", criticality=1.0,
                      fungible=True, tag_readable=False, size=(0.1, 0.1, 0.01)) for i in range(n_sponges)]
    items += [
        ItemSpec("clamp_1", "clamp", "instrument_steel_polished", "mayo:top", criticality=0.8,
                 metallic=True, size=(0.16, 0.05, 0.01)),
        ItemSpec("needle_driver_1", "needle_driver", "instrument_steel_polished", "mayo:top",
                 criticality=0.9, metallic=True, size=(0.18, 0.04, 0.01)),
        ItemSpec("specimen_1", "specimen", "plastic_hdpe", "hand:surgeon", criticality=1.5,
                 size=(0.08, 0.08, 0.1)),
        ItemSpec("implant_box_1", "implant_box", "plastic_hdpe", "cart_2:top", criticality=1.2,
                 size=(0.25, 0.15, 0.06)),
    ]
    for s in staff:
        slots.append(Slot(f"hand:{s.name}", "hand", s.name, np.array([s.home[0], s.home[1], 1.0]),
                          sterile=s.sterile, radius=0.35))
    slots.append(Slot("elsewhere", "elsewhere", "none", np.array([np.nan, np.nan, np.nan]),
                      hidden_from_camera=True, radius=0.0))

    # --- landmarks (surveyed wall fiducials) ------------------------------
    lms = []
    for i, (x, y) in enumerate([(0.0, D * 0.3), (0.0, D * 0.75), (W * 0.25, D), (W * 0.7, D),
                                (W, D * 0.35), (W, D * 0.8), (W * 0.2, 0.0), (W * 0.75, 0.0)]):
        lms.append(Landmark(f"tag_{i}", np.array([x, y, 1.6])))

    lights = [Light("surgical_1", np.array([tx - 0.3, ty, 1.9]), float(V.uniform(40000, 120000))),
              Light("surgical_2", np.array([tx + 0.5, ty + 0.2, 1.95]), float(V.uniform(40000, 120000))),
              Light("ambient", np.array([W / 2, D / 2, 2.9]), float(V.uniform(300, 1200)), kind="ambient")]

    spec = SceneSpec(
        seed=streams.seed, room=(W, D, H), objects=objs, slots=slots, sterile_zones=zones,
        landmarks=lms, staff=staff, items=items, lights=lights,
        # charging dock beside the corridor door on the east wall
        robot_start=np.array([W - 0.75, D * 0.5, np.pi]), dock=np.array([W - 0.7, D * 0.5]),
        hidden_cause=dict(cfg.get("hidden_cause", {"factor": "none", "value": "none"})),
        nuisance={"haze": float(V.uniform(*cfg.get("nuisance", {}).get("haze", [0.0, 0.15]))),
                  "glare_gain": float(V.uniform(*cfg.get("nuisance", {}).get("glare_gain", [0.8, 1.3]))),
                  "material_scale": float(cfg.get("nuisance", {}).get("material_perturb_scale", 1.0))},
        family=cfg.get("family", "nominal"),
        scenario_id=cfg.get("scenario_id", ""),
    )
    _apply_hidden_cause_geometry(spec, streams)
    return spec


def _apply_hidden_cause_geometry(spec: SceneSpec, streams: RngStreams) -> None:
    """Geometry-level consequences of the hidden cause (CF-B only).

    The other factors act through the workflow ground truth (``world.workflow``)
    or the fault model (``sim.faults``); CF-B needs a physical object.
    """
    hc = spec.hidden_cause
    if hc.get("factor") != "CF-B":
        return
    screen = spec.object("steel_screen")
    hamper = spec.object("linen_hamper")
    # Mirror the hamper across the screen plane (x = screen.x): the ghost lies
    # at the mirror image.  From east-side viewpoints the real obstacle at the
    # same place is occluded by the screen, so lidar evidence is identical.
    sx = screen.box.center[0]
    ghost_xy = np.array([2 * sx - hamper.box.center[0], hamper.box.center[1]])
    spec.hidden_cause["aisle_point"] = ghost_xy.tolist()
    if hc.get("value") == "real_obstacle":
        spec.objects.append(SceneObject("aisle_obstacle", "clutter",
                                        _box((ghost_xy[0], ghost_xy[1], 0.4), (0.2, 0.2, 0.4)),
                                        "plastic_hdpe", "clutter", movable=True,
                                        rigid_body=True, mass_kg=10, tags=["hidden_cause"]))
    # The screen is equally specular in both arms of the pair (matched evidence).
    screen.tags.append("strong_specular")


def perturbed_materials(spec: SceneSpec, streams: RngStreams) -> dict:
    """Nuisance-randomised material table for this episode (causal classes fixed)."""
    rng = streams["materials"]
    scale = spec.nuisance.get("material_scale", 1.0)
    return {k: m.perturbed(rng, scale) for k, m in MATERIALS.items()}
