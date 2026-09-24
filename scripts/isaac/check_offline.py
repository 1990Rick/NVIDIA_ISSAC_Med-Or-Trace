#!/usr/bin/env python3
"""Offline checks of the Isaac integration (plain Python + usd-core; no Isaac Sim needed).

    PYTHONPATH=. python scripts/isaac/check_offline.py [--keep DIR]

Causal lock / nuisance randomiser (the pxr-only parts of ``medortrace.isaac.replicator_randomizers``):
  * CF-B stages (both arms) randomise for several frames without CausalViolation;
  * a randomiser that moves a locked prim, an out-of-band move of a locked item, an edit of a locked
    material input and pushing clutter into the CF-B keep-clear disc all raise CausalViolation;
  * moving unlocked clutter does not; ``causal_edit()`` accepts deliberate item moves;
  * matched pairs (CF-B, CF-D): same nuisance seed -> identical nuisance signature per frame in both
    arms, different seed -> different; per-frame randomisation is idempotent (no drift);
  * PreviewSurface and MDL (what RTX renders) receive the same appearance values.
Pure helpers: lidar grid binning, pinhole rays, the gt-surrogate on a Replicator-style structured
array, custody placement parity with ``LiteBackend``, ``CausalLabelWriter`` output, ROS 2 graph wiring.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from _common import DEFAULT_REGISTRY

from pxr import Gf, Usd, UsdGeom, UsdShade

from medortrace.eval.registry import load_registry
from medortrace.isaac.replicator_randomizers import (CausalLabelWriter, CausalLock, CausalViolation,
                                                     NuisanceRandomizer)
from medortrace.sim.episode import build_episode
from medortrace.usd.robot_rig import build_rig
from medortrace.usd.scene_builder import build_stage

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        def run(*a, **kw):
            try:
                msg = fn(*a, **kw) or ""
                RESULTS.append((name, True, msg))
                print(f"PASS  {name}  {msg}")
            except Exception as e:
                RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL  {name}  {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
        return run
    return deco


def expect_violation(fn) -> str:
    try:
        fn()
    except CausalViolation as e:
        return str(e)
    raise AssertionError("expected CausalViolation, none raised")


def translate_op(prim):
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            return op
    raise KeyError(prim.GetPath())


class Stages:
    def __init__(self, work: Path, registry: str):
        self.work = work
        self.reg = {e.scenario_id: e for e in load_registry(registry)}
        build_rig(work / "robot" / "medortrace_rig.usda")
        self.eps = {}

    def build(self, sid: str, tag: str = "") -> Usd.Stage:
        e = self.reg[sid]
        if sid not in self.eps:
            self.eps[sid] = build_episode(e.resolve(), e.seed)
        ep = self.eps[sid]
        return build_stage(ep.spec, ep.materials, self.work / "scenes" / f"{sid}{tag}.usda",
                           robot_rig="../robot/medortrace_rig.usda")


# ---------------------------------------------------------------------------
@check("cfb.apply_no_violation")
def c_apply(S: Stages):
    out = []
    for sid in ("cf_b__p0000__specular_ghost", "cf_b__p0000__real_obstacle"):
        st = S.build(sid)
        lock = CausalLock(st)
        rnd = NuisanceRandomizer(st, np.random.default_rng(0), lock)
        sigs = {rnd.apply()["causal_signature"] for _ in range(4)}
        assert len(sigs) == 1, "causal signature changed across frames"
        need = {"/World/Furniture/steel_screen", "/World/Furniture/linen_hamper"}
        assert need <= set(lock.prims), f"CF-B anchors not locked: {need - set(lock.prims)}"
        if sid.endswith("real_obstacle"):
            assert "/World/Furniture/aisle_obstacle" in lock.prims
        out.append(f"{sid.split('__')[-1]}: {len(lock.prims)} prims/{len(lock.materials)} materials locked")
    return "; ".join(out)


@check("cfb.rogue_randomizer_moving_locked_prim_raises")
def c_rogue(S: Stages):
    st = S.build("cf_b__p0000__real_obstacle", "_rogue")

    class Rogue(NuisanceRandomizer):
        def rogue(self):
            op = translate_op(self.stage.GetPrimAtPath("/World/Furniture/aisle_obstacle"))
            op.Set(op.Get() + Gf.Vec3d(0.3, 0.0, 0.0))

    rnd = Rogue(st, seed=1)
    msg = expect_violation(lambda: rnd.apply(which=("materials", "rogue")))
    assert "rogue" in msg and "aisle_obstacle" in msg, msg
    return msg[:110]


@check("cfb.out_of_band_locked_item_move_raises")
def c_oob(S: Stages):
    st = S.build("cf_b__p0000__specular_ghost", "_oob")
    rnd = NuisanceRandomizer(st, seed=2)
    rnd.apply()
    op = translate_op(st.GetPrimAtPath("/World/Items/sponge_1"))
    op.Set(op.Get() + Gf.Vec3d(0.0, 0.05, 0.0))
    return expect_violation(rnd.apply)[:110]


@check("cfb.locked_material_edit_raises")
def c_mat(S: Stages):
    st = S.build("cf_b__p0000__specular_ghost", "_mat")
    rnd = NuisanceRandomizer(st, seed=3)
    sh = UsdShade.Shader(st.GetPrimAtPath("/World/Looks/instrument_steel_polished/MDL"))
    sh.GetInput("reflection_roughness_constant").Set(0.5)
    return expect_violation(rnd.apply)[:110]


@check("cfb.unlocked_clutter_move_ok_but_keep_clear_intrusion_raises")
def c_clutter(S: Stages):
    st = S.build("cf_b__p0000__specular_ghost", "_clutter")
    lock = CausalLock(st)
    rnd = NuisanceRandomizer(st, seed=4, lock=lock)
    clutter = [p for p in st.GetPrimAtPath("/World/Furniture").GetChildren()
               if p.GetAttribute("medortrace:kind").Get() == "clutter" and not lock.locked(str(p.GetPath()))]
    assert clutter, "no unlocked clutter in this scene"
    op = translate_op(clutter[0])
    op.Set(op.Get() + Gf.Vec3d(0.05, 0.0, 0.0))
    rnd.apply()
    (xy, _r), = lock.keep_clear
    op.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), op.Get()[2]))
    msg = expect_violation(rnd.apply)
    assert "<keep_clear>" in msg, msg
    return f"moved {clutter[0].GetName()} freely; intrusion -> {msg[:60]}"


@check("cfb.causal_edit_accepts_item_move")
def c_edit(S: Stages):
    st = S.build("cf_b__p0000__specular_ghost", "_edit")
    lock = CausalLock(st)
    rnd = NuisanceRandomizer(st, seed=5, lock=lock)
    s0 = rnd.apply()["causal_signature"]
    with lock.causal_edit():
        op = translate_op(st.GetPrimAtPath("/World/Items/sponge_1"))
        op.Set(op.Get() + Gf.Vec3d(0.0, 0.05, 0.0))
    s1 = rnd.apply()["causal_signature"]
    assert s0 != s1
    return "signature re-based after deliberate edit"


@check("cfb.layer_metadata_only_medortrace_keys_locked")
def c_layer(S: Stages):
    st = S.build("cf_b__p0000__specular_ghost", "_layer")
    rnd = NuisanceRandomizer(st, seed=6)
    rnd.apply()
    layer = st.GetRootLayer()
    layer.customLayerData = {**layer.customLayerData, "renderSettings": {"rtx:fog:enabled": True}}  # what Kit does
    rnd.apply()
    layer.customLayerData = {**layer.customLayerData, "medortrace:hidden_value": "real_obstacle"}
    msg = expect_violation(rnd.apply)
    assert "<layer>" in msg, msg
    return "Kit renderSettings ignored; hidden-value relabel caught"


def _pair_signatures(S: Stages, arms, seed: int, frames=(0, 1, 2), tag=""):
    out = {}
    for sid in arms:
        st = S.build(sid, f"_pair{seed}{tag}")
        rnd = NuisanceRandomizer(st, seed=seed)
        out[sid] = [rnd.apply(frame=f)["nuisance_signature"] for f in frames]
    return out


@check("pairs.matched_nuisance_cfb_cfd")
def c_pairs(S: Stages):
    msgs = []
    for arms in (("cf_b__p0000__specular_ghost", "cf_b__p0000__real_obstacle"),
                 ("cf_d__p0000__loc_drift", "cf_d__p0000__cart_moved")):
        a, b = _pair_signatures(S, arms, 11).values()
        assert a == b, f"{arms}: nuisance differs between arms"
        assert len(set(a)) == len(a), "frames not distinct"
        c = _pair_signatures(S, arms[:1], 12, tag="b")[arms[0]]
        assert c != a, "different seed gave identical nuisance"
        msgs.append(f"{arms[0].split('__')[0]}: equal across arms, distinct per frame/seed")
    st = S.build("cf_d__p0000__cart_moved", "_lock")
    assert "/World/Furniture/cart_1" in CausalLock(st).prims
    return "; ".join(msgs)


@check("randomizer.idempotent_per_frame_and_mdl_mirrored")
def c_idem(S: Stages):
    st = S.build("nominal__0000", "_idem")
    rnd = NuisanceRandomizer(st, seed=7)
    s = [rnd.apply(frame=f)["nuisance_signature"] for f in (0, 1, 0)]
    assert s[0] == s[2] and s[0] != s[1], "per-frame nuisance not reproducible (drift across frames)"
    n = 0
    for m in st.GetPrimAtPath("/World/Looks").GetChildren():
        prev = UsdShade.Shader(st.GetPrimAtPath(m.GetPath().AppendChild("PreviewSurface")))
        mdl = UsdShade.Shader(st.GetPrimAtPath(m.GetPath().AppendChild("MDL")))
        if mdl.GetInput("diffuse_color_constant"):
            assert np.allclose(prev.GetInput("diffuseColor").Get(), mdl.GetInput("diffuse_color_constant").Get())
            assert np.isclose(prev.GetInput("roughness").Get(), mdl.GetInput("reflection_roughness_constant").Get())
            n += 1
    return f"frame 0 reproduced after frame 1; {n} materials PreviewSurface==MDL"


# ---------------------------------------------------------------------------
@check("sensors.grid_scan")
def c_grid():
    from medortrace.isaac.sensors import grid_scan, lidar_elevations
    from medortrace.common.config import CONFIG_DIR
    el = lidar_elevations(CONFIG_DIR / "sensors" / "rtx_lidar_or16.json")
    pts = np.array([[2.0, 0.0, 0.0], [2.5, 0.001, 0.0], [0.0, 3.0, 3.0 * np.tan(np.deg2rad(15))], [0, 0, 0]])
    p, i, ring, dirs, ranges, _ = grid_scan(pts, np.array([10.0, 20.0, 5.0, 1.0]), el, 2.0, 20.0)
    assert dirs.shape == (16 * 180, 3) and ranges.shape == (16 * 180,)
    assert np.isfinite(ranges).sum() == 2, "nearest return per cell expected"
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert set(ring.tolist()) == {7, 15} or set(ring.tolist()) == {8, 15}
    assert np.isclose(sorted(np.linalg.norm(p, axis=1))[0], 2.0) and i.max() <= 1.0
    from medortrace.isaac.sensors import spherical_points
    sp = spherical_points({"distance": [2.0, 3.0], "azimuth": [0.0, 90.0], "elevation": [0.0, 15.0]})
    assert np.allclose(sp[1], [0.0, 3 * np.cos(np.deg2rad(15)), 3 * np.sin(np.deg2rad(15))], atol=1e-9)
    return f"{len(p)} returns on {len(ranges)} rays, rings {sorted(set(ring.tolist()))}; spherical fields in deg ok"


@check("sensors.pixel_rays")
def c_rays():
    from medortrace.isaac.sensors import pixel_rays
    b, e, n = pixel_rays(640, 480, 1280, 960, np.deg2rad(90), np.deg2rad(-25))
    assert abs(b) < 1e-12 and np.isclose(e, np.deg2rad(-25)) and np.isclose(n, 1.0)
    b2, e2, _ = pixel_rays(1280, 480, 1280, 960, np.deg2rad(90), 0.0)
    assert np.isclose(b2, -np.pi / 4) and abs(e2) < 1e-12        # right image edge = -hfov/2 bearing
    return "centre ray = pitch, right edge = -hfov/2"


@check("sensors.gt_surrogate_structured_array")
def c_surrogate():
    from medortrace.isaac.sensors import gt_surrogate_detections
    dt = np.dtype([("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4"),
                   ("occlusionRatio", "<f4")])
    data = np.array([(1, 600, 600, 680, 660, 0.0), (2, 100, 100, 140, 140, 0.2), (3, 10, 10, 20, 20, 0.0)], dtype=dt)
    boxes = {"data": data, "info": {"idToLabels": {"1": {"class": "sponge"}, "2": {"class": "clamp"},
                                                   "3": {"class": "person"}},
                                    "primPaths": ["/World/Items/sponge_1", "/World/Items/clamp_1", "/World/Staff/x"]}}
    depth = np.full((960, 1280), 1.2, np.float32)
    depth[:50, :50] = np.inf
    items = {"sponge_1": {"cls": "sponge", "size": (0.1, 0.1, 0.02), "glare": 0.0, "tag_readable": True,
                          "prim": "/World/Items/sponge_1"},
             "clamp_1": {"cls": "clamp", "size": (0.15, 0.05, 0.02), "glare": 0.3, "tag_readable": False,
                         "prim": "/World/Items/clamp_1"}}
    from medortrace.isaac.sensors import pixel_rays
    expect = {iid: pixel_rays(0.5 * (bx[1] + bx[3]), 0.5 * (bx[2] + bx[4]), 1280, 960, np.deg2rad(90),
                              np.deg2rad(-25)) for iid, bx in (("sponge_1", data[0]), ("clamp_1", data[1]))}
    hits = 0
    for s in range(20):
        dets = gt_surrogate_detections(boxes, depth, 1280, 960, np.deg2rad(90), np.deg2rad(-25),
                                       np.random.default_rng(s), items, {"fp_rate": 0.0})
        for d in dets:
            assert d.gt_item_id in ("sponge_1", "clamp_1") and d.logits.shape == (5,)
            b, e, n = expect[d.gt_item_id]
            assert abs(d.range - 1.2 * n) < 0.15 and abs(d.bearing - b) < 0.05 and abs(d.elevation - e) < 0.05
            assert d.item_id_hint in (None, "sponge_1"), "unreadable tag decoded"
        hits += len(dets)
    assert hits > 10, "surrogate almost never detects visible items"
    return f"{hits} detections over 20 draws, person/off-class boxes ignored"


@check("truth.parity_with_lite_backend")
def c_truth(S: Stages):
    from medortrace.isaac.truth import item_offsets, item_position, staff_positions
    from medortrace.sim.lite_backend import LiteBackend
    ep = S.eps.get("cf_b__p0000__real_obstacle") or build_episode(S.reg["cf_b__p0000__real_obstacle"].resolve(),
                                                                   S.reg["cf_b__p0000__real_obstacle"].seed)
    lb = LiteBackend(ep.cfg)
    lb.reset(ep)
    offs = item_offsets(ep)
    staff = staff_positions(lb.actual)
    for it in ep.spec.items:
        assert np.allclose(offs[it.id], lb._item_offsets[it.id])
        a = item_position(ep.spec, it.id, lb.item_slot[it.id], offs, staff)
        b = lb.item_position(it.id)
        assert (np.all(np.isnan(a)) and np.all(np.isnan(b))) or np.allclose(a, b), it.id
    return f"{len(ep.spec.items)} items placed identically"


@check("writer.causal_label_writer")
def c_writer(work: Path):
    out = work / "writer"
    w = CausalLabelWriter(out, label_fn=lambda fid: {"scenario_id": "x", "frame": fid})
    dt = np.dtype([("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4"),
                   ("occlusionRatio", "<f4")])
    ann = {"rgb": np.random.default_rng(0).integers(0, 255, (8, 12, 4), dtype=np.uint8),
           "distance_to_image_plane": np.where(np.eye(8, 12) > 0, np.inf, 1.5).astype(np.float32),
           "bounding_box_2d_tight": {"data": np.array([(1, 1, 2, 3, 4, 0.5)], dtype=dt),
                                     "info": {"idToLabels": {"1": {"class": "sponge"}}, "bboxIds": np.array([0])}},
           "semantic_segmentation": {"data": np.zeros((8, 12), np.uint32), "info": {"idToLabels": {}}}}
    res = w.write(ann, labels={"t": np.float64(1.5), "nan": float("nan")})
    png = (out / res["files"]["rgb"]).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    lab = json.loads((out / res["files"]["labels"]).read_text())
    assert lab["frame"] == "000000" and lab["t"] == 1.5 and lab["nan"] is None
    bb = json.loads((out / res["files"]["bbox2d"]).read_text())
    assert bb["boxes"][0]["occlusionRatio"] == 0.5
    return f"files {sorted(res['files'])}"


@check("ros2.graph_wiring")
def c_graph():
    from medortrace.isaac.ros2_bridge import graph_spec
    ns = {"bridge": "isaacsim.ros2.bridge", "core": "isaacsim.core.nodes", "wheeled": "isaacsim.robot.wheeled_robots"}
    for drive in (True, False):
        nodes, conns, vals = graph_spec(ns, lidar_render_product="/Render/rp0", camera_render_product="/Render/rp1",
                                        drive_from_cmd_vel=drive)
        names = [n for n, _ in nodes]
        assert len(names) == len(set(names)), "duplicate node names"
        for a, b in conns:
            assert a.split(".")[0] in names and b.split(".")[0] in names, (a, b)
        for k, _ in vals:
            assert k.split(".")[0] in names, k
        assert ("SubscribeTwist" in names) == drive
    return f"{len(names)} nodes (monitoring) / cmd_vel subscriber only when driving"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--keep", default=None, help="write test stages here (default: temp dir)")
    a = ap.parse_args(argv)
    work = Path(a.keep or tempfile.mkdtemp(prefix="medortrace_isaac_check_"))
    S = Stages(work, a.registry)
    for fn in (c_apply, c_rogue, c_oob, c_mat, c_clutter, c_edit, c_layer, c_pairs, c_idem):
        fn(S)
    c_grid()
    c_rays()
    c_surrogate()
    c_truth(S)
    c_writer(work)
    c_graph()
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed (work dir {work})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
