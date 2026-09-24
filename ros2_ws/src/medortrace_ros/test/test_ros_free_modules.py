"""Plain-pytest tests (no ROS) of the ROS-free parts of medortrace_ros and of the interface files.

* interface definitions: syntax / naming rules of rosidl, CMake registration, and field coverage of
  the medortrace dataclasses they mirror;
* topic registry & QoS: names match medortrace.isaac.ros2_bridge.TOPICS, QoS policy per class;
* convert: lossless lidar layout, scan-pattern refill, attribute-level conversions on fake messages;
* mission JSON, workflow-gateway parsing, operator policy, and the runtime's verification services.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
MSGS = PKG.parent / "medortrace_msgs"


# =====================================================================================================================
# interface definitions
# =====================================================================================================================
PRIMITIVES = {"bool", "byte", "char", "float32", "float64", "int8", "uint8", "int16", "uint16", "int32", "uint32",
              "int64", "uint64", "string", "wstring"}
EXTERNAL = {"std_msgs", "geometry_msgs", "nav_msgs", "builtin_interfaces"}
FIELD_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
CONST_RE = re.compile(r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$")


def _parse_interface(path: Path) -> dict:
    fields, consts = {}, {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip() if "=" not in raw.split("#", 1)[0] else raw.strip()
        if not line or line == "---":
            continue
        typ, rest = line.split(None, 1)
        if "=" in rest:
            name, value = (s.strip() for s in rest.split("=", 1))
            assert "#" not in value, f"{path.name}: comment after constant {name}"
            assert CONST_RE.match(name), f"{path.name}: bad constant name {name}"
            consts[name] = value.strip('"')
            continue
        name = rest.split()[0]
        assert FIELD_RE.match(name), f"{path.name}: bad field name {name}"
        base = re.sub(r"(\[\d*\]|\[<=\d+\])$", "", typ)
        if "/" in base:
            assert base.split("/")[0] in EXTERNAL, f"{path.name}: unexpected dependency {base}"
        else:
            assert base in PRIMITIVES or (MSGS / "msg" / f"{base}.msg").is_file(), f"{path.name}: unknown type {base}"
        fields[name] = typ
    return {"fields": fields, "consts": consts}


def _iface(name: str) -> dict:
    return _parse_interface(MSGS / "msg" / f"{name}.msg")


def test_interface_files_are_valid_and_registered():
    cmake = (MSGS / "CMakeLists.txt").read_text()
    files = sorted((MSGS / "msg").glob("*.msg")) + sorted((MSGS / "srv").glob("*.srv"))
    required = {"ItemBelief", "ItemBeliefArray", "ClaimVerdict", "ProvenanceEvent", "SafetyState", "WorkflowEvent",
                "CameraDetection", "CameraDetectionArray", "RadarDetection", "RadarDetectionArray", "AcousticEcho",
                "AcousticFrame", "LandmarkObservation", "LandmarkObservationArray", "NextBestView", "SceneGraph",
                "UncertaintyGrid", "OperatorAck", "VerifyClaim", "ExplainVerdict"}
    assert required <= {f.stem for f in files}
    for f in files:
        assert f'"{f.parent.name}/{f.name}"' in cmake, f"{f.name} not in CMakeLists.txt"
        if f.suffix == ".srv":
            req, resp = f.read_text().split("\n---\n")
            for part in (req, resp):
                tmp = f.with_suffix(".part")
                try:
                    tmp.write_text(part)
                    _parse_interface(tmp)
                finally:
                    tmp.unlink()
        else:
            _parse_interface(f)
    pkg = (MSGS / "package.xml").read_text()
    assert "<member_of_group>rosidl_interface_packages</member_of_group>" in pkg
    for dep in EXTERNAL:
        assert f"<depend>{dep}</depend>" in pkg and dep in cmake


def test_messages_cover_the_medortrace_dataclasses():
    import dataclasses

    from medortrace.common import msgs as M

    mapping = {   # dataclass -> (msg, renames)
        M.CameraDetection: ("CameraDetection", {}),
        M.CameraFrame: ("CameraDetectionArray", {}),
        M.RadarDetection: ("RadarDetection", {}),
        M.RadarFrame: ("RadarDetectionArray", {}),
        M.AcousticEcho: ("AcousticEcho", {}),
        M.AcousticFrame: ("AcousticFrame", {}),
        M.LandmarkObservation: ("LandmarkObservation", {}),
        M.LandmarkFrame: ("LandmarkObservationArray", {}),
        M.ContactState: ("ContactState", {}),
        M.WorkflowEvent: ("WorkflowEvent", {"t": "header", "payload": "payload_json"}),
    }
    for dc, (msg, ren) in mapping.items():
        fields = _iface(msg)["fields"]
        for f in dataclasses.fields(dc):
            assert ren.get(f.name, f.name) in fields, f"{msg} lacks {f.name}"


def test_message_constants_match_python_enums():
    from medortrace_ros.records import MODE_CODE, VERDICT_CODE

    from medortrace.common.msgs import WorkflowEventType
    from medortrace.provenance.graph import Verdict
    from medortrace.safety.supervisor import Mode

    safety = {k: int(v) for k, v in _iface("SafetyState")["consts"].items()}
    assert safety == {m.value: MODE_CODE[m.value] for m in Mode}
    verdict = {k: int(v) for k, v in _iface("ClaimVerdict")["consts"].items()}
    assert verdict == {v.value: VERDICT_CODE[v.value] for v in Verdict}
    wf = _iface("WorkflowEvent")["consts"]
    assert {wf[t.name] for t in WorkflowEventType} == {t.value for t in WorkflowEventType}


# =====================================================================================================================
# topics & QoS
# =====================================================================================================================
def test_topic_registry_matches_isaac_bridge_and_namespace():
    from medortrace_ros.topics import SERVICES, SPECS, TOPICS

    from medortrace.isaac.ros2_bridge import TOPICS as ISAAC

    assert all(TOPICS[k] == v for k, v in ISAAC.items())
    standard = {"/clock", "/tf", "/diagnostics", "/initialpose"}
    for name in list(TOPICS.values()) + [s[0] for s in SERVICES.values()]:
        assert name in standard or name.startswith("/medortrace/"), name
    assert len(set(TOPICS.values())) == len(TOPICS)
    assert all(s.type.count("/") == 2 for s in SPECS.values())


def test_qos_policy_per_topic_class():
    from medortrace_ros.qos import load_qos_config, resolve
    from medortrace_ros.topics import SPECS

    cfg = load_qos_config()
    for key in SPECS:
        resolve(key, cfg)
    for key in ("lidar_points", "camera_detections", "radar", "imu", "odom", "landmarks", "acoustic"):
        assert resolve(key, cfg)["reliability"] == "best_effort" and resolve(key, cfg)["durability"] == "volatile"
    for key in ("verdicts", "provenance", "workflow", "mission"):
        q = resolve(key, cfg)
        assert q["reliability"] == "reliable" and q["durability"] == "transient_local"
    assert resolve("cmd_vel", cfg)["durability"] == "volatile"


def test_params_yaml_names_existing_nodes():
    import yaml

    p = yaml.safe_load((PKG / "config" / "params.yaml").read_text())
    assert set(p) == {"medortrace_autonomy", "medortrace_sim_bridge", "medortrace_workflow_gateway",
                      "medortrace_operator_console"}
    for mod, name in (("autonomy_node", "medortrace_autonomy"), ("sim_bridge_node", "medortrace_sim_bridge"),
                      ("workflow_gateway_node", "medortrace_workflow_gateway"),
                      ("operator_console_node", "medortrace_operator_console")):
        assert f'"{name}"' in (PKG / "medortrace_ros" / f"{mod}.py").read_text()


# =====================================================================================================================
# convert (pure parts)
# =====================================================================================================================
@pytest.fixture(scope="module")
def lite_scan():
    from medortrace.common.msgs import Header
    from medortrace.sim.raycast import RayScene
    from medortrace.sim.sensors_lite import LidarConfig, simulate_lidar
    from medortrace.world.materials import MATERIALS

    scene = RayScene(np.array([[3.0, 0.0, 0.5]]), np.array([[0.3, 2.0, 0.5]]), np.zeros(1),
                     np.zeros((0, 2)), np.zeros(0), np.zeros(0), 3.0)
    cfg = LidarConfig()
    rng = np.random.default_rng(0)
    mats = [MATERIALS["painted_wall"], MATERIALS["floor_vinyl"], MATERIALS["painted_wall"]]
    ranges, d_s, ring, inten, ghost, obj = simulate_lidar(scene, mats, [[]], np.array([1.0, 0.0, 0.9]), 0.0, cfg, rng)
    from medortrace.common.msgs import LidarScan

    fin = np.isfinite(ranges)
    return LidarScan(Header(1.0, 1.0, "lidar_link"), d_s[fin] * ranges[fin, None], inten[fin], ring[fin], d_s,
                     ranges, 0.9, ghost, obj), cfg


def test_lidar_columns_round_trip_is_lossless(lite_scan):
    from medortrace_ros.convert import lidar_from_columns, lidar_to_columns

    scan, _ = lite_scan
    assert np.isinf(scan.ranges).any() and np.isfinite(scan.ranges).any()
    cols = lidar_to_columns(scan)
    assert set(cols) >= {"x", "y", "z", "range", "dir_x", "dir_y", "dir_z", "gt_ghost", "gt_object"}
    back = lidar_from_columns(cols, scan.header, 0.9)
    np.testing.assert_array_equal(np.isinf(back.ranges), np.isinf(scan.ranges))
    np.testing.assert_allclose(back.directions, scan.directions, atol=1e-6)
    np.testing.assert_allclose(back.points, scan.points, atol=1e-4)
    np.testing.assert_array_equal(back.ring, scan.ring)
    np.testing.assert_array_equal(back.gt_is_ghost, scan.gt_is_ghost)


def test_driver_cloud_without_no_returns_is_refilled_from_scan_pattern(lite_scan):
    from medortrace_ros.convert import lidar_from_columns, lidar_to_columns

    scan, cfg = lite_scan
    full = lidar_to_columns(scan, include_gt=False)
    fin = np.isfinite(scan.ranges)
    xyz_only = {k: full[k][fin] for k in ("x", "y", "z", "intensity")}          # what a real driver sends
    pattern = {"rings": cfg.rings, "elev_min_deg": cfg.elev_min_deg, "elev_max_deg": cfg.elev_max_deg,
               "az_res_deg": cfg.az_res_deg}
    back = lidar_from_columns(xyz_only, scan.header, 0.9, pattern)
    assert len(back.ranges) == len(scan.ranges)
    assert np.isinf(back.ranges).sum() == np.isinf(scan.ranges).sum()
    miss_ref = scan.directions[~fin]
    miss = back.directions[np.isinf(back.ranges)]
    d = np.abs(miss_ref[:, None, :] - miss[None, :, :]).sum(-1).min(axis=1)
    assert d.max() < 1e-6
    no_fill = lidar_from_columns(xyz_only, scan.header, 0.9)
    assert np.isfinite(no_fill.ranges).all() and len(no_fill.ranges) == fin.sum()


def _stamp(t):
    return NS(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))


def _hdr(t, frame="x"):
    return NS(stamp=_stamp(t), frame_id=frame)


def test_attribute_level_conversions_on_fake_messages():
    from medortrace_ros import convert as cv

    from medortrace.belief.items import CLASSES
    from medortrace.common.msgs import WorkflowEventType

    t0 = 100.0
    det = NS(cls="clamp", item_id_hint="", bearing=0.1, elevation=-0.2, range=1.5, logits=[0.0, 5.0, 1.0],
             visible_fraction=1.0, glare=0.0, gt_item_id="")
    cam = cv.camera_from_ros(NS(header=_hdr(t0 + 2.0, "camera_link"), class_names=["sponge", "clamp", "specimen"],
                                detections=[det], surfaces=[], fov_h=1.5, max_range=5.0, exposure_ok=True), 2.05, t0)
    assert cam.header.stamp == pytest.approx(2.0) and cam.header.recv_stamp == 2.05
    lg = cam.detections[0].logits
    assert len(lg) == len(CLASSES) and lg[CLASSES.index("clamp")] == 5.0 and lg[CLASSES.index("specimen")] == 1.0
    assert lg[CLASSES.index("implant_box")] < -10 and cam.detections[0].item_id_hint is None
    ev = cv.workflow_from_ros(NS(header=_hdr(t0 + 12.5), type="discard", item_id="sponge_2", src="field:top",
                                 dst="kick_bucket_1:inside", reporter="surgeon", confidence=0.8, event_id="e1",
                                 payload_json='{"phase": "x"}', source="voice"), t0)
    assert ev.t == pytest.approx(12.5) and ev.type == WorkflowEventType.DISCARD and ev.payload["source"] == "voice"
    assert cv.workflow_from_ros(NS(header=_hdr(t0), type="teleport", item_id="", src="", dst="", reporter="",
                                   confidence=1.0, event_id="", payload_json="", source=""), t0) is None
    q = (0.0, 0.0, np.sin(0.4), np.cos(0.4))
    od = NS(header=_hdr(t0 + 1.0, "odom"), child_frame_id="base_link",
            twist=NS(twist=NS(linear=NS(x=0.3), angular=NS(z=-0.2))),
            pose=NS(pose=NS(position=NS(x=1.0, y=2.0), orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3]))))
    wo = cv.odom_from_ros(od, 1.0, t0)
    assert (wo.v, wo.omega, wo.header.frame_id) == (0.3, -0.2, "base_link")
    np.testing.assert_allclose(cv.pose2d_from_pose(od.pose.pose), [1.0, 2.0, 0.8])
    batt = NS(charge=float("nan"), voltage=25.6, percentage=0.5)
    assert cv.battery_from_ros(batt, 480.0) == pytest.approx(240.0)
    assert cv.battery_from_ros(NS(charge=10.0, voltage=25.6, percentage=0.1)) == pytest.approx(256.0)
    ac = cv.acoustic_from_ros(NS(header=_hdr(t0 + 1), echoes=[NS(target_region="patient_drape", energy=0.3,
                                                                  delay_s=0.01, path_occluded=True,
                                                                  gt_hard_reflector=-1)]), 1.0, t0)
    assert ac.echoes[0].gt_hard_reflector is None and ac.echoes[0].path_occluded
    rc = cv.radar_from_columns({"x": np.array([3.0, np.nan]), "y": np.array([4.0, 0.0]), "z": np.array([0.0, 0.0]),
                                "velocity": np.array([-0.5, 0.0]), "intensity": np.array([100.0, 1.0])},
                               cv.header_from_ros(_hdr(t0 + 1), 1.0, t0))
    assert len(rc.detections) == 1 and rc.detections[0].range == pytest.approx(5.0)
    assert rc.detections[0].azimuth == pytest.approx(np.arctan2(4, 3)) and rc.detections[0].rcs_dbsm == 20.0


def test_map_to_odom_composition():
    from medortrace_ros.frames import compose_map_to_odom

    def compose(a, b):
        c, s = np.cos(a[2]), np.sin(a[2])
        return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])

    map_base, odom_base = np.array([4.0, 3.0, 2.5]), np.array([-1.2, 0.7, -0.4])
    m_o = compose_map_to_odom(map_base, odom_base)
    got = compose(m_o, odom_base)
    np.testing.assert_allclose(got[:2], map_base[:2], atol=1e-12)
    assert np.isclose(np.cos(got[2] - map_base[2]), 1.0)


# =====================================================================================================================
# mission
# =====================================================================================================================
@pytest.fixture(scope="module")
def mission():
    from medortrace_ros.mission import mission_for_scenario

    return mission_for_scenario("scenarios/nominal.yaml", seed=2, duration=4.0)


def test_mission_json_is_strict_and_round_trips(mission):
    import dataclasses

    from medortrace_ros.mission import (
        ROBOT_CFG_KEYS,
        episode_config,
        mission_from_json,
        mission_to_json,
        stack_setup,
    )

    from medortrace.autonomy.stack import stack_inputs_from_episode
    from medortrace.sim.episode import build_episode

    s = mission_to_json(mission)
    json.loads(s, parse_constant=lambda c: pytest.fail(f"non-standard JSON constant {c}"))
    assert set(mission["cfg"]) <= set(ROBOT_CFG_KEYS) and "faults" not in mission["cfg"]
    inputs, cfg = stack_setup(mission_from_json(s), overrides={"autonomy": {"policy": "passive"}})
    assert cfg["autonomy"]["policy"] == "passive"
    ref = stack_inputs_from_episode(build_episode(episode_config("scenarios/nominal.yaml", 2, duration=4.0), 2))

    def same(a, b):
        if dataclasses.is_dataclass(a):
            return all(same(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True)
        return a == b

    for f in dataclasses.fields(ref):
        assert same(getattr(inputs, f.name), getattr(ref, f.name)), f.name


def test_mission_validation_and_file_io(mission, tmp_path):
    from medortrace_ros.mission import load_mission_file, save_mission_file, validate_mission

    p = save_mission_file(mission, tmp_path / "survey.json")
    assert load_mission_file(p)["inputs"]["duration"] == mission["inputs"]["duration"]
    with pytest.raises(ValueError):
        validate_mission({**mission, "schema": "other/1"})
    with pytest.raises(ValueError):
        validate_mission({**mission, "inputs": {k: v for k, v in mission["inputs"].items() if k != "slots"}})


# =====================================================================================================================
# runtime services (verify / explain / operator ack / audit export) on a tiny lite episode
# =====================================================================================================================
def test_runtime_services_and_audit_export(mission, tmp_path):
    from medortrace_ros.assembler import explode_bundle
    from medortrace_ros.autonomy_node import AutonomyRuntime
    from medortrace_ros.mission import episode_config, stack_setup

    from medortrace.sim.episode import build_episode
    from medortrace.sim.lite_backend import LiteBackend

    cfg = episode_config("scenarios/nominal.yaml", 2, duration=4.0)
    be = LiteBackend(cfg)
    bundle = be.reset(build_episode(cfg, 2))
    rt = AutonomyRuntime(*stack_setup(mission))
    item, slot = "clamp_1", mission["inputs"]["initial_placement"]["clamp_1"]
    assert not rt.verify_claim("nope", slot, None, None, None)["accepted"]
    assert not rt.verify_claim(item, "nowhere", None, None, None)["accepted"]
    verdicts = []
    for k in range(20):
        for ch, m in explode_bundle(bundle):
            rt.push(ch, m, bundle.t)
        res = rt.tick(bundle.t)
        verdicts += res.verdicts
        if k == 2:
            r = rt.verify_claim(item, slot, None, 0.6, "q_test")
            assert r["accepted"] and not r["decided"] and 0.0 <= r["posterior"] <= 1.0
            assert rt.verify_claim(item, slot, None, 0.6, "q_test")["message"] == "already pending"
            assert rt.explain("q_test")["found"] is False
        bundle = be.step(res.cmd)
    q = [v for v in verdicts if v.claim_id == "q_test"]
    assert len(q) == 1 and q[0].verdict in ("VERIFIED", "REFUTED", "ABSTAIN") and q[0].node_hash
    ex = rt.explain("q_test", 3)
    assert ex["found"] and json.loads(ex["explanation_json"])["verdict"]["claim"]["claim_id"] == "q_test"
    assert json.loads(ex["custody_chain_json"])["item"] == item
    assert rt.verify_claim(item, slot, None, None, "q_test")["decided"]
    ok, msg, mode = rt.operator_ack(np.array([1.0, 1.0, 0.0]), bundle.t)
    assert ok and "no handover pending" in msg and mode in ("NOMINAL", "CAUTION", "STOP", "RETREAT", "HANDOVER")
    np.testing.assert_allclose(rt.stack.ekf.x, [1.0, 1.0, 0.0])
    fin, prov = rt.finalize(bundle.t)
    assert not rt.stack.verifier.open and rt.finalize(bundle.t) == ([], [])
    op = [p for p in prov if p.node_id.startswith("operator:")]
    assert len(op) == 1 and op[0].sensor == "operator" and op[0].attributed_to == "operator"
    d = rt.export_audit(tmp_path / "audit")
    for name in ("provenance.json", "scene_graph.json", "verdicts.jsonl", "safety_events.jsonl",
                 "middleware_health.json"):
        assert (d / name).is_file()
    assert json.loads((d / "middleware_health.json").read_text())["provenance_chain_valid"] is True


# =====================================================================================================================
# workflow gateway & operator console (ROS-free logic)
# =====================================================================================================================
@pytest.fixture(scope="module")
def vocab():
    from medortrace_ros.workflow_gateway_node import load_vocabulary

    return load_vocabulary(PKG / "config" / "voice_vocabulary.yaml")


@pytest.mark.parametrize("text,speaker,expect", [
    ("Passing sponge two to the surgeon", "scrub_nurse", ("handoff", "sponge_2", "hand:scrub_nurse", "hand:surgeon")),
    ("clamp back on the mayo", None, ("place", "clamp_1", None, "mayo:top")),
    ("sponge 3 into kick bucket 2", "surgeon", ("discard", "sponge_3", None, "kick_bucket_2:inside")),
    ("implant box opened onto the back table", "circulator", ("open", "implant_box_1", None, "back_table:tray")),
    ("specimen is out", "surgeon", ("specimen_out", "specimen_1", "elsewhere", "hand:surgeon")),
    ("needle driver from the mayo to the surgeon", "scrub_nurse",
     ("handoff", "needle_driver_1", "mayo:top", "hand:surgeon")),
])
def test_voice_transcript_grammar(vocab, text, speaker, expect):
    from medortrace_ros.workflow_gateway_node import parse_transcript

    r = parse_transcript(text, vocab, speaker)
    assert (r["type"].value, r["item_id"], r["src"], r["dst"]) == expect


def test_count_and_unparsable_transcripts(vocab):
    from medortrace_ros.workflow_gateway_node import parse_transcript

    assert parse_transcript("starting the final count", vocab)["payload"] == {"phase": "final_count"}
    assert parse_transcript("closing count please", vocab)["payload"] == {"phase": "first_closing_count"}
    assert parse_transcript("could you dim the lights", vocab) is None


def test_gateway_record_formats_and_truth_is_never_replayed(vocab):
    from medortrace_ros.workflow_gateway_node import parse_jsonl

    lines = [
        json.dumps({"kind": "workflow_log", "t": 10.0, "type": "handoff", "item": "sponge_1", "src": "hand:scrub_nurse",
                    "dst": "field:top", "id": "wf_001"}),
        json.dumps({"kind": "truth_move", "t": 11.0, "item": "sponge_1", "src": "field:top",
                    "dst": "field:under_drape", "cause": "hidden_cause"}),
        json.dumps({"kind": "verdict", "claim_id": "c1"}),
        json.dumps({"t": 20.0, "reported_t": 22.5, "type": "place", "item_id": "specimen_1",
                    "dst": "back_table:specimen_cup", "confidence": 0.95, "event_id": "or_17"}),
        json.dumps({"kind": "voice", "t": 30.0, "text": "passing sponge four to the surgeon",
                    "speaker": "scrub_nurse", "asr_confidence": 0.8}),
        json.dumps({"kind": "voice", "t": 31.0, "text": "suction please"}),
        json.dumps({"stamp": 1040.0, "type": "count", "payload": {"phase": "final_count"}}),
        "not json",
        "",
    ]
    recs = parse_jsonl(lines, vocab, t0=1000.0)
    evs = [r.event for r in recs if r.event is not None]
    assert [e.event_id for e in evs[:2]] == ["wf_001", "or_17"]
    assert all(e.dst != "field:under_drape" for e in evs)
    assert recs[1].publish_t == 22.5 and evs[1].t == 20.0
    voice = [r for r in recs if r.source == "voice"]
    assert len(voice) == 2 and voice[0].event.item_id == "sponge_4" and voice[1].event is None
    assert voice[0].event.confidence == pytest.approx(0.8 * 0.85) and voice[0].event.reporter == "scrub_nurse"
    assert evs[-1].t == pytest.approx(40.0) and evs[-1].type.value == "count"


def test_operator_policy_modes():
    from medortrace_ros.operator_console_node import OperatorPolicy, parse_operator_input

    op = OperatorPolicy("auto", (8.0, 20.0), seed=0)
    assert op.on_state(100.0, True) and not op.on_state(100.1, True)
    assert op.poll(100.5, np.zeros(3)) is None
    dec = op.poll(125.0, np.array([1.0, 2.0, 0.5]))
    assert dec.relocalize and np.allclose(dec.pose, [1.0, 2.0, 0.5], atol=0.1)
    assert 8.0 <= op.log[0]["duration"] <= 25.0 and op.poll(126.0, np.zeros(3)) is None
    rv = OperatorPolicy("rviz", ack_without_pose_after_s=30.0)
    rv.on_state(0.0, True)
    assert rv.poll(5.0) is None and rv.poll(6.0, rviz_pose=np.array([3.0, 3.0, 0.0])).relocalize
    rv.on_state(50.0, True)
    assert rv.poll(90.0).relocalize is False
    rv.on_state(100.0, True)
    rv.on_state(101.0, False)                                    # resolved elsewhere
    assert rv.pending_since is None
    assert parse_operator_input("y").relocalize is False
    assert np.allclose(parse_operator_input("r 1 2 0.5").pose, [1, 2, 0.5])
    assert parse_operator_input("maybe") is None and parse_operator_input("r 1 x 2") is None
    with pytest.raises(ValueError):
        OperatorPolicy("telepathy")
