"""Plain-pytest tests (no ROS) of medortrace_ros.assembler.SensorBundleAssembler.

Unit tests pin down the delivery semantics per channel (consume-once, sample-and-hold, stream),
staleness, timing and bookkeeping.  The loopback test replays a short lite-simulator episode
message by message through the assembler into ``AutonomyRuntime`` (mission JSON round trip
included) and checks that the stack commands are identical to the in-process episode loop - i.e.
that the ROS middleware path loses nothing but the transport.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from medortrace_ros.assembler import (
    CHANNELS,
    CONSUME_ONCE,
    AssemblerConfig,
    SensorBundleAssembler,
    explode_bundle,
)

from medortrace.common.msgs import (
    CameraFrame,
    ContactState,
    Header,
    ImuSample,
    LidarScan,
    RadarDetection,
    RadarFrame,
    SensorBundle,
    WheelOdometry,
    WorkflowEvent,
    WorkflowEventType,
)


def hdr(stamp: float, recv: float | None = None, frame: str = "x") -> Header:
    return Header(stamp, stamp if recv is None else recv, frame)


def radar(stamp: float, recv: float | None = None, n: int = 1) -> RadarFrame:
    return RadarFrame(hdr(stamp, recv, "radar_link"), [RadarDetection(2.0 + k, 0.1, 0.0, 0.0, -10.0) for k in range(n)])


def lidar(stamp: float, recv: float | None = None) -> LidarScan:
    d = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = np.array([2.0, np.inf])
    return LidarScan(hdr(stamp, recv, "lidar_link"), d[:1] * 2.0, np.ones(1), np.zeros(1, int), d, r)


def odom(stamp: float, v: float = 0.3, recv: float | None = None) -> WheelOdometry:
    return WheelOdometry(hdr(stamp, recv, "base_link"), v, 0.1)


def contact(stamp: float, force: float, recv: float | None = None) -> ContactState:
    return ContactState(hdr(stamp, recv, "bumper"), force > 1.0, force)


def imu(stamp: float, recv: float | None = None) -> ImuSample:
    return ImuSample(hdr(stamp, recv, "imu_link"), np.zeros(3), np.array([0.0, 0.0, stamp]))


def wf(t: float, eid: str = "", item: str = "sponge_1") -> WorkflowEvent:
    return WorkflowEvent(t, WorkflowEventType.HANDOFF, item, "back_table:tray", "field:top", event_id=eid)


# ---------------------------------------------------------------------------------------------------------------------
# consume-once channels
# ---------------------------------------------------------------------------------------------------------------------
def test_consume_once_newest_wins_and_is_delivered_once():
    a = SensorBundleAssembler()
    a.push("radar", radar(0.00, n=1), recv_t=0.02)
    a.push("radar", radar(0.05, n=2), recv_t=0.07)
    b, _ = a.assemble(0.1)
    assert b.radar is not None and len(b.radar.detections) == 2          # newest
    assert a.stats["radar"].superseded == 1 and a.stats["radar"].delivered == 1
    b2, _ = a.assemble(0.2)
    assert b2.radar is None                                              # never re-delivered (no double counting)


def test_consume_once_stale_message_is_dropped():
    a = SensorBundleAssembler(AssemblerConfig.from_dict({"max_age_s": {"lidar": 0.3}}))
    a.push("lidar", lidar(0.0), recv_t=0.0)
    b, _ = a.assemble(0.5)
    assert b.lidar is None and a.stats["lidar"].stale == 1


def test_message_received_after_tick_waits_for_next_tick():
    a = SensorBundleAssembler()
    a.push("camera", CameraFrame(hdr(0.15), [], 1.5, 5.0), recv_t=0.15)
    b, _ = a.assemble(0.1)
    assert b.camera is None
    b, _ = a.assemble(0.2)
    assert b.camera is not None and b.camera.header.stamp == pytest.approx(0.15)


def test_skewed_sensor_stamp_is_preserved_for_the_time_sync_monitor():
    """The assembler judges freshness on receive time and never rewrites the sensor stamp."""
    a = SensorBundleAssembler()
    a.push("radar", radar(stamp=0.35), recv_t=0.10)                     # clock 0.25 s ahead
    b, _ = a.assemble(0.1)
    assert b.radar.header.stamp == pytest.approx(0.35)
    assert b.radar.header.recv_stamp == pytest.approx(0.10)


def test_push_does_not_mutate_callers_message_and_assigns_unique_seq():
    a = SensorBundleAssembler()
    m1, m2 = radar(0.0), radar(0.0)
    m1.header.seq = m2.header.seq = 7
    a.push("radar", m1, recv_t=0.01)
    b1, _ = a.assemble(0.1)
    a.push("radar", m2, recv_t=0.11)
    b2, _ = a.assemble(0.2)
    assert m1.header.seq == 7 and m1.header.recv_stamp == 0.0           # caller's object untouched
    assert b1.radar.header.seq != b2.radar.header.seq                    # unique provenance evidence ids
    assert b1.radar.header.recv_stamp == pytest.approx(0.01)


# ---------------------------------------------------------------------------------------------------------------------
# sample-and-hold channels
# ---------------------------------------------------------------------------------------------------------------------
def test_odometry_is_held_within_max_age_then_reported_missing():
    a = SensorBundleAssembler(AssemblerConfig.from_dict({"max_age_s": {"odom": 0.25}}))
    a.push("odom", odom(0.0, v=0.4), recv_t=0.0)
    b0, _ = a.assemble(0.0)
    b1, _ = a.assemble(0.1)
    b2, _ = a.assemble(0.2)
    b3, _ = a.assemble(0.3)
    assert b0.odom.v == b1.odom.v == b2.odom.v == 0.4                  # zero-order hold
    assert b3.odom is None                                               # dropout becomes visible to the EKF
    assert a.stats["odom"].held == 2 and a.stats["odom"].stale == 1


def test_contact_latches_strongest_hit_between_ticks():
    a = SensorBundleAssembler()
    a.push("contact", contact(0.02, 0.0), recv_t=0.02)
    a.push("contact", contact(0.05, 40.0), recv_t=0.05)                # short bumper hit
    a.push("contact", contact(0.08, 0.0), recv_t=0.08)
    b, _ = a.assemble(0.1)
    assert b.contact.in_contact and b.contact.force_n == 40.0
    b, _ = a.assemble(0.2)
    assert not b.contact.in_contact                                      # hold uses the latest sample


def test_battery_hold():
    a = SensorBundleAssembler()
    a.push("battery", 321.0, recv_t=0.0)
    assert a.assemble(0.0)[0].battery_wh == 321.0
    assert a.assemble(30.0)[0].battery_wh == 321.0
    assert a.assemble(61.0)[0].battery_wh is None
    with pytest.raises(ValueError):
        a.push("battery", 1.0)


# ---------------------------------------------------------------------------------------------------------------------
# streams
# ---------------------------------------------------------------------------------------------------------------------
def test_imu_samples_accumulate_in_stamp_order_and_stale_ones_drop():
    a = SensorBundleAssembler(AssemblerConfig.from_dict({"max_age_s": {"imu": 0.2}}))
    for k in (3, 1, 2):
        a.push("imu", imu(0.1 + 0.01 * k), recv_t=0.1 + 0.01 * k)
    a.push("imu", imu(0.0), recv_t=0.0)                                  # too old at t=0.25
    b, _ = a.assemble(0.25)
    assert [s.header.stamp for s in b.imu] == pytest.approx([0.11, 0.12, 0.13])
    assert a.stats["imu"].stale == 1
    assert a.assemble(0.35)[0].imu == []


def test_imu_window_is_capped():
    a = SensorBundleAssembler(AssemblerConfig.from_dict({"max_imu_samples": 5}))
    for k in range(12):
        a.push("imu", imu(0.01 * k), recv_t=0.01 * k)
    b, _ = a.assemble(0.12)
    assert len(b.imu) == 5 and b.imu[-1].header.stamp == pytest.approx(0.11)


def test_workflow_events_exactly_once_deduplicated_and_time_ordered():
    a = SensorBundleAssembler()
    a.push("workflow", wf(5.0, "wf_002"), recv_t=6.0)
    a.push("workflow", wf(3.0, "wf_001"), recv_t=6.0)
    a.push("workflow", wf(3.0, "wf_001"), recv_t=6.05)                 # transient-local replay duplicate
    a.push("workflow", wf(4.0, ""), recv_t=6.05)                       # no id: one is assigned
    b, _ = a.assemble(6.1)
    assert [e.t for e in b.workflow] == [3.0, 4.0, 5.0]
    assert b.workflow[1].event_id.startswith("ros_")
    assert a.stats["workflow"].duplicates == 1
    assert a.assemble(6.2)[0].workflow == []
    a.push("workflow", wf(3.0, "wf_001"), recv_t=6.3)                  # late duplicate still ignored
    assert a.assemble(6.4)[0].workflow == []


# ---------------------------------------------------------------------------------------------------------------------
# timing & bookkeeping
# ---------------------------------------------------------------------------------------------------------------------
def test_dt_nominal_first_then_measured_and_clamped():
    a = SensorBundleAssembler(AssemblerConfig.from_dict({"control_rate_hz": 10.0, "dt_clamp": [0.5, 3.0]}))
    assert a.assemble(1.0)[1] == pytest.approx(0.1)
    assert a.assemble(1.12)[1] == pytest.approx(0.12)
    assert a.assemble(1.13)[1] == pytest.approx(0.05)                   # clamped up (0.5 x period)
    assert a.assemble(3.0)[1] == pytest.approx(0.3)                     # clamped down (3 x period)


def test_clock_jump_backwards_resets_buffers():
    a = SensorBundleAssembler()
    a.assemble(50.0)
    a.push("radar", radar(50.05), recv_t=50.05)
    a.push("workflow", wf(40.0, "wf_x"), recv_t=50.05)
    b, dt = a.assemble(1.0)                                              # simulation restarted
    assert a.n_resets == 1 and b.radar is None and b.workflow == [] and dt == pytest.approx(0.1)
    a.push("workflow", wf(0.5, "wf_x"), recv_t=1.05)                   # same id is new after a reset
    assert len(a.assemble(1.1)[0].workflow) == 1


def test_ready_health_and_config_validation():
    a = SensorBundleAssembler()
    assert not a.ready(("odom",))
    a.push("odom", odom(0.0), recv_t=0.0)
    assert a.ready(("odom",))
    a.assemble(0.1)
    h = a.health(0.1)
    assert set(h) == set(CHANNELS) and h["odom"]["age_s"] == pytest.approx(0.1) and h["lidar"]["age_s"] is None
    with pytest.raises(ValueError):
        AssemblerConfig.from_dict({"max_age_s": {"sonar": 1.0}})
    with pytest.raises(KeyError):
        a.push("sonar", None, 0.0)


def test_explode_then_assemble_round_trip():
    b = SensorBundle(t=2.0, lidar=lidar(1.95), radar=radar(1.98), imu=[imu(1.91), imu(1.96)], odom=odom(2.0),
                     contact=contact(2.0, 0.0), workflow=[wf(1.0, "wf_7")], battery_wh=100.0)
    a = SensorBundleAssembler()
    for ch, m in explode_bundle(b):
        a.push(ch, m, recv_t=b.t)
    out, _ = a.assemble(b.t)
    for ch in CONSUME_ONCE + ("odom", "contact"):
        src, dst = getattr(b, ch), getattr(out, ch)
        assert (src is None) == (dst is None)
        if src is not None:
            assert dst.header.stamp == src.header.stamp and dst.header.frame_id == src.header.frame_id
    assert len(out.imu) == 2 and out.workflow[0].event_id == "wf_7" and out.battery_wh == 100.0


# ---------------------------------------------------------------------------------------------------------------------
# loopback: lite simulator -> per-topic messages -> assembler -> AutonomyRuntime (tiny episode)
# ---------------------------------------------------------------------------------------------------------------------
def test_loopback_matches_in_process_episode_loop():
    from medortrace_ros.autonomy_node import AutonomyRuntime
    from medortrace_ros.mission import (
        episode_config,
        mission_from_episode,
        mission_from_json,
        mission_to_json,
        stack_setup,
    )
    from medortrace_ros.records import verify_provenance_stream

    from medortrace.autonomy.stack import AutonomyStack, stack_inputs_from_episode
    from medortrace.common.msgs import VelocityCommand
    from medortrace.sim.episode import build_episode
    from medortrace.sim.lite_backend import LiteBackend

    cfg = episode_config("scenarios/nominal.yaml", seed=1, duration=3.0)
    ep = build_episode(cfg, 1)
    be = LiteBackend(cfg)
    bundle = be.reset(ep)
    # reference: the in-process loop of medortrace.eval.runner.run_episode
    ref = AutonomyStack(stack_inputs_from_episode(ep), cfg)
    # middleware path: mission JSON round trip + assembler
    inputs, rcfg = stack_setup(mission_from_json(mission_to_json(mission_from_episode(ep, cfg))))
    rt = AutonomyRuntime(inputs, rcfg)
    ref_cmds, rt_cmds, prov, n_workflow = [], [], [], 0
    cmd = VelocityCommand()
    while be.t < ep.workflow.duration - 1e-9:
        for ch, m in explode_bundle(bundle):
            rt.push(ch, m, bundle.t)                      # recv = sim time, as under /clock
        n_workflow += len(bundle.workflow)
        cmd = ref.step(copy.copy(bundle), be.dt)
        res = rt.tick(bundle.t)
        ref_cmds.append((cmd.v, cmd.omega, cmd.acoustic_probe_target))
        rt_cmds.append((res.cmd.v, res.cmd.omega, res.cmd.acoustic_probe_target))
        prov += res.provenance
        assert res.safety.mode in ("NOMINAL", "CAUTION", "STOP", "RETREAT", "HANDOVER")
        bundle = be.step(cmd)
    assert len(rt_cmds) == 30
    np.testing.assert_allclose([c[:2] for c in rt_cmds], [c[:2] for c in ref_cmds], atol=1e-9)
    assert [c[2] for c in rt_cmds] == [c[2] for c in ref_cmds]
    assert rt.asm.stats["workflow"].delivered == n_workflow
    assert rt.asm.stats["lidar"].stale == rt.asm.stats["lidar"].superseded == 0
    # the streamed provenance events re-verify as a hash chain, like the graph itself
    assert prov and prov[0].index == 0 and verify_provenance_stream(prov) and rt.stack.prov.verify_chain()
    tampered = copy.deepcopy(prov)
    tampered[3].json_attrs = tampered[3].json_attrs.replace("}", ', "x": 1}', 1)
    assert not verify_provenance_stream(tampered)
