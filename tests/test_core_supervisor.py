"""Safety supervisor mode machine: escalation, hysteresis, categories, handover, gating."""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.safety.supervisor import Envelope, Mode, SafetyInputs, SafetySupervisor

CLEAR = {}
HUMAN_CLOSE = {"human_clearance": 0.3}          # below the 0.45 m hard limit
LOC_LOST = {"loc_std": 0.5}                     # above the 0.3 m hard limit


def _run(sup, schedule, t0=0.0, t1=10.0, dt=0.1):
    """Feed ``schedule(t) -> dict of SafetyInputs overrides``; returns [(t, mode)]."""
    out = []
    for t in np.arange(t0, t1 + 1e-9, dt):
        t = round(float(t), 6)
        out.append((t, sup.update(SafetyInputs(t=t, **schedule(t)))))
    return out


def _first(trace, mode):
    return next((t for t, m in trace if m == mode), None)


def test_immediate_escalation_and_hysteretic_deescalation():
    sup = SafetySupervisor()
    trace = _run(sup, lambda t: HUMAN_CLOSE if t < 1.0 else CLEAR, 0.0, 6.0)
    assert trace[0] == (0.0, Mode.STOP)                                  # no delay on escalation
    ev = sup.events[0]
    assert (ev.mode_from, ev.mode_to, ev.category) == ("NOMINAL", "STOP", "human_proximity")
    t_hold = sup.env.t_hold
    # stays stopped for t_hold after the condition cleared, then CAUTION (never straight to NOMINAL)
    assert _first(trace, Mode.CAUTION) == pytest.approx(1.0 + t_hold)
    assert all(m == Mode.STOP for t, m in trace if t < 1.0 + t_hold)
    # CAUTION needs its own clear period (timer starts on the tick after the transition)
    assert _first(trace, Mode.NOMINAL) == pytest.approx(1.0 + t_hold + 0.1 + t_hold, abs=0.1 + 1e-6)
    assert [e.mode_to for e in sup.events] == ["STOP", "CAUTION", "NOMINAL"]
    assert sup.events[1].category == "recovery"


def test_flicker_resets_the_hold_timer():
    sup = SafetySupervisor()
    bad = lambda t: HUMAN_CLOSE if (t < 1.0 or 2.0 <= t < 2.1) else CLEAR   # noqa: E731
    trace = _run(sup, bad, 0.0, 6.0)
    # clear at 1.0, relapse at 2.0 (< t_hold), clear again at 2.1 -> earliest release 2.1 + t_hold
    assert _first(trace, Mode.CAUTION) == pytest.approx(2.1 + sup.env.t_hold)


def test_uncertainty_category_and_soft_limits():
    sup = SafetySupervisor()
    assert sup.update(SafetyInputs(0.0, **LOC_LOST)) == Mode.STOP
    assert sup.events[-1].category == "uncertainty" and "localisation" in sup.events[-1].reasons[0]
    sup = SafetySupervisor()
    assert sup.update(SafetyInputs(0.0, lidar_age=2.0)) == Mode.STOP and sup.events[-1].category == "uncertainty"
    sup = SafetySupervisor()
    assert sup.update(SafetyInputs(0.0, path_entropy=0.4)) == Mode.CAUTION     # soft only (bits beyond the prior)
    assert sup.events[-1].category == "uncertainty"
    sup = SafetySupervisor()
    assert sup.update(SafetyInputs(0.0, human_clearance=0.6)) == Mode.CAUTION
    assert sup.events[-1].category != "uncertainty"
    # a physical hazard takes precedence in the label even when localisation is also lost
    sup = SafetySupervisor()
    sup.update(SafetyInputs(0.0, collision_prob=0.5, **LOC_LOST))
    assert sup.events[-1].category == "collision_risk" and len(sup.last_reasons) == 2
    # nominal inputs: nothing happens
    sup = SafetySupervisor()
    assert sup.update(SafetyInputs(0.0)) == Mode.NOMINAL and sup.events == [] and not sup.degraded()


@pytest.mark.parametrize("t0", [0.0, 5.0])     # a stop starting at t = 0 must escalate like any other
def test_persistent_uncertainty_stop_hands_over_to_operator(t0):
    sup = SafetySupervisor()
    trace = _run(sup, lambda t: LOC_LOST, t0, t0 + 14.0)
    env = sup.env
    assert _first(trace, Mode.RETREAT) is None                       # uncertainty never triggers a retreat
    t_ho = _first(trace, Mode.HANDOVER)
    assert t_ho == pytest.approx(t0 + env.t_handover_after + 0.1, abs=1e-6)
    assert sup.handover_requests == 1
    # HANDOVER is latched until the operator acknowledges, even if the cause clears
    later = _run(sup, lambda t: CLEAR, t0 + 14.1, t0 + 20.0)
    assert all(m == Mode.HANDOVER for _, m in later)
    assert sup.update(SafetyInputs(t0 + 20.1, operator_ack=True)) == Mode.STOP
    assert sup.events[-1].category == "operator"
    rec = _run(sup, lambda t: CLEAR, t0 + 20.2, t0 + 25.0)
    assert rec[-1][1] == Mode.NOMINAL


@pytest.mark.parametrize("t0", [0.0, 5.0])
def test_persistent_human_block_retreats_then_hands_over(t0):
    sup = SafetySupervisor()
    trace = _run(sup, lambda t: HUMAN_CLOSE, t0, t0 + 13.0)
    assert _first(trace, Mode.RETREAT) == pytest.approx(t0 + sup.env.t_retreat_after + 0.1, abs=1e-6)
    assert _first(trace, Mode.HANDOVER) == pytest.approx(t0 + sup.env.t_handover_after + 0.1, abs=1e-6)


def test_contact_forces_immediate_retreat():
    sup = SafetySupervisor()
    assert sup.update(SafetyInputs(0.0, contact_force=60.0)) == Mode.RETREAT
    assert sup.events[-1].category == "contact"


def test_gate_per_mode():
    sup = SafetySupervisor()
    assert sup.gate(0.6, 1.0) == (0.6, 1.0)
    sup.mode = Mode.CAUTION
    s = sup.env.caution_speed_scale
    v, w = sup.gate(0.6, 1.0)
    assert v == pytest.approx(0.7 * s) and w == pytest.approx(1.0 * max(s, 0.6))
    assert sup.gate(-0.5, 0.0)[0] == pytest.approx(-0.2)
    assert sup.gate(0.1, 0.0)[0] == pytest.approx(0.1)
    sup.mode = Mode.STOP
    assert sup.gate(0.6, 1.0, retreat_cmd=(-0.1, 0.2)) == (0.0, 0.0)
    sup.mode = Mode.RETREAT
    assert sup.gate(0.6, 1.0, retreat_cmd=(-0.1, 0.2)) == (-0.1, 0.2)
    assert sup.gate(0.6, 1.0) == (0.0, 0.0)
    sup.mode = Mode.HANDOVER
    assert sup.gate(0.6, 1.0, retreat_cmd=(-0.1, 0.2)) == (0.0, 0.0)


def test_disabled_supervisor_and_envelope_overrides():
    off = SafetySupervisor(enabled=False)
    assert off.update(SafetyInputs(0.0, contact_force=100.0, **LOC_LOST)) == Mode.NOMINAL and off.events == []
    env = Envelope.from_dict({"t_hold": 0.5, "loc_std_hard": 0.6, "unknown_key": 3})
    assert env.t_hold == 0.5 and env.loc_std_hard == 0.6 and not hasattr(env, "unknown_key")
    sup = SafetySupervisor(env)
    assert sup.update(SafetyInputs(0.0, **LOC_LOST)) == Mode.CAUTION     # 0.5 m is now only a soft violation


def test_retreat_persists_while_the_hazard_persists():
    """A person keeps crowding the robot: STOP -> RETREAT after t_retreat_after, then
    RETREAT holds (no RETREAT <-> STOP oscillation) until the hand-over timeout."""
    env = Envelope()
    sup = SafetySupervisor(env)
    modes = []
    t = 0.0
    while t < env.t_handover_after + 2.0:
        modes.append(sup.update(SafetyInputs(t, human_clearance=0.3)))
        t += 0.1
    first_retreat = modes.index(Mode.RETREAT)
    first_handover = modes.index(Mode.HANDOVER)
    assert all(m == Mode.RETREAT for m in modes[first_retreat:first_handover])
    assert abs(first_retreat * 0.1 - env.t_retreat_after) < 0.25
    # once clear for t_hold, it steps down (RETREAT -> CAUTION)
    sup = SafetySupervisor(env)
    for k in range(int(env.t_retreat_after / 0.1) + 5):
        sup.update(SafetyInputs(k * 0.1, human_clearance=0.3))
    assert sup.mode == Mode.RETREAT
    t0 = env.t_retreat_after + 0.5
    for k in range(int((env.t_hold + 0.5) / 0.1)):
        sup.update(SafetyInputs(t0 + k * 0.1, human_clearance=3.0))
    assert sup.mode == Mode.CAUTION
