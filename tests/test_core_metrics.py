"""Evaluation metrics on hand-computable inputs (calibration, safety, verification,
people-side effects) and the bootstrap aggregation helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from medortrace.eval.aggregate import bootstrap_ci, paired_delta, summarize
from medortrace.eval.metrics import HUMAN_R, ROBOT_R, TruthLog, brier, compute_metrics, ece
from medortrace.provenance.graph import Verdict
from medortrace.provenance.verifier import VerdictRecord
from medortrace.sim.faults import FaultModel
from medortrace.world.workflow import Claim, TruthMove, WorkflowScript


def test_ece_known_values():
    assert ece(np.full(10, 0.2), np.array([1, 1] + [0] * 8)) == pytest.approx(0.0)      # perfectly calibrated
    assert ece(np.full(4, 0.9), np.zeros(4)) == pytest.approx(0.9)                      # maximally overconfident
    # two equally populated bins: |0.1 - 0.5| and |0.9 - 1.0|
    assert ece(np.array([0.1, 0.1, 0.9, 0.9]), np.array([0, 1, 1, 1])) == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)
    assert ece(np.array([1.0, 0.0]), np.array([1, 0])) == pytest.approx(0.0)            # bin edges included
    assert ece(np.array([0.3, 0.7]), np.array([0, 1]), n_bins=1) == pytest.approx(0.0)  # one bin: |0.5 - 0.5|
    assert np.isnan(ece(np.zeros(0), np.zeros(0)))


def test_brier_known_values():
    assert brier(np.array([1.0, 0.0, 0.5]), np.array([1, 0, 1])) == pytest.approx(0.25 / 3)
    assert brier(np.array([0.0]), np.array([1])) == 1.0
    assert np.isnan(brier(np.zeros(0), np.zeros(0)))


def _agent(roaming, arrivals, distance):
    return SimpleNamespace(spec=SimpleNamespace(roaming=roaming), arrivals=arrivals, distance=distance)


def _synthetic_episode():
    """One minute at 10 Hz: a static robot, one roaming person passing close twice."""
    n = 600
    t = np.round(np.arange(1, n + 1) * 0.1, 6)
    center = np.full(n, 3.0)
    center[100:150] = 0.7                       # surface clearance 0.7 - 0.53 = 0.17 m  (< 0.3)
    center[300:320] = 0.6                       # 0.07 m: the minimum
    agents = np.stack([np.c_[center, np.zeros(n)], np.c_[np.full(n, 5.0), np.full(n, 5.0)]], 1)
    shadow = agents.copy()
    shadow[:, 0, 1] += 0.2                      # the robot-free twin walked 0.2 m to the side
    tl = TruthLog()
    tl.t = list(t)
    tl.robot = [np.zeros(3)] * n
    tl.agents = list(agents)
    tl.shadow = list(shadow)
    tl.collision_agent = [False] * n
    tl.collision_static = [False] * n
    for a, b in ((10, 13), (50, 51)):
        tl.collision_static[a:b] = [True] * (b - a)
    tl.in_keepout = [False] * n
    tl.in_keepout[200:205] = [True] * 5
    tl.in_keepout_margin = [False] * n
    tl.in_keepout_margin[195:210] = [True] * 15
    tl.battery = list(np.linspace(300.0, 240.0, n))
    tl.energy = list(np.linspace(0.0, 60.0, n))
    tl.item_slots = [{}] * n
    tl.fault_any = [False] * n
    claims = [Claim("c1", 10.0, 30.0, "a", "s1", kind="handoff"),
              Claim("c2", 10.0, 30.0, "b", "s1", kind="handoff"),
              Claim("c3", 30.0, 50.0, "a", "s1", kind="count")]
    wf = WorkflowScript(60.0, [TruthMove(5.0, "b", "s1", "s2")], [], {}, claims, {"a": "s1", "b": "s1"})
    ep = SimpleNamespace(workflow=wf, spec=SimpleNamespace(hidden_cause={"factor": "none", "value": "none"}),
                         faults=FaultModel(dropouts={"lidar": [(19.0, 21.0)]}))

    def vr(cid, t, verdict, p, direct):
        c = next(c for c in claims if c.id == cid)
        return VerdictRecord(cid, t, verdict, p, "", c.item_id, c.slot_id, c.t_ref, c.kind, "s1", direct)

    verdicts = [vr("c1", 20.0, Verdict.VERIFIED, 0.95, True),       # correct, on time, during a lidar dropout
                vr("c2", 25.0, Verdict.VERIFIED, 0.92, True),       # wrong: b left s1 at t = 5
                vr("c3", 40.0, Verdict.ABSTAIN, 0.5, False)]        # warranted: no direct evidence
    ev = lambda to, cat: SimpleNamespace(mode_to=to, category=cat)   # noqa: E731
    stack = SimpleNamespace(
        sup=SimpleNamespace(events=[ev("STOP", "uncertainty"), ev("CAUTION", "recovery"),
                                    ev("STOP", "collision_risk"), ev("HANDOVER", "uncertainty")]),
        ghost_confusion=lambda: np.array([8, 2, 4, 6]),
        telemetry=[SimpleNamespace(pose_est=np.array([0.3, 0.4, 0.0]))] * n)
    backend = SimpleNamespace(
        actual=SimpleNamespace(agents=[_agent(True, {0: 12.0, 1: 30.0}, 20.0), _agent(False, {}, 1.0)]),
        shadow=SimpleNamespace(agents=[_agent(True, {0: 10.0, 1: 31.0, 2: 50.0}, 18.0), _agent(False, {}, 1.0)]))
    return ep, tl, stack, verdicts, backend


def test_compute_metrics_on_known_episode():
    ep, tl, stack, verdicts, backend = _synthetic_episode()
    m = compute_metrics(ep, tl, stack, verdicts, backend, [{"duration": 30.0}], battery_cap=480.0)
    assert m["min_human_clearance_m"] == pytest.approx(0.6 - ROBOT_R - HUMAN_R)
    assert m["near_collision_events"] == 2 and m["near_collision_rate_per_min"] == pytest.approx(2.0)
    # delays: 12-10, max(0, 30-31), and an unfinished task (60 - 50)
    assert m["task_delay_s"] == pytest.approx(12.0) and m["task_delay_mean_s"] == pytest.approx(4.0)
    assert m["human_path_disruption_m"] == pytest.approx(2.0)          # only roaming staff count
    assert m["human_path_deviation_m"] == pytest.approx(0.2)
    assert m["uncertainty_safe_stop_rate_per_min"] == pytest.approx(1.0)
    assert m["safe_stop_rate_per_min"] == pytest.approx(2.0) and m["handover_requests"] == 1
    assert m["intervention_cost"] == pytest.approx(1.0 + 30.0 / 60.0)
    assert m["energy_reserve_frac"] == pytest.approx(0.5) and m["energy_used_wh"] == pytest.approx(60.0)
    assert m["claims_total"] == 3 and m["claims_answered"] == 3
    assert m["decision_accuracy"] == 0.5 and m["wrong_assertion_rate"] == 0.5
    assert m["abstention_rate"] == pytest.approx(1 / 3) and m["correct_abstention_frac"] == 1.0
    assert m["handoff_success"] == pytest.approx(0.5)
    assert m["brier"] == pytest.approx(np.mean([0.05 ** 2, 0.92 ** 2, 0.5 ** 2]))
    assert m["calibration_ece_under_fault"] == pytest.approx(0.05)     # c1 decided during the dropout
    assert m["calibration_ece_nominal"] == pytest.approx(0.5 * 0.92 + 0.5 * 0.5)
    assert m["collisions_static"] == 2 and m["collisions_agent"] == 0
    assert m["sterile_breach_s"] == pytest.approx(0.5) and m["keepout_margin_violation_s"] == pytest.approx(1.5)
    assert m["ghost_precision"] == pytest.approx(0.8) and m["ghost_recall"] == pytest.approx(8 / 12)
    assert m["loc_error_mean_m"] == pytest.approx(0.5) and m["loc_error_max_m"] == pytest.approx(0.5)
    assert m["distance_travelled_m"] == 0.0 and m["hc_factor"] == "none"


def test_bootstrap_and_paired_delta():
    mean, lo, hi = bootstrap_ci(np.array([1.0, 2.0, 3.0, np.nan, 4.0]))
    assert mean == pytest.approx(2.5) and 1.0 <= lo < mean < hi <= 4.0
    assert np.isnan(bootstrap_ci(np.array([np.nan]))[0])
    rows = []
    for k in range(20):
        for pol, delay in (("active", 1.0 + k), ("fixed_route", 3.0 + k)):
            rows.append({"scenario_id": f"s{k}", "policy": pol, "family": "f", "metrics": {"task_delay_s": delay}})
    d = paired_delta(rows, "active", "fixed_route", metrics=["task_delay_s"])
    assert d["n_pairs"] == 20
    r = d["task_delay_s"]
    assert r["delta"] == pytest.approx(-2.0) and r["ci95"] == pytest.approx([-2.0, -2.0])
    assert r["a_better"] and r["p_boot"] == 0.0                         # lower delay is better
    s = summarize(rows, metrics=["task_delay_s"])
    assert [(x["policy"], x["n"]) for x in s] == [("active", 20), ("fixed_route", 20)]
    assert s[0]["task_delay_s"]["mean"] == pytest.approx(10.5)
