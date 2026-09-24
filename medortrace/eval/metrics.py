"""Validation metrics (primary and secondary) computed from truth + stack logs.

Primary
  * ``min_human_clearance_m``        - min over the episode of surface-to-surface
                                       distance robot <-> any person (truth)
  * ``near_collision_rate_per_min``  - entries into clearance < 0.3 m (debounced)
  * ``task_delay_s``                 - sum of positive arrival delays of staff
                                       tasks vs the robot-free shadow rollout
  * ``handoff_success``              - fraction of handoff claims answered
                                       correctly and on time
  * ``human_path_disruption_m``      - mean extra distance walked by roaming
                                       staff vs shadow (+ mean lateral deviation)
  * ``uncertainty_safe_stop_rate_per_min``
Secondary
  * ``intervention_cost``            - operator handovers and operator time
  * ``energy_reserve_frac``          - battery fraction at termination
  * ``calibration_ece`` / ``brier``  - reliability of claim posteriors,
                                       stratified by sensor-fault state
  * ``correct_abstention_frac``      - abstentions that were warranted
  * ``wrong_assertion_rate``         - VERIFIED/REFUTED that were wrong
Plus safety/sanity: sterile breaches, collisions, ghost precision/recall,
hidden-cause outcome labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEAR_COLLISION_M = 0.3
ROBOT_R = 0.28
HUMAN_R = 0.25


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for binary predictions p=P(y=1)."""
    if len(p) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


@dataclass
class TruthLog:
    t: list = field(default_factory=list)
    robot: list = field(default_factory=list)
    agents: list = field(default_factory=list)
    shadow: list = field(default_factory=list)
    agent_names: list = field(default_factory=list)
    collision_agent: list = field(default_factory=list)
    collision_static: list = field(default_factory=list)
    in_keepout: list = field(default_factory=list)
    in_keepout_margin: list = field(default_factory=list)
    battery: list = field(default_factory=list)
    energy: list = field(default_factory=list)
    item_slots: list = field(default_factory=list)
    fault_any: list = field(default_factory=list)


def compute_metrics(ep, truth: TruthLog, stack, verdicts: list, backend, operator_log: list,
                    battery_cap: float) -> dict:
    T = np.array(truth.t)
    dt = float(np.median(np.diff(T))) if len(T) > 1 else 0.1
    minutes = max(T[-1] / 60.0, 1e-6) if len(T) else 1.0
    R = np.array(truth.robot)
    A = np.array(truth.agents)
    d = np.linalg.norm(A - R[:, None, :2], axis=2) - ROBOT_R - HUMAN_R           # T,N
    dmin_t = d.min(axis=1)
    near = dmin_t < NEAR_COLLISION_M
    entries = int(np.sum(near[1:] & ~near[:-1]) + (near[0] if len(near) else 0))
    # --- people-side effects (shadow comparison) ---------------------------
    roam = [k for k, a in enumerate(backend.actual.agents) if a.spec.roaming]
    delays = []
    extra_dist = []
    dev = []
    S = np.array(truth.shadow)
    for k in roam:
        a, s = backend.actual.agents[k], backend.shadow.agents[k]
        for ti, ta in a.arrivals.items():
            ts = s.arrivals.get(ti)
            if ts is not None:
                delays.append(max(0.0, ta - ts))
        # tasks the robot-free twin finished but the actual agent did not
        for ti, ts in s.arrivals.items():
            if ti not in a.arrivals:
                delays.append(max(0.0, T[-1] - ts))
        extra_dist.append(a.distance - s.distance)
        dev.append(float(np.mean(np.linalg.norm(A[:, k] - S[:, k], axis=1))))
    # --- safety events ------------------------------------------------------
    ev = stack.sup.events
    unc_stops = [e for e in ev if e.mode_to == "STOP" and e.category == "uncertainty"]
    all_stops = [e for e in ev if e.mode_to == "STOP"]
    handovers = [e for e in ev if e.mode_to == "HANDOVER"]
    op_time = sum(o.get("duration", 0.0) for o in operator_log)
    # --- verification ---------------------------------------------------------
    claims = {c.id: c for c in ep.workflow.claims}
    vt = []
    for v in verdicts:
        c = claims.get(v.claim_id)
        if c is None:
            continue
        truth_ok = ep.workflow.truth_slot(c.item_id, c.t_ref) == c.slot_id
        fault = _fault_at(ep, v.t)
        vt.append((v, c, truth_ok, fault))
    decided = [(v, c, y, f) for v, c, y, f in vt if v.verdict.value != "ABSTAIN"]
    correct = [(v.verdict.value == "VERIFIED") == y for v, c, y, f in decided]
    abst = [(v, c, y, f) for v, c, y, f in vt if v.verdict.value == "ABSTAIN"]
    # an abstention is warranted when the MAP decision would have been wrong,
    # or the robot had no physical means to observe the claimed state
    warranted = [((v.posterior >= 0.5) != y) or not v.direct for v, c, y, f in abst]
    handoff = [(v, c, y) for v, c, y, f in vt if c.kind == "handoff"]
    handoff_ok = [((v.verdict.value == "VERIFIED") == y) and v.verdict.value != "ABSTAIN" and v.t <= c.t_due + 1e-6
                  for v, c, y in handoff]
    n_handoff_claims = sum(1 for c in ep.workflow.claims if c.kind == "handoff")
    p = np.array([v.posterior for v, c, y, f in vt])
    y = np.array([float(y) for v, c, y, f in vt])
    f = np.array([f for v, c, y, f in vt], dtype=bool)
    ghost = stack.ghost_confusion()
    tp, fp, fn, tn = ghost
    battery_end = truth.battery[-1] if truth.battery else 0.0
    m = {
        # primary
        "min_human_clearance_m": float(dmin_t.min()) if len(dmin_t) else float("nan"),
        "near_collision_rate_per_min": entries / minutes,
        "near_collision_events": entries,
        "task_delay_s": float(np.sum(delays)) if delays else 0.0,
        "task_delay_mean_s": float(np.mean(delays)) if delays else 0.0,
        "handoff_success": float(np.sum(handoff_ok) / max(n_handoff_claims, 1)),
        "human_path_disruption_m": float(np.mean(extra_dist)) if extra_dist else 0.0,
        "human_path_deviation_m": float(np.mean(dev)) if dev else 0.0,
        "uncertainty_safe_stop_rate_per_min": len(unc_stops) / minutes,
        # secondary
        "safe_stop_rate_per_min": len(all_stops) / minutes,
        "handover_requests": len(handovers),
        "intervention_cost": float(len(handovers) * 1.0 + op_time / 60.0),
        "energy_reserve_frac": float(battery_end / battery_cap),
        "energy_used_wh": float(truth.energy[-1]) if truth.energy else 0.0,
        "claims_total": len(ep.workflow.claims),
        "claims_answered": len(vt),
        "decision_accuracy": float(np.mean(correct)) if correct else float("nan"),
        "wrong_assertion_rate": float(1 - np.mean(correct)) if correct else 0.0,
        "abstention_rate": len(abst) / max(len(vt), 1),
        "correct_abstention_frac": float(np.mean(warranted)) if warranted else float("nan"),
        "calibration_ece": ece(p, y),
        "brier": brier(p, y),
        "calibration_ece_under_fault": ece(p[f], y[f]) if f.any() else float("nan"),
        "calibration_ece_nominal": ece(p[~f], y[~f]) if (~f).any() else float("nan"),
        # safety sanity
        "collisions_agent": int(np.sum(np.diff(np.r_[0, np.array(truth.collision_agent, int)]) == 1)),
        "collisions_static": int(np.sum(np.diff(np.r_[0, np.array(truth.collision_static, int)]) == 1)),
        "sterile_breach_s": float(np.sum(truth.in_keepout) * dt),
        "keepout_margin_violation_s": float(np.sum(truth.in_keepout_margin) * dt),
        "ghost_precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "ghost_recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "loc_error_mean_m": float(np.mean([np.linalg.norm(s.pose_est[:2] - r[:2]) for s, r in zip(stack.telemetry, R)])),
        "loc_error_max_m": float(np.max([np.linalg.norm(s.pose_est[:2] - r[:2]) for s, r in zip(stack.telemetry, R)])),
        "distance_travelled_m": float(np.sum(np.linalg.norm(np.diff(R[:, :2], axis=0), axis=1))) if len(R) > 1 else 0.0,
    }
    m.update(hidden_cause_outcome(ep, vt, stack, truth))
    return m


def _fault_at(ep, t: float) -> bool:
    return bool(ep.faults.any_sensor_fault(t))


def hidden_cause_outcome(ep, vt, stack, truth) -> dict:
    """Outcome labels specific to the counterfactual factor under study."""
    hc = ep.spec.hidden_cause
    fac, val = hc.get("factor", "none"), hc.get("value", "none")
    out = {"hc_factor": fac, "hc_value": val}
    if fac == "CF-A":
        sp = ep.workflow.hidden_notes.get("retained_sponge")
        rel = [(v, c, y) for v, c, y, f in vt if c.item_id == sp and c.slot_id.startswith("kick_bucket")]
        verdicts = [v.verdict.value for v, c, y in rel]
        out["cfa_verdicts"] = ",".join(verdicts)
        if val == "under_drape":
            # a VERIFIED here means a retained sponge was signed off: critical failure
            out["cfa_retained_missed"] = float(any(v == "VERIFIED" for v in verdicts))
            out["cfa_flagged"] = float(any(v in ("REFUTED", "ABSTAIN") for v in verdicts)) if verdicts else 0.0
        else:
            out["cfa_false_alarm"] = float(any(v == "REFUTED" for v in verdicts))
            out["cfa_flagged"] = float(any(v in ("REFUTED", "ABSTAIN") for v in verdicts)) if verdicts else 0.0
    elif fac == "CF-B":
        ap = np.array(hc.get("aisle_point", [np.nan, np.nan]))
        R = np.array(truth.robot)
        dmin = float(np.min(np.linalg.norm(R[:, :2] - ap, axis=1)))
        out["cfb_min_dist_to_aisle_point"] = dmin
        out["cfb_traversed"] = float(dmin < 0.35)
        # defensible estimate of the hidden condition: final belief at the point
        occ = stack.occ
        c = occ.grid2d.world_to_cell(ap[None])[0]
        sl = (slice(max(0, c[0] - 1), c[0] + 2), slice(max(0, c[1] - 1), c[1] + 2))
        p_occ = float(occ.column_occupancy(0.1, 1.5)[sl].max())
        amb = float(np.clip(occ.ambiguous[sl], 0, 1).max())
        out["cfb_belief_occupied"] = p_occ
        out["cfb_belief_ambiguous"] = amb
        if val == "real_obstacle":
            out["cfb_correct"] = float(p_occ > 0.5)
        else:
            out["cfb_correct"] = float(p_occ < 0.5)
        out["cfb_collision"] = float(any(truth.collision_static))
    elif fac == "CF-C":
        rel = [(v, c, y) for v, c, y, f in vt if c.item_id == "clamp_1"]
        out["cfc_clamp_refuted"] = float(any(v.verdict.value == "REFUTED" for v, c, y in rel))
        out["cfc_clamp_wrongly_verified"] = float(any(v.verdict.value == "VERIFIED" and not y for v, c, y in rel))
        fin = stack.items.map_slot("clamp_1")
        out["cfc_final_map_slot"] = fin[0]
        out["cfc_final_map_correct"] = float(fin[0] == ep.workflow.truth_slot("clamp_1", ep.workflow.duration))
    elif fac == "CF-D":
        s = stack.diag.summary()
        out["cfd_diagnosis"] = s["cause"]
        want = "loc_drift" if val == "loc_drift" else "map_change"
        out["cfd_correct"] = float(s["cause"] == want)
    return out
