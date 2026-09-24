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
LIDAR_MOUNT_Z = 0.9          # sensors_lite.LidarConfig.mount_height
LOS_RANGE_M = 8.0            # beyond this a 0.7 m cart gets too few returns to explain
CFD_MIN_LOS_S = 10.0         # seconds of line of sight for a displaced cart to count as observable


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
    robot_at_stack: list = field(default_factory=list)   # true pose at the time of each stack step
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
    entries = _debounced_entries(dmin_t, dt)
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
    # localisation error: estimate at each stack step vs the true pose *at that time*
    # (truth.robot is logged after the simulator step, one tick later)
    R_est = truth.robot_at_stack if len(truth.robot_at_stack) == len(stack.telemetry) else truth.robot
    loc_err = np.array([np.linalg.norm(s.pose_est[:2] - np.asarray(r)[:2]) for s, r in zip(stack.telemetry, R_est)])
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
        # undefined (NaN) when nothing was decided: 0 would reward a policy that always abstains
        "wrong_assertion_rate": float(1 - np.mean(correct)) if correct else float("nan"),
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
        "loc_error_mean_m": float(np.mean(loc_err)) if len(loc_err) else float("nan"),
        "loc_error_max_m": float(np.max(loc_err)) if len(loc_err) else float("nan"),
        "distance_travelled_m": float(np.sum(np.linalg.norm(np.diff(R[:, :2], axis=0), axis=1))) if len(R) > 1 else 0.0,
    }
    m.update(hidden_cause_outcome(ep, vt, stack, truth))
    return m


def _cart_line_of_sight_seconds(ep, truth: TruthLog, obj_name: str, face_z: tuple = (0.5, 0.85)) -> float:
    """Seconds (1 Hz truth samples) with lidar line of sight to ``obj_name``.

    Rays go from the lidar origin to five points on the object (centre and the
    four face midpoints) at each height in ``face_z``; a sample counts if any
    ray reaches the object first (static boxes of the true scene + people as
    cylinders) within ``LOS_RANGE_M``.
    """
    from medortrace.common.geometry import rot2
    from medortrace.sim.raycast import RayScene, segments_blocked

    objs = ep.spec.objects
    k = next((i for i, o in enumerate(objs) if o.name == obj_name), None)
    if k is None or not truth.t:
        return 0.0
    box = objs[k].box
    Rw = rot2(box.yaw)
    offs = np.array([[0, 0], [box.half[0], 0], [-box.half[0], 0], [0, box.half[1]], [0, -box.half[1]]]) * 0.95
    xy = box.center[:2] + offs @ Rw.T
    tgt = np.vstack([np.c_[xy, np.full(len(xy), z)] for z in face_z])
    bc = np.array([o.box.center for o in objs])
    bh = np.array([o.box.half for o in objs])
    by = np.array([o.box.yaw for o in objs])
    T = np.array(truth.t)
    step = max(1, int(round(1.0 / max(float(np.median(np.diff(T))) if len(T) > 1 else 1.0, 1e-3))))
    seen = 0
    for i in range(0, len(T), step):
        r = np.asarray(truth.robot[i], float)
        o = np.array([r[0] + 0.1 * np.cos(r[2]), r[1] + 0.1 * np.sin(r[2]), LIDAR_MOUNT_Z])
        if np.min(np.linalg.norm(tgt[:, :2] - o[:2], axis=1)) > LOS_RANGE_M:
            continue
        ppl = np.asarray(truth.agents[i], float).reshape(-1, 2)
        sc = RayScene(bc, bh, by, ppl, np.full(len(ppl), HUMAN_R), np.full(len(ppl), 1.75),
                      ceiling=float(ep.spec.room[2]))
        a = np.repeat(o[None], len(tgt), 0)
        blocked = segments_blocked(sc, a, tgt, own=np.full(len(tgt), k), own_tol=0.6)
        seen += int(not blocked.all())
    return float(seen) * step * (float(np.median(np.diff(T))) if len(T) > 1 else 1.0)


def _debounced_entries(clearance: np.ndarray, dt: float, enter: float = NEAR_COLLISION_M,
                       exit_: float = NEAR_COLLISION_M + 0.05, min_gap_s: float = 1.0) -> int:
    """Number of distinct near-collision episodes: an entry (clearance < ``enter``)
    counts only after the robot was clear (> ``exit_``, hysteresis) for at least
    ``min_gap_s`` - a person lingering at the boundary is one event, not many."""
    n, inside, clear_for = 0, False, np.inf
    for c in clearance:
        if inside:
            if c > exit_:
                inside, clear_for = False, dt
        else:
            if c < enter:
                if clear_for >= min_gap_s:
                    n += 1
                inside = True
            elif c > exit_:
                clear_for += dt
    return n


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
        # was the point ever sensed (a lidar return or free-space carving in the
        # robot's height band, or ghost-suspect mass there)?  An unobserved point
        # keeps the prior "free", which is not a decision.
        k1 = int(np.ceil(1.5 / occ.res))
        out["cfb_observed"] = float(bool(occ.observed[sl][:, :, 1:k1].any()) or amb > 0.05)
        # same decision rule as the pair analysis (counterfactual_analysis._decide_b):
        # occupied (p > 0.5), ambiguous (ghost-suspect mass >= 0.5) or free.  Treating the
        # point as possibly occupied is the safe answer for a real obstacle; only "free"
        # is the right answer for a specular ghost.
        decision = "occupied" if p_occ > 0.5 else ("ambiguous" if amb >= 0.5 else "free")
        if val == "real_obstacle":
            out["cfb_correct"] = float(decision in ("occupied", "ambiguous"))
        else:
            out["cfb_correct"] = float(decision == "free")
        # a collision *at the hidden-cause location* (not anywhere in the episode)
        cs = np.array(truth.collision_static, dtype=bool)
        near_ap = np.linalg.norm(R[:, :2] - ap, axis=1) < 1.0 if len(R) else np.zeros(0, bool)
        out["cfb_collision"] = float(bool(np.any(cs & near_ap))) if len(cs) == len(near_ap) else 0.0
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
        out["cfd_object"] = s.get("object") or "none"
        want = "loc_drift" if val == "loc_drift" else "map_change"
        out["cfd_correct"] = float(s["cause"] == want)
        if val != "loc_drift":
            # a map change attributed to the wrong object is not a correct diagnosis
            out["cfd_object_correct"] = float(s["cause"] == want and s.get("object") == "cart_1")
        # observability of the hidden cause: drift corrupts every scan once it
        # starts; a displaced cart is only observable if the lidar had line of
        # sight to it (truth poses and people, 1 Hz samples, within LOS_RANGE_M)
        R = np.array(truth.robot)
        cart = ep.spec.object("cart_1").box
        out["cfd_min_dist_cart_m"] = float(np.min(cart.distance_xy(R[:, :2]))) if len(R) else float("inf")
        los_s = _cart_line_of_sight_seconds(ep, truth, "cart_1")
        out["cfd_cart_los_s"] = los_s
        out["cfd_observable"] = 1.0 if val == "loc_drift" else float(los_s >= CFD_MIN_LOS_S)
    return out
