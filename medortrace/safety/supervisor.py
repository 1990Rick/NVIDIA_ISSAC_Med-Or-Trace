"""Safety supervisor: a monitored envelope with a hysteretic mode machine.

    NOMINAL --(soft limit)--> CAUTION --(hard limit)--> STOP --(persisting / contact)--> RETREAT
       ^                          |                       |                                   |
       +------(clear for t_hold)--+-----------------------+-----(persisting > t_handover)-> HANDOVER

Monitored signals (each has a soft and a hard threshold in the envelope):
  * predicted collision probability & minimum predicted human clearance (MPC)
  * measured human clearance (tracker)
  * localisation health: position std-dev and windowed NIS
  * perception freshness: age of last lidar integration (dropout)
  * path uncertainty: mean entropy of the costmap along the next metre
  * time-sync health (uncorrectable skew)
  * bumper contact force
  * battery reserve

Every transition is logged as a :class:`SafetyEvent`, and stops caused by
uncertainty (localisation / entropy / sensing) are tagged ``uncertainty``
so the *uncertainty-triggered safe-stop rate* can be measured.  The
supervisor is the final arbiter of velocity: it clamps or overrides the MPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Mode(str, Enum):
    NOMINAL = "NOMINAL"
    CAUTION = "CAUTION"
    STOP = "STOP"
    RETREAT = "RETREAT"
    HANDOVER = "HANDOVER"


SEVERITY = {Mode.NOMINAL: 0, Mode.CAUTION: 1, Mode.STOP: 2, Mode.RETREAT: 3, Mode.HANDOVER: 4}


@dataclass
class Envelope:
    collision_prob_soft: float = 0.05
    collision_prob_hard: float = 0.25
    pred_clearance_soft: float = 0.6
    pred_clearance_hard: float = 0.2
    human_clearance_soft: float = 0.8
    human_clearance_hard: float = 0.45
    loc_std_soft: float = 0.12
    loc_std_hard: float = 0.3
    nis_soft: float = 8.0
    nis_hard: float = 25.0
    lidar_age_soft: float = 0.5
    lidar_age_hard: float = 1.2
    # path uncertainty in bits *beyond the surveyed prior* (OccupancyBelief.
    # excess_uncertainty_field): conflicting evidence ~0.3, ghost-suspect mass up to 1
    path_entropy_soft: float = 0.25
    path_entropy_hard: float = 0.6
    contact_force_hard: float = 25.0
    battery_soft_frac: float = 0.2
    t_hold: float = 1.5
    t_retreat_after: float = 6.0
    t_handover_after: float = 12.0
    caution_speed_scale: float = 0.45

    @staticmethod
    def from_dict(d: dict | None) -> "Envelope":
        e = Envelope()
        for k, v in (d or {}).items():
            if hasattr(e, k):
                setattr(e, k, float(v))
        return e


@dataclass
class SafetyEvent:
    t: float
    mode_from: str
    mode_to: str
    reasons: list[str]
    category: str                  # collision_risk | uncertainty | contact | human_proximity | energy | operator
    values: dict = field(default_factory=dict)


@dataclass
class SafetyInputs:
    t: float
    collision_prob: float = 0.0
    pred_clearance: float = 10.0
    human_clearance: float = 10.0
    loc_std: float = 0.0
    nis: float = 2.0
    lidar_age: float = 0.0
    path_entropy: float = 0.0
    skew_uncorrectable: bool = False
    contact_force: float = 0.0
    battery_frac: float = 1.0
    operator_ack: bool = False


class SafetySupervisor:
    def __init__(self, env: Envelope | None = None, enabled: bool = True):
        self.env = env or Envelope()
        self.enabled = enabled
        self.mode = Mode.NOMINAL
        self.events: list[SafetyEvent] = []
        self._since_bad = None
        self._since_clear = None
        self._stop_started = None
        self.last_reasons: list[str] = []
        self.handover_requests = 0

    def _classify(self, x: SafetyInputs) -> tuple[Mode, list[str], str]:
        e = self.env
        hard, soft = [], []
        cat = "collision_risk"
        if x.contact_force > e.contact_force_hard:
            return Mode.RETREAT, [f"contact {x.contact_force:.0f}N"], "contact"
        if x.human_clearance < e.human_clearance_hard:
            hard.append(f"human clearance {x.human_clearance:.2f}m")
            cat = "human_proximity"
        elif x.human_clearance < e.human_clearance_soft:
            soft.append("human near")
        if x.collision_prob > e.collision_prob_hard or x.pred_clearance < e.pred_clearance_hard:
            hard.append(f"predicted collision p={x.collision_prob:.2f} c={x.pred_clearance:.2f}")
        elif x.collision_prob > e.collision_prob_soft or x.pred_clearance < e.pred_clearance_soft:
            soft.append("predicted proximity")
        unc_hard = []
        if x.loc_std > e.loc_std_hard or x.nis > e.nis_hard:
            unc_hard.append(f"localisation std={x.loc_std:.2f} nis={x.nis:.1f}")
        elif x.loc_std > e.loc_std_soft or x.nis > e.nis_soft:
            soft.append("localisation degraded")
        if x.lidar_age > e.lidar_age_hard:
            unc_hard.append(f"lidar stale {x.lidar_age:.1f}s")
        elif x.lidar_age > e.lidar_age_soft:
            soft.append("lidar late")
        if x.path_entropy > e.path_entropy_hard:
            unc_hard.append(f"path entropy {x.path_entropy:.2f} bits")
        elif x.path_entropy > e.path_entropy_soft:
            soft.append("path uncertain")
        if x.skew_uncorrectable:
            soft.append("clock skew")
        if x.battery_frac < e.battery_soft_frac:
            soft.append("battery low")
        if unc_hard and not hard:
            return Mode.STOP, unc_hard, "uncertainty"
        if hard:
            return Mode.STOP, hard + unc_hard, cat
        if soft:
            return (
                Mode.CAUTION,
                soft,
                "uncertainty"
                if any(s in ("localisation degraded", "lidar late", "path uncertain", "clock skew") for s in soft)
                else cat,
            )
        return Mode.NOMINAL, [], ""

    def update(self, x: SafetyInputs) -> Mode:
        if not self.enabled:
            self.mode = Mode.NOMINAL
            return self.mode
        want, reasons, cat = self._classify(x)
        e = self.env
        prev = self.mode
        new = prev
        if self.mode == Mode.HANDOVER:
            if x.operator_ack:
                new = Mode.STOP
                self._stop_started = x.t
                reasons, cat = ["operator acknowledged"], "operator"
        elif SEVERITY[want] > SEVERITY[self.mode]:
            new = want
            self._since_clear = None
            if want in (Mode.STOP, Mode.RETREAT):
                self._stop_started = x.t
        else:
            # de-escalate only after the condition has been clear for t_hold
            if SEVERITY[want] < SEVERITY[self.mode]:
                if self._since_clear is None:
                    self._since_clear = x.t
                if x.t - self._since_clear >= e.t_hold:
                    new = Mode.CAUTION if self.mode in (Mode.STOP, Mode.RETREAT) and want == Mode.NOMINAL else want
                    self._since_clear = None
            else:
                self._since_clear = None
            if self.mode in (Mode.STOP, Mode.RETREAT) and want in (Mode.STOP, Mode.RETREAT):
                # (a stop that began at t = 0.0 is a real start time, not "unset")
                dur = x.t - (self._stop_started if self._stop_started is not None else x.t)
                if dur > e.t_handover_after:
                    new = Mode.HANDOVER
                    self.handover_requests += 1
                    reasons = reasons + ["persisting stop -> operator"]
                elif (
                    dur > e.t_retreat_after and self.mode == Mode.STOP and cat in ("human_proximity", "collision_risk")
                ):
                    new = Mode.RETREAT
        if new != prev:
            self.events.append(SafetyEvent(x.t, prev.value, new.value, reasons, cat or "recovery",
                                           {"collision_prob": x.collision_prob, "pred_clearance": x.pred_clearance,
                                            "human_clearance": x.human_clearance, "loc_std": x.loc_std,
                                            "nis": x.nis, "lidar_age": x.lidar_age, "path_entropy": x.path_entropy}))
            self.mode = new
        self.last_reasons = reasons
        return self.mode

    def gate(self, v: float, w: float, retreat_cmd: tuple[float, float] | None = None) -> tuple[float, float]:
        if self.mode == Mode.NOMINAL:
            return v, w
        if self.mode == Mode.CAUTION:
            s = self.env.caution_speed_scale
            return float(np.clip(v, -0.2, 0.7 * s)), float(w * max(s, 0.6))
        if self.mode == Mode.RETREAT and retreat_cmd is not None:
            return retreat_cmd
        return 0.0, 0.0

    def degraded(self) -> bool:
        return self.mode != Mode.NOMINAL
