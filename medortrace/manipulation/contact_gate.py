"""Force/contact gating for the robot's (non-sterile) manipulation task.

The only manipulation MED-OR-TRACE performs is *retrieval of a dropped,
non-sterile item from the floor into a kick bucket*, and only after explicit
operator approval.  The gate is a small state machine that allows each phase
to proceed only when contact/force/effort signals are inside an envelope:

    IDLE -> APPROACH (v <= v_max_approach, no contact)
         -> PRE_GRASP (item localised with p >= p_min, human clearance ok)
         -> GRASP     (contact force rises into [f_min, f_max] within t_max)
         -> LIFT      (wrist effort consistent with expected item mass)
         -> PLACE / ABORT

Any out-of-envelope sample (unexpected contact, force spike, effort
mismatch that suggests the wrong object or a snag on a drape) aborts and
releases.  In Isaac Sim the signals come from ``isaacsim.sensors.physics``
ContactSensor and articulation joint efforts (see medortrace.isaac.sensors).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(str, Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    PRE_GRASP = "PRE_GRASP"
    GRASP = "GRASP"
    LIFT = "LIFT"
    PLACE = "PLACE"
    DONE = "DONE"
    ABORT = "ABORT"


@dataclass
class GateConfig:
    v_max_approach: float = 0.15
    p_item_min: float = 0.85
    human_clearance_min: float = 1.0
    f_grasp_min: float = 3.0
    f_grasp_max: float = 25.0
    t_grasp_max: float = 3.0
    effort_tol_frac: float = 0.5
    f_unexpected: float = 8.0


@dataclass
class GateInputs:
    t: float
    base_speed: float
    item_prob: float
    human_clearance: float
    contact_force: float
    in_contact: bool
    wrist_effort: float = 0.0
    expected_effort: float = 0.0
    operator_approved: bool = False
    at_pre_grasp: bool = False
    at_place: bool = False


class ContactGate:
    def __init__(self, cfg: GateConfig | None = None):
        self.cfg = cfg or GateConfig()
        self.phase = Phase.IDLE
        self.t_phase = 0.0
        self.abort_reason = ""
        self.log: list[tuple[float, str, str]] = []

    def _go(self, p: Phase, t: float, why: str = "") -> None:
        self.log.append((t, self.phase.value, p.value + (f" ({why})" if why else "")))
        self.phase = p
        self.t_phase = t

    def step(self, x: GateInputs) -> Phase:
        c = self.cfg
        if self.phase in (Phase.DONE, Phase.ABORT):
            return self.phase
        if self.phase != Phase.IDLE and x.human_clearance < c.human_clearance_min:
            self.abort_reason = "human too close"
            self._go(Phase.ABORT, x.t, self.abort_reason)
            return self.phase
        if self.phase == Phase.IDLE:
            if x.operator_approved:
                self._go(Phase.APPROACH, x.t)
        elif self.phase == Phase.APPROACH:
            if x.in_contact and x.contact_force > c.f_unexpected:
                self.abort_reason = "unexpected contact during approach"
                self._go(Phase.ABORT, x.t, self.abort_reason)
            elif x.at_pre_grasp and x.base_speed <= c.v_max_approach:
                self._go(Phase.PRE_GRASP, x.t)
        elif self.phase == Phase.PRE_GRASP:
            if x.item_prob >= c.p_item_min:
                self._go(Phase.GRASP, x.t)
            elif x.t - self.t_phase > 10.0:
                self.abort_reason = "item not confirmed at grasp location"
                self._go(Phase.ABORT, x.t, self.abort_reason)
        elif self.phase == Phase.GRASP:
            if x.contact_force > c.f_grasp_max:
                self.abort_reason = f"grasp force {x.contact_force:.1f}N above limit"
                self._go(Phase.ABORT, x.t, self.abort_reason)
            elif c.f_grasp_min <= x.contact_force <= c.f_grasp_max and x.in_contact:
                self._go(Phase.LIFT, x.t)
            elif x.t - self.t_phase > c.t_grasp_max:
                self.abort_reason = "no stable grasp contact"
                self._go(Phase.ABORT, x.t, self.abort_reason)
        elif self.phase == Phase.LIFT:
            if (
                x.expected_effort > 0
                and abs(x.wrist_effort - x.expected_effort) > c.effort_tol_frac * x.expected_effort
            ):
                self.abort_reason = "wrist effort mismatch (snag or wrong object)"
                self._go(Phase.ABORT, x.t, self.abort_reason)
            elif x.at_place:
                self._go(Phase.PLACE, x.t)
        elif self.phase == Phase.PLACE:
            if not x.in_contact:
                self._go(Phase.DONE, x.t)
        return self.phase
