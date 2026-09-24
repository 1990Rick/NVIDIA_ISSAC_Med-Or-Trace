"""Surgical workflow model: ground-truth item custody, reported events, claims.

Three layers are produced for every episode:

* ``truth``  - the *actual* custody timeline of every critical item (what really
  happened, including unlogged drops and the counterfactual hidden cause);
* ``log``    - the workflow events as reported to the robot by OR staff / the
  OR information system, with realistic omissions, time jitter and label
  errors (the robot must not trust them blindly);
* ``claims`` - the verification queries the robot must answer (VERIFIED /
  REFUTED / ABSTAIN) with a deadline, e.g. "specimen_1 is in the specimen cup
  on the back table at t=121 s" or the closing count.

The robot never sees ``truth``; evaluation compares verdicts against it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medortrace.common.msgs import WorkflowEvent, WorkflowEventType as WT
from medortrace.common.rng import RngStreams
from medortrace.world.scene import SceneSpec


@dataclass
class TruthMove:
    t: float
    item_id: str
    src: str
    dst: str
    cause: str = "workflow"       # workflow | hidden_cause | drop


@dataclass
class StaffTask:
    t_start: float
    goal: np.ndarray
    dwell: float
    purpose: str = ""


@dataclass
class Claim:
    id: str
    t_ref: float                  # time the assertion refers to
    t_due: float                  # verdict deadline
    item_id: str
    slot_id: str
    source_event: str | None = None
    kind: str = "location"        # location | handoff | count


@dataclass
class WorkflowScript:
    duration: float
    truth: list[TruthMove]
    log: list[WorkflowEvent]
    staff_tasks: dict[str, list[StaffTask]]
    claims: list[Claim]
    initial: dict[str, str]
    hidden_notes: dict = field(default_factory=dict)

    def truth_slot(self, item_id: str, t: float) -> str:
        slot = self.initial[item_id]
        for m in self.truth:
            if m.item_id == item_id and m.t <= t:
                slot = m.dst
        return slot


def generate_workflow(spec: SceneSpec, cfg: dict, streams: RngStreams) -> WorkflowScript:
    R = streams["workflow"]
    Hd = streams["hidden"]
    wcfg = cfg.get("workflow", {})
    T = float(cfg.get("episode", {}).get("duration_s", 180.0))
    p_missing = float(wcfg.get("p_log_missing", 0.05))
    jitter = float(wcfg.get("log_time_jitter_s", 1.5))
    p_mislabel = float(wcfg.get("p_log_mislabel", 0.02))

    truth: list[TruthMove] = []
    events: list[tuple[WorkflowEvent, bool]] = []   # (event, force_logged)
    tasks: dict[str, list[StaffTask]] = {s.name: [] for s in spec.staff}
    initial = {i.id: i.initial_slot for i in spec.items}
    initial["specimen_1"] = "elsewhere"
    hc = spec.hidden_cause
    notes: dict = {}

    def slot_xy(sid: str) -> np.ndarray:
        return spec.slot(sid).position[:2]

    def approach(sid: str, offset=(0.0, -0.6)) -> np.ndarray:
        return slot_xy(sid) + np.array(offset)

    def move(t, item, src, dst, logged_type: WT | None, cause="workflow", reporter="circulating_nurse"):
        truth.append(TruthMove(t, item, src, dst, cause))
        if logged_type is not None:
            events.append((WorkflowEvent(t, logged_type, item, src, dst, reporter=reporter), False))

    # ---- 1. implant box opened onto the sterile back table ------------------
    t0 = float(R.uniform(0.06, 0.12) * T)
    tasks["circulator"].append(StaffTask(t0 - 8.0, approach("cart_2:top", (0.0, -0.65)), 4.0, "fetch implant"))
    move(t0 - 4.0, "implant_box_1", "cart_2:top", "hand:circulator", None)
    bt_edge = spec.slot("back_table:tray").position[:2] + np.array([1.35, -0.25])
    tasks["circulator"].append(StaffTask(t0, bt_edge, 5.0, "open implant"))
    move(t0 + 5.0, "implant_box_1", "hand:circulator", "back_table:tray", WT.OPEN)

    # ---- 2. sponges to the field and discards --------------------------------
    sponges = [i.id for i in spec.items if i.cls == "sponge"]
    t_use = np.sort(R.uniform(0.15 * T, 0.40 * T, size=len(sponges)))
    retained = None
    if hc.get("factor") == "CF-A":
        retained = sponges[int(Hd.integers(0, len(sponges)))]
        notes["retained_sponge"] = retained
    for sid, tu in zip(sponges, t_use):
        move(float(tu), sid, "back_table:tray", "hand:scrub_nurse", None)
        move(float(tu) + 2.0, sid, "hand:scrub_nurse", "field:top", WT.HANDOFF, reporter="scrub_nurse")
        td = float(tu) + float(R.uniform(10.0, 25.0))
        bucket = "kick_bucket_1:inside" if R.random() < 0.6 else "kick_bucket_2:inside"
        if sid == retained and hc.get("value") == "under_drape":
            # Log says discarded, but the sponge slipped under the drape.
            move(td, sid, "field:top", "field:under_drape", None, cause="hidden_cause")
            events.append((WorkflowEvent(td, WT.DISCARD, sid, "field:top", bucket, reporter="surgeon"), True))
            notes["claimed_bucket"] = bucket
        else:
            move(td, sid, "field:top", bucket, WT.DISCARD, reporter="surgeon")
            if sid == retained:
                notes["claimed_bucket"] = bucket
                truth[-1].cause = "hidden_cause"
                events[-1] = (events[-1][0], True)

    # ---- 3. needle driver round trip (logged) --------------------------------
    tn = float(R.uniform(0.2, 0.3) * T)
    move(tn, "needle_driver_1", "mayo:top", "hand:surgeon", WT.HANDOFF, reporter="scrub_nurse")
    move(tn + float(R.uniform(15, 30)), "needle_driver_1", "hand:surgeon", "mayo:top", WT.HANDOFF,
         reporter="scrub_nurse")

    # ---- 4. clamp leaves the mayo stand (CF-C) -------------------------------
    if hc.get("factor") == "CF-C":
        tc = float(Hd.uniform(0.3, 0.4) * T)
        notes["t_clamp"] = tc
        if hc.get("value") == "dropped_floor":
            move(tc, "clamp_1", "mayo:top", "floor:foot_of_table", None, cause="hidden_cause")
        else:
            move(tc, "clamp_1", "mayo:top", "hand:assistant", None, cause="hidden_cause")
            move(tc + 12.0, "clamp_1", "hand:assistant", "back_table:tray", None, cause="hidden_cause")

    # ---- 5. specimen chain of custody ----------------------------------------
    ts = float(R.uniform(0.45, 0.55) * T)
    move(ts, "specimen_1", "elsewhere", "hand:surgeon", WT.SPECIMEN_OUT, reporter="surgeon")
    move(ts + 3.0, "specimen_1", "hand:surgeon", "hand:scrub_nurse", WT.HANDOFF, reporter="scrub_nurse")
    move(ts + 8.0, "specimen_1", "hand:scrub_nurse", "back_table:specimen_cup", WT.PLACE, reporter="scrub_nurse")
    tcirc = ts + float(R.uniform(20, 35))
    cup_edge = spec.slot("back_table:specimen_cup").position[:2] + np.array([0.95, -0.2])
    tasks["circulator"].append(StaffTask(tcirc - 6.0, cup_edge, 5.0, "collect specimen"))
    move(tcirc, "specimen_1", "back_table:specimen_cup", "hand:circulator", WT.HANDOFF)
    tasks["circulator"].append(StaffTask(tcirc + 1.0, approach("specimen_counter:top", (0.0, 0.6)), 6.0, "label specimen"))
    move(tcirc + 10.0, "specimen_1", "hand:circulator", "specimen_counter:top", WT.PLACE)

    # ---- 6. circulator & anesthetist background roaming ----------------------
    circ = next(s for s in spec.staff if s.name == "circulator")
    t = 2.0
    while t < T:
        busy = any(abs(t - k.t_start) < 15 for k in tasks["circulator"])
        if not busy:
            wp = circ.waypoints[int(R.integers(0, len(circ.waypoints)))]
            tasks["circulator"].append(StaffTask(t, wp + R.normal(0, 0.1, 2), float(R.uniform(3, 10)), "roam"))
        t += float(R.uniform(12, 25))
    anes = next(s for s in spec.staff if s.name == "anesthetist")
    t = float(R.uniform(10, 30))
    while t < T:
        wp = anes.waypoints[int(R.integers(0, len(anes.waypoints)))]
        tasks["anesthetist"].append(StaffTask(t, wp + R.normal(0, 0.1, 2), float(R.uniform(5, 15)), "roam"))
        t += float(R.uniform(25, 50))
    for k in tasks:
        tasks[k].sort(key=lambda s: s.t_start)

    # ---- 7. count checkpoints -------------------------------------------------
    t_count1 = 0.62 * T
    t_count2 = 0.92 * T
    events.append((WorkflowEvent(t_count1, WT.COUNT, None, None, None, reporter="circulating_nurse",
                                 payload={"phase": "first_closing_count"}), True))
    events.append((WorkflowEvent(t_count2, WT.COUNT, None, None, None, reporter="circulating_nurse",
                                 payload={"phase": "final_count"}), True))

    truth.sort(key=lambda m: m.t)

    # ---- reported log with realistic imperfections ---------------------------
    log: list[WorkflowEvent] = []
    slot_ids = [s.id for s in spec.slots if s.kind in ("surface", "container")]
    for n, (ev, forced) in enumerate(sorted(events, key=lambda e: e[0].t)):
        if not forced and R.random() < p_missing:
            continue
        ev.event_id = f"wf_{n:03d}"
        ev.t = float(max(0.0, ev.t + R.normal(0.0, jitter))) if not forced else ev.t
        if not forced and ev.dst and R.random() < p_mislabel:
            ev.dst = str(R.choice(slot_ids))
            ev.payload["mislabelled"] = True
        ev.confidence = float(np.clip(R.normal(0.9, 0.05), 0.6, 0.99))
        log.append(ev)
    log.sort(key=lambda e: e.t)

    # ---- claims (same deterministic rule the robot applies online) -----------
    grace = float(wcfg.get("claim_grace_s", 25.0))
    claims = claims_from_log(log, initial, T, grace)

    return WorkflowScript(T, truth, log, tasks, claims, initial, notes)


COUNT_TAGS = {"first_closing_count": "count1", "final_count": "count2"}


def handoff_claim(ev: WorkflowEvent, T: float, grace: float) -> Claim | None:
    if ev.type in (WT.HANDOFF, WT.PLACE, WT.DISCARD, WT.OPEN) and ev.item_id and ev.dst:
        if ev.dst.startswith("hand:"):
            return None  # custody in hands is transient; verified via the next placement
        t_ref = ev.t + 6.0
        return Claim(f"c_{ev.event_id}", t_ref, min(T, t_ref + grace), ev.item_id, ev.dst,
                     source_event=ev.event_id, kind="handoff")
    return None


def count_claims(count_ev: WorkflowEvent, log_so_far: list[WorkflowEvent], initial: dict[str, str],
                 T: float, grace: float) -> list[Claim]:
    """Each item must be where the last reported event before the count put it."""
    tag = COUNT_TAGS.get(count_ev.payload.get("phase", ""), f"count_{count_ev.event_id}")
    tc = count_ev.t
    out = []
    for iid, s0 in initial.items():
        last = s0
        for ev in log_so_far:
            if ev.item_id == iid and ev.t <= tc and ev.dst:
                last = ev.dst
        if last == "elsewhere" or last.startswith("hand:"):
            continue
        out.append(Claim(f"{tag}_{iid}", tc, min(T, tc + grace), iid, last, kind="count"))
    return out


def claims_from_log(log: list[WorkflowEvent], initial: dict[str, str], T: float, grace: float) -> list[Claim]:
    claims: list[Claim] = []
    for ev in log:
        c = handoff_claim(ev, T, grace)
        if c:
            claims.append(c)
        if ev.type == WT.COUNT:
            claims.extend(count_claims(ev, [e for e in log if e.type != WT.COUNT], initial, T, grace))
    claims.sort(key=lambda c: c.t_due)
    return claims
