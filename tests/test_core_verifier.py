"""Claim verifier: VERIFIED / REFUTED / ABSTAIN semantics and fixed-lag smoothing."""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.belief.items import ItemBelief
from medortrace.provenance.graph import ProvenanceGraph, Verdict
from medortrace.provenance.verifier import ClaimVerifier
from medortrace.world.scene import ItemSpec, Slot
from medortrace.world.workflow import Claim

SLOTS = [Slot("table:top", "surface", "table", (1.0, 1.0, 0.9)),
         Slot("bin:inside", "container", "bin", (3.0, 1.0, 0.3)),
         Slot("floor:near_table", "floor", "table", (1.5, 1.0, 0.02)),
         Slot("elsewhere", "elsewhere", "none", (np.nan, np.nan, np.nan))]
ITEM = ItemSpec("clamp_1", "clamp", "instrument_steel_polished", "table:top", metallic=True)


def _setup(n_events=1, **kw):
    bel = ItemBelief([ITEM], [Slot(s.id, s.kind, s.anchor, s.position) for s in SLOTS], {"clamp_1": "table:top"})
    prov = ProvenanceGraph()
    ver = ClaimVerifier(bel, prov, **kw)
    for k in range(n_events):
        prov.add_evidence(f"wf:{k}", 5.0, "workflow", {"dst": "bin:inside"}, attributed_to="scrub_nurse")
        bel.apply_workflow_event("clamp_1", "table:top", "bin:inside", 0.99, f"wf:{k}", 5.0)
    ver.add_claim(Claim("c1", t_ref=10.0, t_due=30.0, item_id="clamp_1", slot_id="bin:inside", kind="handoff"))
    return bel, prov, ver


def _camera(bel, prov, t, count, pd=0.9, slot="bin:inside", eid=None, tag=False):
    eid = eid or f"cam:{t:.1f}"
    prov.add_evidence(eid, t, "camera", {"count": count})
    k = bel.sidx[slot]
    vp = np.zeros(bel.n)
    vp[k] = pd
    n = np.zeros(bel.n)
    n[k] = count
    bel.update_camera_classes({"clamp": vp}, {"clamp": n}, [("clamp_1", k)] if tag else [], eid, t)
    return eid


def test_workflow_log_alone_never_verifies():
    bel, prov, ver = _setup(n_events=3)
    assert bel.prob("clamp_1", "bin:inside") > 0.95            # the log is very confident ...
    ver.add_claim(Claim("c1", 10.0, 30.0, "clamp_1", "bin:inside"))  # duplicate id ignored
    assert len(ver.pending()) == 1
    assert ver.step(9.0) == [] and ver.step(10.0) == [] and ver.step(29.9) == []
    (v,) = ver.step(30.0)                                       # ... but the deadline yields ABSTAIN
    assert v.verdict == Verdict.ABSTAIN and not v.direct and v.posterior > 0.95
    assert "no direct sensor evidence" in v.reason
    assert ver.pending() == [] and "c1" in ver.done
    node = prov.nodes["verdict:c1"]
    assert node.attrs["verdict"] == "ABSTAIN" and prov.custody == {}
    # ablation: without the direct-evidence rule the same belief is (over-)asserted immediately
    bel2, prov2, ver2 = _setup(n_events=3, require_direct_evidence=False)
    (v2,) = ver2.step(10.0)
    assert v2.verdict == Verdict.VERIFIED and v2.t == 10.0


def test_verified_with_direct_positive_evidence():
    bel, prov, ver = _setup()
    assert ver.step(10.0) == []                                 # snapshot at t_ref, no evidence yet
    assert ver.urgency(10.0)["clamp_1"] > 0 and "bin:inside" in ver.urgent_slots(10.0)
    e1 = _camera(bel, prov, 11.0, count=1.0)
    out = ver.step(11.0)
    assert len(out) == 1
    v = out[0]
    assert v.verdict == Verdict.VERIFIED and v.direct and v.posterior >= 0.9 and v.t == 11.0
    assert v.map_slot == "bin:inside"
    ex = prov.explain("verdict:c1")
    assert [s["evidence"] for s in ex["supporting"]] == [e1] and ex["contradicting"] == []
    assert prov.handoff_chain("clamp_1")["chain"][0]["slot"] == "bin:inside"


def test_refuted_with_negative_evidence():
    bel, prov, ver = _setup()
    ver.step(10.0)
    verdicts, t = [], 10.0
    while not verdicts and t < 29.0:
        t += 0.5
        _camera(bel, prov, t, count=0.0)                        # bin clearly visible, nothing there
        verdicts = ver.step(t)
    assert verdicts, "negative evidence never produced a verdict"
    v = verdicts[0]
    assert v.verdict == Verdict.REFUTED and v.posterior <= 0.1 and v.direct and v.t < 30.0
    assert v.map_slot == "table:top"
    ex = prov.explain("verdict:c1", k=3)
    assert len(ex["contradicting"]) == 3 and ex["supporting"] == []
    w = [c["weight"] for c in ex["contradicting"]]
    assert w == sorted(w) and all(x < 0 for x in w)
    assert prov.custody == {}


def test_workflow_log_delivered_after_t_ref_is_not_direct_evidence():
    # count claims take t_ref = the count time; a log entry reported <= 1 s later and delivered
    # late reaches the belief *after* the t_ref snapshot and is not cut off by note_item_event
    bel, prov, ver = _setup(n_events=0)
    ver.open["c1"].claim.kind = "count"                         # (a handoff claim would be superseded)
    assert ver.step(10.0) == []                                 # snapshot: the clamp is on the table
    for k in range(2):
        prov.add_evidence(f"wf:late{k}", 10.5, "workflow", {"dst": "bin:inside"}, attributed_to="scrub_nurse")
        bel.apply_workflow_event("clamp_1", "table:top", "bin:inside", 0.99, f"wf:late{k}", 10.5)
    ver.note_item_event("clamp_1", 10.4)                        # within t_ref + 1 s: smoothing window stays open
    assert bel.prob("clamp_1", "bin:inside") > 0.95             # the filtered belief follows the log ...
    for t in (10.5, 12.0, 20.0, 29.9):                          # ... but no sensor ever looked at the bin
        assert ver.step(t) == [] and "bin:inside" in ver.urgent_slots(t)
    (v,) = ver.step(30.0)
    assert v.verdict == Verdict.ABSTAIN and v.direct is False and v.t == 30.0
    assert "no direct sensor evidence" in v.reason and prov.custody == {}


def test_evidence_about_another_slot_is_not_direct():
    # seeing the *source* slot empty pushes the claimed-slot posterior above tau_verify,
    # but the item could equally be on the floor: the claimed slot itself was never observed
    bel, prov, ver = _setup()
    ver.step(10.0)
    t = 10.0
    while t < 29.0:
        t += 0.5
        _camera(bel, prov, t, count=0.0, slot="table:top")
        assert ver.step(t) == [] and "bin:inside" in ver.urgent_slots(t)
    (v,) = ver.step(30.0)
    assert v.verdict == Verdict.ABSTAIN and v.direct is False and v.posterior >= 0.9
    assert "no direct sensor evidence" in v.reason and prov.custody == {}


def test_direct_evidence_against_the_claim_never_verifies_it():
    # the log keeps the posterior at ~0.996; a partial view of the bin that does *not* see the clamp
    # is direct evidence, but against the claim - it must not unlock a VERIFIED on the log's say-so
    bel, prov, ver = _setup(n_events=3)
    ver.step(10.0)
    _camera(bel, prov, 11.0, count=0.0, pd=0.3)
    assert ver.step(11.0) == [] and ver.step(29.9) == []
    (v,) = ver.step(30.0)
    assert v.verdict == Verdict.ABSTAIN and v.direct and v.posterior > 0.95
    assert "disagrees" in v.reason and prov.custody == {}
    # a later positive sighting of the clamp in the bin then does verify
    bel, prov, ver = _setup(n_events=3)
    ver.step(10.0)
    _camera(bel, prov, 11.0, count=0.0, pd=0.3)
    e2 = _camera(bel, prov, 12.0, count=1.0)
    (v,) = ver.step(12.0)
    assert v.verdict == Verdict.VERIFIED and v.direct
    assert [s["evidence"] for s in prov.explain("verdict:c1")["supporting"]] == [e2]
    # symmetric: a log that says "not in the bin" plus a weak sighting in the bin never refutes
    bel, prov, ver = _setup(n_events=0)
    ver.step(10.0)
    _camera(bel, prov, 11.0, count=1.0, pd=0.3)
    assert ver.step(11.0) == []
    (v,) = ver.step(30.0)
    assert v.verdict == Verdict.ABSTAIN and v.direct and v.posterior < 0.1


def test_identity_read_of_the_item_elsewhere_is_direct_evidence():
    # positive control for the two tests above: a tag read of *this* item at another slot is direct
    bel, prov, ver = _setup()
    ver.step(10.0)
    e1 = _camera(bel, prov, 11.0, count=0.0, pd=0.0, slot="table:top", tag=True)
    assert ver.step(11.0) == []                                 # one read: posterior ~0.17, undecided ...
    assert "bin:inside" not in ver.urgent_slots(11.0)           # ... but the claim now has direct evidence
    e2 = _camera(bel, prov, 12.0, count=0.0, pd=0.0, slot="table:top", tag=True)
    (v,) = ver.step(12.0)
    assert v.verdict == Verdict.REFUTED and v.direct and v.posterior <= 0.1 and v.map_slot == "table:top"
    # the explanation cites the camera frames that carried the tag reads
    ex = prov.explain("verdict:c1")
    assert {c["evidence"] for c in ex["contradicting"]} == {e1, e2} and ex["supporting"] == []
    assert all(c["weight"] == pytest.approx(-np.log(25.0)) for c in ex["contradicting"])


def test_degraded_mode_raises_the_verify_threshold():
    bel, prov, ver = _setup()
    ver.degraded = True
    ver.step(10.0)
    _camera(bel, prov, 11.0, count=1.0)                         # posterior ~0.92: enough nominally ...
    assert ver.step(11.0) == []                                 # ... not when degraded (tau 0.95)
    bel2, prov2, ver2 = _setup()
    ver2.step(10.0)
    _camera(bel2, prov2, 11.0, count=1.0)
    (v,) = ver2.step(11.0)
    assert 0.9 <= v.posterior < 0.95


def _smoothing_run(report_move: bool, move_t: float = 12.0):
    bel, prov, ver = _setup()
    ver.step(10.0)                                              # snapshot of the belief at t_ref
    _camera(bel, prov, 11.0, count=0.0, pd=0.0, tag=True, eid="cam:tag")   # identity read in the bin
    if report_move:
        ver.note_item_event("clamp_1", move_t)                  # item reported moved at 12 s
    for k in range(16):                                         # afterwards the bin is seen empty
        _camera(bel, prov, 13.0 + 0.25 * k, count=0.0)
    out = ver.step(17.0) or ver.step(30.0)
    return out[0], bel


def test_fixed_lag_smoothing_ignores_evidence_after_reported_move():
    v, bel = _smoothing_run(report_move=True)
    assert v.verdict == Verdict.VERIFIED and v.t == 17.0
    # the smoothed posterior is snapshot x tag likelihood ratio only
    odds = 0.85 * 0.99 / (1 - 0.85 * 0.99) * 25.0
    assert v.posterior == pytest.approx(odds / (1 + odds), abs=2e-3)
    # while the *current* filtered belief (rightly) says the item left the bin
    assert bel.prob("clamp_1", "bin:inside") < 0.9
    # control: without the reported move the later empty-bin evidence counts
    v_ctrl, _ = _smoothing_run(report_move=False)
    assert v_ctrl.verdict != Verdict.VERIFIED and v_ctrl.posterior < 0.9
    # a move reported at (or just after) t_ref does not invalidate the claim window
    v_early, _ = _smoothing_run(report_move=True, move_t=10.5)
    assert v_early.verdict != Verdict.VERIFIED


def test_handoff_claim_superseded_by_a_move_reported_before_t_ref():
    """A sponge discarded seconds after reaching the field: the handoff claim's
    placement was transient, so it is closed as ABSTAIN (never checked against the
    post-move state); a move reported after t_ref ends the smoothing window, after
    which nothing can change the claim, so it is closed with the evidence it had."""
    bel, prov, ver = _setup(n_events=0)                         # claim c1: clamp at bin:inside, t_ref = 10
    ver.note_item_event("clamp_1", 8.0)                         # moved on before t_ref
    (v,) = ver.step(8.5)                                        # closed at once, before t_ref
    assert v.verdict == Verdict.ABSTAIN and "superseded" in v.reason and "c1" in ver.done
    bel, prov, ver = _setup(n_events=0)
    assert ver.step(10.0) == []                                 # snapshot at t_ref
    ver.note_item_event("clamp_1", 12.0)                        # after t_ref + 1 s: window cut at 12.0
    assert ver.open["c1"].moved_t == 12.0 and not ver.open["c1"].superseded
    (v,) = ver.step(12.5)                                       # frozen: decided now, not at t_due
    assert v.verdict == Verdict.ABSTAIN and "moved on" in v.reason and v.t == 12.5
    bel, prov, ver = _setup(n_events=0)
    ver.open["c1"].claim.kind = "count"                         # counts are never superseded
    ver.note_item_event("clamp_1", 8.0)
    assert ver.step(8.5) == [] and not ver.open["c1"].superseded

