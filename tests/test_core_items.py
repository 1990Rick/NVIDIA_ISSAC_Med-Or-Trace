"""Item-custody belief (discrete Bayes filter over slots).

The expected numbers are recomputed from the documented model (mass transfer
``rho * conf``; mean-field Poisson-binomial camera count likelihood; tag
likelihood ratio), so
the tests pin the *semantics* rather than incidental values.
"""

from __future__ import annotations

import numpy as np
import pytest

from medortrace.belief.items import CLASSES, ItemBelief, poisson_logpmf
from medortrace.world.scene import ItemSpec, Slot


def _slots():
    return [Slot("table:top", "surface", "table", (1.0, 1.0, 0.9)),
            Slot("bin:inside", "container", "bin", (3.0, 1.0, 0.3)),
            Slot("floor:near_table", "floor", "table", (1.5, 1.0, 0.02)),
            Slot("hand:nurse", "hand", "nurse", (2.0, 2.0, 1.0)),
            Slot("far:shelf", "surface", "shelf", (9.0, 9.0, 1.0)),
            Slot("elsewhere", "elsewhere", "none", (np.nan, np.nan, np.nan))]


def _belief(items=None, initial=None, **params):
    items = items or [ItemSpec("clamp_1", "clamp", "instrument_steel_polished", "table:top", metallic=True)]
    return ItemBelief(items, _slots(), initial, params=params or None)


def _sponges(n=3):
    return [ItemSpec(f"sponge_{i}", "sponge", "cotton_sponge", "table:top", fungible=True, tag_readable=False)
            for i in range(n)]


def _cam(bel, cls, slot, pd, count, eid="cam:1", t=1.0, tags=()):
    vp = np.zeros(bel.n)
    vp[bel.sidx[slot]] = pd
    n = np.zeros(bel.n)
    n[bel.sidx[slot]] = count
    bel.update_camera_classes({cls: vp}, {cls: n}, list(tags), eid, t)


def test_initial_belief_is_concentrated_and_normalised():
    bel = _belief()
    b = bel.items["clamp_1"].b
    assert b.sum() == pytest.approx(1.0) and np.all(b >= ItemBelief.FLOOR * 0.99)
    assert bel.map_slot("clamp_1") == ("table:top", pytest.approx(1 - 5 * ItemBelief.FLOOR, abs=1e-6))
    assert bel.entropy("clamp_1") < 0.01
    # the pre-operative count sheet overrides the spec default
    assert _belief(initial={"clamp_1": "bin:inside"}).map_slot("clamp_1")[0] == "bin:inside"


def test_workflow_event_mass_transfer():
    bel = _belief()
    b0 = bel.items["clamp_1"].b.copy()
    bel.apply_workflow_event("clamp_1", "table:top", "bin:inside", conf=0.9, evidence_id="wf:1", t=5.0)
    k = bel.rho * 0.9
    s, d = bel.sidx["table:top"], bel.sidx["bin:inside"]
    move = b0 * k * 0.5                          # known src: half of the non-src mass may also move
    move[s] = b0[s] * k
    move[d] = 0.0
    expect = b0 - move
    expect[d] += move.sum()
    expect = np.maximum(expect, ItemBelief.FLOOR)
    expect /= expect.sum()
    assert np.allclose(bel.items["clamp_1"].b, expect)
    assert bel.prob("clamp_1", "bin:inside") == pytest.approx(0.765, abs=2e-3)
    rec = bel.records[-1]
    assert rec.sensor == "workflow" and rec.evidence_id == "wf:1" and rec.note == "table:top->bin:inside"
    assert rec.llr[d] > 0 and rec.llr[s] < 0
    # hand source: the item may have reached the hand unlogged -> full rho*conf share moves
    bel2 = _belief()
    bel2.apply_workflow_event("clamp_1", "hand:nurse", "bin:inside", conf=1.0, evidence_id="wf:2", t=5.0)
    assert bel2.prob("clamp_1", "bin:inside") == pytest.approx(bel2.rho, abs=2e-3)
    # unknown item / slot: no-op
    n = len(bel.records)
    bel.apply_workflow_event("ghost_item", None, "bin:inside", 0.9, "wf:3", 6.0)
    bel.apply_workflow_event("clamp_1", None, "not_a_slot", 0.9, "wf:4", 6.0)
    assert len(bel.records) == n and np.allclose(bel.items["clamp_1"].b, expect)


def test_camera_positive_and_negative_evidence():
    initial = {"clamp_1": "table:top"}
    bel = _belief(initial=initial)
    bel.apply_workflow_event("clamp_1", "table:top", "bin:inside", 0.6, "wf:1", 0.0)   # now split table/bin
    p_bin0 = bel.prob("clamp_1", "bin:inside")
    assert 0.3 < p_bin0 < 0.7
    # detection of a clamp at the bin -> more mass at the bin
    pos = _belief(initial=initial)
    pos.items["clamp_1"].b = bel.items["clamp_1"].b.copy()
    _cam(pos, "clamp", "bin:inside", pd=0.8, count=1.0)
    assert pos.prob("clamp_1", "bin:inside") > p_bin0 + 0.2
    # the bin was clearly visible and nothing was detected -> mass leaves the bin
    neg = _belief(initial=initial)
    neg.items["clamp_1"].b = bel.items["clamp_1"].b.copy()
    _cam(neg, "clamp", "bin:inside", pd=0.8, count=0.0, t=2.0)
    rec = neg.records[-1]
    k = neg.sidx["bin:inside"]
    # exact count LLR for n = 0 (no other clamps, identity confusion): the clamp,
    # if present, is missed with probability 1 - pd; false positives cancel:
    # log[(1 - pd) e^-fp] - log[e^-fp] = log(1 - pd), discounted by the camera
    # decorrelation factor
    pd = 0.8 * neg.q_vis
    expect_llr = np.log(1 - pd) * neg.decorrelation["camera"]
    assert rec.llr[k] == pytest.approx(expect_llr)
    odds = p_bin0 / (1 - p_bin0) * np.exp(expect_llr)
    assert neg.prob("clamp_1", "bin:inside") == pytest.approx(odds / (1 + odds), abs=1e-6)
    assert neg.prob("clamp_1", "bin:inside") < p_bin0 - 0.04
    assert np.count_nonzero(rec.llr) == 1                       # only the visible slot is touched
    assert neg.items["clamp_1"].last_direct_obs_t == {k: 2.0}
    # invisible slots (pd <= 0.03) carry no evidence at all
    none = _belief()
    _cam(none, "clamp", "bin:inside", pd=0.02, count=0.0)
    assert not none.records


def test_fungible_mean_field_counts():
    """Two sponges are known to be on the table; a third is uncertain (table vs bin).
    Every item is detected at most once (Poisson-binomial count model), so:
    seeing 1 sponge is evidence against the third being there; seeing 2 is nearly
    uninformative at pd = 0.64 (P(2 of 3) = 0.44 vs P(2 of 2) = 0.41); seeing 3 is a
    surplus the two known sponges cannot explain - as informative as a lone
    detection of a non-fungible item."""
    def fresh():
        bel = _belief(_sponges(3))
        bel.apply_workflow_event("sponge_2", "table:top", "bin:inside", 0.59, "wf", 0.0)
        return bel

    p0 = fresh().prob("sponge_2", "table:top")
    assert 0.4 < p0 < 0.6
    low, mid, high = fresh(), fresh(), fresh()
    _cam(low, "sponge", "table:top", pd=0.8, count=1.0)
    _cam(mid, "sponge", "table:top", pd=0.8, count=2.0)
    _cam(high, "sponge", "table:top", pd=0.8, count=3.0)
    assert low.prob("sponge_2", "table:top") < p0 - 0.03
    assert mid.prob("sponge_2", "table:top") == pytest.approx(p0, abs=0.02)
    assert high.prob("sponge_2", "table:top") > p0
    # lone non-fungible item with the same belief and a single detection
    lone = _belief()
    lone.apply_workflow_event("clamp_1", "table:top", "bin:inside", 0.59, "wf", 0.0)
    _cam(lone, "clamp", "table:top", pd=0.8, count=1.0)
    k = high.sidx["table:top"]
    llr_high = [r for r in high.records if r.item_id == "sponge_2" and r.sensor == "camera"][0].llr[k]
    llr_lone = [r for r in lone.records if r.sensor == "camera"][0].llr[k]
    assert llr_high > 0 and llr_high == pytest.approx(llr_lone, rel=0.15)
    # the certain sponges are barely affected by the ambiguous count
    assert high.prob("sponge_0", "table:top") > 0.99
    assert high.expected_class_count("sponge")[k] == pytest.approx(
        sum(high.prob(f"sponge_{i}", "table:top") for i in range(3)))


def test_tag_read_is_identity_specific_strong_evidence():
    bel = _belief(_sponges(1) + [ItemSpec("clamp_1", "clamp", "instrument_steel_polished", "table:top")])
    bel.apply_workflow_event("clamp_1", "table:top", "bin:inside", 0.6, "wf", 0.0)
    b0 = bel.items["clamp_1"].b.copy()
    s0 = bel.items["sponge_0"].b.copy()
    k = bel.sidx["table:top"]
    _cam(bel, "clamp", "table:top", pd=0.0, count=0.0, tags=[("clamp_1", k)], t=3.0)
    post = bel.items["clamp_1"].b
    j = bel.sidx["bin:inside"]
    assert post[k] / post[j] == pytest.approx(bel.tag_lr * b0[k] / b0[j], rel=1e-9)   # likelihood ratio 25
    assert post[k] > 0.95
    assert np.allclose(bel.items["sponge_0"].b, s0)            # other items untouched
    assert bel.items["clamp_1"].last_seen_t == 3.0
    assert bel.records[-1].sensor == "camera_tag" and bel.records[-1].evidence_id == "cam:1:tag"


def test_predict_leak_rates():
    bel = _belief(initial={"clamp_1": "table:top"})
    bel.predict(10.0)
    b = bel.items["clamp_1"].b
    lost = 1 - np.exp(-bel.leak * 10.0)
    assert b[bel.sidx["table:top"]] == pytest.approx((1 - 5 * ItemBelief.FLOOR) * (1 - lost), rel=1e-4)
    # leaked mass goes to nearby floor/hand slots, never to far or unrelated slots
    assert b[bel.sidx["floor:near_table"]] > b[bel.sidx["hand:nurse"]] > 2 * ItemBelief.FLOOR
    assert b[bel.sidx["far:shelf"]] < 1.01 * ItemBelief.FLOOR
    gain = b - ItemBelief.FLOOR
    assert gain[bel.sidx["bin:inside"]] < 0.1 * gain[bel.sidx["hand:nurse"]]    # only via the hand, second order
    # items in hands are transient: much faster leak
    hand = _belief(initial={"clamp_1": "hand:nurse"})
    hand.predict(10.0)
    assert hand.prob("clamp_1", "hand:nurse") == pytest.approx(np.exp(-hand.hand_leak * 10.0), abs=0.01)
    # the temporal model can be ablated
    frozen = _belief(initial={"clamp_1": "hand:nurse"}, use_temporal_model=False)
    frozen.predict(10.0)
    assert frozen.prob("clamp_1", "hand:nurse") > 0.99


def test_radar_fabric_and_acoustic_evidence():
    bel = _belief(initial={"clamp_1": "table:top"})
    bel.apply_workflow_event("clamp_1", "table:top", "bin:inside", 0.6, "wf", 0.0)
    p0 = bel.prob("clamp_1", "bin:inside")
    pd = np.zeros(bel.n)
    pd[bel.sidx["bin:inside"]] = 1.0
    hits = np.zeros(bel.n)
    hits[bel.sidx["bin:inside"]] = 2
    bel.update_radar_fabric(pd, hits, "radar:1", 1.0)
    assert bel.prob("clamp_1", "bin:inside") > p0
    # acoustic: measured energy matching "item present" raises the posterior
    ac = _belief(initial={"clamp_1": "table:top"})
    ac.apply_workflow_event("clamp_1", "table:top", "bin:inside", 0.6, "wf", 0.0)

    def expected(others, include=None):
        return 0.1 + (0.5 if include == "clamp_1" else 0.0)

    ac.update_acoustic([ac.sidx["bin:inside"]], 0.6, expected, "ac:1", 1.0)
    assert ac.prob("clamp_1", "bin:inside") > p0 + 0.2
    assert ac.records[-1].sensor == "acoustic"


def test_poisson_logpmf_matches_definition():
    from math import exp, factorial, log
    for n, lam in [(0, 0.5), (3, 2.0), (7, 4.5)]:
        assert poisson_logpmf(n, lam) == pytest.approx(log(lam ** n * exp(-lam) / factorial(n)))
    assert CLASSES[0] == "sponge"
