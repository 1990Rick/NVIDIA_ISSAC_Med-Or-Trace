"""Tamper-evident provenance graph: hash chain, explanations, PROV-JSON export."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from medortrace.provenance.graph import ProvenanceGraph, Verdict


def _graph() -> ProvenanceGraph:
    g = ProvenanceGraph()
    g.add_evidence("wf:1", 1.0, "workflow", {"type": "handoff", "item": "clamp_1", "dst": "mayo:top"},
                   attributed_to="scrub_nurse")
    g.add_evidence("cam:1", 2.0, "camera", {"n_det": 2, "assoc": [3, -1]}, pose=np.array([1.0, 2.0, 0.5]))
    g.add_evidence("cam:2", 2.2, "camera", {"n_det": 0, "assoc": []}, pose=np.array([1.1, 2.0, 0.5]))
    g.add_evidence("radar:1", 2.4, "radar", {"metal_hits": [0.0, 1.0]})
    g.add_evidence("ac:1", 2.6, "acoustic", {"region": "patient_drape", "energy": 0.31, "nlos": False})
    g.add_evidence("lidar:1", 2.8, "lidar", {"n_points": 900})
    claim = {"claim_id": "c1", "item_id": "clamp_1", "slot_id": "mayo:top", "t_ref": 1.5, "kind": "handoff"}
    g.add_verdict("verdict:c1", 3.0, claim, Verdict.VERIFIED, 0.97, "posterior above verify threshold",
                  [("cam:1", 0.5), ("radar:1", 2.0), ("cam:2", -1.0), ("ac:1", -3.0), ("lidar:1", 1.0),
                   ("not_recorded", 9.0)])
    return g


def test_chain_links_and_verifies():
    g = _graph()
    assert g.verify_chain()
    prev = "GENESIS"
    for nid in g.order:
        n = g.nodes[nid]
        assert n.prev_hash == prev and len(n.hash) == 64
        prev = n.hash
    assert g.head == prev
    # re-adding an existing id is idempotent (no fork, no new link)
    n_before, head = len(g.order), g.head
    g.add("cam:1", "evidence", 99.0, {"x": 1})
    assert len(g.order) == n_before and g.head == head and g.verify_chain()


@pytest.mark.parametrize("field", ["attr_value", "attr_added", "attr_removed", "t", "kind", "prev_hash"])
def test_tampering_any_node_breaks_chain(field):
    base = _graph()
    for nid in base.order:
        g = copy.deepcopy(base)
        n = g.nodes[nid]
        if field == "attr_value":
            k = sorted(n.attrs)[0]
            v = n.attrs[k]
            n.attrs[k] = (v + 1) if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{v}!"
        elif field == "attr_added":
            n.attrs["injected"] = True
        elif field == "attr_removed":
            n.attrs.pop(sorted(n.attrs)[-1])
        elif field == "t":
            n.t += 0.5
        elif field == "kind":
            n.kind = "verdict" if n.kind != "verdict" else "evidence"
        else:
            n.prev_hash = "0" * 64
        assert not g.verify_chain(), (field, nid)


def test_rehashing_a_tampered_node_is_still_detected():
    """An attacker who edits a node and recomputes *its* hash still breaks the next link."""
    g = _graph()
    nid = g.order[2]
    n = g.nodes[nid]
    n.attrs["n_det"] = 7
    payload = json.dumps({"id": n.id, "kind": n.kind, "t": round(n.t, 6), "attrs": n.attrs, "prev": n.prev_hash},
                         sort_keys=True)
    n.hash = hashlib.sha256(payload.encode()).hexdigest()
    assert not g.verify_chain()
    # reordering or dropping nodes is detected too
    g2 = _graph()
    g2.order[1], g2.order[2] = g2.order[2], g2.order[1]
    assert not g2.verify_chain()
    g3 = _graph()
    g3.order.pop(3)
    assert not g3.verify_chain()


def test_numpy_payloads_are_canonical_and_hashable():
    a, b = ProvenanceGraph(), ProvenanceGraph()
    a.add_evidence("e", 1.0, "radar", {"hits": np.array([1.0, 2.0]), "n": np.int64(3), "x": np.float32(0.5),
                                       "flag": np.bool_(True), "nested": {"v": np.float64(1 / 3)}})
    b.add_evidence("e", 1.0, "radar", {"hits": [1.0, 2.0], "n": 3, "x": 0.5, "flag": True,
                                       "nested": {"v": 1 / 3}})
    assert a.head == b.head                                    # numpy vs python types hash identically
    assert a.nodes["e"].attrs["content_digest"] == b.nodes["e"].attrs["content_digest"]
    assert a.verify_chain() and json.dumps(a.to_prov_json())


def test_explain_orders_support_and_contradiction():
    g = _graph()
    ex = g.explain("verdict:c1")
    assert ex["verdict"]["verdict"] == "VERIFIED" and ex["verdict"]["posterior"] == pytest.approx(0.97)
    assert [s["evidence"] for s in ex["supporting"]] == ["radar:1", "lidar:1", "cam:1"]      # strongest first
    assert [c["evidence"] for c in ex["contradicting"]] == ["ac:1", "cam:2"]                  # most negative first
    assert ex["supporting"][0]["weight"] == 2.0 and ex["supporting"][0]["sensor"] == "radar"
    assert ex["contradicting"][0]["region"] == "patient_drape"
    assert "not_recorded" not in json.dumps(ex)               # unknown evidence ids are never linked
    assert [s["evidence"] for s in g.explain("verdict:c1", k=1)["supporting"]] == ["radar:1"]


def test_prov_json_export():
    g = _graph()
    doc = json.loads(json.dumps(g.to_prov_json()))
    assert doc["prefix"]["mot"].startswith("https://")
    assert set(doc["entity"]) == {"wf:1", "cam:1", "cam:2", "radar:1", "ac:1", "lidar:1", "verdict:c1"}
    assert set(doc["agent"]) == {"agent:medortrace_robot_0"}
    for nid, rec in {**doc["entity"], **doc["agent"]}.items():
        assert rec["mot:hash"] == g.nodes[nid].hash and rec["mot:prev"] == g.nodes[nid].prev_hash
        assert rec["prov:type"] == g.nodes[nid].kind
    assert doc["entity"]["cam:1"]["mot:robot_pose_est"] == [1.0, 2.0, 0.5]
    derived = {(r["prov:generatedEntity"], r["prov:usedEntity"]): r["mot:weight"]
               for r in doc["wasDerivedFrom"].values()}
    assert derived == {("verdict:c1", "cam:1"): 0.5, ("verdict:c1", "radar:1"): 2.0, ("verdict:c1", "lidar:1"): 1.0}
    contra = {(r["mot:verdict"], r["mot:evidence"]) for r in doc["mot:contradicts"].values()}
    assert contra == {("verdict:c1", "cam:2"), ("verdict:c1", "ac:1")}
    attributed = {(r["prov:entity"], r["prov:agent"]) for r in doc["wasAttributedTo"].values()}
    assert ("wf:1", "agent:scrub_nurse") in attributed and ("cam:1", "agent:medortrace_robot_0") in attributed
    assert ("verdict:c1", "agent:medortrace_robot_0") in attributed


def test_custody_chain_and_gaps():
    g = ProvenanceGraph()
    for vid, t, v in [("v1", 10.0, Verdict.VERIFIED), ("v2", 50.0, Verdict.ABSTAIN), ("v3", 200.0, Verdict.VERIFIED)]:
        g.add_verdict(vid, t, {"item_id": "specimen_1", "slot_id": f"s_{vid}", "t_ref": t}, v, 0.9, "", [])
    hc = g.handoff_chain("specimen_1")
    assert [c["verdict_id"] for c in hc["chain"]] == ["v1", "v3"]          # only VERIFIED links custody
    assert len(hc["gaps"]) == 1 and hc["gaps"][0]["gap_s"] == pytest.approx(190.0) and not hc["complete"]
    assert g.handoff_chain("nothing") == {"item": "nothing", "chain": [], "gaps": [], "complete": False}


def test_explanations_custody_and_head_are_bound_to_the_chain():
    """Relinking a verdict's evidence, forging custody or dropping the newest node all
    break verify_chain, although no node payload was touched."""
    from medortrace.provenance.graph import Edge, ProvenanceGraph, Verdict

    def make():
        g = ProvenanceGraph()
        g.add_evidence("cam:1", 1.0, "camera", {"n": 1}, attributed_to="medortrace_robot_0")
        g.add_evidence("cam:2", 2.0, "camera", {"n": 0})
        g.add_verdict("verdict:c1", 3.0, {"item_id": "clamp_1", "slot_id": "mayo:top", "t_ref": 2.5},
                      Verdict.VERIFIED, 0.95, "ok", [("cam:1", 1.2), ("cam:2", -0.3)])
        return g

    g = make()
    assert g.verify_chain()
    assert g.to_prov_json()["mot:head"] == g.head
    g.edges = [e for e in g.edges if not (e.src == "verdict:c1" and e.dst == "cam:2")]        # hide contradiction
    assert not g.verify_chain()
    g = make()
    g.edges = [Edge(e.src, e.dst, e.rel, 5.0) if e.dst == "cam:1" and e.src == "verdict:c1" else e for e in g.edges]
    assert not g.verify_chain()                                                              # inflate support
    g = make()
    g.custody.setdefault("sponge_1", []).append({"t": 1.0, "slot": "field:top", "verdict_id": "verdict:c1"})
    assert not g.verify_chain()                                                              # forged custody
    g = make()
    last = g.order.pop()
    del g.nodes[last]
    assert not g.verify_chain()                                   # truncated: the stored head no longer matches
