"""Tamper-evident provenance graph for item custody claims.

Model (aligned with W3C PROV-O so it can be exported as PROV-JSON):

* ``Evidence``  (prov:Entity)   - a sensor observation summary or a workflow
  report, with its source, stamp, the robot pose estimate and a content hash;
* ``Belief``    (prov:Entity)   - a snapshot of an item's posterior;
* ``Verdict``   (prov:Entity)   - the robot's answer to a claim
  (VERIFIED / REFUTED / ABSTAIN + reason), ``wasDerivedFrom`` the evidence
  whose log-likelihood contributions moved the posterior most;
* ``Activity``  (prov:Activity) - a perception update or a decision step;
* ``Agent``     (prov:Agent)    - the robot, staff roles, the OR log system.

Every node is appended to a hash chain (``prev_hash`` -> ``hash``) so that
post-hoc edits are detectable (``verify_chain``).  ``explain(verdict)``
returns the supporting and contradicting evidence, making the robot's
estimate *defensible* to a human reviewer.

Custody chains: for each item the graph also keeps the sequence of verified
holder transitions, and ``handoff_chain(item)`` reports gaps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    ABSTAIN = "ABSTAIN"


@dataclass
class Node:
    id: str
    kind: str                     # evidence | belief | verdict | activity | agent
    t: float
    attrs: dict[str, Any]
    prev_hash: str = ""
    hash: str = ""


@dataclass
class Edge:
    src: str
    dst: str
    rel: str                      # wasDerivedFrom | wasGeneratedBy | wasAttributedTo | used | contradicts
    weight: float = 0.0


def _canon(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return [round(float(x), 6) for x in o.ravel()]
    if isinstance(o, np.bool_):                 # not JSON-serialisable; must hash like a Python bool
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    if isinstance(o, Enum):
        return o.value
    return o


class ProvenanceGraph:
    def __init__(self, robot_id: str = "medortrace_robot_0"):
        self.nodes: dict[str, Node] = {}
        self.order: list[str] = []
        self.edges: list[Edge] = []
        self.head = "GENESIS"
        self.robot_id = robot_id
        self.add("agent:" + robot_id, "agent", 0.0, {"type": "prov:SoftwareAgent", "role": "verifier"})
        self.custody: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    def add(self, nid: str, kind: str, t: float, attrs: dict[str, Any]) -> Node:
        if nid in self.nodes:
            return self.nodes[nid]
        n = Node(nid, kind, float(t), _canon(attrs), prev_hash=self.head)
        payload = json.dumps({"id": nid, "kind": kind, "t": round(float(t), 6), "attrs": n.attrs,
                              "prev": self.head}, sort_keys=True)
        n.hash = hashlib.sha256(payload.encode()).hexdigest()
        self.nodes[nid] = n
        self.order.append(nid)
        self.head = n.hash
        return n

    def link(self, src: str, dst: str, rel: str, weight: float = 0.0) -> None:
        self.edges.append(Edge(src, dst, rel, float(weight)))

    def verify_chain(self) -> bool:
        """Re-hash every node in order, then check that the *derived* structures are
        the ones the hashed nodes commit to: each verdict's explanation edges equal
        the evidence list (ids, weights, evidence hashes) in its hashed attributes,
        each evidence node is attributed to the agent named in its attributes, and
        the custody index is exactly the VERIFIED verdicts.  (Tail truncation is only
        detectable against an externally anchored head - see ``to_prov_json``.)"""
        prev = "GENESIS"
        for nid in self.order:
            n = self.nodes[nid]
            payload = json.dumps({"id": n.id, "kind": n.kind, "t": round(n.t, 6), "attrs": n.attrs,
                                  "prev": prev}, sort_keys=True)
            if n.prev_hash != prev or hashlib.sha256(payload.encode()).hexdigest() != n.hash:
                return False
            prev = n.hash
        if prev != self.head:
            return False
        derived = {}
        attributed = {}
        for e in self.edges:
            if e.rel in ("wasDerivedFrom", "contradicts"):
                derived.setdefault(e.src, []).append((e.dst, round(e.weight, 6)))
            elif e.rel == "wasAttributedTo":
                attributed.setdefault(e.src, set()).add(e.dst)
        custody: dict[str, list] = {}
        for nid in self.order:
            n = self.nodes[nid]
            if n.kind == "verdict" and "evidence" in n.attrs:
                want = sorted((ev["id"], round(float(ev["w"]), 6)) for ev in n.attrs["evidence"])
                if sorted(derived.get(nid, [])) != want:
                    return False
                if any(ev["id"] not in self.nodes or self.nodes[ev["id"]].hash != ev["hash"]
                       for ev in n.attrs["evidence"]):
                    return False
                c = n.attrs.get("claim", {})
                if n.attrs.get("verdict") == Verdict.VERIFIED.value and c.get("item_id"):
                    custody.setdefault(c["item_id"], []).append(
                        {"t": c.get("t_ref", n.t), "slot": c.get("slot_id"), "verdict_id": nid})
            if n.kind == "evidence" and "attributed_to" in n.attrs:
                if "agent:" + str(n.attrs["attributed_to"]) not in attributed.get(nid, set()):
                    return False
        return _canon(custody) == _canon(self.custody)

    # ------------------------------------------------------------------
    def add_evidence(self, eid: str, t: float, sensor: str, summary: dict, pose: np.ndarray | None = None,
                     attributed_to: str | None = None) -> None:
        attrs = {"sensor": sensor, **summary, "attributed_to": attributed_to or self.robot_id}
        if pose is not None:
            attrs["robot_pose_est"] = pose
        digest = hashlib.sha256(json.dumps(_canon(attrs), sort_keys=True).encode()).hexdigest()[:16]
        attrs["content_digest"] = digest
        self.add(eid, "evidence", t, attrs)
        self.link(eid, "agent:" + (attributed_to or self.robot_id), "wasAttributedTo")

    def add_verdict(self, vid: str, t: float, claim: dict, verdict: Verdict, posterior: float, reason: str,
                    contributions: list[tuple[str, float]]) -> None:
        # the explanation (which evidence supported / contradicted the verdict, and the
        # hashes of those evidence nodes) is part of the hashed verdict node
        ev = [{"id": eid, "w": round(float(w), 6), "hash": self.nodes[eid].hash}
              for eid, w in contributions if eid in self.nodes]
        self.add(vid, "verdict", t, {"claim": claim, "verdict": verdict.value, "posterior": posterior,
                                     "reason": reason, "evidence": ev})
        for e in ev:
            self.link(vid, e["id"], "wasDerivedFrom" if e["w"] >= 0 else "contradicts", e["w"])
        self.link(vid, "agent:" + self.robot_id, "wasAttributedTo")
        item = claim.get("item_id")
        if item and verdict == Verdict.VERIFIED:
            self.custody.setdefault(item, []).append({"t": claim.get("t_ref", t), "slot": claim.get("slot_id"),
                                                      "verdict_id": vid})

    # ------------------------------------------------------------------
    def explain(self, vid: str, k: int = 5) -> dict:
        sup = sorted([e for e in self.edges if e.src == vid and e.rel == "wasDerivedFrom"], key=lambda e: -e.weight)[:k]
        con = sorted([e for e in self.edges if e.src == vid and e.rel == "contradicts"], key=lambda e: e.weight)[:k]
        fmt = lambda es: [{"evidence": e.dst, "weight": e.weight, **self.nodes[e.dst].attrs} for e in es]
        return {"verdict": self.nodes[vid].attrs, "supporting": fmt(sup), "contradicting": fmt(con)}

    def handoff_chain(self, item_id: str) -> dict:
        chain = sorted(self.custody.get(item_id, []), key=lambda c: c["t"])
        gaps = []
        for a, b in zip(chain[:-1], chain[1:]):
            if b["t"] - a["t"] > 120.0:
                gaps.append({"from": a, "to": b, "gap_s": b["t"] - a["t"]})
        return {"item": item_id, "chain": chain, "gaps": gaps, "complete": len(chain) > 0 and not gaps}

    def to_prov_json(self) -> dict:
        ent, act, ag = {}, {}, {}
        for nid in self.order:
            n = self.nodes[nid]
            rec = {
                "prov:type": n.kind,
                "mot:t": n.t,
                "mot:hash": n.hash,
                "mot:prev": n.prev_hash,
                **{f"mot:{k}": v for k, v in n.attrs.items()},
            }
            (ag if n.kind == "agent" else act if n.kind == "activity" else ent)[nid] = rec
        rels = {"wasDerivedFrom": {}, "wasAttributedTo": {}, "mot:contradicts": {}}
        for i, e in enumerate(self.edges):
            key = e.rel if e.rel in rels else "mot:contradicts"
            if e.rel == "wasDerivedFrom":
                rels[key][f"_:d{i}"] = {"prov:generatedEntity": e.src, "prov:usedEntity": e.dst, "mot:weight": e.weight}
            elif e.rel == "wasAttributedTo":
                rels[key][f"_:a{i}"] = {"prov:entity": e.src, "prov:agent": e.dst}
            else:
                rels[key][f"_:c{i}"] = {"mot:verdict": e.src, "mot:evidence": e.dst, "mot:weight": e.weight}
        return {"prefix": {"mot": "https://medortrace.example/ns#"}, "entity": ent, "activity": act,
                "agent": ag, **rels, "mot:head": self.head, "mot:n_nodes": len(self.order)}
