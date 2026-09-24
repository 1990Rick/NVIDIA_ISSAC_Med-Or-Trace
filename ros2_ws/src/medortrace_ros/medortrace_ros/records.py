"""Publishable records extracted from the autonomy stack (no ROS imports).

``autonomy_node`` turns the stack's state after every control tick into these
plain dataclasses; ``convert`` maps them 1:1 to ``medortrace_msgs``.  Keeping
the extraction ROS-free lets the same code feed rosbag-free audit logs and be
tested without a ROS installation.

Provenance events are streamed incrementally (:class:`ProvenanceCursor`) and
carry everything needed to re-verify the hash chain on the receiving side
(:func:`verify_provenance_stream` re-implements
``ProvenanceGraph.verify_chain`` over the message stream).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from medortrace.data.writer import MODE_CODE

VERDICT_CODE = {"VERIFIED": 0, "REFUTED": 1, "ABSTAIN": 2}
VERDICT_NAME = {v: k for k, v in VERDICT_CODE.items()}
MODE_NAME = {v: k for k, v in MODE_CODE.items()}


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class VerdictOut:
    claim_id: str
    kind: str
    item_id: str
    slot_id: str
    t_ref: float
    t: float
    verdict: str
    posterior: float
    reason: str
    direct: bool
    map_slot: str
    supporting: list[tuple[str, float]] = field(default_factory=list)
    contradicting: list[tuple[str, float]] = field(default_factory=list)
    node_id: str = ""
    node_hash: str = ""

    @property
    def code(self) -> int:
        return VERDICT_CODE[self.verdict]


def verdict_out(stack, vr, k: int = 8) -> VerdictOut:
    """VerdictRecord + its provenance evidence (ProvenanceGraph.explain)."""
    vid = f"verdict:{vr.claim_id}"
    sup, con, h = [], [], ""
    if vid in stack.prov.nodes:
        ex = stack.prov.explain(vid, k)
        sup = [(e["evidence"], float(e["weight"])) for e in ex["supporting"]]
        con = [(e["evidence"], float(e["weight"])) for e in ex["contradicting"]]
        h = stack.prov.nodes[vid].hash
    v = vr.verdict.value if hasattr(vr.verdict, "value") else str(vr.verdict)
    return VerdictOut(vr.claim_id, vr.kind, vr.item_id, vr.slot_id, float(vr.t_ref), float(vr.t), v,
                      float(vr.posterior), vr.reason, bool(vr.direct), vr.map_slot, sup, con, vid, h)


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class ProvenanceOut:
    index: int
    node_id: str
    kind: str
    t: float
    sensor: str
    attributed_to: str
    content_digest: str
    prev_hash: str
    hash: str
    json_attrs: str


class ProvenanceCursor:
    """Yields the provenance nodes appended since the previous poll, in chain order."""

    def __init__(self):
        self.node_ptr = 0
        self.edge_ptr = 0
        self._attrib: dict[str, str] = {}

    def poll(self, prov) -> list[ProvenanceOut]:
        for e in prov.edges[self.edge_ptr:]:
            if e.rel == "wasAttributedTo":
                self._attrib.setdefault(e.src, e.dst.removeprefix("agent:"))
        self.edge_ptr = len(prov.edges)
        out = []
        for i in range(self.node_ptr, len(prov.order)):
            nid = prov.order[i]
            n = prov.nodes[nid]
            sensor = str(n.attrs.get("sensor", "")) if n.kind == "evidence" else (
                "verifier" if n.kind == "verdict" else "")
            out.append(ProvenanceOut(i, nid, n.kind, float(n.t), sensor, self._attrib.pop(nid, ""),
                                     str(n.attrs.get("content_digest", "")), n.prev_hash, n.hash,
                                     json.dumps(n.attrs, sort_keys=True)))
        self.node_ptr = len(prov.order)
        return out


def verify_provenance_stream(events: list) -> bool:
    """Re-verify the hash chain from received ProvenanceEvent-like records (index order, starting at 0)."""
    prev = "GENESIS"
    for k, e in enumerate(sorted(events, key=lambda e: e.index)):
        if e.index != k or e.prev_hash != prev:
            return False
        payload = json.dumps({"id": e.node_id, "kind": e.kind, "t": round(float(e.t), 6),
                              "attrs": json.loads(e.json_attrs), "prev": prev}, sort_keys=True)
        if hashlib.sha256(payload.encode()).hexdigest() != e.hash:
            return False
        prev = e.hash
    return True


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class SafetyOut:
    t: float
    mode: str
    reasons: list[str]
    category: str
    collision_prob: float
    pred_clearance: float
    human_clearance: float
    loc_std: float
    nis: float
    lidar_age: float
    path_entropy: float
    contact_force: float
    battery_frac: float
    operator_request_open: bool
    handover_requests: int
    transitions: int

    @property
    def code(self) -> int:
        return MODE_CODE[self.mode]


def safety_out(stack, bundle) -> SafetyOut:
    tel = stack.telemetry[-1]
    sup = stack.sup
    mode = sup.mode.value
    cat = sup.events[-1].category if (sup.events and mode != "NOMINAL") else ""
    batt = bundle.battery_wh if bundle.battery_wh is not None else stack.battery_cap
    return SafetyOut(
        t=float(tel.t), mode=mode, reasons=list(sup.last_reasons), category=cat,
        collision_prob=float(tel.mpc_coll_prob), pred_clearance=float(tel.mpc_min_clear),
        human_clearance=float(tel.human_clearance_est), loc_std=float(tel.pose_std), nis=float(tel.nis),
        lidar_age=float(tel.t - stack.last_lidar_t) if "lidar" in stack.modalities else 0.0,
        path_entropy=float(tel.path_entropy),
        contact_force=float(bundle.contact.force_n) if bundle.contact is not None else 0.0,
        battery_frac=float(batt / stack.battery_cap), operator_request_open=bool(stack.operator_request_open),
        handover_requests=int(sup.handover_requests), transitions=len(sup.events))


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class ItemOut:
    item_id: str
    cls: str
    criticality: float
    probabilities: np.ndarray
    map_slot: str
    map_probability: float
    entropy_bits: float
    last_seen: float | None


@dataclass
class BeliefsOut:
    t: float
    slot_ids: list[str]
    slot_kinds: list[str]
    items: list[ItemOut]


def beliefs_out(stack) -> BeliefsOut:
    ib = stack.items
    items = []
    for iid, st in ib.items.items():
        ms, mp = ib.map_slot(iid)
        items.append(ItemOut(iid, st.spec.cls, float(st.spec.criticality), st.b.astype(np.float32), ms, mp,
                             ib.entropy(iid), float(st.last_seen_t) if st.last_seen_t > -1e8 else None))
    return BeliefsOut(float(stack.t), list(ib.slot_ids), [s.kind for s in ib.slots], items)


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class NbvOut:
    t: float
    pose: np.ndarray
    path: np.ndarray
    target_slot: str
    probe_region: str
    score: float
    breakdown: dict[str, float]


def nbv_out(stack) -> NbvOut | None:
    g = stack.goal
    if g is None:
        return None
    bd = {str(k): float(v) for k, v in (g.breakdown or {}).items() if np.isscalar(v)}
    return NbvOut(float(stack.t), np.asarray(g.pose, float), np.asarray(g.path, float).reshape(-1, 2),
                  g.target_slot or "", g.probe_region or "", float(g.score), bd)


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class GridOut:
    t: float
    res: float
    origin: np.ndarray        # world xy of cell (0, 0) lower-left corner
    data: np.ndarray          # (nx, ny), indexed [ix, iy]


def uncertainty_out(stack) -> GridOut:
    occ = stack.occ
    return GridOut(float(stack.t), float(occ.res), np.asarray(occ.grid2d.origin, float), occ.uncertainty_field())


def occupancy_out(stack) -> GridOut:
    """Column occupancy probability in [0, 1] over the robot's height band (prior map included)."""
    occ = stack.occ
    return GridOut(float(stack.t), float(occ.res), np.asarray(occ.grid2d.origin, float),
                   occ.column_occupancy().astype(np.float32))
