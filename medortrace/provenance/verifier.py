"""Claim verification with explicit abstention.

The verifier answers each claim with VERIFIED / REFUTED / ABSTAIN.  It never
turns weak evidence into an assertion:

* a verdict requires *direct* sensor evidence about the claimed slot (or an
  identity read of the item elsewhere) gathered after the reference time -
  a workflow log entry alone is not enough (``require_direct_evidence``);
* posteriors are fixed-lag smoothed: evidence gathered between ``t_ref`` and
  the decision time is applied to the belief snapshot at ``t_ref`` as long as
  no reported event moved the item in between;
* the decision thresholds (``tau_verify``/``tau_refute``) are applied to a
  *calibrated* posterior (see ``eval.calibration``), and are inflated when
  the safety supervisor reports degraded localisation or sensing.

An ABSTAIN is a request for human confirmation - it is the correct, safe
answer when the physics does not allow the robot to know (e.g. a sponge under
a drape is acoustically and radar-transparent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from medortrace.belief.items import ItemBelief
from medortrace.provenance.graph import ProvenanceGraph, Verdict
from medortrace.world.workflow import Claim


@dataclass
class OpenClaim:
    claim: Claim
    snapshot: np.ndarray | None = None
    acc_llr: np.ndarray | None = None
    rec_ptr: int = 0
    direct: bool = False
    moved: bool = False
    moved_t: float = float("inf")
    contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class VerdictRecord:
    claim_id: str
    t: float
    verdict: Verdict
    posterior: float
    reason: str
    item_id: str
    slot_id: str
    t_ref: float
    kind: str
    map_slot: str
    direct: bool


class ClaimVerifier:
    def __init__(self, belief: ItemBelief, prov: ProvenanceGraph, tau_verify: float = 0.9,
                 tau_refute: float = 0.1, require_direct_evidence: bool = True, early_decision: bool = True):
        self.belief = belief
        self.prov = prov
        self.tau_v = tau_verify
        self.tau_r = tau_refute
        self.require_direct = require_direct_evidence
        self.early = early_decision
        self.open: dict[str, OpenClaim] = {}
        self.done: dict[str, VerdictRecord] = {}
        self.degraded = False

    def add_claim(self, c: Claim) -> None:
        if c.id in self.open or c.id in self.done:
            return
        self.open[c.id] = OpenClaim(c)

    def note_item_event(self, item_id: str, t_event: float) -> None:
        """A reported move after t_ref invalidates smoothing for claims on that item."""
        for oc in self.open.values():
            if oc.claim.item_id == item_id and t_event > oc.claim.t_ref + 1.0:
                oc.moved = True
                oc.moved_t = min(oc.moved_t, t_event)

    def pending(self) -> list[Claim]:
        return [oc.claim for oc in self.open.values()]

    def step(self, t: float) -> list[VerdictRecord]:
        out = []
        recs = self.belief.records
        for cid in list(self.open):
            oc = self.open[cid]
            c = oc.claim
            if t < c.t_ref:
                continue
            k = self.belief.sidx.get(c.slot_id)
            if k is None:
                continue
            st = self.belief.items[c.item_id]
            if oc.snapshot is None:
                oc.snapshot = st.b.copy()
                oc.acc_llr = np.zeros_like(st.b)
                oc.rec_ptr = len(recs)
            # accumulate sensor evidence recorded since t_ref
            for r in recs[oc.rec_ptr:]:
                if r.item_id != c.item_id or r.sensor == "workflow" or r.t > oc.moved_t:
                    continue
                oc.acc_llr += r.llr
                if abs(r.llr[k]) > 0.05:
                    oc.direct = True
                    oc.contributions[r.evidence_id] = oc.contributions.get(r.evidence_id, 0.0) + float(r.llr[k])
                if r.sensor == "camera_tag" and r.llr.max() > 1.0:
                    oc.direct = True
                    oc.contributions[r.evidence_id] = oc.contributions.get(r.evidence_id, 0.0) + float(r.llr[k] - r.llr.max())
            oc.rec_ptr = len(recs)
            # fixed-lag smoothing: snapshot at t_ref x evidence gathered until the
            # item was next reported moved (later evidence is about a different state)
            lb = np.log(oc.snapshot) + oc.acc_llr
            lb -= lb.max()
            post = np.exp(lb)
            post /= post.sum()
            p = float(post[k])
            tau_v = min(0.99, self.tau_v + (0.05 if self.degraded else 0.0))
            tau_r = max(0.01, self.tau_r - (0.05 if self.degraded else 0.0))
            decided = None
            if oc.direct or not self.require_direct:
                if p >= tau_v:
                    decided = (Verdict.VERIFIED, "posterior above verify threshold with direct evidence")
                elif p <= tau_r:
                    decided = (Verdict.REFUTED, "posterior below refute threshold with direct evidence")
            if decided is None and t >= c.t_due:
                reason = "no direct sensor evidence of the claimed slot" if not oc.direct else \
                    f"ambiguous evidence (p={p:.2f})"
                decided = (Verdict.ABSTAIN, reason)
            if decided is None or (not self.early and t < c.t_due):
                continue
            ms = self.belief.slot_ids[int(np.argmax(post))]
            vr = VerdictRecord(cid, t, decided[0], p, decided[1], c.item_id, c.slot_id, c.t_ref, c.kind, ms, oc.direct)
            self.done[cid] = vr
            del self.open[cid]
            contrib = sorted(oc.contributions.items(), key=lambda kv: -abs(kv[1]))[:8]
            self.prov.add_verdict(f"verdict:{cid}", t, {"claim_id": cid, "item_id": c.item_id, "slot_id": c.slot_id,
                                                         "t_ref": c.t_ref, "kind": c.kind},
                                  vr.verdict, p, vr.reason, contrib)
            out.append(vr)
        return out

    def urgency(self, t: float) -> dict[str, float]:
        """Per-item urgency weight for active perception (pending, soon-due claims)."""
        u: dict[str, float] = {}
        for oc in self.open.values():
            c = oc.claim
            if t < c.t_ref - 10.0:
                continue
            slack = max(1.0, c.t_due - t)
            w = (2.0 if not oc.direct else 0.7) * (10.0 / slack + 0.5)
            u[c.item_id] = u.get(c.item_id, 0.0) + w
        return u

    def urgent_slots(self, t: float) -> dict[str, float]:
        u: dict[str, float] = {}
        for oc in self.open.values():
            c = oc.claim
            if t < c.t_ref - 10.0 or oc.direct:
                continue
            u[c.slot_id] = u.get(c.slot_id, 0.0) + 10.0 / max(1.0, c.t_due - t)
        return u
