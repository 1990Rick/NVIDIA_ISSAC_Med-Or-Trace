"""Item-custody belief: a discrete Bayes filter per critical item over slots.

State space
    For every item ``i`` a categorical distribution ``b_i(s)`` over the slot set
    ``S`` (surfaces, containers, under-drape pockets, floor zones, staff hands,
    and ``elsewhere``).

Process model
    * *Persistence with leak*: items on surfaces stay with high probability; a
      small hazard moves mass to nearby floor zones (unlogged drops) and to
      nearby hands (unlogged pickups).  Items in hands are transient.
    * *Workflow events*: a reported event (item, src -> dst, confidence c)
      moves mass ``rho * c`` from ``src`` (or from anywhere, if src is unknown)
      to ``dst``.  ``rho`` < 1 encodes that logs are sometimes wrong.

Observation models (all returned as per-slot log-likelihood ratios so that
contributions can be recorded in the provenance graph)
    * *Camera class counts* (mean-field Poisson): for visible slot ``s`` with
      expected detection probability ``pd_s`` and soft class count ``n_{c,s}``,
      ``lambda_with = pd_s (1 + m_{-i,s}) + fp`` and
      ``lambda_without = pd_s m_{-i,s} + fp`` where ``m_{-i,s}`` is the expected
      number of *other* same-class items at ``s``.  This handles fungible
      items (sponges) without an explicit data-association search.
    * *Tag reads*: identity-specific strong evidence.
    * *Radar through fabric*: metallic items in under-drape pockets.
    * *Acoustic echo energy*: Gaussian likelihood around the expected energy of
      the probed region given the region's contents.

Everything is in log space; beliefs are floored to keep them recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import gammaln

from medortrace.world.scene import ItemSpec, Slot

CLASSES = ["sponge", "clamp", "needle_driver", "specimen", "implant_box"]


@dataclass
class EvidenceRecord:
    evidence_id: str
    t: float
    sensor: str
    item_id: str
    llr: np.ndarray               # per-slot log-likelihood ratio contribution
    note: str = ""


@dataclass
class ItemBeliefState:
    spec: ItemSpec
    b: np.ndarray
    last_direct_obs_t: dict[int, float] = field(default_factory=dict)   # slot idx -> last informative obs time
    last_seen_t: float = -1e9


class ItemBelief:
    FLOOR = 1e-4

    def __init__(self, items: list[ItemSpec], slots: list[Slot], initial: dict[str, str] | None = None,
                 params: dict | None = None):
        p = params or {}
        self.slots = slots
        self.slot_ids = [s.id for s in slots]
        self.sidx = {s: i for i, s in enumerate(self.slot_ids)}
        self.n = len(slots)
        self.leak = p.get("leak_per_s", 2e-4)
        self.hand_leak = p.get("hand_leak_per_s", 0.03)
        self.rho = p.get("event_reliability", 0.85)
        self.fp = p.get("camera_fp_per_slot", 0.02)
        self.pd0 = p.get("camera_pd0", 0.85)
        self.tag_lr = p.get("tag_lr", 25.0)
        self.radar_pd = p.get("radar_pd_metal_fabric", 0.35)
        self.radar_fp = p.get("radar_fp", 0.05)
        self.ac_sigma = p.get("acoustic_sigma", 0.06)
        self.use_temporal = p.get("use_temporal_model", True)
        # Consecutive frames are strongly correlated (same occluder, same glare);
        # treating them as independent makes negative evidence overconfident.
        # Each sensor's per-frame LLR is discounted by an effective-sample factor.
        self.decorrelation = {"camera": 0.3, "camera_tag": 1.0, "radar": 0.15, "acoustic": 0.5}
        self.decorrelation.update(p.get("decorrelation", {}))
        self.q_vis = p.get("visibility_confidence", 0.8)   # P(visibility prediction is right)
        # soft confusion C[true, pred]: expected soft class mass a detection of a
        # true-class item puts on each class (from detector calibration)
        self.confusion = np.asarray(p.get("soft_confusion", np.eye(len(CLASSES))), dtype=float)
        self.items: dict[str, ItemBeliefState] = {}
        for it in items:
            b = np.full(self.n, self.FLOOR)
            s0 = (initial or {}).get(it.id, it.initial_slot)
            b[self.sidx[s0]] = 1.0
            self.items[it.id] = ItemBeliefState(it, b / b.sum())
        # neighbourhood structure for leak: floor zones near each surface
        pos = np.array([s.position[:2] if np.all(np.isfinite(s.position[:2])) else [1e3, 1e3] for s in slots])
        self.floor_idx = [i for i, s in enumerate(slots) if s.kind == "floor"]
        self.hand_idx = [i for i, s in enumerate(slots) if s.kind == "hand"]
        self.else_idx = self.sidx.get("elsewhere")
        self.dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)
        self.records: list[EvidenceRecord] = []

    # ------------------------------------------------------------------
    def update_hand_positions(self, hand_xy: dict[str, np.ndarray]) -> None:
        for name, xy in hand_xy.items():
            k = self.sidx.get(f"hand:{name}")
            if k is not None:
                self.slots[k].position = np.array([xy[0], xy[1], 1.0])
        pos = np.array([s.position[:2] if np.all(np.isfinite(s.position[:2])) else [1e3, 1e3] for s in self.slots])
        self.dist = np.linalg.norm(pos[:, None] - pos[None], axis=2)

    def predict(self, dt: float) -> None:
        if not self.use_temporal:
            return
        for st in self.items.values():
            b = st.b
            nb = b.copy()
            for i in range(self.n):
                s = self.slots[i]
                if b[i] < 1e-6 or s.kind == "elsewhere":
                    continue
                if s.kind == "hand":
                    rate = self.hand_leak
                    targets = [j for j in range(self.n) if j != i and self.slots[j].kind in ("surface", "container", "hand")
                               and self.dist[i, j] < 1.5]
                else:
                    rate = self.leak
                    targets = [j for j in self.floor_idx + self.hand_idx if j != i and self.dist[i, j] < 2.5]
                if not targets:
                    continue
                m = b[i] * (1 - np.exp(-rate * dt))
                nb[i] -= m
                w = np.exp(-self.dist[i, targets])
                nb[targets] += m * w / w.sum()
            st.b = self._norm(nb)

    def apply_workflow_event(self, item_id: str, src: str | None, dst: str | None, conf: float,
                             evidence_id: str, t: float) -> None:
        st = self.items.get(item_id)
        if st is None or dst not in self.sidx:
            return
        k = self.rho * conf
        before = st.b.copy()
        d = self.sidx[dst]
        # Mass that moves: all mass at src, plus a share of the rest (the item may
        # have reached src through an unlogged step - always the case for hands).
        gamma = 1.0 if (src is None or src not in self.sidx or src.startswith("hand:")) else 0.5
        s = self.sidx.get(src, -1)
        move = st.b * (k * gamma)
        if s >= 0:
            move[s] = st.b[s] * k
        move[d] = 0.0
        st.b = st.b - move
        st.b[d] += move.sum()
        st.b = self._norm(st.b)
        self.records.append(EvidenceRecord(evidence_id, t, "workflow", item_id,
                                           np.log(st.b + 1e-12) - np.log(before + 1e-12), f"{src}->{dst}"))

    # ------------------------------------------------------------------
    def update_camera_classes(self, pd_by_class: dict[str, np.ndarray], class_counts: dict[str, np.ndarray],
                              tag_reads: list[tuple[str, int]], evidence_id: str, t: float,
                              glare_expect: dict[str, float] | None = None) -> None:
        """``pd_by_class[c]``: (S,) predicted detection probability of a class-c item at
        each slot (0 where not visible).  ``class_counts[c]``: (S,) soft count of class-c
        detections associated to each slot.  ``tag_reads``: list of (item_id, slot_idx)."""
        by_class: dict[str, list[str]] = {}
        for iid, st in self.items.items():
            by_class.setdefault(st.spec.cls, []).append(iid)
        # expected (soft) detections per class and slot from each class's items
        exp_det = {}
        for cls, iids in by_class.items():
            vp = pd_by_class.get(cls, np.zeros(self.n))
            g = np.mean([(glare_expect or {}).get(j, 0.0) for j in iids])
            exp_det[cls] = vp * (1 - 0.45 * g) * self.q_vis * np.sum([self.items[j].b for j in iids], axis=0)
        ci = {c: k for k, c in enumerate(CLASSES)}
        for cls, iids in by_class.items():
            vp = pd_by_class.get(cls)
            if vp is None:
                continue
            vis = vp > 0.03
            if not vis.any():
                continue
            c = ci[cls]
            n = class_counts.get(cls, np.zeros(self.n))
            # confusion-induced soft counts from items of *other* classes + clutter false positives
            lam_conf = self.fp + sum(self.confusion[ci[o], c] * exp_det[o] for o in by_class if o != cls)
            total = np.sum([self.items[j].b for j in iids], axis=0)
            for iid in iids:
                st = self.items[iid]
                g = (glare_expect or {}).get(iid, 0.0)
                pd = vp * (1 - 0.45 * g) * self.q_vis * self.confusion[c, c]
                m_other = np.maximum(total - st.b, 0.0)
                lam_w = pd * (1 + m_other) + lam_conf
                lam_wo = pd * m_other + lam_conf
                llr = np.where(vis, n * np.log(lam_w / lam_wo) - (lam_w - lam_wo), 0.0)
                llr = np.clip(llr, -6.0, 6.0)
                self._apply(st, llr, evidence_id, t, "camera", vis)
        for iid, k in tag_reads:
            st = self.items.get(iid)
            if st is None:
                continue
            llr = np.zeros(self.n)
            llr[k] = np.log(self.tag_lr)
            self._apply(st, llr, evidence_id + ":tag", t, "camera_tag", llr != 0)
            st.last_seen_t = t

    def update_camera(self, visible_pd: np.ndarray, class_counts, tag_reads, evidence_id, t, glare_expect=None):
        """Class-agnostic visibility variant (same pd for every class)."""
        self.update_camera_classes({c: visible_pd for c in CLASSES}, class_counts, tag_reads, evidence_id, t,
                                   glare_expect)

    def update_radar_fabric(self, slot_pd: np.ndarray, metal_hits: np.ndarray, evidence_id: str, t: float) -> None:
        """Metallic returns through drapes: slot_pd (S,) and hit counts (S,)."""
        vis = slot_pd > 0.03
        if not vis.any():
            return
        metal = [iid for iid, st in self.items.items() if st.spec.metallic]
        total = np.sum([self.items[j].b for j in metal], axis=0) if metal else np.zeros(self.n)
        for iid in metal:
            st = self.items[iid]
            m_other = np.maximum(total - st.b, 0)
            lam_w = slot_pd * self.radar_pd * (1 + m_other) + self.radar_fp
            lam_wo = slot_pd * self.radar_pd * m_other + self.radar_fp
            llr = np.where(vis, metal_hits * np.log(lam_w / lam_wo) - (lam_w - lam_wo), 0.0)
            self._apply(st, np.clip(llr, -4, 4), evidence_id, t, "radar", vis)

    def update_acoustic(self, region_slots: list[int], energy: float, expected_fn, evidence_id: str, t: float) -> None:
        """``expected_fn(contents_materials_weights) -> expected energy`` is the calibrated
        forward model; we compare 'item i in region' vs 'not in region' under mean-field
        expectations of the other items."""
        if not region_slots:
            return
        mask = np.zeros(self.n, dtype=bool)
        mask[region_slots] = True
        p_in = {iid: float(st.b[mask].sum()) for iid, st in self.items.items()}
        for iid, st in self.items.items():
            others = {j: p for j, p in p_in.items() if j != iid}
            mu_w = expected_fn(others, include=iid)
            mu_wo = expected_fn(others, include=None)
            if abs(mu_w - mu_wo) < 1e-4:
                continue
            s2 = self.ac_sigma ** 2
            l_w = -0.5 * (energy - mu_w) ** 2 / s2
            l_wo = -0.5 * (energy - mu_wo) ** 2 / s2
            llr = np.where(mask, np.clip(l_w - l_wo, -5, 5), 0.0)
            self._apply(st, llr, evidence_id, t, "acoustic", mask)

    # ------------------------------------------------------------------
    def _apply(self, st: ItemBeliefState, llr: np.ndarray, eid: str, t: float, sensor: str, observed: np.ndarray):
        if not np.any(llr):
            return
        llr = llr * self.decorrelation.get(sensor, 1.0)
        lb = np.log(st.b) + llr
        lb -= lb.max()
        st.b = self._norm(np.exp(lb))
        for k in np.where(observed)[0]:
            st.last_direct_obs_t[int(k)] = t
        if np.max(np.abs(llr)) > 0.05:
            self.records.append(EvidenceRecord(eid, t, sensor, st.spec.id, llr.copy()))
            if len(self.records) > 20000:
                self.records = self.records[-10000:]

    def _norm(self, b: np.ndarray) -> np.ndarray:
        b = np.maximum(b, self.FLOOR)
        return b / b.sum()

    # ------------------------------------------------------------------
    def prob(self, item_id: str, slot_id: str) -> float:
        return float(self.items[item_id].b[self.sidx[slot_id]])

    def entropy(self, item_id: str) -> float:
        b = self.items[item_id].b
        return float(-(b * np.log2(b)).sum())

    def map_slot(self, item_id: str) -> tuple[str, float]:
        b = self.items[item_id].b
        k = int(np.argmax(b))
        return self.slot_ids[k], float(b[k])

    def matrix(self) -> np.ndarray:
        return np.array([st.b for st in self.items.values()])

    def expected_class_count(self, cls: str) -> np.ndarray:
        return np.sum([st.b for st in self.items.values() if st.spec.cls == cls], axis=0)


def poisson_logpmf(n, lam):
    return n * np.log(lam) - lam - gammaln(n + 1)
