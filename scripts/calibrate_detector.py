#!/usr/bin/env python3
"""Fit detector calibration (temperature + soft confusion) and report reliability before/after.

    python scripts/calibrate_detector.py --source lite
    python scripts/calibrate_detector.py --source replicator --detections runs/detector/eval_val.npz

Sources
  * ``lite``: the lite surrogate detector (``medortrace.perception.calibration.calibrate_lite``) fitted on
    ``lite_validation_set(seed)``; before/after metrics are reported on an independent draw (``seed + 1``).
  * ``replicator``: an ``.npz`` written by ``scripts/train_detector.py --eval`` with ``logits`` (N, C) raw class
    logits over ``medortrace.perception.frontend.CLASSES`` and ``labels`` (N,) true class indices; optional
    ``classes`` (class-name order, remapped if it differs) and ``groups`` (pair-level group key per row).
    The rows are split into fit / held-out parts by *group* (``--holdout``; matched counterfactual frames
    never straddle the split); parameters are fitted on the fit part and the before/after metrics are
    reported on the held-out part (in-sample when ``--holdout 0``).

Metrics: NLL of the true class, top-label ECE (15 bins), multi-class Brier score; "before" is T=1.

The output (default ``configs/perception/detector_calibration.yaml``) keeps the keys the autonomy stack
reads (``temperature``, ``classes``, ``soft_confusion``) and adds a ``report:`` block.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import yaml

from medortrace.common.config import REPO_ROOT
from medortrace.common.rng import stable_hash
from medortrace.eval.metrics import ece
from medortrace.perception.calibration import calibrate_lite, fit_temperature, lite_validation_set, soft_confusion
from medortrace.perception.frontend import CLASSES, softmax


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["lite", "replicator"], required=True)
    ap.add_argument("--detections", default=None, help="npz with logits+labels (required for --source replicator)")
    ap.add_argument("--logits-key", default="logits",
                    help="npz logits array ('roi_logits': box-head logits at GT boxes; labels/groups use the prefix)")
    ap.add_argument("--holdout", type=float, default=0.3, help="held-out fraction for the report (replicator)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out", default=str(REPO_ROOT / "configs/perception/detector_calibration.yaml"))
    return ap.parse_args(argv)


def reliability(logits: np.ndarray, labels: np.ndarray, T: float = 1.0, bins: int = 15) -> dict:
    P = softmax(logits, T)
    n = len(labels)
    p_true = P[np.arange(n), labels]
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == labels).astype(float)
    onehot = np.eye(P.shape[1])[labels]
    return {"nll": float(-np.mean(np.log(p_true + 1e-12))), "ece": ece(conf, correct, bins),
            "brier": float(np.mean(np.sum((P - onehot) ** 2, axis=1))), "accuracy": float(correct.mean()),
            "mean_confidence": float(conf.mean())}


def load_detections(path: str | Path, logits_key: str = "logits"
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z = np.load(path, allow_pickle=False)
    prefix = logits_key[: -len("logits")] if logits_key.endswith("logits") else ""
    labels_key, groups_key = f"{prefix}labels", f"{prefix}groups"
    if logits_key not in z or labels_key not in z:
        raise SystemExit(f"{path}: expected arrays {logits_key!r} and {labels_key!r}, found {sorted(z.files)}")
    logits = np.asarray(z[logits_key], float)
    labels = np.asarray(z[labels_key]).astype(int)
    if "classes" in z:
        names = [str(c) for c in z["classes"]]
        if names != CLASSES:
            missing = [c for c in CLASSES if c not in names]
            if missing:
                raise SystemExit(f"{path}: classes {names} lack {missing}")
            order = [names.index(c) for c in CLASSES]
            logits = logits[:, order]
            remap = np.full(len(names), -1)
            remap[order] = np.arange(len(CLASSES))
            ok = (labels >= 0) & (labels < len(names))
            labels = np.where(ok, remap[np.clip(labels, 0, len(names) - 1)], -1)
    if logits.ndim != 2 or logits.shape[1] != len(CLASSES) or len(labels) != len(logits):
        raise SystemExit(f"{path}: logits {logits.shape} / labels {labels.shape} do not match {len(CLASSES)} classes")
    keep = (labels >= 0) & (labels < len(CLASSES)) & np.all(np.isfinite(logits), axis=1)
    groups = np.asarray(z[groups_key]).astype(str) if groups_key in z else None
    return logits[keep], labels[keep], (groups[keep] if groups is not None else None)


def group_holdout(n: int, groups: np.ndarray | None, frac: float, seed: int) -> np.ndarray:
    """Boolean held-out mask; rows sharing a group key (a matched pair) always land on the same side."""
    if frac <= 0:
        return np.zeros(n, dtype=bool)
    keys = groups if groups is not None else np.arange(n).astype(str)
    u = np.array([(stable_hash(f"calib:{seed}:{k}") % 10_000) / 10_000 for k in keys])
    mask = u < frac
    if mask.all() or not mask.any():         # degenerate tiny sets: fall back to in-sample
        return np.zeros(n, dtype=bool)
    return mask


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.source == "lite":
        res = calibrate_lite(seed=a.seed)
        T = float(res["temperature"])
        C = np.asarray(res["soft_confusion"], float)
        lg, lb = lite_validation_set(seed=a.seed + 1)
        n_fit, n_eval, eval_set = 20000, len(lb), f"lite_validation_set(seed={a.seed + 1})"
        src = {"source": "lite", "fit": f"calibrate_lite(seed={a.seed})"}
    else:
        if not a.detections:
            raise SystemExit("--source replicator needs --detections FILE (from scripts/train_detector.py --eval)")
        logits, labels, groups = load_detections(a.detections, a.logits_key)
        if len(labels) < 2 * len(CLASSES):
            raise SystemExit(f"only {len(labels)} labelled detections in {a.detections}: too few to calibrate")
        hold = group_holdout(len(labels), groups, a.holdout, a.seed)
        fit = ~hold
        T = fit_temperature(logits[fit], labels[fit])
        C = soft_confusion(logits[fit], labels[fit], T).round(4)
        ev = hold if hold.any() else fit
        lg, lb = logits[ev], labels[ev]
        n_fit, n_eval = int(fit.sum()), int(ev.sum())
        eval_set = "held-out groups" if hold.any() else "in-sample (no hold-out)"
        missing = [CLASSES[c] for c in range(len(CLASSES)) if not (labels[fit] == c).any()]
        if missing:
            print(f"[calibrate] WARNING: no fit samples for {missing}; their confusion rows are identity")
        sha = hashlib.sha256(Path(a.detections).read_bytes()).hexdigest()[:16]
        src = {"source": "replicator", "detections": str(a.detections), "detections_sha256": sha,
               "logits_key": a.logits_key,
               "holdout_frac": a.holdout, "grouped": groups is not None}
    before = reliability(lg, lb, 1.0, a.bins)
    after = reliability(lg, lb, T, a.bins)
    print(f"[calibrate] source={a.source} T={T:.4f} fit n={n_fit}, report on {eval_set} n={n_eval}")
    print(f"{'':10s}{'NLL':>9s}{'ECE':>9s}{'Brier':>9s}{'acc':>8s}{'conf':>8s}")
    for name, r in (("before", before), ("after", after)):
        print(f"{name:10s}{r['nll']:9.4f}{r['ece']:9.4f}{r['brier']:9.4f}{r['accuracy']:8.3f}{r['mean_confidence']:8.3f}")
    doc = {"temperature": float(T), "classes": list(CLASSES),
           "soft_confusion": [[float(x) for x in row] for row in C],
           "report": {**src, "n_fit": n_fit, "n_eval": n_eval, "eval_set": eval_set, "ece_bins": a.bins,
                      "before": before, "after": after,
                      "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# Generated by scripts/calibrate_detector.py --source {a.source}\n")
        yaml.safe_dump(doc, f, sort_keys=False, width=120)
    print(f"[calibrate] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
