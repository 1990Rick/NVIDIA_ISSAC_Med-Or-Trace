#!/usr/bin/env python3
"""Train / evaluate the item detector (torchvision Faster R-CNN) on Replicator ``CausalLabelWriter`` data.

    python scripts/train_detector.py --data datasets/cf_all --out runs/detector --epochs 12
    python scripts/train_detector.py --data datasets/cf_all --out runs/detector --eval \
        --checkpoint runs/detector/model.pt
    python scripts/train_detector.py --data datasets/cf_all --dry-run        # parse + split report, no torch
    python scripts/calibrate_detector.py --source replicator --detections runs/detector/eval_val.npz

Data: ``medortrace.perception.dataset.load_samples`` (boxes of the five item classes of
``medortrace.perception.frontend.CLASSES``; staff/furniture boxes dropped).  Splits are pair-level:
``--split-mode registry`` (default) uses the registry split recorded in each frame's labels (val is carved out
of train by pair-level hashing when the dataset has no val frames); ``--split-mode hash`` re-splits every
frame by hashing its pair-level group key.  Matched counterfactual frames never straddle a split.

Model: ``fasterrcnn_resnet50_fpn`` with ``len(CLASSES) + 1`` outputs (index 0 = background), i.e. exactly the
architecture ``medortrace.isaac.detector.TorchDetector`` loads (``CameraAdapter(detector="model:<path>")``).
``model.pt`` is ``{"model": state_dict, "classes": CLASSES, ...}``.  ``--init imagenet`` (default) downloads the
ImageNet ResNet-50 backbone, ``coco`` the COCO detector (heads re-initialised), ``none`` trains from scratch.

Evaluation (``--eval``, also run after training) writes per split ``eval_<split>.npz``:
  * ``logits``/``labels``: detections matched to a ground-truth box (IoU >= 0.5, score >= ``--score-thresh``)
    converted to pseudo-logits exactly as ``TorchDetector.predict`` does at run time
    (``log(score)`` for the predicted class, ``log((1-score)/(C-1))`` elsewhere), with the matched box's
    true class - so the temperature/confusion fitted by ``calibrate_detector.py`` apply to what the
    autonomy stack will see;
  * ``roi_logits``/``roi_labels``: raw box-head class logits (background column dropped) at every
    ground-truth box - classification calibration independent of detection (``--logits-key roi_logits``);
  * ``groups`` / ``roi_groups`` (pair-level keys for grouped hold-out), ``classes``;
and ``eval_<split>.json`` with AP@0.5 per class and mAP.

torch / torchvision are imported lazily (``pip install -e .[train]``); ``--dry-run`` needs neither.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

from medortrace.perception.dataset import (
    DetectionSample,
    average_precision,
    box_iou,
    class_counts,
    load_rgb,
    load_samples,
    pair_level_split,
    registry_splits,
)
from medortrace.perception.frontend import CLASSES


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root (generate_synthetic_data.py --out) or one scenario")
    ap.add_argument("--out", default="runs/detector")
    ap.add_argument("--split-mode", choices=["registry", "hash"], default="registry")
    ap.add_argument("--val-frac", type=float, default=0.15, help="pair-level val fraction carved from train if needed")
    ap.add_argument("--min-box", type=float, default=4.0, help="drop boxes smaller than this (px)")
    ap.add_argument("--max-occlusion", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=None, help="max training frames (debug)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--init", choices=["imagenet", "coco", "none"], default="imagenet")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval", action="store_true", help="evaluate --checkpoint only (no training)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--eval-splits", nargs="+", default=["val", "test"])
    ap.add_argument("--score-thresh", type=float, default=0.3, help="TorchDetector's run-time score threshold")
    ap.add_argument("--dry-run", action="store_true", help="parse the dataset and report splits; no torch")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
def make_splits(samples: list[DetectionSample], mode: str, val_frac: float, seed: int) -> dict[str, list]:
    if mode == "hash":
        return pair_level_split(samples, {"train": 1 - 2 * val_frac, "val": val_frac, "test": val_frac}, seed)
    splits = registry_splits(samples, default="train")
    if not splits.get("val") and splits.get("train"):
        sub = pair_level_split(splits["train"], {"train": 1 - val_frac, "val": val_frac}, seed)
        splits["train"], splits["val"] = sub["train"], sub["val"]
    return {k: v for k, v in splits.items() if v}


def split_report(splits: dict[str, list]) -> dict:
    return {k: {"frames": len(v), "groups": len({s.group for s in v}), "boxes": int(sum(s.n_boxes for s in v)),
                "classes": class_counts(v)} for k, v in splits.items()}


class TorchDetectionDataset:
    """Map-style dataset (DataLoader protocol) over :class:`DetectionSample`; labels are ``class + 1``."""

    def __init__(self, samples: list[DetectionSample], train: bool = False):
        self.samples = samples
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        import torch
        s = self.samples[i]
        img = load_rgb(s.image_path)
        boxes = s.boxes.copy()
        if self.train and float(torch.rand(())) < 0.5:     # horizontal flip (torch RNG: seeded per worker)
            img = img[:, ::-1]
            W = img.shape[1]
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        tgt = {"boxes": torch.from_numpy(boxes.reshape(-1, 4)).float(),
               "labels": torch.from_numpy(s.classes + 1).long(), "image_id": torch.tensor(i)}
        return x, tgt


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
def _import_torch():
    try:
        import torch
        import torchvision
    except ImportError as e:
        raise SystemExit(f"train_detector.py needs torch + torchvision ({e}). Install the training extras:\n"
                         "    pip install -e '.[train]'\n"
                         "or run with --dry-run to only parse and split the dataset.") from None
    return torch, torchvision


def build_model(torchvision, init: str):
    det = torchvision.models.detection
    n = len(CLASSES) + 1
    try:
        if init == "coco":
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            m = det.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, n)
            return m
        wb = "DEFAULT" if init == "imagenet" else None
        return det.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=wb, num_classes=n)
    except Exception as e:  # typically: no network for the weight download
        raise SystemExit(f"could not build the model with --init {init}: {e}\n(use --init none offline)") from None


def pseudo_logits(score: float, label0: int, C: int = len(CLASSES)) -> np.ndarray:
    """Run-time logit contract of ``medortrace.isaac.detector.TorchDetector.predict`` (label0 is 0-based)."""
    lg = np.full(C, np.log((1 - score) / (C - 1) + 1e-6))
    lg[label0] = np.log(score + 1e-6)
    return lg


def evaluate(torch, model, samples: list[DetectionSample], device, score_thresh: float) -> tuple[dict, dict]:
    model.eval()
    preds, gts = [], []
    det_lg, det_lb, det_g, roi_lg, roi_lb, roi_g, uids = [], [], [], [], [], [], []
    C = len(CLASSES)
    with torch.no_grad():
        for s in samples:
            x = torch.from_numpy(load_rgb(s.image_path)).permute(2, 0, 1).float().div(255.0).to(device)
            out = model([x])[0]
            pb = out["boxes"].cpu().numpy()
            pl = out["labels"].cpu().numpy() - 1
            ps = out["scores"].cpu().numpy()
            ok = (pl >= 0) & (pl < C)
            pb, pl, ps = pb[ok], pl[ok], ps[ok]
            preds.append((pb, pl, ps))
            gts.append((s.boxes, s.classes))
            # detections matched to GT (greedy by score, IoU >= 0.5), as TorchDetector would report them
            keep = ps >= score_thresh
            if keep.any() and s.n_boxes:
                iou = box_iou(pb[keep], s.boxes)
                used = np.zeros(s.n_boxes, bool)
                for k in np.argsort(-ps[keep]):
                    j = int(np.argmax(iou[k]))
                    if iou[k, j] >= 0.5 and not used[j]:
                        used[j] = True
                        det_lg.append(pseudo_logits(float(ps[keep][k]), int(pl[keep][k])))
                        det_lb.append(int(s.classes[j]))
                        det_g.append(s.group)
                        uids.append(s.uid)
            # raw box-head class logits at the GT boxes
            if s.n_boxes:
                images, targets = model.transform([x], [{"boxes": torch.from_numpy(s.boxes).float().to(device),
                                                         "labels": torch.from_numpy(s.classes + 1).to(device)}])
                feats = model.backbone(images.tensors)
                bf = model.roi_heads.box_roi_pool(feats, [targets[0]["boxes"]], images.image_sizes)
                cls_logits, _ = model.roi_heads.box_predictor(model.roi_heads.box_head(bf))
                roi_lg.append(cls_logits[:, 1:].cpu().numpy())
                roi_lb += s.classes.tolist()
                roi_g += [s.group] * s.n_boxes
    ap = average_precision(preds, gts)
    arrays = {"logits": np.asarray(det_lg, np.float32).reshape(-1, C), "labels": np.asarray(det_lb, np.int64),
              "groups": np.asarray(det_g, dtype=str), "uids": np.asarray(uids, dtype=str),
              "roi_logits": (np.concatenate(roi_lg) if roi_lg else np.zeros((0, C))).astype(np.float32),
              "roi_labels": np.asarray(roi_lb, np.int64), "roi_groups": np.asarray(roi_g, dtype=str),
              "classes": np.asarray(CLASSES)}
    acc = float(np.mean(np.argmax(arrays["logits"], 1) == arrays["labels"])) if det_lb else None
    report = {"frames": len(samples), "ap50": dict(zip(CLASSES, ap["ap"])), "map50": ap["map"],
              "matched_detections": len(det_lb), "gt_boxes": len(roi_lb), "matched_accuracy": acc}
    return report, arrays


def train(torch, torchvision, a, splits, out: Path):
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(a.seed)
    model = build_model(torchvision, a.init).to(device)
    tr = splits["train"][: a.limit] if a.limit else splits["train"]
    dl = torch.utils.data.DataLoader(TorchDetectionDataset(tr, train=True), batch_size=a.batch_size,
                                     shuffle=True, num_workers=a.workers, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=a.weight_decay)
    steps = max(1, a.epochs * len(dl))
    warm = min(500, steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(i, steps) / steps)))
    amp_on = a.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp_on)
    log = open(out / "train_log.jsonl", "a")
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for imgs, tgts in dl:
            imgs = [i.to(device) for i in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]
            with torch.autocast(device.type, enabled=scaler.is_enabled()):
                losses = model(imgs, tgts)
                loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss)
            n += 1
        rec = {"epoch": ep, "loss": tot / max(n, 1), "lr": opt.param_groups[0]["lr"], "wall_s": time.time() - t0}
        print(f"[train_detector] epoch {ep + 1}/{a.epochs} loss={rec['loss']:.4f} ({rec['wall_s']:.0f}s)", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        save(torch, model, out / "model.pt", a, splits, ep)
    log.close()
    return model, device


def save(torch, model, path: Path, a, splits, epoch: int) -> None:
    torch.save({"model": model.state_dict(), "classes": CLASSES, "arch": "fasterrcnn_resnet50_fpn",
                "num_classes": len(CLASSES) + 1, "epoch": epoch, "data": str(a.data), "init": a.init,
                "splits": {k: len(v) for k, v in splits.items()}}, path)


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.out)
    samples = load_samples(a.data, min_size=a.min_box, max_occlusion=a.max_occlusion)
    if not samples:
        raise SystemExit(f"no frames with rgb/ + bbox2d/ found under {a.data}")
    splits = make_splits(samples, a.split_mode, a.val_frac, a.seed)
    rep = split_report(splits)
    print(json.dumps({"data": str(a.data), "split_mode": a.split_mode, "splits": rep}, indent=1))
    if a.dry_run:
        s0 = next((s for s in samples if s.image_path is not None), None)
        if s0 is not None:
            img = load_rgb(s0.image_path)
            print(f"[train_detector] decoded {s0.uid}: {img.shape} {img.dtype}")
        return 0
    if "train" not in splits and not a.eval:
        raise SystemExit("no training frames")
    torch, torchvision = _import_torch()
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps({k: [s.uid for s in v] for k, v in splits.items()}))
    if a.eval:
        if not a.checkpoint:
            raise SystemExit("--eval needs --checkpoint")
        device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = build_model(torchvision, "none").to(device)
        ck = torch.load(a.checkpoint, map_location=device)
        model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    else:
        model, device = train(torch, torchvision, a, splits, out)
    for sp in a.eval_splits:
        if not splits.get(sp):
            continue
        report, arrays = evaluate(torch, model, splits[sp], device, a.score_thresh)
        np.savez_compressed(out / f"eval_{sp}.npz", **arrays)
        (out / f"eval_{sp}.json").write_text(json.dumps(report, indent=1))
        print(f"[train_detector] {sp}: mAP50={report['map50']:.3f} matched={report['matched_detections']} "
              f"-> {out / f'eval_{sp}.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
