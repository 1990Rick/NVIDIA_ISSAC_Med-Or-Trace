"""Detection dataset from ``CausalLabelWriter`` output (pure numpy; no torch, no PIL required).

``scripts/isaac/generate_synthetic_data.py`` writes, per scenario ``OUT/<scenario_id>/``::

    rgb/NNNNNN.png        8-bit RGB
    bbox2d/NNNNNN.json    {"boxes": [{"semanticId", "x_min", "y_min", "x_max", "y_max", "occlusionRatio"}, ...],
                           "info": {"idToLabels": {"<semanticId>": {"class": "sponge"}}, "primPaths": [...]}}
    labels/NNNNNN.json    causal labels: scenario_id, family, pair_id, split, pair_frame_key, hidden factor/value,
                          t, items {id: {cls, slot, ...}}, camera {resolution, ...}

plus ``OUT/index.json``.  :func:`load_samples` turns this into
:class:`DetectionSample` records whose box classes index
``medortrace.perception.frontend.CLASSES`` (boxes of other semantic classes -
staff, furniture - are dropped).  ``idToLabels`` values may be a dict with a
``class`` entry, a bare string, or a comma-separated list; the first token that
is a known class wins (the same rule as the Isaac camera adapter).

Splits are **pair-level**: the group of a sample is its render unit (the
counterfactual ``pair_id`` for CF arms, else the scenario id), so both arms of a
matched pair and all their frames always share a split.  :func:`registry_splits`
uses the split recorded by the generator (the registry split);
:func:`pair_level_split` re-splits deterministically by hashing the group key
(e.g. train/val *inside* the registry train split); :func:`assert_no_leakage`
checks either.

Also here (numpy, for ``scripts/train_detector.py`` and tests): a PNG reader
(PIL when installed, else a zlib decoder for 8-bit non-interlaced PNGs), box IoU
and VOC-style AP@IoU.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from medortrace.common.rng import stable_hash
from medortrace.perception.frontend import CLASSES

BOX_COLUMNS = ("semanticId", "x_min", "y_min", "x_max", "y_max", "occlusionRatio")
SPLITS = ("train", "val", "test")


@dataclass
class DetectionSample:
    uid: str                          # "<scenario_id>/<frame_id>"
    scenario_id: str
    frame_id: str
    image_path: Path | None           # None if rgb/ is missing for this frame
    boxes: np.ndarray                 # (N,4) float32 x_min, y_min, x_max, y_max (pixels)
    classes: np.ndarray               # (N,) int64 index into CLASSES
    occlusion: np.ndarray             # (N,) float32 occlusionRatio in [0,1]
    item_ids: list                    # (N,) item id when the prim path identifies one, else None
    group: str                        # pair-level split key (pair_id or scenario_id)
    split: str | None                 # split recorded by the generator (registry split)
    image_size: tuple[int, int] | None = None      # (W, H)
    meta: dict = field(default_factory=dict)       # family, pair_id, hidden factor/value, t, pair_frame_key

    @property
    def n_boxes(self) -> int:
        return len(self.classes)


# ---------------------------------------------------------------------------
def class_index(label, classes: list[str] = CLASSES) -> int | None:
    """``idToLabels`` value -> class index (None for non-item classes)."""
    if isinstance(label, dict):
        label = label.get("class")
    if not label:
        return None
    for tok in str(label).split(","):
        tok = tok.strip()
        if tok in classes:
            return classes.index(tok)
    return None


def _lookup_label(id_to_labels: dict, sid):
    for key in (str(sid), sid):
        if key in id_to_labels:
            return id_to_labels[key]
    try:
        return id_to_labels.get(int(sid))
    except (TypeError, ValueError):
        return None


def _rows(boxes) -> list[dict]:
    if boxes is None:
        return []
    out = []
    for r in boxes:
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, (list, tuple)):
            out.append(dict(zip(BOX_COLUMNS, r)))
    return out


def _item_from_prim(path, item_ids: list[str]) -> str | None:
    if not path:
        return None
    parts = str(path).rstrip("/").split("/")
    if "Items" not in parts:
        return None
    leaf = parts[-1]
    for iid in item_ids:
        if leaf == iid or leaf == _safe(iid):
            return iid
    return None


def _safe(name: str) -> str:
    """Prim-name rule of ``medortrace.usd.scene_builder.safe`` (duplicated to keep this module pxr-free)."""
    n = name.translate(str.maketrans({":": "_", "-": "_", " ": "_", ".": "_"}))
    return n if n and n[0].isalpha() else "_" + n


def parse_bbox_json(doc: dict, item_ids: list[str] | None = None, min_size: float = 2.0,
                    max_occlusion: float = 0.95, image_size: tuple[int, int] | None = None,
                    classes: list[str] = CLASSES) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """One ``bbox2d/NNNNNN.json`` -> (boxes (N,4), class idx (N,), occlusion (N,), item ids).

    Drops non-item classes, boxes smaller than ``min_size`` px, and boxes more occluded than
    ``max_occlusion`` (occlusionRatio < 0 or NaN means "unknown" and is kept as 0).
    """
    info = doc.get("info", {}) or {}
    id2 = info.get("idToLabels", {}) or {}
    paths = list(info.get("primPaths", []) or [])
    boxes, cls, occ, iids = [], [], [], []
    for k, r in enumerate(_rows(doc.get("boxes"))):
        ci = class_index(_lookup_label(id2, r.get("semanticId", -1)), classes)
        if ci is None:
            continue
        try:
            b = [float(r[c]) for c in ("x_min", "y_min", "x_max", "y_max")]
        except (KeyError, TypeError, ValueError):
            continue
        if image_size is not None:
            W, H = image_size
            b = [min(max(b[0], 0.0), W), min(max(b[1], 0.0), H), min(max(b[2], 0.0), W), min(max(b[3], 0.0), H)]
        if b[2] - b[0] < min_size or b[3] - b[1] < min_size:
            continue
        o = r.get("occlusionRatio", 0.0)
        o = float(o) if o is not None else 0.0
        o = o if np.isfinite(o) and o >= 0 else 0.0
        if o > max_occlusion:
            continue
        boxes.append(b)
        cls.append(ci)
        occ.append(min(o, 1.0))
        iids.append(_item_from_prim(paths[k], item_ids or []) if k < len(paths) else None)
    return (np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(cls, np.int64),
            np.asarray(occ, np.float32), iids)


def unit_key(labels: dict, scenario_id: str) -> str:
    """Pair-level group key: the render unit (``pair_frame_key`` minus the frame), else pair_id / scenario."""
    pfk = labels.get("pair_frame_key")
    if isinstance(pfk, str) and "/" in pfk:
        return pfk.rsplit("/", 1)[0]
    return str(labels.get("pair_id") or labels.get("scenario_id") or scenario_id)


# ---------------------------------------------------------------------------
def scenario_dirs(root: str | Path) -> list[Path]:
    """``root`` itself if it is one scenario's output, else its scenario sub-directories (sorted)."""
    root = Path(root)
    if (root / "bbox2d").is_dir():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "bbox2d").is_dir())


def load_samples(root: str | Path, min_size: float = 2.0, max_occlusion: float = 0.95, require_image: bool = True,
                 keep_empty: bool = True, classes: list[str] = CLASSES) -> list[DetectionSample]:
    """Parse every frame under ``root`` (one scenario dir or a dataset root with many)."""
    out = []
    for sd in scenario_dirs(root):
        for bj in sorted((sd / "bbox2d").glob("*.json")):
            fid = bj.stem
            lab_p = sd / "labels" / f"{fid}.json"
            labels = json.loads(lab_p.read_text()) if lab_p.exists() else {}
            img = sd / "rgb" / f"{fid}.png"
            if not img.exists():
                if require_image:
                    continue
                img = None
            size = None
            res = (labels.get("camera") or {}).get("resolution")
            if isinstance(res, (list, tuple)) and len(res) == 2:
                size = (int(res[0]), int(res[1]))
            elif img is not None:
                size = png_size(img)
            sid = str(labels.get("scenario_id") or sd.name)
            items = list((labels.get("items") or {}).keys())
            boxes, cls, occ, iids = parse_bbox_json(json.loads(bj.read_text()), items, min_size, max_occlusion,
                                                    size, classes)
            if not keep_empty and len(cls) == 0:
                continue
            meta = {k: labels.get(k) for k in ("family", "pair_id", "pair_frame_key", "hidden_factor", "hidden_value",
                                               "t", "seed", "cfg_hash")}
            out.append(DetectionSample(f"{sid}/{fid}", sid, fid, img, boxes, cls, occ, iids, unit_key(labels, sid),
                                       labels.get("split"), size, meta))
    return out


# ---------------------------------------------------------------------------
def registry_splits(samples: list[DetectionSample], default: str | None = None) -> dict[str, list[DetectionSample]]:
    """Group by the split recorded in the labels (samples without one go to ``default`` or are dropped)."""
    out: dict[str, list[DetectionSample]] = defaultdict(list)
    for s in samples:
        sp = s.split or default
        if sp:
            out[sp].append(s)
    assert_no_leakage(out)
    return dict(out)


def pair_level_split(samples: list[DetectionSample], fractions: dict[str, float] | None = None,
                     seed: int = 0) -> dict[str, list[DetectionSample]]:
    """Deterministic split by hashing each sample's pair-level group key."""
    fractions = fractions or {"train": 0.8, "val": 0.1, "test": 0.1}
    names = list(fractions)
    cum = np.cumsum([float(fractions[n]) for n in names])
    cum = cum / cum[-1]
    out: dict[str, list[DetectionSample]] = {n: [] for n in names}
    for s in samples:
        u = (stable_hash(f"dataset_split:{seed}:{s.group}") % 1_000_000) / 1_000_000
        out[names[int(np.searchsorted(cum, u, side="right"))]].append(s)
    assert_no_leakage(out)
    return out


def assert_no_leakage(splits: dict[str, list[DetectionSample]]) -> None:
    owner: dict[str, str] = {}
    for name, ss in splits.items():
        for s in ss:
            prev = owner.setdefault(s.group, name)
            if prev != name:
                raise ValueError(f"group {s.group!r} appears in splits {prev!r} and {name!r} (pair-level leakage)")


def class_counts(samples: list[DetectionSample], classes: list[str] = CLASSES) -> dict[str, int]:
    n = np.zeros(len(classes), int)
    for s in samples:
        n += np.bincount(s.classes, minlength=len(classes))[: len(classes)]
    return dict(zip(classes, n.tolist()))


# ---------------------------------------------------------------------------
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy boxes -> (N,M) IoU."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area_a = np.prod(np.clip(a[:, 2:] - a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(b[:, 2:] - b[:, :2], 0, None), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None] - inter, 1e-9)


def average_precision(preds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                      gts: list[tuple[np.ndarray, np.ndarray]], n_classes: int = len(CLASSES),
                      iou_thr: float = 0.5) -> dict:
    """VOC-style (all-point interpolated) AP per class.

    ``preds[i] = (boxes (K,4), labels (K,), scores (K,))`` and ``gts[i] = (boxes (N,4), labels (N,))`` for image i;
    labels index ``CLASSES``.  Returns ``{"ap": [per class, NaN if no GT], "map": nanmean}``.
    """
    ap = np.full(n_classes, np.nan)
    for c in range(n_classes):
        recs = []
        n_gt = 0
        for (pb, pl, ps), (gb, gl) in zip(preds, gts):
            g = np.asarray(gb).reshape(-1, 4)[np.asarray(gl) == c]
            n_gt += len(g)
            m = np.asarray(pl) == c
            p, s = np.asarray(pb).reshape(-1, 4)[m], np.asarray(ps)[m]
            used = np.zeros(len(g), bool)
            iou = box_iou(p, g) if len(g) and len(p) else np.zeros((len(p), len(g)))
            for k in np.argsort(-s, kind="stable"):
                j = int(np.argmax(iou[k])) if len(g) else -1
                tp = j >= 0 and iou[k, j] >= iou_thr and not used[j]
                if tp:
                    used[j] = True
                recs.append((float(s[k]), tp))
        if n_gt == 0:
            continue
        recs.sort(key=lambda r: -r[0])
        tp = np.cumsum([r[1] for r in recs]) if recs else np.zeros(0)
        fp = np.cumsum([not r[1] for r in recs]) if recs else np.zeros(0)
        rec = tp / n_gt
        prec = tp / np.maximum(tp + fp, 1e-9)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[0.0], prec, [0.0]])
        mpre = np.maximum.accumulate(mpre[::-1])[::-1]
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap[c] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {"ap": ap.tolist(), "map": float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan")}


# ---------------------------------------------------------------------------
def png_size(path: str | Path) -> tuple[int, int] | None:
    """(W, H) from the IHDR chunk, without decoding."""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    w, h = struct.unpack(">II", head[16:24])
    return int(w), int(h)


def load_rgb(path: str | Path) -> np.ndarray:
    """(H, W, 3) uint8 image; PIL if installed, else :func:`decode_png`."""
    try:
        from PIL import Image
    except ImportError:
        img = decode_png(Path(path).read_bytes())
    else:
        with Image.open(path) as im:
            img = np.asarray(im.convert("RGB"))
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.shape[2] == 2:                  # grey + alpha
        img = np.repeat(img[..., :1], 3, axis=2)
    return np.ascontiguousarray(img[..., :3])


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = np.abs(p - a), np.abs(p - b), np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))


def decode_png(data: bytes) -> np.ndarray:
    """Minimal PNG decoder: 8-bit grey / grey+alpha / RGB / RGBA, non-interlaced, all five row filters."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos, idat, hdr = 8, [], None
    while pos < len(data):
        n, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + n
    if hdr is None:
        raise ValueError("PNG without IHDR")
    w, h, depth, color, _, _, interlace = hdr
    ch = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if depth != 8 or ch is None or interlace:
        raise ValueError(f"unsupported PNG (bit depth {depth}, colour type {color}, interlace {interlace}); "
                         "install pillow")
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    stride = w * ch
    rows = raw.reshape(h, stride + 1)
    out = np.zeros((h, stride), np.int32)
    prev = np.zeros(stride, np.int32)
    for y in range(h):
        f, line = rows[y, 0], rows[y, 1:].astype(np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 0xFF
        elif f in (1, 3, 4):
            cur = np.zeros(stride, np.int32)
            for x in range(0, stride, ch):          # left neighbour dependency: per-pixel, vectorised over channels
                left = cur[x - ch:x] if x else np.zeros(ch, np.int32)
                if f == 1:
                    pred = left
                elif f == 3:
                    pred = (left + prev[x:x + ch]) >> 1
                else:
                    ul = prev[x - ch:x] if x else np.zeros(ch, np.int32)
                    pred = _paeth(left, prev[x:x + ch], ul)
                cur[x:x + ch] = (line[x:x + ch] + pred) & 0xFF
        else:
            raise ValueError(f"bad PNG filter type {f}")
        out[y] = cur
        prev = cur
    return out.astype(np.uint8).reshape(h, w, ch)
