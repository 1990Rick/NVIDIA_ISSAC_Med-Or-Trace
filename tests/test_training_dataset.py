"""Detection dataset parsing (CausalLabelWriter layout), pair-level splits, PNG decoding, AP, detector scripts."""

from __future__ import annotations

import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medortrace.perception.dataset import (  # noqa: E402
    DetectionSample,
    assert_no_leakage,
    average_precision,
    box_iou,
    class_counts,
    class_index,
    decode_png,
    load_rgb,
    load_samples,
    pair_level_split,
    png_size,
    registry_splits,
)
from medortrace.perception.frontend import CLASSES  # noqa: E402

ID2LABELS = {"1": {"class": "sponge"}, "2": {"class": "clamp,metal"}, "3": {"class": "person"}, "4": "specimen",
             "5": {"class": "or_table"}}


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def encode_png(img: np.ndarray, filters=(0,)) -> bytes:
    """Test PNG encoder that applies the given row filters cyclically (0 None, 1 Sub, 2 Up, 3 Average, 4 Paeth)."""
    a = np.ascontiguousarray(img, np.uint8)
    h, w, ch = a.shape
    color = {1: 0, 2: 4, 3: 2, 4: 6}[ch]
    rows = a.reshape(h, w * ch).astype(int)
    raw = bytearray()
    for y in range(h):
        f = filters[y % len(filters)]
        r = rows[y]
        prev = rows[y - 1] if y else np.zeros_like(r)
        out = []
        for x in range(len(r)):
            left = r[x - ch] if x >= ch else 0
            up = prev[x]
            ul = prev[x - ch] if x >= ch else 0
            pred = [0, left, up, (left + up) // 2, _paeth(left, up, ul)][f]
            out.append((r[x] - pred) & 0xFF)
        raw += bytes([f]) + bytes(out)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


def write_frame(sd: Path, fid: str, boxes: list, labels: dict, prim_paths=None, size=(64, 48), rgb=True) -> None:
    for d in ("rgb", "bbox2d", "labels"):
        (sd / d).mkdir(parents=True, exist_ok=True)
    if rgb:
        img = (np.arange(size[0] * size[1] * 3).reshape(size[1], size[0], 3) % 251).astype(np.uint8)
        (sd / "rgb" / f"{fid}.png").write_bytes(encode_png(img))
    info = {"idToLabels": ID2LABELS, "primPaths": prim_paths or [], "bboxIds": list(range(len(boxes)))}
    (sd / "bbox2d" / f"{fid}.json").write_text(json.dumps({"boxes": boxes, "info": info}))
    (sd / "labels" / f"{fid}.json").write_text(json.dumps({**labels, "frame_id": fid}))


def box(sid, x0, y0, x1, y1, occ=0.0):
    return {"semanticId": sid, "x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "occlusionRatio": occ}


def frame_labels(scenario_id, frame, split, pair_id=None, unit=None, family="counterfactual"):
    return {"schema": "medortrace.synthetic_frame/1", "scenario_id": scenario_id, "family": family,
            "pair_id": pair_id, "split": split, "pair_frame_key": f"{unit or scenario_id}/{frame}",
            "hidden_factor": "CF-A" if pair_id else "none", "hidden_value": scenario_id.rsplit("__", 1)[-1],
            "t": 12.5, "items": {"sponge_1": {"cls": "sponge"}, "clamp_1": {"cls": "clamp"}},
            "camera": {"resolution": [64, 48]}}


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    items = ["/World/Items/sponge_1", "/World/Items/clamp_1", "/World/Staff/nurse", "/World/Items/sponge_1",
             "/World/Items/clamp_1", "/World/Env/or_table", "/World/Items/sponge_1"]
    for arm in ("under_drape", "kick_bucket"):
        sid = f"cf_a__p0001__{arm}"
        for k in range(2):
            fid = f"{k:06d}"
            rows = [box(1, 10, 10, 20, 18, 0.1),              # sponge
                    box(2, 30.5, 5, 50, 25, None),            # clamp via "clamp,metal"; unknown occlusion -> 0
                    box(3, 0, 0, 30, 47),                     # person -> dropped
                    box(1, 5, 5, 6, 6),                       # 1 px -> dropped
                    box(2, 40, 30, 60, 46, 0.99),             # fully occluded -> dropped
                    box(5, 0, 30, 64, 48),                    # furniture -> dropped
                    box(1, 50, 40, 90, 60, float("nan"))]     # clipped to the image; NaN occlusion -> 0
            write_frame(root / sid, fid, rows, frame_labels(sid, fid, "train", "CF-A/p0001", "CF-A/p0001"), items)
    sid = "nominal__0003"
    write_frame(root / sid, "000000", [[4, 1, 2, 11, 12, 0.0]], frame_labels(sid, "000000", "test", family="nominal"))
    write_frame(root / sid, "000001", [], frame_labels(sid, "000001", "test", family="nominal"))
    write_frame(root / sid, "000002", [box(1, 1, 1, 9, 9)], frame_labels(sid, "000002", "test"), rgb=False)
    (root / "usd").mkdir()                                     # non-scenario dirs are ignored
    return root


# ---------------------------------------------------------------------------
def test_class_index_label_forms():
    assert class_index({"class": "sponge"}) == CLASSES.index("sponge")
    assert class_index("clamp,metal") == CLASSES.index("clamp")
    assert class_index("metal, needle_driver") == CLASSES.index("needle_driver")
    assert class_index({"class": "person"}) is None and class_index(None) is None and class_index("") is None


def test_load_samples_parses_boxes_classes_and_groups(dataset: Path):
    samples = load_samples(dataset)
    assert len(samples) == 6                                   # rgb-less frame skipped
    by = {s.uid: s for s in samples}
    s = by["cf_a__p0001__under_drape/000000"]
    assert s.image_path is not None and s.image_path.exists() and s.image_size == (64, 48)
    assert s.classes.tolist() == [CLASSES.index("sponge"), CLASSES.index("clamp"), CLASSES.index("sponge")]
    np.testing.assert_allclose(s.boxes[0], [10, 10, 20, 18])
    np.testing.assert_allclose(s.boxes[1], [30.5, 5, 50, 25])
    np.testing.assert_allclose(s.boxes[2], [50, 40, 64, 48])   # clipped to 64x48
    np.testing.assert_allclose(s.occlusion, [0.1, 0.0, 0.0], atol=1e-6)
    assert s.item_ids == ["sponge_1", "clamp_1", "sponge_1"]
    assert s.boxes.dtype == np.float32 and s.classes.dtype == np.int64
    assert s.group == "CF-A/p0001" and s.split == "train" and s.meta["hidden_factor"] == "CF-A"
    assert by["cf_a__p0001__kick_bucket/000001"].group == "CF-A/p0001"
    n = by["nominal__0003/000000"]
    assert n.classes.tolist() == [CLASSES.index("specimen")] and n.item_ids == [None]   # list-form rows
    assert n.group == "nominal__0003" and n.split == "test"
    assert by["nominal__0003/000001"].n_boxes == 0 and by["nominal__0003/000001"].boxes.shape == (0, 4)
    assert len(load_samples(dataset, keep_empty=False)) == 5
    assert len(load_samples(dataset, require_image=False)) == 7
    one = load_samples(dataset / "nominal__0003")               # a single scenario directory works too
    assert {s.scenario_id for s in one} == {"nominal__0003"}
    cc = class_counts(samples)
    assert cc["sponge"] == 8 and cc["clamp"] == 4 and cc["specimen"] == 1


def test_registry_and_pair_level_splits(dataset: Path):
    samples = load_samples(dataset)
    sp = registry_splits(samples)
    assert {k: len(v) for k, v in sp.items()} == {"train": 4, "test": 2}
    # many synthetic pairs: every group lands in exactly one split, deterministically
    many = []
    for g in range(200):
        for arm in ("a", "b"):
            for f in range(3):
                many.append(DetectionSample(f"s{g}{arm}/{f}", f"s{g}{arm}", str(f), None, np.zeros((0, 4)),
                                            np.zeros(0, int), np.zeros(0), [], f"CF-B/p{g:04d}", None))
    s1 = pair_level_split(many, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=3)
    s2 = pair_level_split(many, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=3)
    assert {k: [s.uid for s in v] for k, v in s1.items()} == {k: [s.uid for s in v] for k, v in s2.items()}
    owners = {}
    for name, ss in s1.items():
        for s in ss:
            assert owners.setdefault(s.group, name) == name
    assert len(owners) == 200
    frac = len(s1["train"]) / len(many)
    assert 0.55 < frac < 0.85
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage({"train": many[:1], "test": many[1:2]})


def test_png_decoder_all_filters(tmp_path: Path):
    rng = np.random.default_rng(0)
    for ch in (1, 2, 3, 4):
        img = rng.integers(0, 256, (7, 5, ch), dtype=np.uint8)
        dec = decode_png(encode_png(img, filters=(0, 1, 2, 3, 4)))
        np.testing.assert_array_equal(dec, img)
    img = rng.integers(0, 256, (6, 9, 4), dtype=np.uint8)
    p = tmp_path / "x.png"
    p.write_bytes(encode_png(img, filters=(4, 3)))
    assert png_size(p) == (9, 6)
    rgb = load_rgb(p)
    assert rgb.shape == (6, 9, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb, img[..., :3])


def test_box_iou_and_average_precision():
    a = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], float)
    iou = box_iou(a, a)
    np.testing.assert_allclose(np.diag(iou), 1.0)
    np.testing.assert_allclose(iou[0, 1], 25 / 175)
    gts = [(np.array([[0, 0, 10, 10]]), np.array([0])), (np.array([[20, 20, 30, 30]]), np.array([0]))]
    perfect = [(g[0], g[1], np.array([0.9])) for g in gts]
    res = average_precision(perfect, gts)
    assert res["ap"][0] == pytest.approx(1.0) and np.isnan(res["ap"][1]) and res["map"] == pytest.approx(1.0)
    # a false positive ranked first: precision 1/2 at recall 1/2, then 2/3 at recall 1 -> AP = 2/3
    fp_first = [(np.array([[0, 0, 10, 10], [40, 40, 50, 50]]), np.array([0, 0]), np.array([0.5, 0.95])),
                (gts[1][0], gts[1][1], np.array([0.4]))]
    assert average_precision(fp_first, gts)["ap"][0] == pytest.approx(2 / 3)
    # duplicate detections of one object: the second is a false positive
    dup = [(np.array([[0, 0, 10, 10], [0, 0, 10, 10]]), np.array([0, 0]), np.array([0.9, 0.8])),
           (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))]
    assert average_precision(dup, gts)["ap"][0] == pytest.approx(0.5)


def test_real_causal_label_writer_output_is_parsed(tmp_path: Path):
    pytest.importorskip("pxr")
    from medortrace.isaac.replicator_randomizers import CausalLabelWriter
    dt = np.dtype([("semanticId", "<u4"), ("x_min", "<i4"), ("y_min", "<i4"), ("x_max", "<i4"), ("y_max", "<i4"),
                   ("occlusionRatio", "<f4")])
    out = tmp_path / "real" / "cf_c__p0002__handed_off"
    w = CausalLabelWriter(out, label_fn=lambda fid: frame_labels("cf_c__p0002__handed_off", fid, "val",
                                                                 "CF-C/p0002", "CF-C/p0002"))
    ann = {"rgb": np.random.default_rng(1).integers(0, 255, (48, 64, 4), dtype=np.uint8),
           "bounding_box_2d_tight": {"data": np.array([(2, 3, 4, 20, 30, 0.25), (3, 0, 0, 10, 10, 0.0)], dtype=dt),
                                     "info": {"idToLabels": ID2LABELS,
                                              "primPaths": ["/World/Items/clamp_1", "/World/Staff/x"]}}}
    w.write(ann)
    (s,) = load_samples(tmp_path / "real")
    assert s.classes.tolist() == [CLASSES.index("clamp")] and s.item_ids == ["clamp_1"]
    np.testing.assert_allclose(s.boxes, [[3, 4, 20, 30]])
    assert s.group == "CF-C/p0002" and s.split == "val"
    np.testing.assert_array_equal(load_rgb(s.image_path), ann["rgb"][..., :3])


def test_train_detector_dry_run_and_splits(dataset: Path, capsys):
    import train_detector
    assert train_detector.main(["--data", str(dataset), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert '"train"' in out and "decoded" in out
    samples = load_samples(dataset)
    sp = train_detector.make_splits(samples, "registry", 0.5, 0)
    assert "test" in sp and sum(len(v) for v in sp.values()) == len(samples)
    assert_no_leakage(sp)                                      # val carved from train at pair level
    np.testing.assert_allclose(np.exp(train_detector.pseudo_logits(0.8, 2))[2], 0.8, rtol=1e-4)


def test_calibrate_detector_replicator_npz(tmp_path: Path):
    import calibrate_detector
    rng = np.random.default_rng(0)
    n, C = 3000, len(CLASSES)
    labels = rng.integers(0, C, n)
    # overconfident detector: margins too large for its actual accuracy
    logits = 4.0 * (np.eye(C)[labels] * rng.uniform(0.2, 1.5, (n, 1)) + rng.normal(0, 0.6, (n, C)))
    groups = np.array([f"CF-A/p{k // 4:04d}" for k in range(n)])
    order = [4, 0, 1, 2, 3]                                     # class order differs from CLASSES
    names = np.array([CLASSES[i] for i in order])
    inv = {c: i for i, c in enumerate(order)}
    f = tmp_path / "eval_val.npz"
    np.savez(f, logits=logits[:, order], labels=np.array([inv[c] for c in labels]), groups=groups, classes=names)
    out = tmp_path / "calib.yaml"
    assert calibrate_detector.main(["--source", "replicator", "--detections", str(f), "--out", str(out)]) == 0
    doc = yaml.safe_load(out.read_text())
    assert doc["classes"] == CLASSES and doc["temperature"] > 1.0
    Cm = np.array(doc["soft_confusion"])
    assert Cm.shape == (C, C) and np.allclose(Cm.sum(1), 1.0, atol=1e-3)
    assert np.all(Cm.argmax(axis=1) == np.arange(C))           # true class carries the most soft mass
    rep = doc["report"]
    assert rep["after"]["ece"] < rep["before"]["ece"] and rep["after"]["nll"] < rep["before"]["nll"]
    assert rep["grouped"] and 0 < rep["n_eval"] < n
    # the lite source reproduces the committed calibration's temperature
    out2 = tmp_path / "lite.yaml"
    assert calibrate_detector.main(["--source", "lite", "--out", str(out2)]) == 0
    committed = yaml.safe_load((ROOT / "configs/perception/detector_calibration.yaml").read_text())
    assert yaml.safe_load(out2.read_text())["temperature"] == pytest.approx(committed["temperature"], rel=1e-6)
