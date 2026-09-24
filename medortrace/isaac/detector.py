"""Learned item detector wrapper (torchvision Faster R-CNN trained on Replicator data).

A detector trained on ``scripts/isaac/generate_synthetic_data.py`` output is
run by ``CameraAdapter(detector="model:<path>")`` on the RTX camera image.
The output contract matches the lite surrogate: per detection a pixel centre,
raw class logits over ``medortrace.perception.frontend.CLASSES`` (so
temperature scaling and the soft confusion matrix apply unchanged) and a
visible-fraction estimate.

``<path>`` is either a bare ``state_dict`` or a checkpoint dict with a
``"model"`` / ``"state_dict"`` entry.  torch/torchvision are imported lazily
(they are not needed for the ground-truth surrogate).
"""

from __future__ import annotations

import numpy as np

from medortrace.perception.frontend import CLASSES


class TorchDetector:
    def __init__(self, path: str, device: str | None = None, score_thresh: float = 0.3):
        import torch
        import torchvision
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # no backbone download: every weight comes from the checkpoint
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(CLASSES) + 1)
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict):
            for key in ("model", "state_dict"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        self.model.load_state_dict(ckpt)
        self.model.to(self.device).eval()
        self.th = score_thresh

    def predict(self, rgb: np.ndarray):
        t = self.torch.from_numpy(np.ascontiguousarray(rgb[..., :3]).astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).to(self.device)
        with self.torch.no_grad():
            out = self.model([t])[0]
        res = []
        C = len(CLASSES)
        for box, lab, sc in zip(out["boxes"].cpu().numpy(), out["labels"].cpu().numpy(), out["scores"].cpu().numpy()):
            if sc < self.th or not 1 <= int(lab) <= C:
                continue
            # Faster R-CNN returns a label + score; convert to pseudo-logits so the
            # belief's calibration (temperature + confusion) applies uniformly.
            logits = np.full(C, np.log((1 - sc) / (C - 1) + 1e-6))
            logits[int(lab) - 1] = np.log(sc + 1e-6)
            u, v = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            res.append((float(u), float(v), logits, 1.0))
        return res


def load_detector(path: str) -> TorchDetector:
    return TorchDetector(path)
