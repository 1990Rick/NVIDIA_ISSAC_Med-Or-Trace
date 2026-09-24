"""Detector calibration: temperature scaling and soft confusion matrix.

The item-custody belief treats detector outputs as *soft class counts*, so it
needs (a) calibrated class posteriors and (b) the expected soft mass a
detection of true class ``i`` puts on class ``j`` (the soft confusion matrix).
Both are estimated from labelled validation data:

* Isaac Sim: Replicator-annotated frames (``scripts/calibrate_detector.py``
  with ``--source replicator``);
* lite backend: samples from the surrogate detector (``--source lite``).

Results are stored in ``configs/perception/detector_calibration.yaml``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from medortrace.perception.frontend import CLASSES, softmax


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Minimise NLL of softmax(logits / T) over T in [0.05, 20]."""
    def nll(T):
        p = softmax(logits, T)
        return -np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12))
    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def soft_confusion(logits: np.ndarray, labels: np.ndarray, T: float, n_classes: int = len(CLASSES)) -> np.ndarray:
    P = softmax(logits, T)
    C = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
        m = labels == c
        C[c] = P[m].mean(0) if m.any() else np.eye(n_classes)[c]
    return C


def lite_validation_set(n: int = 20000, seed: int = 0):
    """Samples from the lite surrogate detector across its operating range."""
    from medortrace.sim.sensors_lite import SIMILARITY, CameraConfig
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, len(CLASSES), n)
    vis = rng.uniform(0.2, 1.0, n)
    rf = rng.uniform(0.1, 1.0, n)
    glare = rng.uniform(0.0, 0.9, n)
    margin = CameraConfig().logit_scale * vis * rf * (1 - 0.6 * glare)
    logits = margin[:, None] * SIMILARITY[labels] + rng.normal(0, 0.8, (n, len(CLASSES)))
    return logits, labels


def calibrate_lite(seed: int = 0) -> dict:
    lg, lb = lite_validation_set(seed=seed)
    T = fit_temperature(lg, lb)
    C = soft_confusion(lg, lb, T)
    return {"temperature": T, "classes": CLASSES, "soft_confusion": C.round(4).tolist()}
