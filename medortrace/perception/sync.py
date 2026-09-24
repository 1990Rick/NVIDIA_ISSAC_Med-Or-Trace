"""Time synchronisation, skew detection and motion compensation.

Sensor clocks in a real robot drift and jump (PTP loss, USB camera clocks,
NTP steps).  The stack keeps a history of *estimated* robot poses keyed by
host receive time and, for every sensor:

* tracks the offset ``stamp - recv_stamp`` with a robust running median;
* flags the sensor as *skewed* when the offset leaves the tolerance band;
* when skewed, re-times the message with ``recv_stamp - nominal_latency``
  (trusting the host clock) instead of the corrupted sensor stamp;
* rejects stale data older than ``max_age``.

The corrected time is used to look up / interpolate the pose at which the
measurement was taken (motion compensation).
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from medortrace.common.geometry import wrap_angle


class PoseHistory:
    def __init__(self, horizon_s: float = 5.0):
        self.buf: deque = deque()
        self.horizon = horizon_s

    def add(self, t: float, pose: np.ndarray) -> None:
        self.buf.append((t, pose.copy()))
        while self.buf and self.buf[0][0] < t - self.horizon:
            self.buf.popleft()

    def at(self, t: float) -> np.ndarray | None:
        if not self.buf:
            return None
        if t <= self.buf[0][0]:
            return self.buf[0][1].copy()
        if t >= self.buf[-1][0]:
            # bounded extrapolation using last two samples
            if len(self.buf) < 2:
                return self.buf[-1][1].copy()
            (t0, p0), (t1, p1) = self.buf[-2], self.buf[-1]
            a = min((t - t1) / max(t1 - t0, 1e-6), 5.0)
            p = p1 + a * np.array([p1[0] - p0[0], p1[1] - p0[1], wrap_angle(p1[2] - p0[2])])
            p[2] = wrap_angle(p[2])
            return p
        for (ta, pa), (tb, pb) in zip(list(self.buf)[:-1], list(self.buf)[1:]):
            if ta <= t <= tb:
                w = (t - ta) / max(tb - ta, 1e-9)
                p = pa + w * np.array([pb[0] - pa[0], pb[1] - pa[1], wrap_angle(pb[2] - pa[2])])
                p[2] = wrap_angle(p[2])
                return p
        return self.buf[-1][1].copy()


class TimeSyncMonitor:
    def __init__(self, tolerance_s: float = 0.05, nominal_latency_s: float = 0.0, max_age_s: float = 0.5,
                 window: int = 15, enabled: bool = True):
        self.tol = tolerance_s
        self.lat = nominal_latency_s
        self.max_age = max_age_s
        self.enabled = enabled
        self.offsets: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.skewed: dict[str, bool] = defaultdict(bool)
        self.rejected: dict[str, int] = defaultdict(int)

    def correct(self, sensor: str, stamp: float, recv: float) -> float | None:
        """Return the time to use for this measurement, or None to reject it."""
        off = stamp - recv
        self.offsets[sensor].append(off)
        med = float(np.median(self.offsets[sensor]))
        self.skewed[sensor] = abs(med - (-self.lat)) > self.tol
        if not self.enabled:
            return stamp
        t_use = recv - self.lat if self.skewed[sensor] else stamp
        if recv - t_use > self.max_age:
            self.rejected[sensor] += 1
            return None
        return t_use

    def health(self) -> dict[str, float]:
        return {s: float(np.median(o)) for s, o in self.offsets.items()}

    def any_skewed(self) -> bool:
        return any(self.skewed.values())
