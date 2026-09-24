"""EKF localisation (odometry + gyro prediction, landmark range/bearing update)
with consistency monitoring.

Outputs a pose, its covariance and two health signals consumed by the safety
supervisor and the drift-vs-map-change diagnosis (CF-D):

* ``nis_avg``  - windowed normalised innovation squared; E[NIS] = 2 for a
  consistent 2-D measurement model, large values indicate drift/corruption;
* ``pos_std``  - sqrt of the largest eigenvalue of the xy covariance.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from medortrace.common.geometry import wrap_angle
from medortrace.common.msgs import ImuSample, LandmarkFrame, WheelOdometry


class EkfLocalizer:
    CHI2_2DOF_99 = 9.21

    def __init__(self, init_pose: np.ndarray, landmarks: dict[str, np.ndarray], q_v: float = 0.02,
                 q_w: float = 0.02, r_range: float = 0.05, r_bearing: float = 0.02,
                 init_std=(0.05, 0.05, 0.02)):
        self.x = np.asarray(init_pose, float).copy()
        self.P = np.diag(np.square(init_std))
        self.lm = {k: np.asarray(v, float)[:2] for k, v in landmarks.items()}
        self.q_v, self.q_w = q_v, q_w
        self.R = np.diag([r_range ** 2, r_bearing ** 2])
        self.nis_hist: deque = deque(maxlen=40)
        self.rejected = 0
        self.accepted = 0
        self.last_update_t = 0.0
        self.innov_hist: deque = deque(maxlen=60)   # (landmark_id, innovation) for drift diagnosis

    def predict(self, odom: WheelOdometry | None, imu: list[ImuSample], dt: float) -> None:
        v = odom.v if odom is not None else 0.0
        w = odom.omega if odom is not None else 0.0
        q_scale = 1.0 if odom is not None else 6.0      # odometry dropout -> inflate process noise
        if imu:
            gz = float(np.mean([s.ang_vel[2] for s in imu]))
            w = 0.5 * w + 0.5 * gz if odom is not None else gz
        th = self.x[2]
        self.x = self.x + np.array([v * dt * np.cos(th), v * dt * np.sin(th), w * dt])
        self.x[2] = wrap_angle(self.x[2])
        F = np.array([[1, 0, -v * dt * np.sin(th)], [0, 1, v * dt * np.cos(th)], [0, 0, 1]])
        G = np.array([[dt * np.cos(th), 0], [dt * np.sin(th), 0], [0, dt]])
        Q = np.diag([(self.q_v * q_scale + 0.05 * abs(v)) ** 2, (self.q_w * q_scale + 0.05 * abs(w)) ** 2])
        self.P = F @ self.P @ F.T + G @ Q @ G.T + np.diag([1e-6, 1e-6, 1e-7])

    def update_landmarks(self, frame: LandmarkFrame, t: float) -> None:
        for ob in frame.observations:
            if ob.landmark_id not in self.lm:
                continue
            L = self.lm[ob.landmark_id]
            dx, dy = L - self.x[:2]
            q = dx * dx + dy * dy
            r = np.sqrt(q)
            z_hat = np.array([r, wrap_angle(np.arctan2(dy, dx) - self.x[2])])
            H = np.array([[-dx / r, -dy / r, 0], [dy / q, -dx / q, -1]])
            y = np.array([ob.range - z_hat[0], wrap_angle(ob.bearing - z_hat[1])])
            S = H @ self.P @ H.T + self.R
            nis = float(y @ np.linalg.solve(S, y))
            self.nis_hist.append(nis)
            self.innov_hist.append((ob.landmark_id, y.copy()))
            if nis > self.CHI2_2DOF_99 * 4:
                self.rejected += 1
                continue
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.x[2] = wrap_angle(self.x[2])
            I_KH = np.eye(3) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T      # Joseph form
            self.accepted += 1
            self.last_update_t = t

    @property
    def nis_avg(self) -> float:
        return float(np.mean(self.nis_hist)) if self.nis_hist else 2.0

    @property
    def pos_std(self) -> float:
        return float(np.sqrt(np.max(np.linalg.eigvalsh(self.P[:2, :2]))))

    @property
    def heading_std(self) -> float:
        return float(np.sqrt(self.P[2, 2]))

    def rejection_rate(self) -> float:
        n = self.rejected + self.accepted
        return self.rejected / n if n else 0.0
