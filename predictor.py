from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KalmanParams:
    q_process: float = 20.0
    r_meas_px: float = 64.0
    p0_pos: float = 5_000.0
    p0_vel: float = 200.0
    max_dt: float = 0.05
    min_dt: float = 1e-3
    drop_grace_frames: int = 8
    vel_hint_decay: float = 0.9


class KalmanPredictor:
    """Simple constant-velocity Kalman filter for 2D aim prediction."""

    def __init__(self, params: Optional[KalmanParams] = None) -> None:
        self.params = params or KalmanParams()
        self._state: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None
        self._gap_frames = 0
        self._inflated = False

    def reset(self) -> None:
        self._state = None
        self._covariance = None
        self._gap_frames = 0
        self._inflated = False

    def configure(self, params: KalmanParams) -> None:
        self.params = params
        self.reset()

    def state_xy(self) -> Optional[np.ndarray]:
        if self._state is None:
            return None
        return self._state[:2].copy()

    def state_velocity(self) -> Optional[np.ndarray]:
        if self._state is None:
            return None
        return self._state[2:].copy()

    def covariance(self) -> Optional[np.ndarray]:
        if self._covariance is None:
            return None
        return self._covariance.copy()

    def step(
        self,
        measurement: Optional[np.ndarray],
        dt: float,
        velocity_hint: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        dt = float(max(self.params.min_dt, min(dt, self.params.max_dt)))
        predicted = self.predict(dt, velocity_hint=velocity_hint)
        if measurement is not None:
            return self.update(measurement)
        return predicted

    def predict(self, dt: float, velocity_hint: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        dt = float(max(self.params.min_dt, min(dt, self.params.max_dt)))

        if self._state is None:
            return None

        mix = max(0.0, min(1.0, self.params.vel_hint_decay))
        if velocity_hint is not None:
            velocity_hint = np.asarray(velocity_hint, dtype=float).reshape(2)
            self._state[2:] = mix * self._state[2:] + (1.0 - mix) * velocity_hint
        else:
            self._state[2:] *= mix

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        control = np.array(
            [
                [0.5 * dt * dt, 0.0],
                [0.0, 0.5 * dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=float,
        )
        q_matrix = (control @ control.T) * self.params.q_process
        self._state = transition @ self._state
        if self._covariance is None:
            self._covariance = self._initial_covariance()
        self._covariance = transition @ self._covariance @ transition.T + q_matrix

        self._gap_frames += 1
        if self._gap_frames > self.params.drop_grace_frames and not self._inflated:
            self._covariance *= 3.0
            self._inflated = True

        return self._state[:2].copy()

    def update(self, measurement: np.ndarray) -> Optional[np.ndarray]:
        measurement = np.asarray(measurement, dtype=float).reshape(2)
        if self._state is None:
            self._state = np.zeros(4, dtype=float)
            self._state[:2] = measurement
            self._covariance = self._initial_covariance()
            self._gap_frames = 0
            self._inflated = False
            return self._state[:2].copy()

        observation = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        r_matrix = np.eye(2, dtype=float) * self.params.r_meas_px
        predicted_measurement = observation @ self._state
        innovation = measurement - predicted_measurement
        innovation_covariance = observation @ self._covariance @ observation.T + r_matrix
        kalman_gain = self._covariance @ observation.T @ np.linalg.inv(innovation_covariance)

        self._state = self._state + kalman_gain @ innovation
        identity = np.eye(4, dtype=float)
        self._covariance = (identity - kalman_gain @ observation) @ self._covariance
        self._gap_frames = 0
        self._inflated = False
        return self._state[:2].copy()

    def _initial_covariance(self) -> np.ndarray:
        return np.diag([self.params.p0_pos, self.params.p0_pos, self.params.p0_vel, self.params.p0_vel]).astype(float)


__all__ = ['KalmanPredictor', 'KalmanParams']
