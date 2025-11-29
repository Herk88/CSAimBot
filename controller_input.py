from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import logging

try:
    from inputs import get_gamepad
except ImportError:  # pragma: no cover - library not present during unit tests
    get_gamepad = None


@dataclass
class ControllerState:
    velocity: np.ndarray
    timestamp: float


class ControllerInputReader:
    """Background reader that tracks right-stick deltas as a velocity hint."""

    def __init__(
        self,
        polling_rate: float = 125.0,
        deadzone: float = 0.15,
        sensitivity: float = 3.0,
        decay_rate: float = 6.0,
        smoothing: float = 0.2,
    ) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._polling_rate = polling_rate
        self._deadzone = deadzone
        self._sensitivity = sensitivity
        self._decay_rate = decay_rate
        self._smoothing = max(0.0, min(1.0, smoothing))
        self._state = ControllerState(velocity=np.zeros(2, dtype=float), timestamp=time.perf_counter())
        self._warned_backend = False
        self._logger = logging.getLogger(__name__)

    def configure(
        self,
        polling_rate: Optional[float] = None,
        deadzone: Optional[float] = None,
        sensitivity: Optional[float] = None,
        decay_rate: Optional[float] = None,
        smoothing: Optional[float] = None,
    ) -> None:
        with self._lock:
            if polling_rate is not None and polling_rate > 0:
                self._polling_rate = polling_rate
            if deadzone is not None:
                self._deadzone = max(0.0, min(deadzone, 1.0))
            if sensitivity is not None and sensitivity > 0:
                self._sensitivity = sensitivity
            if decay_rate is not None and decay_rate > 0:
                self._decay_rate = decay_rate
            if smoothing is not None:
                self._smoothing = max(0.0, min(1.0, smoothing))

    def start(self) -> None:
        if self._running:
            return
        if get_gamepad is None and not self._warned_backend:
            self._logger.warning(
                'Controller input backend not available. Install the `inputs` package to enable velocity hints.'
            )
            self._warned_backend = True
        self._running = True
        self._thread = threading.Thread(target=self._loop, name='controller-input-reader', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=0.5)
        self._thread = None

    def get_velocity(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._running:
                return None
            now = time.perf_counter()
            dt = max(0.0, now - self._state.timestamp)
            if dt > 0.5:
                return None
            decay = math.exp(-dt * self._decay_rate)
            self._state.velocity *= decay
            self._state.timestamp = now
            return self._state.velocity.copy()

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                poll_interval = 1.0 / max(self._polling_rate, 1.0)
            if get_gamepad is None:
                if not self._warned_backend:
                    self._logger.warning(
                        'Controller input backend not available. Install the `inputs` package to enable velocity hints.'
                    )
                    self._warned_backend = True
                time.sleep(poll_interval)
                continue
            try:
                events = get_gamepad()
            except Exception:
                time.sleep(poll_interval)
                continue

            vx = vy = 0.0
            for event in events:
                if event.code in ('ABS_RX', 'ABS_X'):
                    vx = self._apply_deadzone(event.state)
                elif event.code in ('ABS_RY', 'ABS_Y'):
                    vy = self._apply_deadzone(event.state)

            with self._lock:
                vector = np.array([vx, -vy], dtype=float) * self._sensitivity
                if np.allclose(vector, 0.0):
                    decay = math.exp(-poll_interval * self._decay_rate)
                    smoothed = self._state.velocity * decay
                else:
                    mix = self._smoothing
                    smoothed = (1.0 - mix) * self._state.velocity + mix * vector
                self._state = ControllerState(velocity=smoothed, timestamp=time.perf_counter())

            time.sleep(poll_interval)

    def _apply_deadzone(self, raw: int) -> float:
        normalized = float(raw) / 32768.0
        if abs(normalized) < self._deadzone:
            return 0.0
        sign = math.copysign(1.0, normalized)
        adjusted = (abs(normalized) - self._deadzone) / (1.0 - self._deadzone)
        return sign * adjusted


__all__ = ['ControllerInputReader']
