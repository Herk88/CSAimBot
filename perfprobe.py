from __future__ import annotations

import csv
import os
import time
from collections import deque
from typing import Deque, Optional


class PerfProbe:
    """Captures timing spans for frame → command latency analysis."""

    def __init__(self, path: Optional[str] = None, window: int = 1000) -> None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.path = path or os.path.join(os.getcwd(), f'perf_{timestamp}.csv')
        self._fh = open(self.path, 'w', newline='')
        self._writer = csv.writer(self._fh)
        self._writer.writerow(['t_frame_ns', 't_detect_ns', 't_predict_ns', 't_cmd_ns', 'had_meas', 'had_hint'])
        self._window: Deque[int] = deque(maxlen=window)

    def log(
        self,
        t_frame_ns: int,
        t_detect_ns: int,
        t_predict_ns: int,
        t_cmd_ns: int,
        had_measurement: bool,
        had_hint: bool,
    ) -> None:
        self._writer.writerow(
            [t_frame_ns, t_detect_ns, t_predict_ns, t_cmd_ns, int(had_measurement), int(had_hint)]
        )
        self._window.append(t_cmd_ns - t_frame_ns)

    def p95_cmd_latency_ms(self) -> float:
        if not self._window:
            return 0.0
        sorted_samples = sorted(self._window)
        idx = min(int(len(sorted_samples) * 0.95), len(sorted_samples) - 1)
        return sorted_samples[idx] / 1_000_000.0

    def jitter_ms(self) -> float:
        if not self._window:
            return 0.0
        sorted_samples = sorted(self._window)
        p95 = sorted_samples[min(int(len(sorted_samples) * 0.95), len(sorted_samples) - 1)]
        p50 = sorted_samples[min(int(len(sorted_samples) * 0.5), len(sorted_samples) - 1)]
        return (p95 - p50) / 1_000_000.0

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.flush()
            self._fh.close()


__all__ = ['PerfProbe']
