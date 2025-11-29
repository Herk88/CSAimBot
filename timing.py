import time
from typing import Optional


class FixedTicker:
    """Maintains a fixed-rate schedule with drift correction."""

    def __init__(self, hz: int) -> None:
        if hz <= 0:
            raise ValueError('Ticker frequency must be positive.')
        self.period_ns = int(1_000_000_000 // hz)
        self.next_ns = time.perf_counter_ns()
        self._hz = hz

    @property
    def hz(self) -> int:
        return self._hz

    def sleep_until_next(self) -> None:
        self.next_ns += self.period_ns
        remaining_ns = self.next_ns - time.perf_counter_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns * 1e-9)
        now = time.perf_counter_ns()
        if now - self.next_ns > 3 * self.period_ns:
            self.next_ns = now


__all__ = ['FixedTicker']
