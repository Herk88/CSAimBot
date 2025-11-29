from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import numpy as np

from predictor import KalmanParams, KalmanPredictor


@dataclass
class SweepResult:
    mse: float
    q_process: float
    r_meas_px: float


def evaluate(
    trace: Sequence[Tuple[int, float, float, float, float, bool]],
    q_process: float,
    r_meas_px: float,
    params: KalmanParams,
) -> float:
    tuning = KalmanParams(
        q_process=q_process,
        r_meas_px=r_meas_px,
        p0_pos=params.p0_pos,
        p0_vel=params.p0_vel,
        max_dt=params.max_dt,
        min_dt=params.min_dt,
        drop_grace_frames=params.drop_grace_frames,
        vel_hint_decay=params.vel_hint_decay,
    )
    kf = KalmanPredictor(tuning)
    last_t = trace[0][0]
    errors = []

    for t_ns, meas_x, meas_y, hint_vx, hint_vy, had_meas in trace:
        dt = (t_ns - last_t) * 1e-9
        last_t = t_ns
        kf.predict(dt, velocity_hint=np.array([hint_vx, hint_vy], dtype=float))
        if had_meas:
            kf.update(np.array([meas_x, meas_y], dtype=float))
        state = kf.state_xy()
        if state is None:
            continue
        errors.append((state[0] - meas_x) ** 2 + (state[1] - meas_y) ** 2)

    return float(np.mean(errors)) if errors else float('inf')


def sweep(
    trace: Sequence[Tuple[int, float, float, float, float, bool]],
    q_values: Iterable[float],
    r_values: Iterable[float],
    base_params: KalmanParams,
) -> SweepResult:
    best = SweepResult(mse=float('inf'), q_process=base_params.q_process, r_meas_px=base_params.r_meas_px)
    for q in q_values:
        for r in r_values:
            mse = evaluate(trace, q, r, base_params)
            if mse < best.mse:
                best = SweepResult(mse=mse, q_process=q, r_meas_px=r)
    return best


__all__ = ['evaluate', 'sweep', 'SweepResult']
