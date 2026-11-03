"""Raw time-delay embedding utilities."""

from __future__ import annotations

import numpy as np


def lag_matrix(anchor_times: np.ndarray, m: int, tau: float) -> np.ndarray:
    if m < 1:
        raise ValueError("m must be at least 1.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    anchor_times = np.asarray(anchor_times, dtype=float)
    if anchor_times.ndim != 1:
        raise ValueError("anchor_times must be one-dimensional.")
    lags = np.arange(m + 1, dtype=float) * float(tau)
    return anchor_times[:, None] + lags[None, :]


def restrict_anchors_to_domain(
    anchor_times: np.ndarray,
    m: int,
    tau: float,
    eval_domain: tuple[float, float],
) -> np.ndarray:
    anchor_times = np.asarray(anchor_times, dtype=float)
    lo, hi = map(float, eval_domain)
    total_span = float(m) * float(tau)
    mask = (anchor_times >= lo) & (anchor_times + total_span <= hi + 1e-12)
    return anchor_times[mask]


def sliding_window_embedding(
    evaluator,
    anchor_times: np.ndarray,
    m: int,
    tau: float,
) -> np.ndarray:
    query = lag_matrix(anchor_times, m=m, tau=tau)
    values = evaluator(query.reshape(-1)).reshape(query.shape)
    return np.asarray(values, dtype=float)
