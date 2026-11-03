"""Sampling and synthetic noise utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import truncnorm

from .config import NoiseConfig


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_uniform_times(
    n: int,
    domain: tuple[float, float],
    seed: int | None = None,
    sort: bool = False,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive.")
    lo, hi = map(float, domain)
    if not lo < hi:
        raise ValueError("domain must satisfy lo < hi.")
    rng = make_rng(seed)
    t = rng.uniform(lo, hi, size=int(n))
    return np.sort(t) if sort else t


def add_noise(values: np.ndarray, config: NoiseConfig, seed: int | None = None) -> np.ndarray:
    config.validate()
    values = np.asarray(values, dtype=float)
    if config.kind == "none" or config.scale == 0:
        return values.copy()

    rng = make_rng(seed)
    scale = float(config.scale)
    if config.kind == "AG":
        noise = rng.normal(0.0, scale, size=values.shape)
        return values + noise
    if config.kind == "AL":
        noise = rng.laplace(0.0, scale / np.sqrt(2.0), size=values.shape)
        return values + noise
    if config.kind == "MG":
        noise = rng.normal(0.0, scale, size=values.shape)
        return values * (1.0 + noise)
    if config.kind == "ML":
        noise = rng.laplace(0.0, scale / np.sqrt(2.0), size=values.shape)
        return values * (1.0 + noise)
    if config.kind in {"BU", "bounded"}:
        noise = rng.uniform(-scale, scale, size=values.shape)
        return values + noise
    if config.kind == "TG":
        std = scale / 2.0
        noise = truncnorm.rvs(
            -scale / std,
            scale / std,
            loc=0.0,
            scale=std,
            size=values.shape,
            random_state=rng,
        )
        return values + noise
    raise AssertionError("NoiseConfig.validate should have caught this.")


def evaluate_signal(func, times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    values = func(times)
    values = np.asarray(values, dtype=float)
    if values.shape != times.shape:
        values = np.broadcast_to(values, times.shape).astype(float)
    return values
