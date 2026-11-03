"""Generalized Lomb-Scargle baseline for scalar observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from scipy.signal import lombscargle


@dataclass(frozen=True)
class GLSConfig:
    """Frequency-grid and testing parameters for the GLS baseline."""

    alpha: float = 0.05
    ofac: float = 10.0
    hifac: float = 1.0
    min_period: Optional[float] = None
    max_period: Optional[float] = None

    def validate(self) -> None:
        if not (0 < self.alpha < 1):
            raise ValueError("GLS alpha must lie in (0, 1).")
        if self.ofac <= 0:
            raise ValueError("GLS ofac must be positive.")
        if self.hifac <= 0:
            raise ValueError("GLS hifac must be positive.")
        if self.min_period is not None and self.min_period <= 0:
            raise ValueError("GLS min_period must be positive.")
        if self.max_period is not None and self.max_period <= 0:
            raise ValueError("GLS max_period must be positive.")
        if (
            self.min_period is not None
            and self.max_period is not None
            and self.min_period >= self.max_period
        ):
            raise ValueError("GLS min_period must be smaller than max_period.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GLSResult:
    reject: bool
    fap: float
    max_power: float
    power_threshold: float
    best_frequency: float
    best_period: float
    n_frequencies: int
    frequency_min: float
    frequency_max: float

    def summary_dict(self) -> dict:
        return {
            "gls_reject": int(self.reject),
            "gls_fap": float(self.fap),
            "gls_max_power": float(self.max_power),
            "gls_power_threshold": float(self.power_threshold),
            "gls_best_frequency": float(self.best_frequency),
            "gls_best_period": float(self.best_period),
            "gls_n_frequencies": int(self.n_frequencies),
            "gls_frequency_min": float(self.frequency_min),
            "gls_frequency_max": float(self.frequency_max),
        }


def build_frequency_grid(times: np.ndarray, config: GLSConfig) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 4:
        raise ValueError("GLS requires at least four one-dimensional time points.")
    tbase = float(np.max(times) - np.min(times))
    if tbase <= 0:
        raise ValueError("GLS time span must be positive.")

    fstep = 1.0 / (tbase * float(config.ofac))
    fnyq = 0.5 * float(times.size) / tbase
    f_min = fstep if config.max_period is None else 1.0 / float(config.max_period)
    f_max = fnyq * float(config.hifac) if config.min_period is None else 1.0 / float(config.min_period)
    if f_max <= f_min:
        raise ValueError(f"Invalid GLS frequency range: f_min={f_min}, f_max={f_max}.")
    return np.arange(f_min, f_max, fstep, dtype=float)


def gls_fap(max_power: float, n_points: int, n_independent: float) -> float:
    """Approximate false-alarm probability used by the classic GLS code.

    For Zechmeister-Kuerster normalized power p, the single-frequency tail is
    approximately (1-p)^((N-3)/2).  The full-band FAP then uses M independent
    frequencies.
    """

    if n_points <= 3 or n_independent <= 0 or not np.isfinite(max_power):
        return np.nan
    p = float(np.clip(max_power, 0.0, 1.0 - 1e-15))
    prob = (1.0 - p) ** ((float(n_points) - 3.0) / 2.0)
    if n_independent * prob <= 0.01:
        return float(n_independent * prob)
    return float(1.0 - (1.0 - prob) ** n_independent)


def gls_power_level(alpha: float, n_points: int, n_independent: float) -> float:
    if n_points <= 3 or n_independent <= 0:
        return np.nan
    prob = 1.0 - (1.0 - float(alpha)) ** (1.0 / float(n_independent))
    return float(1.0 - prob ** (2.0 / (float(n_points) - 3.0)))


def run_gls_periodicity_test(
    times: np.ndarray,
    values: np.ndarray,
    config: GLSConfig,
) -> GLSResult:
    """Run a GLS periodogram test on scalar observations."""

    config.validate()
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape:
        raise ValueError("GLS times and values must have the same shape.")
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if times.size < 4:
        raise ValueError("GLS requires at least four finite observations.")

    frequencies = build_frequency_grid(times, config)
    if frequencies.size == 0:
        raise ValueError("GLS frequency grid is empty.")

    centered_times = times - np.min(times)
    if float(np.var(values)) <= 0:
        max_power = 0.0
        best_frequency = float(frequencies[0])
    else:
        power = lombscargle(
            centered_times,
            values,
            2.0 * np.pi * frequencies,
            floating_mean=True,
            normalize=True,
        )
        power = np.asarray(power, dtype=float)
        power = np.where(np.isfinite(power), power, -np.inf)
        best_index = int(np.argmax(power))
        max_power = float(max(power[best_index], 0.0))
        best_frequency = float(frequencies[best_index])

    tbase = float(np.max(times) - np.min(times))
    n_independent = max(1.0, float((frequencies[-1] - frequencies[0]) * tbase))
    fap = gls_fap(max_power=max_power, n_points=times.size, n_independent=n_independent)
    threshold = gls_power_level(alpha=config.alpha, n_points=times.size, n_independent=n_independent)
    reject = bool(np.isfinite(fap) and fap <= config.alpha)
    return GLSResult(
        reject=reject,
        fap=float(fap),
        max_power=float(max_power),
        power_threshold=float(threshold),
        best_frequency=best_frequency,
        best_period=float(1.0 / best_frequency),
        n_frequencies=int(frequencies.size),
        frequency_min=float(frequencies[0]),
        frequency_max=float(frequencies[-1]),
    )
