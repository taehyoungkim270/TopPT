"""Classical p-value based periodogram tests for scalar observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
from scipy.special import gammaln


@dataclass(frozen=True)
class FisherGResult:
    reject: bool
    p_value: float
    statistic: float
    best_frequency: float
    best_period: float
    n_frequencies: int

    def summary_dict(self) -> dict:
        return {
            "fisher_reject": int(self.reject),
            "fisher_p_value": float(self.p_value),
            "fisher_g": float(self.statistic),
            "fisher_best_frequency": float(self.best_frequency),
            "fisher_best_period": float(self.best_period),
            "fisher_n_frequencies": int(self.n_frequencies),
        }


@dataclass(frozen=True)
class PermutationPeriodogramConfig:
    alpha: float = 0.05
    n_permutations: int = 500
    seed: Optional[int] = None

    def validate(self) -> None:
        if not (0 < self.alpha < 1):
            raise ValueError("Permutation periodogram alpha must lie in (0, 1).")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PermutationPeriodogramResult:
    reject: bool
    p_value: float
    statistic: float
    best_frequency: float
    best_period: float
    n_frequencies: int
    n_permutations: int

    def summary_dict(self) -> dict:
        return {
            "perm_reject": int(self.reject),
            "perm_p_value": float(self.p_value),
            "perm_statistic": float(self.statistic),
            "perm_best_frequency": float(self.best_frequency),
            "perm_best_period": float(self.best_period),
            "perm_n_frequencies": int(self.n_frequencies),
            "perm_n_permutations": int(self.n_permutations),
        }


def _prepare_regular_series(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape:
        raise ValueError("Periodogram test times and values must have the same shape.")
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]
    if times.size < 4:
        raise ValueError("Periodogram tests require at least four finite observations.")
    diffs = np.diff(times)
    spacing = float(np.median(diffs))
    if spacing <= 0:
        raise ValueError("Periodogram test times must be strictly increasing.")
    if not np.allclose(diffs, spacing, rtol=1e-5, atol=1e-10):
        raise ValueError("Fisher and permutation periodogram tests require a regular time grid.")
    return times, values, spacing


def _fourier_periodogram_from_values(values: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    centered = values - float(np.mean(values))
    spectrum = np.fft.rfft(centered)
    frequencies = np.fft.rfftfreq(n, d=spacing)

    if n % 2 == 0:
        stop = n // 2
    else:
        stop = n // 2 + 1
    powers = np.abs(spectrum[1:stop]) ** 2
    freqs = frequencies[1:stop]
    return freqs.astype(float), powers.astype(float)


def _max_periodogram_statistic(values: np.ndarray, spacing: float) -> tuple[float, float, float, int]:
    freqs, powers = _fourier_periodogram_from_values(values, spacing)
    if powers.size == 0:
        return 0.0, np.nan, np.nan, 0
    total = float(np.sum(powers))
    if total <= 0 or not np.isfinite(total):
        return 0.0, float(freqs[0]), float(1.0 / freqs[0]), int(freqs.size)
    best_index = int(np.argmax(powers))
    statistic = float(powers[best_index] / total)
    best_frequency = float(freqs[best_index])
    return statistic, best_frequency, float(1.0 / best_frequency), int(freqs.size)


def fisher_g_p_value(g_statistic: float, n_frequencies: int) -> float:
    """Exact Fisher g-test tail probability under the white-noise null."""

    if n_frequencies <= 0 or not np.isfinite(g_statistic):
        return 1.0
    g = float(np.clip(g_statistic, 0.0, 1.0))
    if g <= 0:
        return 1.0
    max_k = min(n_frequencies, int(np.floor(1.0 / g)))
    total = 0.0
    for k in range(1, max_k + 1):
        base = 1.0 - float(k) * g
        if base <= 0:
            continue
        log_term = (
            gammaln(n_frequencies + 1)
            - gammaln(k + 1)
            - gammaln(n_frequencies - k + 1)
            + float(n_frequencies - 1) * np.log(base)
        )
        term = float(np.exp(log_term))
        total += term if k % 2 == 1 else -term
    return float(np.clip(total, 0.0, 1.0))


def run_fisher_g_test(times: np.ndarray, values: np.ndarray, alpha: float = 0.05) -> FisherGResult:
    """Run Fisher's g-test on a regularly sampled scalar time series."""

    if not (0 < alpha < 1):
        raise ValueError("Fisher alpha must lie in (0, 1).")
    _, clean_values, spacing = _prepare_regular_series(times, values)
    statistic, best_frequency, best_period, n_frequencies = _max_periodogram_statistic(clean_values, spacing)
    p_value = fisher_g_p_value(statistic, n_frequencies)
    return FisherGResult(
        reject=bool(p_value <= alpha),
        p_value=float(p_value),
        statistic=float(statistic),
        best_frequency=float(best_frequency),
        best_period=float(best_period),
        n_frequencies=int(n_frequencies),
    )


def run_permutation_max_periodogram_test(
    times: np.ndarray,
    values: np.ndarray,
    config: PermutationPeriodogramConfig,
) -> PermutationPeriodogramResult:
    """Permutation test using the maximum normalized Fourier periodogram ordinate."""

    config.validate()
    _, clean_values, spacing = _prepare_regular_series(times, values)
    statistic, best_frequency, best_period, n_frequencies = _max_periodogram_statistic(clean_values, spacing)
    if n_frequencies <= 0:
        p_value = 1.0
    else:
        rng = np.random.default_rng(config.seed)
        centered = clean_values - float(np.mean(clean_values))
        permuted = np.empty((int(config.n_permutations), centered.size), dtype=float)
        for row in range(int(config.n_permutations)):
            permuted[row, :] = rng.permutation(centered)

        spectrum = np.fft.rfft(permuted, axis=1)
        if centered.size % 2 == 0:
            stop = centered.size // 2
        else:
            stop = centered.size // 2 + 1
        powers = np.abs(spectrum[:, 1:stop]) ** 2
        totals = np.sum(powers, axis=1)
        perm_stats = np.divide(
            np.max(powers, axis=1),
            totals,
            out=np.zeros_like(totals, dtype=float),
            where=totals > 0,
        )
        p_value = float((1 + np.count_nonzero(perm_stats >= statistic - 1e-15)) / (config.n_permutations + 1))

    return PermutationPeriodogramResult(
        reject=bool(p_value <= config.alpha),
        p_value=float(p_value),
        statistic=float(statistic),
        best_frequency=float(best_frequency),
        best_period=float(best_period),
        n_frequencies=int(n_frequencies),
        n_permutations=int(config.n_permutations),
    )


def with_seed(config: PermutationPeriodogramConfig, seed: int | None) -> PermutationPeriodogramConfig:
    return replace(config, seed=seed)
