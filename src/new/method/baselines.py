"""Scalar periodogram tests compared with TopPT.

Each test runs on the same scalar observations used to build the delay cloud.
"""

import numpy as np
from scipy.signal import lombscargle
from scipy.special import gammaln


def _gls_false_alarm(max_power: float, n_points: int, n_independent: float) -> float:
    """Return the Zechmeister-Kuerster false-alarm probability of the largest power."""
    power = np.clip(max_power, 0.0, 1.0 - 1e-15)
    tail = (1.0 - power) ** ((n_points - 3.0) / 2.0)
    if n_independent * tail <= 0.01:
        return float(n_independent * tail)
    return float(1.0 - (1.0 - tail) ** n_independent)


def gls_test(
    times: np.ndarray,
    values: np.ndarray,
    alpha: float,
    oversampling: float = 10.0,
    min_period: float | None = None,
    max_period: float | None = None,
) -> dict:
    """Run the generalized Lomb-Scargle test.

    By default the search runs from one cycle per ``oversampling`` time spans up
    to the average Nyquist frequency. ``min_period`` and ``max_period`` replace
    the upper and lower ends of that range.
    """
    span = times[-1] - times[0]
    step = 1.0 / (span * oversampling)
    low = step if max_period is None else 1.0 / max_period
    high = 0.5 * times.size / span if min_period is None else 1.0 / min_period
    frequencies = np.arange(low, high, step)
    power = lombscargle(
        times - times[0],
        values,
        2.0 * np.pi * frequencies,
        floating_mean=True,
        normalize=True,
    )
    n_independent = max(1.0, (frequencies[-1] - frequencies[0]) * span)
    false_alarm = _gls_false_alarm(power.max(), times.size, n_independent)
    return {
        "gls_reject": false_alarm <= alpha,
        "gls_fap": false_alarm,
        "gls_max_power": max(float(power.max()), 0.0),
    }


def _check_regular_grid(times: np.ndarray) -> None:
    gaps = np.diff(times)
    if not np.allclose(gaps, np.median(gaps), rtol=1e-5, atol=1e-10):
        raise ValueError("The Fisher and permutation tests need a regular time grid.")


def _max_normalized_power(centered: np.ndarray) -> np.ndarray:
    """Return the largest share of power at any Fourier frequency, along the last axis.

    The zero frequency, and for even lengths the Nyquist frequency, are left out.
    """
    stop = (centered.shape[-1] + 1) // 2
    power = np.abs(np.fft.rfft(centered, axis=-1)[..., 1:stop]) ** 2
    return power.max(axis=-1) / power.sum(axis=-1)


def fisher_g_test(times: np.ndarray, values: np.ndarray, alpha: float) -> dict:
    """Run Fisher's g-test, with its exact p-value under the white-noise null."""
    _check_regular_grid(times)
    g = float(_max_normalized_power(values - np.mean(values)))
    n = (values.size + 1) // 2 - 1
    p_value = 0.0
    for k in range(1, min(n, int(np.floor(1.0 / g))) + 1):
        base = 1.0 - k * g
        if base > 0:
            log_choose = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            term = float(np.exp(log_choose + (n - 1) * np.log(base)))
            p_value += term if k % 2 == 1 else -term
    p_value = float(np.clip(p_value, 0.0, 1.0))
    return {"fisher_reject": p_value <= alpha, "fisher_p_value": p_value, "fisher_g": g}


def permutation_test(
    times: np.ndarray,
    values: np.ndarray,
    alpha: float,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    """Run a permutation test on the same statistic as Fisher's g-test."""
    _check_regular_grid(times)
    centered = values - np.mean(values)
    statistic = float(_max_normalized_power(centered))
    permuted = np.array([rng.permutation(centered) for _ in range(n_permutations)])
    exceed = np.count_nonzero(_max_normalized_power(permuted) >= statistic - 1e-15)
    p_value = (1 + exceed) / (n_permutations + 1)
    return {"perm_reject": p_value <= alpha, "perm_p_value": p_value, "perm_statistic": statistic}
