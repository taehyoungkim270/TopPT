"""Deterministic signals from Appendix I of the paper.

Each name starts with the signal's notation in the paper.
"""

import numpy as np

DOMAIN = (0.0, 2.0 * np.pi)
PERIOD = 2.0 * np.pi / 5.0


# Periodic alternatives
def f1_rational(t: np.ndarray) -> np.ndarray:
    return 3.0 / (2.0 - np.cos(5.0 * t))


def f2_log_exp(t: np.ndarray) -> np.ndarray:
    return 5.0 * np.log(np.sin(10.0 * t) + 5.0) + np.exp(np.cos(5.0 * t))


# Non-periodic nulls
def h1_linear(t: np.ndarray) -> np.ndarray:
    return (t - np.pi) / np.pi


def h2_quadratic(t: np.ndarray) -> np.ndarray:
    return h1_linear(t) ** 2


def h3_logistic(t: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-5.0 * h1_linear(t))) - 0.5


def h4_aperiodic(t: np.ndarray) -> np.ndarray:
    return 0.45 * h1_linear(t) + 0.35 * np.sin(1.7 * t + 0.2)


# Near-periodic stress cases: not periodic, and not part of the main null set
def g1_phase_drift(t: np.ndarray) -> np.ndarray:
    return np.sin(5.0 * t + 0.18 * t**2)


def g2_amplitude_modulated(t: np.ndarray) -> np.ndarray:
    return (1.0 + 0.18 * np.cos(0.5 * t) + 0.08 * t) * np.sin(5.0 * t)


SIGNALS = {
    signal.__name__: signal
    for signal in (
        f1_rational,
        f2_log_exp,
        h1_linear,
        h2_quadratic,
        h3_logistic,
        h4_aperiodic,
        g1_phase_drift,
        g2_amplitude_modulated,
    )
}
PERIODIC = {"f1_rational", "f2_log_exp"}


def known_period_tau(m: int, multiplier: float = 1.0) -> float:
    """Return the delay ``multiplier * PERIOD / (m + 1)``, used for every signal."""
    return multiplier * PERIOD / (m + 1)
