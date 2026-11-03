"""Synthetic signal definitions.

Each signal is a vectorized callable with metadata.  Periodic examples expose
their fundamental period.  Non-periodic examples expose a reference period used
only for choosing comparable synthetic delays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


ArrayFunc = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    func: ArrayFunc
    domain: tuple[float, float]
    is_periodic: bool
    period: Optional[float] = None
    reference_period: Optional[float] = None
    description: str = ""

    def delay_period(self) -> float:
        if self.period is not None:
            return float(self.period)
        if self.reference_period is not None:
            return float(self.reference_period)
        raise ValueError(f"Signal {self.name} has no period or reference_period.")


def f1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 3.0 / (2.0 - np.cos(5.0 * x))


def f2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 5.0 * np.log(np.sin(10.0 * x) + 5.0) + np.exp(np.cos(5.0 * x))


def h0_linear_trend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - np.pi) / np.pi


def h0_quadratic_trend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    u = (x - np.pi) / np.pi
    return u**2


def h0_logistic_trend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    u = (x - np.pi) / np.pi
    return 1.0 / (1.0 + np.exp(-5.0 * u)) - 0.5


def h0_aperiodic_trend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    u = (x - np.pi) / np.pi
    return 0.45 * u + 0.35 * np.sin(1.7 * x + 0.2)


def f7_phase_drift(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sin(5.0 * x + 0.18 * x**2)


def f8_amplitude_modulated(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (1.0 + 0.18 * np.cos(0.5 * x) + 0.08 * x) * np.sin(5.0 * x)


DEFAULT_DOMAIN = (0.0, 2.0 * np.pi)
PERIOD_5 = 2.0 * np.pi / 5.0


SIGNALS: dict[str, SignalSpec] = {
    "f1_periodic": SignalSpec(
        name="f1_periodic",
        func=f1,
        domain=DEFAULT_DOMAIN,
        is_periodic=True,
        period=PERIOD_5,
        description="Rational cosine periodic signal from the paper simulations.",
    ),
    "f2_periodic": SignalSpec(
        name="f2_periodic",
        func=f2,
        domain=DEFAULT_DOMAIN,
        is_periodic=True,
        period=PERIOD_5,
        description="Log-sine plus exponential-cosine periodic signal.",
    ),
    "h0_linear_trend": SignalSpec(
        name="h0_linear_trend",
        func=h0_linear_trend,
        domain=DEFAULT_DOMAIN,
        is_periodic=False,
        reference_period=PERIOD_5,
        description="Clean H0: centered linear trend with no recurrent loop structure.",
    ),
    "h0_quadratic_trend": SignalSpec(
        name="h0_quadratic_trend",
        func=h0_quadratic_trend,
        domain=DEFAULT_DOMAIN,
        is_periodic=False,
        reference_period=PERIOD_5,
        description="Clean H0: smooth quadratic trend on the finite interval.",
    ),
    "h0_logistic_trend": SignalSpec(
        name="h0_logistic_trend",
        func=h0_logistic_trend,
        domain=DEFAULT_DOMAIN,
        is_periodic=False,
        reference_period=PERIOD_5,
        description="Clean H0: saturating monotone trend.",
    ),
    "h0_aperiodic_trend": SignalSpec(
        name="h0_aperiodic_trend",
        func=h0_aperiodic_trend,
        domain=DEFAULT_DOMAIN,
        is_periodic=False,
        reference_period=PERIOD_5,
        description="Harder clean H0: finite-interval trend with one aperiodic oscillatory component.",
    ),
    "f7_phase_drift": SignalSpec(
        name="f7_phase_drift",
        func=f7_phase_drift,
        domain=DEFAULT_DOMAIN,
        is_periodic=False,
        reference_period=PERIOD_5,
        description="Appendix stress case: nonstationary oscillatory signal with slowly drifting phase.",
    ),
    "f8_amplitude_modulated": SignalSpec(
        name="f8_amplitude_modulated",
        func=f8_amplitude_modulated,
        domain=DEFAULT_DOMAIN,
        is_periodic=False,
        reference_period=PERIOD_5,
        description="Appendix stress case: sinusoid with non-periodic amplitude modulation.",
    ),
}


def get_signal(name: str) -> SignalSpec:
    try:
        return SIGNALS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown signal {name!r}. Available: {sorted(SIGNALS)}") from exc
