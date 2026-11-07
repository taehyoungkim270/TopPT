"""Data-generating processes for the synthetic experiments."""

from .noise import NoiseModel
from .signals import DOMAIN, PERIOD, PERIODIC, SIGNALS, known_period_tau
from .synthetic import generate_observations, interpolation_error_bound

__all__ = [
    "DOMAIN",
    "NoiseModel",
    "PERIOD",
    "PERIODIC",
    "SIGNALS",
    "generate_observations",
    "interpolation_error_bound",
    "known_period_tau",
]
