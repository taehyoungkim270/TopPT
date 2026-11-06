"""Noisy scalar observations of a signal on a regular grid."""

from collections.abc import Callable

import numpy as np

from .noise import NoiseModel
from .signals import DOMAIN


def generate_observations(
    signal: Callable[[np.ndarray], np.ndarray],
    noise: NoiseModel,
    n_observations: int,
    extension: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return observation times and noisy values on a regular grid.

    The grid covers ``DOMAIN`` plus ``extension`` past its end. Setting
    ``extension = m * tau`` gives every observed time in ``DOMAIN`` a full delay
    window. Randomness comes only from ``rng``.
    """
    times = np.linspace(DOMAIN[0], DOMAIN[1] + extension, n_observations)
    return times, signal(times) + noise.sample(times.shape, rng)


def interpolation_error_bound(
    signal: Callable[[np.ndarray], np.ndarray],
    noise: NoiseModel,
    times: np.ndarray,
    m: int,
) -> float:
    """Return zeta, the bound on the error of a delay vector, for the robust rule.

    ``zeta = sqrt(m + 1) * (sigma + L * delta**2 / 8)``, where ``sigma`` is the noise
    bound and ``delta`` is the largest gap between observed times. ``L`` is a numerical
    estimate of the largest second derivative, computed from the true signal, so this
    belongs here and not with the method.
    """
    grid = np.linspace(times[0], times[-1], 5000)
    first = np.gradient(signal(grid), grid, edge_order=2)
    second = np.gradient(first, grid, edge_order=2)
    delta = np.max(np.diff(times))
    return float(np.sqrt(m + 1) * (noise.scale + np.abs(second).max() * delta**2 / 8.0))
