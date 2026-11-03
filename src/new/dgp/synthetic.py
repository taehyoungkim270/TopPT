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
