"""Bounded additive noise for scalar observations."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm

KINDS = ("none", "BU", "TG")


@dataclass(frozen=True)
class NoiseModel:
    """Additive noise whose absolute value never exceeds ``scale``.

    Attributes
    ----------
    kind : str
        ``"none"`` for no noise, ``"BU"`` for uniform on ``[-scale, scale]``, or
        ``"TG"`` for a Gaussian with standard deviation ``scale / 2``, truncated
        at two standard deviations.
    scale : float
        The deterministic bound on the noise, written sigma_N in the paper. Must
        be zero when ``kind`` is ``"none"`` and positive otherwise.
    """

    kind: str = "none"
    scale: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Unknown noise kind {self.kind!r}. Allowed: {KINDS}")
        if self.kind == "none" and self.scale != 0:
            raise ValueError(f"Noise kind 'none' needs scale 0, got {self.scale}.")
        if self.kind != "none" and not self.scale > 0:
            raise ValueError(f"Noise kind {self.kind!r} needs scale > 0, got {self.scale}.")

    def sample(self, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        """Draw noise of the given shape. ``"none"`` leaves ``rng`` unused."""
        if self.kind == "none":
            return np.zeros(shape)
        if self.kind == "BU":
            return rng.uniform(-self.scale, self.scale, size=shape)
        return truncnorm.rvs(-2.0, 2.0, scale=self.scale / 2.0, size=shape, random_state=rng)
