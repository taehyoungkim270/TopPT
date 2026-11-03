"""Configuration objects for synthetic simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class TDAConfig:
    """Parameters for one raw time-delay embedding test."""

    m: int
    tau: float
    alpha: float = 0.05
    b: int = 250
    n_subsamples: int = 1000
    max_edge_length: Optional[float] = None
    seed: Optional[int] = None

    def validate(self, n_points: int) -> None:
        if self.m < 1:
            raise ValueError("m must be at least 1.")
        if self.tau <= 0:
            raise ValueError("tau must be positive.")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must lie in (0, 1).")
        if not (1 <= self.b <= n_points):
            raise ValueError(f"b must lie in [1, n_points]; got b={self.b}, n_points={n_points}.")
        if self.n_subsamples < 1:
            raise ValueError("n_subsamples must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NoiseConfig:
    """Noise model for synthetic scalar observations or embedding entries."""

    kind: str = "none"
    scale: float = 0.0

    def validate(self) -> None:
        allowed = {"none", "AG", "AL", "MG", "ML", "BU", "TG", "bounded"}
        if self.kind not in allowed:
            raise ValueError(f"Unknown noise kind {self.kind!r}. Allowed: {sorted(allowed)}")
        if self.scale < 0:
            raise ValueError("noise scale must be nonnegative.")

    def deterministic_bound(self) -> Optional[float]:
        """Return a sup-norm error bound when the noise model has one."""

        self.validate()
        if self.scale == 0 or self.kind == "none":
            return 0.0
        if self.kind in {"BU", "TG", "bounded"}:
            return float(self.scale)
        return None

    def to_dict(self) -> dict:
        return asdict(self)
