"""Periodicity test based on raw time-delay embeddings and subsampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TDAConfig
from .persistence import PersistenceInterval, best_interval, compute_rips_persistence, intervals_by_dim
from .subsampling import confidence_radius, subsampling_hausdorff_radii


@dataclass
class TDATestResult:
    reject: bool
    c_alpha: float
    radius_quantile: float
    best_h1: PersistenceInterval | None
    h1_intervals: list[PersistenceInterval]
    radii: np.ndarray
    diagram: list[PersistenceInterval]
    config: TDAConfig
    filtration: str = "rips"

    def summary_dict(self) -> dict:
        return {
            "reject": int(self.reject),
            "c_alpha": float(self.c_alpha),
            "radius_quantile": float(self.radius_quantile),
            "n_h1": len(self.h1_intervals),
            "best_birth": np.nan if self.best_h1 is None else float(self.best_h1.birth),
            "best_death": np.nan if self.best_h1 is None else float(self.best_h1.death),
            "best_persistence": np.nan if self.best_h1 is None else float(self.best_h1.persistence),
            "filtration": self.filtration,
            **{f"config_{k}": v for k, v in self.config.to_dict().items()},
        }


def h1_rejection_region(interval: PersistenceInterval, c_alpha: float) -> bool:
    return bool(interval.birth <= c_alpha and interval.persistence > 2.0 * c_alpha)


def run_tda_periodicity_test(x: np.ndarray, config: TDAConfig) -> TDATestResult:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be a two-dimensional point cloud.")
    config.validate(n_points=x.shape[0])

    radii = subsampling_hausdorff_radii(
        x,
        b=config.b,
        n_subsamples=config.n_subsamples,
        seed=config.seed,
    )
    c_alpha = confidence_radius(radii, alpha=config.alpha)
    radius_quantile = c_alpha / 2.0

    diagram = compute_rips_persistence(
        x,
        max_dimension=2,
        max_edge_length=config.max_edge_length,
    )
    h1 = intervals_by_dim(diagram, dim=1, finite_only=False)
    best = best_interval(h1)
    reject = any(h1_rejection_region(interval, c_alpha=c_alpha) for interval in h1)

    return TDATestResult(
        reject=reject,
        c_alpha=float(c_alpha),
        radius_quantile=float(radius_quantile),
        best_h1=best,
        h1_intervals=h1,
        radii=radii,
        diagram=diagram,
        config=config,
    )
