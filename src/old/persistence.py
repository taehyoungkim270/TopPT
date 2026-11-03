"""Persistence diagram computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PersistenceInterval:
    dim: int
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        return float(self.death - self.birth)


def compute_rips_persistence(
    x: np.ndarray,
    max_dimension: int = 2,
    max_edge_length: float | None = None,
) -> list[PersistenceInterval]:
    try:
        import gudhi as gd
    except ImportError as exc:
        raise ImportError("gudhi is required for Rips persistence. Install gudhi first.") from exc

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("point cloud must be a two-dimensional array.")
    edge = float("inf") if max_edge_length is None else float(max_edge_length)
    rips = gd.RipsComplex(points=x, max_edge_length=edge)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)
    diagram = simplex_tree.persistence()
    return [
        PersistenceInterval(dim=int(dim), birth=float(pair[0]), death=float(pair[1]))
        for dim, pair in diagram
    ]


def intervals_by_dim(
    diagram: Iterable[PersistenceInterval],
    dim: int,
    finite_only: bool = False,
) -> list[PersistenceInterval]:
    out = []
    for interval in diagram:
        if interval.dim != dim:
            continue
        if finite_only and not np.isfinite(interval.death):
            continue
        out.append(interval)
    return out


def best_interval(intervals: Iterable[PersistenceInterval]) -> PersistenceInterval | None:
    intervals = list(intervals)
    if not intervals:
        return None
    return max(intervals, key=lambda z: z.persistence)
