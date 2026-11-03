"""Subsampling confidence radii for finite point clouds."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("point cloud must be a two-dimensional array.")
    return cdist(x, x, metric="euclidean")


def hausdorff_full_to_subset(distance_matrix: np.ndarray, subset: np.ndarray) -> float:
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    subset = np.asarray(subset, dtype=int)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be square.")
    if subset.size == 0:
        raise ValueError("subset must be nonempty.")
    return float(distance_matrix[:, subset].min(axis=1).max())


def subsampling_hausdorff_radii(
    x: np.ndarray,
    b: int,
    n_subsamples: int,
    seed: int | None = None,
    distance_matrix: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if not (1 <= b <= n):
        raise ValueError(f"b must lie in [1, n]; got b={b}, n={n}.")
    if n_subsamples < 1:
        raise ValueError("n_subsamples must be positive.")
    dmat = pairwise_distances(x) if distance_matrix is None else np.asarray(distance_matrix, dtype=float)
    rng = np.random.default_rng(seed)
    radii = np.empty(int(n_subsamples), dtype=float)
    for i in range(int(n_subsamples)):
        subset = rng.choice(n, size=int(b), replace=False)
        radii[i] = hausdorff_full_to_subset(dmat, subset)
    return radii


def empirical_inverse_tail(radii: np.ndarray, alpha: float) -> float:
    """Compute inf{t: empirical P(radius > t) <= alpha}.

    The strict tail is used to match the paper's convention
    L(t)=P(d_H > t).
    """

    radii = np.asarray(radii, dtype=float)
    if radii.size == 0:
        raise ValueError("radii must be nonempty.")
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0, 1).")
    for value in np.unique(np.sort(radii)):
        if np.mean(radii > value) <= alpha:
            return float(value)
    return float(np.max(radii))


def confidence_radius(radii: np.ndarray, alpha: float) -> float:
    return 2.0 * empirical_inverse_tail(radii, alpha=alpha)
