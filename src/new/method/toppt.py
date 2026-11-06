"""The Topological Periodicity Test (Sections 5 and 6 of the paper)."""

import gudhi
import numpy as np
from scipy.spatial.distance import cdist


def subsampling_radii(
    cloud: np.ndarray,
    b: int,
    n_subsamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return Hausdorff distances from the cloud to random subsets of size ``b``."""
    distances = cdist(cloud, cloud)
    radii = np.empty(n_subsamples)
    for i in range(n_subsamples):
        subset = rng.choice(len(cloud), size=b, replace=False)
        radii[i] = distances[:, subset].min(axis=1).max()
    return radii


def confidence_radius(radii: np.ndarray, alpha: float) -> float:
    """Return ``c_alpha``, twice the smallest ``t`` with empirical P(radius > t) <= alpha."""
    return 2.0 * next(float(t) for t in np.unique(radii) if np.mean(radii > t) <= alpha)


def h1_intervals(cloud: np.ndarray) -> np.ndarray:
    """Return the Vietoris-Rips H1 intervals as rows of (birth, death)."""
    simplex_tree = gudhi.RipsComplex(points=cloud).create_simplex_tree(max_dimension=2)
    diagram = simplex_tree.persistence()
    return np.array([pair for dim, pair in diagram if dim == 1]).reshape(-1, 2)


def _rejects(intervals: np.ndarray, radius: float) -> bool:
    """Check for an interval in the rejection region: birth <= r, death - birth > 2r."""
    births, deaths = intervals.T
    return bool(np.any((births <= radius) & (deaths - births > 2.0 * radius)))


def toppt_test(
    cloud: np.ndarray,
    alpha: float,
    b: int,
    n_subsamples: int,
    rng: np.random.Generator,
    zeta: float | None = None,
) -> dict:
    """Run the test on a delay cloud and return its results as a flat dictionary.

    If the interpolation-error bound ``zeta`` is given, the result also has the
    bounded-error rule, which uses the inflated radius ``c_alpha + 5 * zeta``.
    """
    c_alpha = confidence_radius(subsampling_radii(cloud, b, n_subsamples, rng), alpha)
    intervals = h1_intervals(cloud)
    persistence = intervals[:, 1] - intervals[:, 0]
    best = intervals[persistence.argmax()] if len(intervals) else (np.nan, np.nan)
    result = {
        "reject": _rejects(intervals, c_alpha),
        "c_alpha": c_alpha,
        "n_h1": len(intervals),
        "best_birth": best[0],
        "best_death": best[1],
        "best_persistence": best[1] - best[0],
    }
    if zeta is not None:
        result["c_alpha_err"] = c_alpha + 5.0 * zeta
        result["robust_reject"] = _rejects(intervals, result["c_alpha_err"])
    return result
