"""Linear interpolation helpers.

No periodic wrapping is implemented here.  Query points outside the observed
time range raise an error by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearInterpolant:
    times: np.ndarray
    values: np.ndarray
    allow_extrapolation: bool = False

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if times.ndim != 1 or values.ndim != 1:
            raise ValueError("times and values must be one-dimensional.")
        if len(times) != len(values):
            raise ValueError("times and values must have the same length.")
        if len(times) < 2:
            raise ValueError("linear interpolation needs at least two points.")

        order = np.argsort(times)
        sorted_times = times[order]
        sorted_values = values[order]
        unique_times, unique_idx = np.unique(sorted_times, return_index=True)
        unique_values = sorted_values[unique_idx]
        if len(unique_times) < 2:
            raise ValueError("linear interpolation needs at least two unique times.")

        object.__setattr__(self, "times", unique_times)
        object.__setattr__(self, "values", unique_values)

    @property
    def domain(self) -> tuple[float, float]:
        return float(self.times[0]), float(self.times[-1])

    def __call__(self, query_times: np.ndarray) -> np.ndarray:
        query = np.asarray(query_times, dtype=float)
        lo, hi = self.domain
        if not self.allow_extrapolation:
            if np.any(query < lo - 1e-12) or np.any(query > hi + 1e-12):
                raise ValueError(
                    "Interpolation query leaves the observed domain. "
                    f"domain=({lo:.6g}, {hi:.6g}), query=({np.min(query):.6g}, {np.max(query):.6g})."
                )
        return np.interp(query, self.times, self.values)


def build_linear_interpolant(
    times: np.ndarray,
    values: np.ndarray,
    allow_extrapolation: bool = False,
) -> LinearInterpolant:
    return LinearInterpolant(times=times, values=values, allow_extrapolation=allow_extrapolation)
