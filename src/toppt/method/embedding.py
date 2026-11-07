"""Time-delay embedding of scalar observations."""

import numpy as np


def delay_cloud(
    times: np.ndarray,
    values: np.ndarray,
    m: int,
    tau: float,
    max_anchors: int | None = None,
) -> np.ndarray:
    """Return the delay vectors ``(f(t), f(t + tau), ..., f(t + m * tau))``.

    ``f`` is the linear interpolant of the observations, and the anchors ``t`` are
    the observed times whose whole window lies within the observed range. There is
    no periodic wrapping. ``times`` must be increasing. If ``max_anchors`` is
    given, the anchors are thinned evenly down to at most that many.
    """
    anchors = times[times + m * tau <= times[-1] + 1e-12]
    if anchors.size == 0:
        raise ValueError(f"No anchor fits a window of length {m * tau} in the observed range.")
    if max_anchors is not None and anchors.size > max_anchors:
        indices = np.linspace(0, anchors.size - 1, max_anchors)
        anchors = anchors[np.unique(np.rint(indices).astype(int))]
    query = anchors[:, None] + np.arange(m + 1) * tau
    return np.interp(query, times, values)
