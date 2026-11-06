"""Candidate periods for one window of a recording (Appendix I.3)."""

import numpy as np
from scipy import signal as spsignal


def _peaks(curve: np.ndarray, **kwargs) -> np.ndarray:
    """Return the indices of a curve's local peaks, or of its maximum if it has none."""
    peaks, _ = spsignal.find_peaks(curve, **kwargs)
    return peaks if peaks.size else np.array([np.argmax(curve)])


def strongest_period(values: np.ndarray, fs: float, bounds: tuple[float, float]) -> float:
    """Return the strongest period in ``bounds`` among autocorrelation and Welch peaks.

    Autocorrelation peaks are scored by their height and Welch peaks by ``log1p`` of
    their power.
    """
    low, high = bounds
    min_lag = max(1, int(np.floor(low * fs)))
    max_lag = max(min_lag + 1, int(np.ceil(high * fs)))
    centered = values - values.mean()
    acf = spsignal.correlate(centered, centered, mode="full", method="fft")[values.size - 1 :]
    acf_window = (acf / max(acf[0], 1e-12))[min_lag : max_lag + 1]
    acf_peaks = _peaks(acf_window, distance=max(1, min_lag // 3))

    segment = min(values.size, max(64, round(8.0 * fs * low)))
    frequencies, power = spsignal.welch(values, fs=fs, nperseg=segment)
    periods = 1.0 / frequencies[1:]
    in_bounds = (periods >= low) & (periods <= high)
    periods, power = periods[in_bounds], power[1:][in_bounds]
    welch_peaks = _peaks(power)

    hints = np.concatenate(((min_lag + acf_peaks) / fs, periods[welch_peaks]))
    scores = np.concatenate((acf_window[acf_peaks], np.log1p(power[welch_peaks])))
    return float(hints[np.argmax(scores)])


def candidate_periods(
    hint: float,
    bounds: tuple[float, float],
    n_points: int,
    scale: float,
) -> list[float]:
    """Return distinct periods on a grid around ``hint``, clipped to ``bounds``."""
    grid = hint * np.linspace(1.0 - scale, 1.0 + scale, n_points)
    return np.unique(np.round(np.clip(grid, *bounds), 4)).tolist()
