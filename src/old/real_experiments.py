"""Real-data experiments for the current no-folding simulation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import signal as spsignal

from .config import TDAConfig
from .embedding import restrict_anchors_to_domain, sliding_window_embedding
from .gls_baseline import GLSConfig, run_gls_periodicity_test
from .interpolation import build_linear_interpolant
from .periodicity_test import TDATestResult, run_tda_periodicity_test
from .spectral_baselines import (
    PermutationPeriodogramConfig,
    run_fisher_g_test,
    run_permutation_max_periodogram_test,
)
from .synthetic_experiments import json_safe


FANTASIA_RECORDS: tuple[str, ...] = tuple(
    [f"f1y{index:02d}" for index in range(1, 11)]
    + [f"f2y{index:02d}" for index in range(1, 11)]
    + [f"f1o{index:02d}" for index in range(1, 11)]
    + [f"f2o{index:02d}" for index in range(1, 11)]
)
BIDMC_RECORDS: tuple[str, ...] = tuple(f"bidmc{index:02d}" for index in range(1, 54))


def fantasia_record_names(first_n: int | None = None) -> list[str]:
    """Return the canonical Fantasia record order used by the simulations."""

    records = list(FANTASIA_RECORDS)
    if first_n is None:
        return records
    if int(first_n) < 1:
        raise ValueError("first_n must be positive.")
    return records[: int(first_n)]


def bidmc_record_names(first_n: int | None = None) -> list[str]:
    """Return the canonical BIDMC record order used by the simulations."""

    records = list(BIDMC_RECORDS)
    if first_n is None:
        return records
    if int(first_n) < 1:
        raise ValueError("first_n must be positive.")
    return records[: int(first_n)]


def download_fantasia_records(
    data_dir: str | Path,
    records: Sequence[str],
    overwrite: bool = False,
) -> list[str]:
    """Download selected Fantasia WFDB records through PhysioNet."""

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required to download Fantasia records. Install wfdb first.") from exc

    record_list = [str(record) for record in records]
    if len(record_list) == 0:
        raise ValueError("records must contain at least one record name.")
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(
        "fantasia",
        dl_dir=str(output),
        records=record_list,
        annotators="all",
        keep_subdirs=True,
        overwrite=bool(overwrite),
    )
    return record_list


def download_bidmc_records(
    data_dir: str | Path,
    records: Sequence[str],
    overwrite: bool = False,
) -> list[str]:
    """Download selected BIDMC WFDB records through PhysioNet."""

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required to download BIDMC records. Install wfdb first.") from exc

    record_list = [str(record) for record in records]
    if len(record_list) == 0:
        raise ValueError("records must contain at least one record name.")
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(
        "bidmc",
        dl_dir=str(output),
        records=record_list,
        annotators="all",
        keep_subdirs=True,
        overwrite=bool(overwrite),
    )
    return record_list


def find_channel_index(sig_names: Sequence[str], keywords: Sequence[str], default: int = 0) -> int:
    names_upper = [name.upper() for name in sig_names]
    for keyword in keywords:
        key = keyword.upper()
        for index, name in enumerate(names_upper):
            if key == name or key in name:
                return int(index)
    return int(default)


def load_wfdb_signal(
    record: str,
    local_dir: str | Path,
    channel_keywords: Sequence[str] | None = None,
    default_channel: int = 0,
    sampto: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one scalar channel from a local WFDB record."""

    try:
        import wfdb
    except ImportError as exc:
        raise ImportError("wfdb is required for real-data WFDB experiments. Install wfdb first.") from exc

    record_path = str(Path(local_dir) / record)
    signal, fields = wfdb.rdsamp(record_path, sampto=sampto)
    fs = float(fields["fs"])
    times = np.arange(signal.shape[0], dtype=float) / fs
    sig_names = list(fields["sig_name"])
    channel_index = (
        int(default_channel)
        if channel_keywords is None
        else find_channel_index(sig_names, channel_keywords, default=int(default_channel))
    )
    values = np.asarray(signal[:, channel_index], dtype=float)
    meta = {
        "fs": fs,
        "sig_name": sig_names,
        "channel_index": int(channel_index),
        "channel_name": sig_names[channel_index],
        "source_record": str(record),
        "local_dir": str(local_dir),
    }
    return times, values, meta


def load_fantasia_resp(
    record: str = "f1y01",
    local_dir: str | Path = "data/fantasia",
    sampto: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    return load_wfdb_signal(
        record=record,
        local_dir=local_dir,
        channel_keywords=["RESP"],
        default_channel=0,
        sampto=sampto,
    )


def load_bidmc_resp(
    record: str = "bidmc01",
    local_dir: str | Path = "data/bidmc",
    sampto: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    return load_wfdb_signal(
        record=record,
        local_dir=local_dir,
        channel_keywords=["RESP"],
        default_channel=0,
        sampto=sampto,
    )


def list_local_wfdb_records(local_dir: str | Path) -> list[str]:
    """List locally available WFDB records by RECORDS file or header stems."""

    local_path = Path(local_dir)
    records_file = local_path / "RECORDS"
    if records_file.exists():
        records = [line.strip() for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if records:
            return records
    return sorted(path.stem for path in local_path.glob("*.hea"))


def preprocess_signal(
    values: np.ndarray,
    fs: float,
    linear_detrend: bool = True,
    bandpass_hz: tuple[float, float] | None = None,
    zscore: bool = True,
) -> np.ndarray:
    """Basic scalar preprocessing for real waveform windows."""

    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y)
    if not np.all(finite):
        fill = float(np.nanmedian(y[finite])) if np.any(finite) else 0.0
        y = np.nan_to_num(y, nan=fill, posinf=fill, neginf=fill)
    if linear_detrend and y.size >= 2:
        y = spsignal.detrend(y, type="linear")
    if bandpass_hz is not None:
        low, high = map(float, bandpass_hz)
        nyquist = 0.5 * float(fs)
        low = max(low, 1e-8)
        high = min(high, 0.99 * nyquist)
        if low < high and y.size >= 16:
            sos = spsignal.butter(4, [low, high], btype="bandpass", fs=float(fs), output="sos")
            y = spsignal.sosfiltfilt(sos, y)
    if zscore:
        scale = float(np.std(y))
        center = float(np.mean(y))
        y = (y - center) / scale if scale > 0 else y - center
    return np.asarray(y, dtype=float)


def resample_uniform(
    times: np.ndarray,
    values: np.ndarray,
    target_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Interpolate a signal onto a lower-rate uniform grid when requested."""

    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 2:
        raise ValueError("Need at least two samples for real-data resampling.")
    if target_hz is None:
        dt = float(np.median(np.diff(times)))
        return times, values, float(1.0 / dt)
    target_hz = float(target_hz)
    if target_hz <= 0:
        raise ValueError("target_hz must be positive.")
    dt = 1.0 / target_hz
    new_times = np.arange(float(times[0]), float(times[-1]) + 0.5 * dt, dt)
    new_times = new_times[new_times <= float(times[-1]) + 1e-12]
    new_values = np.interp(new_times, times, values)
    return new_times, new_values, target_hz


def compute_autocorrelation(values: np.ndarray, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float)
    y = y - float(np.mean(y))
    if y.size == 0:
        return np.array([0], dtype=int), np.array([0.0], dtype=float)
    if np.allclose(y, 0.0):
        corr = np.zeros(y.size, dtype=float)
        corr[0] = 1.0
    else:
        corr = spsignal.correlate(y, y, mode="full", method="fft")
        corr = corr[y.size - 1 :]
        corr = corr / max(float(corr[0]), 1e-12)
    lags = np.arange(corr.size, dtype=int)
    if max_lag is not None:
        stop = min(int(max_lag) + 1, corr.size)
        return lags[:stop], corr[:stop]
    return lags, corr


def _select_top_unique(values: Sequence[float], scores: Sequence[float], max_items: int, rtol: float = 0.05) -> list[float]:
    pairs = sorted(zip(values, scores), key=lambda item: item[1], reverse=True)
    selected: list[tuple[float, float]] = []
    for value, score in pairs:
        value = float(value)
        if not np.isfinite(value):
            continue
        if any(abs(value - previous) <= rtol * max(abs(previous), abs(value), 1e-12) for previous, _ in selected):
            continue
        selected.append((value, float(score)))
        if len(selected) >= int(max_items):
            break
    return [value for value, _ in selected]


def estimate_period_hints(
    values: np.ndarray,
    fs: float,
    period_bounds: tuple[float, float],
    top_k: int = 1,
) -> dict:
    """Estimate candidate periods from ACF and Welch peaks."""

    y = np.asarray(values, dtype=float)
    p_min, p_max = map(float, period_bounds)
    if y.size < max(32, int(np.ceil(3.0 * p_min * fs))):
        return {"period_hints": [], "acf_period": np.nan, "acf_peak": np.nan, "spec_period": np.nan, "spec_ratio": np.nan}

    hints: list[float] = []
    scores: list[float] = []

    min_lag = max(1, int(np.floor(p_min * fs)))
    max_lag = max(min_lag + 1, int(np.ceil(p_max * fs)))
    lags, acf = compute_autocorrelation(y, max_lag=max_lag)
    acf_period = np.nan
    acf_peak = np.nan
    if max_lag < acf.size:
        acf_window = acf[min_lag : max_lag + 1]
        peaks, _ = spsignal.find_peaks(acf_window, distance=max(1, min_lag // 3))
        if peaks.size == 0 and acf_window.size:
            peaks = np.array([int(np.argmax(acf_window))])
        if peaks.size:
            periods = (min_lag + peaks) / float(fs)
            peak_scores = acf_window[peaks]
            acf_period = float(periods[int(np.argmax(peak_scores))])
            acf_peak = float(np.max(peak_scores))
            hints.extend(periods.tolist())
            scores.extend(peak_scores.tolist())

    nperseg = min(y.size, max(64, int(round(8.0 * fs * p_min))))
    freqs, power = spsignal.welch(y, fs=float(fs), nperseg=nperseg)
    period = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
    mask = (freqs > 0) & (period >= p_min) & (period <= p_max)
    spec_period = np.nan
    spec_ratio = np.nan
    if np.any(mask):
        freqs_sel = freqs[mask]
        power_sel = power[mask]
        peaks, _ = spsignal.find_peaks(power_sel)
        if peaks.size == 0 and power_sel.size:
            peaks = np.array([int(np.argmax(power_sel))])
        if peaks.size:
            peak_scores = power_sel[peaks]
            periods = 1.0 / freqs_sel[peaks]
            best = int(np.argmax(peak_scores))
            spec_period = float(periods[best])
            spec_ratio = float(peak_scores[best] / max(float(np.nanmedian(power_sel)), 1e-12))
            hints.extend(periods.tolist())
            scores.extend(np.log1p(np.maximum(peak_scores, 0.0)).tolist())

    selected = _select_top_unique(hints, scores, max_items=top_k, rtol=0.05) if hints else []
    return {
        "period_hints": [float(x) for x in selected],
        "acf_period": float(acf_period) if np.isfinite(acf_period) else np.nan,
        "acf_peak": float(acf_peak) if np.isfinite(acf_peak) else np.nan,
        "spec_period": float(spec_period) if np.isfinite(spec_period) else np.nan,
        "spec_ratio": float(spec_ratio) if np.isfinite(spec_ratio) else np.nan,
    }


def build_period_grid_from_hints(
    period_hints: Sequence[float],
    period_bounds: tuple[float, float],
    grid_points: int = 5,
    grid_scale: float = 0.10,
) -> list[float]:
    p_min, p_max = map(float, period_bounds)
    periods: list[float] = []
    for hint in period_hints:
        if not np.isfinite(hint):
            continue
        local = float(hint) * np.linspace(1.0 - float(grid_scale), 1.0 + float(grid_scale), int(grid_points))
        periods.extend(np.clip(local, p_min, p_max).tolist())
    if not periods:
        periods = np.linspace(p_min, p_max, int(grid_points)).tolist()
    return sorted({float(np.round(value, 4)) for value in periods})


def summarize_real_record_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compact record-level summary without double-counting scalar baselines."""

    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for record, group in df.groupby("record", dropna=False):
        keys = ["record"]
        if "window_start_seconds" in group.columns:
            keys.append("window_start_seconds")
        baseline_rows = group.drop_duplicates(subset=keys)

        if "window_start_seconds" in group.columns:
            tda_window_rejections = (
                group.groupby("window_start_seconds", dropna=False)["reject"].max().astype(int).sum()
            )
            n_windows = int(group["window_start_seconds"].nunique(dropna=False))
        else:
            tda_window_rejections = int(group["reject"].astype(bool).any())
            n_windows = 1

        tda_margin = group["best_persistence"].astype(float) - 2.0 * group["c_alpha"].astype(float)
        row = {
            "record": record,
            "n_windows": n_windows,
            "n_tda_tests": int(group.shape[0]),
            "tda_rejecting_windows": int(tda_window_rejections),
            "global_tda_reject": bool(group["reject"].astype(bool).any()) if "reject" in group else False,
            "min_c_alpha": float(group["c_alpha"].min()) if "c_alpha" in group else np.nan,
            "max_persistence": float(group["best_persistence"].max()) if "best_persistence" in group else np.nan,
            "best_tda_margin": float(tda_margin.max()) if not tda_margin.empty else np.nan,
        }
        for prefix, column in (
            ("gls", "gls_reject"),
            ("fisher", "fisher_reject"),
            ("perm", "perm_reject"),
        ):
            if column in baseline_rows:
                count = int(baseline_rows[column].astype(int).sum())
                row[f"{prefix}_rejecting_windows"] = count
                row[f"global_{prefix}_reject"] = bool(count > 0)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_real_window_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compact window-level summary for local real-data tests."""

    if df.empty or "window_start_seconds" not in df.columns:
        return pd.DataFrame()

    keys = ["record", "window_start_seconds"]
    work = df.assign(
        reject=df["reject"].astype(int),
        tda_margin=df["best_persistence"].astype(float) - 2.0 * df["c_alpha"].astype(float),
    )
    tda_summary = (
        work.groupby(keys, dropna=False)
        .agg(
            n_tda_tests=("reject", "size"),
            tda_rejections=("reject", "sum"),
            tda_window_reject=("reject", "max"),
            best_tda_margin=("tda_margin", "max"),
            min_c_alpha=("c_alpha", "min"),
            max_persistence=("best_persistence", "max"),
        )
        .reset_index()
    )

    baseline_cols = [col for col in ("gls_reject", "fisher_reject", "perm_reject") if col in df.columns]
    if baseline_cols:
        baseline_summary = df.drop_duplicates(subset=keys)[keys + baseline_cols]
        tda_summary = tda_summary.merge(baseline_summary, on=keys, how="left")

    return tda_summary


def select_time_window(
    times: np.ndarray,
    values: np.ndarray,
    start_seconds: float,
    window_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    start = float(start_seconds)
    stop = start + float(window_seconds)
    mask = (times >= start - 1e-12) & (times <= stop + 1e-12)
    if int(np.count_nonzero(mask)) < 4:
        raise ValueError("Selected real-data window has too few observations.")
    out_times = times[mask] - start
    out_values = values[mask]
    return out_times, out_values


def run_scalar_baseline_tests(
    times: np.ndarray,
    values: np.ndarray,
    alpha: float,
    period_bounds: tuple[float, float],
    gls_ofac: float,
    gls_hifac: float,
    periodogram_permutations: int,
    seed: int,
) -> dict:
    """Run scalar baselines once for one local scalar window."""

    gls_cfg = GLSConfig(
        alpha=float(alpha),
        ofac=float(gls_ofac),
        hifac=float(gls_hifac),
        min_period=float(period_bounds[0]),
        max_period=float(period_bounds[1]),
    )
    gls_result = run_gls_periodicity_test(times, values, gls_cfg)
    fisher_result = run_fisher_g_test(times, values, alpha=float(alpha))
    perm_result = run_permutation_max_periodogram_test(
        times,
        values,
        PermutationPeriodogramConfig(
            alpha=float(alpha),
            n_permutations=int(periodogram_permutations),
            seed=int(seed) + 50_000,
        ),
    )
    return {
        **gls_result.summary_dict(),
        **fisher_result.summary_dict(),
        **perm_result.summary_dict(),
    }


def build_real_embedding(
    times: np.ndarray,
    values: np.ndarray,
    m: int,
    tau: float,
    max_anchors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    interpolant = build_linear_interpolant(times, values)
    anchors = restrict_anchors_to_domain(times, m=m, tau=tau, eval_domain=interpolant.domain)
    if anchors.size == 0:
        raise ValueError("No valid anchors remain after enforcing the delay span.")
    if max_anchors is not None and anchors.size > int(max_anchors):
        indices = np.linspace(0, anchors.size - 1, int(max_anchors))
        anchors = anchors[np.unique(np.rint(indices).astype(int))]
    x = sliding_window_embedding(interpolant, anchors, m=m, tau=tau)
    return anchors, x


def run_real_period_case(
    dataset: str,
    record: str,
    channel_name: str,
    times: np.ndarray,
    values: np.ndarray,
    fs_original: float,
    fs_effective: float,
    period_seconds: float,
    m: int,
    alpha: float,
    b: int | None,
    b_fraction: float,
    n_subsamples: int,
    max_anchors: int | None,
    seed: int,
    period_bounds: tuple[float, float],
    gls_ofac: float,
    gls_hifac: float,
    periodogram_permutations: int,
    max_edge_length: float | None = None,
    baseline_alpha: float | None = None,
    baseline_summary: dict | None = None,
) -> tuple[dict, TDATestResult]:
    tau = float(period_seconds) / float(m + 1)
    anchors, x = build_real_embedding(times, values, m=m, tau=tau, max_anchors=max_anchors)
    n_points = int(x.shape[0])
    if n_points < 4:
        raise ValueError("Real-data embedding has too few points.")
    if b is None:
        b_current = int(np.floor(float(b_fraction) * n_points))
    else:
        b_current = int(b)
    b_current = max(1, min(b_current, n_points))
    cfg = TDAConfig(
        m=int(m),
        tau=float(tau),
        alpha=float(alpha),
        b=b_current,
        n_subsamples=int(n_subsamples),
        max_edge_length=max_edge_length,
        seed=int(seed),
    )
    tda_result = run_tda_periodicity_test(x, cfg)

    baseline_alpha_current = float(alpha) if baseline_alpha is None else float(baseline_alpha)
    if baseline_summary is None:
        baseline_summary = run_scalar_baseline_tests(
            times=times,
            values=values,
            alpha=baseline_alpha_current,
            period_bounds=period_bounds,
            gls_ofac=gls_ofac,
            gls_hifac=gls_hifac,
            periodogram_permutations=periodogram_permutations,
            seed=seed,
        )

    row = {
        "dataset": dataset,
        "record": record,
        "channel_name": channel_name,
        "fs_original": float(fs_original),
        "fs_effective": float(fs_effective),
        "n_observations": int(times.size),
        "n_points": n_points,
        "max_anchors": np.nan if max_anchors is None else int(max_anchors),
        "candidate_period_seconds": float(period_seconds),
        "m": int(m),
        "tau": float(tau),
        "alpha": float(alpha),
        "alpha_tda": float(alpha),
        "alpha_baseline": baseline_alpha_current,
        "b": int(b_current),
        "b_fraction": float(b_fraction),
        "n_subsamples": int(n_subsamples),
        "period_bound_min": float(period_bounds[0]),
        "period_bound_max": float(period_bounds[1]),
        **tda_result.summary_dict(),
        **baseline_summary,
    }
    return row, tda_result


def run_fantasia_experiment(
    data_dir: str | Path,
    record: str = "f1y01",
    use_seconds: float = 90.0,
    window_start_seconds: float = 0.0,
    window_seconds: float = 60.0,
    downsample_to_hz: float | None = 12.0,
    linear_detrend: bool = True,
    zscore: bool = True,
    bandpass_hz: tuple[float, float] | None = None,
    period_bounds: tuple[float, float] = (2.0, 8.0),
    period_grid: Sequence[float] | None = None,
    period_strategy: str = "window",
    top_periods: int = 1,
    grid_points: int = 5,
    grid_scale: float = 0.10,
    m: int = 20,
    alpha: float = 0.05,
    b: int | None = None,
    b_fraction: float = 0.75,
    n_subsamples: int = 300,
    max_anchors: int | None = 220,
    seed: int = 0,
    gls_ofac: float = 10.0,
    gls_hifac: float = 1.0,
    periodogram_permutations: int = 500,
    max_edge_length: float | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    sampto = int(np.ceil(float(use_seconds) * 250.0)) if use_seconds is not None else None
    raw_times, raw_values, meta = load_fantasia_resp(record=record, local_dir=data_dir, sampto=sampto)
    if use_seconds is not None:
        mask = raw_times <= float(use_seconds) + 1e-12
        raw_times = raw_times[mask]
        raw_values = raw_values[mask]

    processed = preprocess_signal(
        raw_values,
        fs=float(meta["fs"]),
        linear_detrend=linear_detrend,
        bandpass_hz=bandpass_hz,
        zscore=zscore,
    )
    work_times, work_values, fs_effective = resample_uniform(raw_times, processed, target_hz=downsample_to_hz)
    window_times, window_values = select_time_window(
        work_times,
        work_values,
        start_seconds=window_start_seconds,
        window_seconds=window_seconds,
    )

    hints = estimate_period_hints(window_values, fs=fs_effective, period_bounds=period_bounds, top_k=top_periods)
    if period_grid is None:
        candidate_periods = build_period_grid_from_hints(
            hints["period_hints"],
            period_bounds=period_bounds,
            grid_points=grid_points,
            grid_scale=grid_scale,
        )
        period_source = "window-auto"
    else:
        candidate_periods = [float(value) for value in period_grid]
        period_source = "fixed"

    rows: list[dict] = []
    baseline_summary = run_scalar_baseline_tests(
        times=window_times,
        values=window_values,
        alpha=alpha,
        period_bounds=period_bounds,
        gls_ofac=gls_ofac,
        gls_hifac=gls_hifac,
        periodogram_permutations=periodogram_permutations,
        seed=seed,
    )
    period_iter = enumerate(candidate_periods)
    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(candidate_periods), desc="fantasia period grid", unit="period")
        except ImportError:
            progress = None

    try:
        for index, period_seconds in period_iter:
            row, _ = run_real_period_case(
                dataset="fantasia",
                record=record,
                channel_name=str(meta["channel_name"]),
                times=window_times,
                values=window_values,
                fs_original=float(meta["fs"]),
                fs_effective=float(fs_effective),
                period_seconds=float(period_seconds),
                m=m,
                alpha=alpha,
                b=b,
                b_fraction=b_fraction,
                n_subsamples=n_subsamples,
                max_anchors=max_anchors,
                seed=int(seed) + index,
                period_bounds=period_bounds,
                gls_ofac=gls_ofac,
                gls_hifac=gls_hifac,
                periodogram_permutations=periodogram_permutations,
                max_edge_length=max_edge_length,
                baseline_alpha=alpha,
                baseline_summary=baseline_summary,
            )
            row.update(
                {
                    "use_seconds": float(use_seconds),
                    "window_start_seconds": float(window_start_seconds),
                    "window_seconds": float(window_seconds),
                    "downsample_to_hz": np.nan if downsample_to_hz is None else float(downsample_to_hz),
                    "linear_detrend": int(linear_detrend),
                    "zscore": int(zscore),
                    "bandpass_low_hz": np.nan if bandpass_hz is None else float(bandpass_hz[0]),
                    "bandpass_high_hz": np.nan if bandpass_hz is None else float(bandpass_hz[1]),
                    "period_source": period_source,
                    "acf_period_seconds": hints["acf_period"],
                    "acf_peak": hints["acf_peak"],
                    "spec_period_seconds": hints["spec_period"],
                    "spec_ratio": hints["spec_ratio"],
                    "periodogram_permutations": int(periodogram_permutations),
                }
            )
            rows.append(row)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    df = pd.DataFrame(rows)
    meta_out = {
        **meta,
        "use_seconds": float(use_seconds),
        "window_start_seconds": float(window_start_seconds),
        "window_seconds": float(window_seconds),
        "fs_effective": float(fs_effective),
        "period_hints": hints,
        "candidate_periods": candidate_periods,
        "period_source": period_source,
        "period_strategy": str(period_strategy),
    }
    return df, meta_out


def run_fantasia_window_grid(
    data_dir: str | Path,
    record: str = "f1y01",
    use_seconds: float = 90.0,
    window_starts: Sequence[float] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
    window_seconds: float = 20.0,
    downsample_to_hz: float | None = 25.0,
    linear_detrend: bool = True,
    zscore: bool = True,
    bandpass_hz: tuple[float, float] | None = None,
    period_bounds: tuple[float, float] = (2.0, 8.0),
    period_grid: Sequence[float] | None = None,
    period_strategy: str = "window",
    top_periods: int = 1,
    grid_points: int = 3,
    grid_scale: float = 0.10,
    m: int = 8,
    alpha: float = 0.05,
    bonferroni: bool = True,
    b: int | None = None,
    b_fraction: float = 0.99,
    n_subsamples: int = 300,
    max_anchors: int | None = 300,
    seed: int = 0,
    gls_ofac: float = 10.0,
    gls_hifac: float = 1.0,
    periodogram_permutations: int = 500,
    max_edge_length: float | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run a fixed Fantasia time-window and period grid.

    The global test rejects if any window-period pair rejects.  When
    ``bonferroni`` is true, the per-test level is ``alpha / K`` where
    ``K`` is the total number of tested window-period pairs.  If
    ``period_grid`` is ``None`` and ``period_strategy`` is ``"window"``,
    each window builds its own candidate periods from its ACF/Welch hints.
    """

    starts = [float(value) for value in window_starts]
    if len(starts) == 0:
        raise ValueError("window_starts must contain at least one value.")

    sampto = int(np.ceil(float(use_seconds) * 250.0)) if use_seconds is not None else None
    raw_times, raw_values, meta = load_fantasia_resp(record=record, local_dir=data_dir, sampto=sampto)
    if use_seconds is not None:
        mask = raw_times <= float(use_seconds) + 1e-12
        raw_times = raw_times[mask]
        raw_values = raw_values[mask]

    processed = preprocess_signal(
        raw_values,
        fs=float(meta["fs"]),
        linear_detrend=linear_detrend,
        bandpass_hz=bandpass_hz,
        zscore=zscore,
    )
    work_times, work_values, fs_effective = resample_uniform(raw_times, processed, target_hz=downsample_to_hz)
    full_hints = estimate_period_hints(work_values, fs=fs_effective, period_bounds=period_bounds, top_k=top_periods)
    strategy = str(period_strategy).lower().replace("_", "-")
    if strategy not in {"full", "window"}:
        raise ValueError("period_strategy must be 'full' or 'window'.")

    fixed_periods: list[float] | None = None
    if period_grid is not None:
        fixed_periods = [float(value) for value in period_grid]
        period_source = "fixed"
    elif strategy == "full":
        fixed_periods = build_period_grid_from_hints(
            full_hints["period_hints"],
            period_bounds=period_bounds,
            grid_points=grid_points,
            grid_scale=grid_scale,
        )
        period_source = "full-auto"
    else:
        period_source = "window-auto"

    if fixed_periods is not None and len(fixed_periods) == 0:
        raise ValueError("No candidate periods were provided or estimated.")

    window_specs: list[dict] = []
    for window_index, window_start in enumerate(starts):
        window_times, window_values = select_time_window(
            work_times,
            work_values,
            start_seconds=window_start,
            window_seconds=window_seconds,
        )
        window_hints = estimate_period_hints(
            window_values,
            fs=fs_effective,
            period_bounds=period_bounds,
            top_k=top_periods,
        )
        if fixed_periods is None:
            candidate_periods = build_period_grid_from_hints(
                window_hints["period_hints"],
                period_bounds=period_bounds,
                grid_points=grid_points,
                grid_scale=grid_scale,
            )
        else:
            candidate_periods = fixed_periods
        if len(candidate_periods) == 0:
            raise ValueError(f"No candidate periods were estimated for window starting at {window_start}.")
        window_specs.append(
            {
                "window_index": int(window_index),
                "window_start": float(window_start),
                "window_times": window_times,
                "window_values": window_values,
                "window_hints": window_hints,
                "candidate_periods": [float(value) for value in candidate_periods],
            }
        )

    n_total_tests = int(sum(len(spec["candidate_periods"]) for spec in window_specs))
    alpha_tda = float(alpha) / float(n_total_tests) if bonferroni else float(alpha)
    alpha_baseline = float(alpha) / float(len(starts)) if bonferroni else float(alpha)

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=n_total_tests, desc="fantasia window grid", unit="test")
        except ImportError:
            progress = None

    rows: list[dict] = []
    try:
        for spec in window_specs:
            window_index = int(spec["window_index"])
            window_start = float(spec["window_start"])
            window_times = spec["window_times"]
            window_values = spec["window_values"]
            window_hints = spec["window_hints"]
            candidate_periods = spec["candidate_periods"]
            baseline_summary = run_scalar_baseline_tests(
                times=window_times,
                values=window_values,
                alpha=alpha_baseline,
                period_bounds=period_bounds,
                gls_ofac=gls_ofac,
                gls_hifac=gls_hifac,
                periodogram_permutations=periodogram_permutations,
                seed=int(seed) + 10_000 * window_index,
            )
            for period_index, period_seconds in enumerate(candidate_periods):
                row, _ = run_real_period_case(
                    dataset="fantasia",
                    record=record,
                    channel_name=str(meta["channel_name"]),
                    times=window_times,
                    values=window_values,
                    fs_original=float(meta["fs"]),
                    fs_effective=float(fs_effective),
                    period_seconds=float(period_seconds),
                    m=m,
                    alpha=alpha_tda,
                    b=b,
                    b_fraction=b_fraction,
                    n_subsamples=n_subsamples,
                    max_anchors=max_anchors,
                    seed=int(seed) + 1000 * window_index + period_index,
                    period_bounds=period_bounds,
                    gls_ofac=gls_ofac,
                    gls_hifac=gls_hifac,
                    periodogram_permutations=periodogram_permutations,
                    max_edge_length=max_edge_length,
                    baseline_alpha=alpha_baseline,
                    baseline_summary=baseline_summary,
                )
                row.update(
                    {
                        "use_seconds": float(use_seconds),
                        "window_index": int(window_index),
                        "window_start_seconds": float(window_start),
                        "window_seconds": float(window_seconds),
                        "downsample_to_hz": np.nan if downsample_to_hz is None else float(downsample_to_hz),
                        "linear_detrend": int(linear_detrend),
                        "zscore": int(zscore),
                        "bandpass_low_hz": np.nan if bandpass_hz is None else float(bandpass_hz[0]),
                        "bandpass_high_hz": np.nan if bandpass_hz is None else float(bandpass_hz[1]),
                        "period_index": int(period_index),
                        "period_source": period_source,
                        "global_alpha": float(alpha),
                        "alpha_per_test": float(alpha_tda),
                        "alpha_tda": float(alpha_tda),
                        "alpha_baseline": float(alpha_baseline),
                        "bonferroni": int(bonferroni),
                        "n_window_tests": int(len(starts)),
                        "n_baseline_tests": int(len(starts)),
                        "n_period_tests": int(len(candidate_periods)),
                        "n_total_tests": int(n_total_tests),
                        "acf_period_seconds": window_hints["acf_period"],
                        "acf_peak": window_hints["acf_peak"],
                        "spec_period_seconds": window_hints["spec_period"],
                        "spec_ratio": window_hints["spec_ratio"],
                        "full_acf_period_seconds": full_hints["acf_period"],
                        "full_acf_peak": full_hints["acf_peak"],
                        "full_spec_period_seconds": full_hints["spec_period"],
                        "full_spec_ratio": full_hints["spec_ratio"],
                        "periodogram_permutations": int(periodogram_permutations),
                    }
                )
                rows.append(row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    df = pd.DataFrame(rows)
    meta_out = {
        **meta,
        "use_seconds": float(use_seconds),
        "window_starts": starts,
        "window_seconds": float(window_seconds),
        "fs_effective": float(fs_effective),
        "period_hints_full_interval": full_hints,
        "candidate_periods": [float(value) for value in fixed_periods] if fixed_periods is not None else None,
        "candidate_periods_by_window": [
            {
                "window_start_seconds": float(spec["window_start"]),
                "period_hints": spec["window_hints"],
                "candidate_periods": [float(value) for value in spec["candidate_periods"]],
            }
            for spec in window_specs
        ],
        "period_source": period_source,
        "period_strategy": strategy,
        "global_alpha": float(alpha),
        "alpha_per_test": float(alpha_tda),
        "alpha_tda": float(alpha_tda),
        "alpha_baseline": float(alpha_baseline),
        "bonferroni": bool(bonferroni),
        "n_window_tests": int(len(starts)),
        "n_baseline_tests": int(len(starts)),
        "n_period_tests": int(len(fixed_periods)) if fixed_periods is not None else None,
        "n_period_tests_per_window": [int(len(spec["candidate_periods"])) for spec in window_specs],
        "n_total_tests": int(n_total_tests),
        "global_reject": bool(df["reject"].astype(bool).any()) if "reject" in df else False,
        "global_gls_reject": bool(df["gls_reject"].astype(bool).any()) if "gls_reject" in df else False,
        "global_fisher_reject": bool(df["fisher_reject"].astype(bool).any()) if "fisher_reject" in df else False,
        "global_perm_reject": bool(df["perm_reject"].astype(bool).any()) if "perm_reject" in df else False,
    }
    return df, meta_out


def run_fantasia_record_grid(
    data_dir: str | Path,
    records: Sequence[str],
    use_seconds: float = 90.0,
    window_starts: Sequence[float] = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
    window_seconds: float = 20.0,
    downsample_to_hz: float | None = 25.0,
    linear_detrend: bool = True,
    zscore: bool = True,
    bandpass_hz: tuple[float, float] | None = None,
    period_bounds: tuple[float, float] = (2.0, 8.0),
    period_grid: Sequence[float] | None = None,
    period_strategy: str = "window",
    top_periods: int = 1,
    grid_points: int = 3,
    grid_scale: float = 0.10,
    m: int = 8,
    alpha: float = 0.05,
    bonferroni: bool = True,
    b: int | None = None,
    b_fraction: float = 0.99,
    n_subsamples: int = 300,
    max_anchors: int | None = 300,
    seed: int = 0,
    gls_ofac: float = 10.0,
    gls_hifac: float = 1.0,
    periodogram_permutations: int = 500,
    max_edge_length: float | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the fixed Fantasia window grid for multiple records.

    Bonferroni correction is applied within each record over its fixed
    window-period grid.  Records are then summarized separately.
    """

    record_list = [str(record) for record in records]
    if len(record_list) == 0:
        raise ValueError("records must contain at least one record name.")

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(record_list), desc="fantasia records", unit="record")
        except ImportError:
            progress = None

    dfs: list[pd.DataFrame] = []
    metadata_by_record: dict[str, dict] = {}
    try:
        for record_index, record in enumerate(record_list):
            df_record, meta_record = run_fantasia_window_grid(
                data_dir=data_dir,
                record=record,
                use_seconds=use_seconds,
                window_starts=window_starts,
                window_seconds=window_seconds,
                downsample_to_hz=downsample_to_hz,
                linear_detrend=linear_detrend,
                zscore=zscore,
                bandpass_hz=bandpass_hz,
                period_bounds=period_bounds,
                period_grid=period_grid,
                period_strategy=period_strategy,
                top_periods=top_periods,
                grid_points=grid_points,
                grid_scale=grid_scale,
                m=m,
                alpha=alpha,
                bonferroni=bonferroni,
                b=b,
                b_fraction=b_fraction,
                n_subsamples=n_subsamples,
                max_anchors=max_anchors,
                seed=int(seed) + 100_000 * record_index,
                gls_ofac=gls_ofac,
                gls_hifac=gls_hifac,
                periodogram_permutations=periodogram_permutations,
                max_edge_length=max_edge_length,
                show_progress=False,
            )
            df_record["record_index"] = int(record_index)
            dfs.append(df_record)
            metadata_by_record[record] = meta_record
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    df = pd.concat(dfs, ignore_index=True)
    record_summary = summarize_real_record_results(df)

    meta_out = {
        "data_dir": str(data_dir),
        "records": record_list,
        "n_records": int(len(record_list)),
        "use_seconds": float(use_seconds),
        "window_starts": [float(value) for value in window_starts],
        "window_seconds": float(window_seconds),
        "candidate_periods": [float(value) for value in period_grid] if period_grid is not None else None,
        "period_strategy": str(period_strategy),
        "global_alpha_per_record": float(alpha),
        "bonferroni_within_record": bool(bonferroni),
        "metadata_by_record": metadata_by_record,
        "record_summary": record_summary.to_dict(orient="records"),
    }
    return df, meta_out


def run_bidmc_window_grid(
    data_dir: str | Path,
    record: str = "bidmc01",
    use_seconds: float = 180.0,
    window_starts: Sequence[float] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0),
    window_seconds: float = 30.0,
    downsample_to_hz: float | None = 25.0,
    linear_detrend: bool = True,
    zscore: bool = True,
    bandpass_hz: tuple[float, float] | None = None,
    period_bounds: tuple[float, float] = (2.0, 10.0),
    period_grid: Sequence[float] | None = None,
    period_strategy: str = "window",
    top_periods: int = 1,
    grid_points: int = 3,
    grid_scale: float = 0.10,
    m: int = 8,
    alpha: float = 0.05,
    bonferroni: bool = True,
    b: int | None = None,
    b_fraction: float = 0.99,
    n_subsamples: int = 300,
    max_anchors: int | None = 300,
    seed: int = 0,
    gls_ofac: float = 10.0,
    gls_hifac: float = 1.0,
    periodogram_permutations: int = 500,
    max_edge_length: float | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run a BIDMC RESP local-window and adaptive-period grid."""

    starts = [float(value) for value in window_starts]
    if len(starts) == 0:
        raise ValueError("window_starts must contain at least one value.")

    sampto = int(np.ceil(float(use_seconds) * 125.0)) if use_seconds is not None else None
    raw_times, raw_values, meta = load_bidmc_resp(record=record, local_dir=data_dir, sampto=sampto)
    if use_seconds is not None:
        mask = raw_times <= float(use_seconds) + 1e-12
        raw_times = raw_times[mask]
        raw_values = raw_values[mask]

    processed = preprocess_signal(
        raw_values,
        fs=float(meta["fs"]),
        linear_detrend=linear_detrend,
        bandpass_hz=bandpass_hz,
        zscore=zscore,
    )
    work_times, work_values, fs_effective = resample_uniform(raw_times, processed, target_hz=downsample_to_hz)
    full_hints = estimate_period_hints(work_values, fs=fs_effective, period_bounds=period_bounds, top_k=top_periods)
    strategy = str(period_strategy).lower().replace("_", "-")
    if strategy not in {"full", "window"}:
        raise ValueError("period_strategy must be 'full' or 'window'.")

    fixed_periods: list[float] | None = None
    if period_grid is not None:
        fixed_periods = [float(value) for value in period_grid]
        period_source = "fixed"
    elif strategy == "full":
        fixed_periods = build_period_grid_from_hints(
            full_hints["period_hints"],
            period_bounds=period_bounds,
            grid_points=grid_points,
            grid_scale=grid_scale,
        )
        period_source = "full-auto"
    else:
        period_source = "window-auto"

    if fixed_periods is not None and len(fixed_periods) == 0:
        raise ValueError("No candidate periods were provided or estimated.")

    window_specs: list[dict] = []
    for window_index, window_start in enumerate(starts):
        window_times, window_values = select_time_window(
            work_times,
            work_values,
            start_seconds=window_start,
            window_seconds=window_seconds,
        )
        window_hints = estimate_period_hints(
            window_values,
            fs=fs_effective,
            period_bounds=period_bounds,
            top_k=top_periods,
        )
        if fixed_periods is None:
            candidate_periods = build_period_grid_from_hints(
                window_hints["period_hints"],
                period_bounds=period_bounds,
                grid_points=grid_points,
                grid_scale=grid_scale,
            )
        else:
            candidate_periods = fixed_periods
        if len(candidate_periods) == 0:
            raise ValueError(f"No candidate periods were estimated for window starting at {window_start}.")
        window_specs.append(
            {
                "window_index": int(window_index),
                "window_start": float(window_start),
                "window_times": window_times,
                "window_values": window_values,
                "window_hints": window_hints,
                "candidate_periods": [float(value) for value in candidate_periods],
            }
        )

    n_total_tests = int(sum(len(spec["candidate_periods"]) for spec in window_specs))
    alpha_tda = float(alpha) / float(n_total_tests) if bonferroni else float(alpha)
    alpha_baseline = float(alpha) / float(len(starts)) if bonferroni else float(alpha)

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=n_total_tests, desc="bidmc window grid", unit="test")
        except ImportError:
            progress = None

    rows: list[dict] = []
    try:
        for spec in window_specs:
            window_index = int(spec["window_index"])
            window_start = float(spec["window_start"])
            window_times = spec["window_times"]
            window_values = spec["window_values"]
            window_hints = spec["window_hints"]
            candidate_periods = spec["candidate_periods"]
            baseline_summary = run_scalar_baseline_tests(
                times=window_times,
                values=window_values,
                alpha=alpha_baseline,
                period_bounds=period_bounds,
                gls_ofac=gls_ofac,
                gls_hifac=gls_hifac,
                periodogram_permutations=periodogram_permutations,
                seed=int(seed) + 10_000 * window_index,
            )
            for period_index, period_seconds in enumerate(candidate_periods):
                row, _ = run_real_period_case(
                    dataset="bidmc",
                    record=record,
                    channel_name=str(meta["channel_name"]),
                    times=window_times,
                    values=window_values,
                    fs_original=float(meta["fs"]),
                    fs_effective=float(fs_effective),
                    period_seconds=float(period_seconds),
                    m=m,
                    alpha=alpha_tda,
                    b=b,
                    b_fraction=b_fraction,
                    n_subsamples=n_subsamples,
                    max_anchors=max_anchors,
                    seed=int(seed) + 1000 * window_index + period_index,
                    period_bounds=period_bounds,
                    gls_ofac=gls_ofac,
                    gls_hifac=gls_hifac,
                    periodogram_permutations=periodogram_permutations,
                    max_edge_length=max_edge_length,
                    baseline_alpha=alpha_baseline,
                    baseline_summary=baseline_summary,
                )
                row.update(
                    {
                        "use_seconds": float(use_seconds),
                        "window_index": int(window_index),
                        "window_start_seconds": float(window_start),
                        "window_seconds": float(window_seconds),
                        "downsample_to_hz": np.nan if downsample_to_hz is None else float(downsample_to_hz),
                        "linear_detrend": int(linear_detrend),
                        "zscore": int(zscore),
                        "bandpass_low_hz": np.nan if bandpass_hz is None else float(bandpass_hz[0]),
                        "bandpass_high_hz": np.nan if bandpass_hz is None else float(bandpass_hz[1]),
                        "period_index": int(period_index),
                        "period_source": period_source,
                        "global_alpha": float(alpha),
                        "alpha_per_test": float(alpha_tda),
                        "alpha_tda": float(alpha_tda),
                        "alpha_baseline": float(alpha_baseline),
                        "bonferroni": int(bonferroni),
                        "n_window_tests": int(len(starts)),
                        "n_baseline_tests": int(len(starts)),
                        "n_period_tests": int(len(candidate_periods)),
                        "n_total_tests": int(n_total_tests),
                        "acf_period_seconds": window_hints["acf_period"],
                        "acf_peak": window_hints["acf_peak"],
                        "spec_period_seconds": window_hints["spec_period"],
                        "spec_ratio": window_hints["spec_ratio"],
                        "full_acf_period_seconds": full_hints["acf_period"],
                        "full_acf_peak": full_hints["acf_peak"],
                        "full_spec_period_seconds": full_hints["spec_period"],
                        "full_spec_ratio": full_hints["spec_ratio"],
                        "periodogram_permutations": int(periodogram_permutations),
                    }
                )
                rows.append(row)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    df = pd.DataFrame(rows)
    meta_out = {
        **meta,
        "use_seconds": float(use_seconds),
        "window_starts": starts,
        "window_seconds": float(window_seconds),
        "fs_effective": float(fs_effective),
        "period_hints_full_interval": full_hints,
        "candidate_periods": [float(value) for value in fixed_periods] if fixed_periods is not None else None,
        "candidate_periods_by_window": [
            {
                "window_start_seconds": float(spec["window_start"]),
                "period_hints": spec["window_hints"],
                "candidate_periods": [float(value) for value in spec["candidate_periods"]],
            }
            for spec in window_specs
        ],
        "period_source": period_source,
        "period_strategy": strategy,
        "global_alpha": float(alpha),
        "alpha_per_test": float(alpha_tda),
        "alpha_tda": float(alpha_tda),
        "alpha_baseline": float(alpha_baseline),
        "bonferroni": bool(bonferroni),
        "n_window_tests": int(len(starts)),
        "n_baseline_tests": int(len(starts)),
        "n_period_tests": int(len(fixed_periods)) if fixed_periods is not None else None,
        "n_period_tests_per_window": [int(len(spec["candidate_periods"])) for spec in window_specs],
        "n_total_tests": int(n_total_tests),
        "global_reject": bool(df["reject"].astype(bool).any()) if "reject" in df else False,
        "global_gls_reject": bool(df["gls_reject"].astype(bool).any()) if "gls_reject" in df else False,
        "global_fisher_reject": bool(df["fisher_reject"].astype(bool).any()) if "fisher_reject" in df else False,
        "global_perm_reject": bool(df["perm_reject"].astype(bool).any()) if "perm_reject" in df else False,
    }
    return df, meta_out


def run_bidmc_record_grid(
    data_dir: str | Path,
    records: Sequence[str],
    use_seconds: float = 180.0,
    window_starts: Sequence[float] = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0),
    window_seconds: float = 30.0,
    downsample_to_hz: float | None = 25.0,
    linear_detrend: bool = True,
    zscore: bool = True,
    bandpass_hz: tuple[float, float] | None = None,
    period_bounds: tuple[float, float] = (2.0, 10.0),
    period_grid: Sequence[float] | None = None,
    period_strategy: str = "window",
    top_periods: int = 1,
    grid_points: int = 3,
    grid_scale: float = 0.10,
    m: int = 8,
    alpha: float = 0.05,
    bonferroni: bool = True,
    b: int | None = None,
    b_fraction: float = 0.99,
    n_subsamples: int = 300,
    max_anchors: int | None = 300,
    seed: int = 0,
    gls_ofac: float = 10.0,
    gls_hifac: float = 1.0,
    periodogram_permutations: int = 500,
    max_edge_length: float | None = None,
    show_progress: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the BIDMC RESP local-window grid for multiple records."""

    record_list = [str(record) for record in records]
    if len(record_list) == 0:
        raise ValueError("records must contain at least one record name.")

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(record_list), desc="bidmc records", unit="record")
        except ImportError:
            progress = None

    dfs: list[pd.DataFrame] = []
    metadata_by_record: dict[str, dict] = {}
    try:
        for record_index, record in enumerate(record_list):
            df_record, meta_record = run_bidmc_window_grid(
                data_dir=data_dir,
                record=record,
                use_seconds=use_seconds,
                window_starts=window_starts,
                window_seconds=window_seconds,
                downsample_to_hz=downsample_to_hz,
                linear_detrend=linear_detrend,
                zscore=zscore,
                bandpass_hz=bandpass_hz,
                period_bounds=period_bounds,
                period_grid=period_grid,
                period_strategy=period_strategy,
                top_periods=top_periods,
                grid_points=grid_points,
                grid_scale=grid_scale,
                m=m,
                alpha=alpha,
                bonferroni=bonferroni,
                b=b,
                b_fraction=b_fraction,
                n_subsamples=n_subsamples,
                max_anchors=max_anchors,
                seed=int(seed) + 100_000 * record_index,
                gls_ofac=gls_ofac,
                gls_hifac=gls_hifac,
                periodogram_permutations=periodogram_permutations,
                max_edge_length=max_edge_length,
                show_progress=False,
            )
            df_record["record_index"] = int(record_index)
            dfs.append(df_record)
            metadata_by_record[record] = meta_record
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    df = pd.concat(dfs, ignore_index=True)
    record_summary = summarize_real_record_results(df)

    meta_out = {
        "data_dir": str(data_dir),
        "records": record_list,
        "n_records": int(len(record_list)),
        "use_seconds": float(use_seconds),
        "window_starts": [float(value) for value in window_starts],
        "window_seconds": float(window_seconds),
        "candidate_periods": [float(value) for value in period_grid] if period_grid is not None else None,
        "period_strategy": str(period_strategy),
        "global_alpha_per_record": float(alpha),
        "bonferroni_within_record": bool(bonferroni),
        "metadata_by_record": metadata_by_record,
        "record_summary": record_summary.to_dict(orient="records"),
    }
    return df, meta_out


def write_real_results(df: pd.DataFrame, output_dir: str | Path, config: dict, metadata: dict) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "real_results.csv", index=False)
    summarize_real_record_results(df).to_csv(output / "record_summary.csv", index=False)
    window_summary = summarize_real_window_results(df)
    if not window_summary.empty:
        window_summary.to_csv(output / "window_summary.csv", index=False)
    (output / "config.json").write_text(json.dumps(config, indent=2, default=json_safe), encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=json_safe), encoding="utf-8")
