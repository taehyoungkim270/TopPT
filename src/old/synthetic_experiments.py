"""Synthetic experiments for the raw time-delay embedding periodicity test."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import NoiseConfig, TDAConfig
from .embedding import sliding_window_embedding
from .gls_baseline import GLSConfig, run_gls_periodicity_test
from .interpolation import build_linear_interpolant
from .periodicity_test import TDATestResult, h1_rejection_region, run_tda_periodicity_test
from .sampling import add_noise, evaluate_signal
from .signals import SignalSpec, get_signal
from .spectral_baselines import (
    PermutationPeriodogramConfig,
    run_fisher_g_test,
    run_permutation_max_periodogram_test,
    with_seed,
)


_L_INT_CACHE: dict[tuple[str, float, float, int], float] = {}


def known_period_tau(signal: SignalSpec, m: int, multiplier: float = 1.0) -> float:
    """Use the paper-style known-period choice tau = Xi / (m + 1)."""

    return float(multiplier) * signal.delay_period() / float(m + 1)


def estimate_second_derivative_bound(
    signal: SignalSpec,
    domain: tuple[float, float],
    grid_size: int = 5000,
) -> float:
    """Numerically estimate a C2 interpolation constant on a finite interval.

    This is simulation bookkeeping for the bounded-error theorem.  It is not a
    proof-certified analytic constant.
    """

    lo, hi = map(float, domain)
    if not lo < hi:
        raise ValueError("domain must satisfy lo < hi.")
    cache_key = (signal.name, lo, hi, int(grid_size))
    if cache_key in _L_INT_CACHE:
        return _L_INT_CACHE[cache_key]
    xs = np.linspace(lo, hi, int(grid_size))
    ys = evaluate_signal(signal.func, xs)
    first = np.gradient(ys, xs, edge_order=2)
    second = np.gradient(first, xs, edge_order=2)
    finite = np.abs(second[np.isfinite(second)])
    value = float(np.max(finite)) if finite.size else np.nan
    _L_INT_CACHE[cache_key] = value
    return value


def robust_rejection_from_intervals(result: TDATestResult, radius: float) -> float:
    if not np.isfinite(radius):
        return np.nan
    return float(any(h1_rejection_region(interval, c_alpha=radius) for interval in result.h1_intervals))


def bounded_error_metrics(
    signal: SignalSpec,
    cfg: TDAConfig,
    noise: NoiseConfig,
    obs_times: np.ndarray | None,
    c_alpha: float,
) -> dict:
    """Compute the interpolation-error quantities used by Theorem 5.4."""

    empty = {
        "bounded_error_applicable": 0,
        "delta_N": np.nan,
        "sigma_N": np.nan,
        "l_int_hat": np.nan,
        "eta_N": np.nan,
        "zeta_mN": np.nan,
        "c_alpha_err": np.nan,
    }
    sigma_n = noise.deterministic_bound()
    if obs_times is None or sigma_n is None:
        return empty

    obs_times = np.asarray(obs_times, dtype=float)
    if obs_times.size < 2:
        return empty
    delta_n = float(np.max(np.diff(np.sort(obs_times))))
    l_int_hat = estimate_second_derivative_bound(signal, domain=(float(obs_times[0]), float(obs_times[-1])))
    eta_n = float(sigma_n) + float(l_int_hat) * delta_n**2 / 8.0
    zeta_mn = np.sqrt(float(cfg.m + 1)) * eta_n
    return {
        "bounded_error_applicable": 1,
        "delta_N": delta_n,
        "sigma_N": float(sigma_n),
        "l_int_hat": float(l_int_hat),
        "eta_N": float(eta_n),
        "zeta_mN": float(zeta_mn),
        "c_alpha_err": float(c_alpha) + 5.0 * float(zeta_mn),
    }


def observed_anchor_times(
    obs_times: np.ndarray,
    anchor_domain: tuple[float, float],
    max_anchors: int | None = None,
) -> np.ndarray:
    """Use observed timestamps as valid sliding-window anchors.

    Observations cover the extended interval ``[T_min, T_max + m*tau]``.
    Anchors live on the original signal interval ``[T_min, T_max]``.  If
    ``max_anchors`` is given, the valid observed anchors are deterministically
    thinned to keep computations manageable.
    """

    obs_times = np.asarray(obs_times, dtype=float)
    lo, hi = map(float, anchor_domain)
    anchors = obs_times[(obs_times >= lo) & (obs_times <= hi + 1e-12)]
    if anchors.size == 0:
        raise ValueError("No observed timestamps are valid sliding-window anchors.")
    if max_anchors is not None and max_anchors > 0 and anchors.size > max_anchors:
        indices = np.linspace(0, anchors.size - 1, num=int(max_anchors))
        anchors = anchors[np.unique(np.rint(indices).astype(int))]
    return anchors


def build_interpolated_embedding(
    signal: SignalSpec,
    n_observations: int,
    m: int,
    tau: float,
    seed: int | None = None,
    noise: NoiseConfig | None = None,
    max_anchors: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an embedding from a noisy scalar linear interpolant.

    The scalar observation grid covers ``[T_min, T_max + m*tau]``.  There is no
    periodic wrapping or phase folding.  Point-cloud anchors are observed
    timestamps on the original signal interval.  This matches the real-data
    workflow: observe, interpolate when delayed coordinates are not observed
    exactly, then embed.
    """

    lo, hi = map(float, signal.domain)
    obs_domain = (lo, hi + float(m) * float(tau))
    obs_times = np.linspace(obs_domain[0], obs_domain[1], int(n_observations))
    obs_true = evaluate_signal(signal.func, obs_times)
    obs_values = add_noise(obs_true, noise or NoiseConfig(), seed=None if seed is None else seed + 20_000)
    interpolant = build_linear_interpolant(obs_times, obs_values)

    anchors = observed_anchor_times(obs_times, signal.domain, max_anchors=max_anchors)
    x = sliding_window_embedding(
        interpolant,
        anchors,
        m=m,
        tau=tau,
    )
    return anchors, x, obs_times, obs_values


def run_single_synthetic(
    signal: SignalSpec,
    tda_config: TDAConfig,
    noise: NoiseConfig,
    seed: int,
    n_observations: int | None = None,
    max_anchors: int | None = None,
    gls_config: GLSConfig | None = None,
    permutation_config: PermutationPeriodogramConfig | None = None,
) -> tuple[dict, TDATestResult]:
    """Run one synthetic replicate."""

    noise.validate()
    cfg = replace(tda_config, seed=seed)
    if n_observations is None:
        n_observations = max(0 if max_anchors is None else int(max_anchors), 2 * cfg.b, 4 * (cfg.m + 1))
    _, x, obs_times, obs_values = build_interpolated_embedding(
        signal,
        n_observations=n_observations,
        m=cfg.m,
        tau=cfg.tau,
        seed=seed,
        noise=noise,
        max_anchors=max_anchors,
    )
    n_obs_used = int(n_observations)

    result = run_tda_periodicity_test(x, cfg)
    gls_result = run_gls_periodicity_test(
        obs_times,
        obs_values,
        gls_config or GLSConfig(alpha=cfg.alpha),
    )
    fisher_result = run_fisher_g_test(obs_times, obs_values, alpha=cfg.alpha)
    perm_result = run_permutation_max_periodogram_test(
        obs_times,
        obs_values,
        with_seed(
            permutation_config or PermutationPeriodogramConfig(alpha=cfg.alpha),
            None if seed is None else int(seed) + 30_000,
        ),
    )
    err = bounded_error_metrics(
        signal=signal,
        cfg=cfg,
        noise=noise,
        obs_times=obs_times,
        c_alpha=result.c_alpha,
    )
    robust_reject = robust_rejection_from_intervals(result, radius=err["c_alpha_err"])
    row = {
        "signal": signal.name,
        "truth_periodic": int(signal.is_periodic),
        "max_anchors": np.nan if max_anchors is None else int(max_anchors),
        "n_points": int(x.shape[0]),
        "n_observations": n_obs_used,
        "seed": int(seed),
        "noise_kind": noise.kind,
        "noise_scale": float(noise.scale),
        "m": int(cfg.m),
        "tau": float(cfg.tau),
        "alpha": float(cfg.alpha),
        "b": int(cfg.b),
        "n_subsamples": int(cfg.n_subsamples),
        **result.summary_dict(),
        **gls_result.summary_dict(),
        **fisher_result.summary_dict(),
        **perm_result.summary_dict(),
        **err,
        "robust_reject": robust_reject,
    }
    return row, result


def run_synthetic_grid(
    signal_names: Sequence[str],
    seeds: Iterable[int],
    noise_kinds: Sequence[str],
    noise_scales: Sequence[float],
    max_anchors: int | None,
    m: int,
    alpha: float,
    b: int,
    n_subsamples: int,
    n_observations: int | None = None,
    n_observations_values: Sequence[int] | None = None,
    m_values: Sequence[int] | None = None,
    tau_values: Sequence[float] | None = None,
    tau_multiplier: float = 1.0,
    tau_multipliers: Sequence[float] | None = None,
    gls_ofac: float = 10.0,
    gls_hifac: float = 1.0,
    gls_min_period: float | None = None,
    gls_max_period: float | None = None,
    periodogram_permutations: int = 500,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Run a grid of synthetic simulations and return one row per replicate."""

    rows: list[dict] = []
    seed_list = [int(seed) for seed in seeds]
    gls_config = GLSConfig(
        alpha=alpha,
        ofac=gls_ofac,
        hifac=gls_hifac,
        min_period=gls_min_period,
        max_period=gls_max_period,
    )
    gls_config.validate()
    permutation_config = PermutationPeriodogramConfig(
        alpha=alpha,
        n_permutations=int(periodogram_permutations),
    )
    permutation_config.validate()
    m_grid = list(m_values) if m_values is not None else [int(m)]
    if n_observations_values is not None:
        obs_grid = [int(value) for value in n_observations_values]
    else:
        obs_grid = [n_observations]

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            total = 0
            for _signal_name in signal_names:
                for m_current in m_grid:
                    if tau_values is not None:
                        tau_count = len(tau_values)
                    else:
                        tau_count = len(tau_multipliers) if tau_multipliers is not None else 1
                    for _ in range(tau_count):
                        for _obs_count in obs_grid:
                            for noise_kind in noise_kinds:
                                scale_count = 1 if noise_kind == "none" else len(noise_scales)
                                total += scale_count * len(seed_list)
            progress = tqdm(total=total, desc="synthetic grid", unit="run")
        except ImportError:
            progress = None

    try:
        for signal_name in signal_names:
            signal = get_signal(signal_name)
            for m_current in m_grid:
                if tau_values is not None:
                    tau_grid = [(float(tau), np.nan) for tau in tau_values]
                else:
                    multipliers = list(tau_multipliers) if tau_multipliers is not None else [tau_multiplier]
                    tau_grid = [
                        (known_period_tau(signal, m=m_current, multiplier=multiplier), float(multiplier))
                        for multiplier in multipliers
                    ]
                for tau, tau_mult in tau_grid:
                    base_cfg = TDAConfig(
                        m=m_current,
                        tau=float(tau),
                        alpha=alpha,
                        b=b,
                        n_subsamples=n_subsamples,
                    )
                    for obs_count in obs_grid:
                        for noise_kind in noise_kinds:
                            scale_grid = [0.0] if noise_kind == "none" else list(noise_scales)
                            for noise_scale in scale_grid:
                                noise = NoiseConfig(kind=noise_kind, scale=float(noise_scale))
                                for seed in seed_list:
                                    row, _ = run_single_synthetic(
                                        signal=signal,
                                        tda_config=base_cfg,
                                        noise=noise,
                                        seed=int(seed),
                                        n_observations=obs_count,
                                        max_anchors=max_anchors,
                                        gls_config=gls_config,
                                        permutation_config=permutation_config,
                                    )
                                    row["tau_multiplier"] = tau_mult
                                    row["gls_alpha"] = float(gls_config.alpha)
                                    row["gls_ofac"] = float(gls_config.ofac)
                                    row["gls_hifac"] = float(gls_config.hifac)
                                    row["gls_min_period"] = (
                                        np.nan if gls_config.min_period is None else float(gls_config.min_period)
                                    )
                                    row["gls_max_period"] = (
                                        np.nan if gls_config.max_period is None else float(gls_config.max_period)
                                    )
                                    row["periodogram_permutations"] = int(permutation_config.n_permutations)
                                    rows.append(row)
                                    if progress is not None:
                                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return pd.DataFrame(rows)


def summarize_synthetic_results(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "signal",
        "truth_periodic",
        "max_anchors",
        "n_points",
        "noise_kind",
        "noise_scale",
        "m",
        "tau",
        "tau_multiplier",
        "alpha",
        "b",
        "n_subsamples",
        "n_observations",
        "gls_ofac",
        "gls_hifac",
        "gls_min_period",
        "gls_max_period",
        "periodogram_permutations",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_reps=("reject", "size"),
            detection_rate=("reject", "mean"),
            robust_detection_rate=("robust_reject", "mean"),
            gls_detection_rate=("gls_reject", "mean"),
            fisher_detection_rate=("fisher_reject", "mean"),
            perm_detection_rate=("perm_reject", "mean"),
            mean_c_alpha=("c_alpha", "mean"),
            sd_c_alpha=("c_alpha", "std"),
            mean_c_alpha_err=("c_alpha_err", "mean"),
            mean_zeta_mN=("zeta_mN", "mean"),
            mean_gls_fap=("gls_fap", "mean"),
            median_gls_fap=("gls_fap", "median"),
            mean_gls_best_period=("gls_best_period", "mean"),
            mean_fisher_p_value=("fisher_p_value", "mean"),
            median_fisher_p_value=("fisher_p_value", "median"),
            mean_fisher_best_period=("fisher_best_period", "mean"),
            mean_perm_p_value=("perm_p_value", "mean"),
            median_perm_p_value=("perm_p_value", "median"),
            mean_perm_best_period=("perm_best_period", "mean"),
            mean_best_persistence=("best_persistence", "mean"),
            median_best_persistence=("best_persistence", "median"),
        )
        .reset_index()
    )
    summary["error_rate"] = np.where(
        summary["truth_periodic"].astype(bool),
        1.0 - summary["detection_rate"],
        summary["detection_rate"],
    )
    summary["robust_error_rate"] = np.where(
        summary["truth_periodic"].astype(bool),
        1.0 - summary["robust_detection_rate"],
        summary["robust_detection_rate"],
    )
    summary["gls_error_rate"] = np.where(
        summary["truth_periodic"].astype(bool),
        1.0 - summary["gls_detection_rate"],
        summary["gls_detection_rate"],
    )
    summary["fisher_error_rate"] = np.where(
        summary["truth_periodic"].astype(bool),
        1.0 - summary["fisher_detection_rate"],
        summary["fisher_detection_rate"],
    )
    summary["perm_error_rate"] = np.where(
        summary["truth_periodic"].astype(bool),
        1.0 - summary["perm_detection_rate"],
        summary["perm_detection_rate"],
    )
    return summary


def write_results(df: pd.DataFrame, output_dir: str | Path, config: dict) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "raw_results.csv", index=False)
    summarize_synthetic_results(df).to_csv(output / "summary.csv", index=False)
    (output / "config.json").write_text(json.dumps(config, indent=2, default=json_safe), encoding="utf-8")


def json_safe(value):
    if isinstance(value, range):
        return list(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")
