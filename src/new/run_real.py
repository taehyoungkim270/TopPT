"""Run a real-data experiment from a configuration file.

Usage: PYTHONPATH=src python -m new.run_real configs/real_fantasia.toml

Records are downloaded to ``data/<database>/`` unless ``--data-dir`` is given, and
results go to ``results/<config name>/`` unless ``--output-dir`` is given. Each
record is a batch, so rerunning the same command after an interruption skips
finished records.
"""

import argparse
from functools import partial
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import download_records, load_series, record_files, select_window
from .method import (
    candidate_periods,
    delay_cloud,
    fisher_g_test,
    gls_test,
    permutation_test,
    strongest_period,
    toppt_test,
)
from .runs import ROOT, load_config, run_batches, start_run

RNG_PLAN = (
    "NeurIPS submission: default_rng(100000 * record + 1000 * window + period) for subsampling, "
    "default_rng(100000 * record + 10000 * window + 50000) for permutation, with zero-based "
    "positions in the configuration."
)


def make_rng(record: int, window: int, period: int | None = None) -> np.random.Generator:
    """Return the generator for subsampling at one period, or for a window's permutations."""
    if period is None:
        return np.random.default_rng(100_000 * record + 10_000 * window + 50_000)
    return np.random.default_rng(100_000 * record + 1_000 * window + period)


def run_record(config: dict, data_dir: Path, indexed_record: tuple[int, str]) -> list[dict]:
    """Test every window of one record and return one row per window and period.

    TopPT is corrected over the record's window-period pairs, and the baselines,
    which run once per window, are corrected over its windows.
    """
    index, record = indexed_record
    m, alpha, bounds = config["m"], config["alpha"], config["period_bounds"]
    times, values = load_series(record, data_dir, config)
    windows = [
        select_window(times, values, start, config["window_seconds"])
        for start in config["window_starts"]
    ]
    periods_by_window = [
        candidate_periods(
            strongest_period(window_values, config["downsample_hz"], bounds),
            bounds,
            config["period_grid_points"],
            config["period_grid_scale"],
        )
        for _, window_values in windows
    ]
    alpha_toppt = alpha / sum(len(periods) for periods in periods_by_window)
    alpha_baseline = alpha / len(windows)

    rows = []
    for window, ((window_times, window_values), periods) in enumerate(zip(windows, periods_by_window)):
        baselines = {
            **gls_test(window_times, window_values, alpha_baseline, min_period=bounds[0], max_period=bounds[1]),
            **fisher_g_test(window_times, window_values, alpha_baseline),
            **permutation_test(
                window_times,
                window_values,
                alpha_baseline,
                config["n_permutations"],
                make_rng(index, window),
            ),
        }
        for period_index, period in enumerate(periods):
            cloud = delay_cloud(window_times, window_values, m, period / (m + 1), config["max_anchors"])
            b = int(np.floor(config["b_fraction"] * len(cloud)))
            rows.append(
                {
                    "record": record,
                    "window_index": window,
                    "window_start": config["window_starts"][window],
                    "period_index": period_index,
                    "period": period,
                    "n_points": len(cloud),
                    "b": b,
                    "alpha_toppt": alpha_toppt,
                    "alpha_baseline": alpha_baseline,
                    **toppt_test(
                        cloud,
                        alpha_toppt,
                        b,
                        config["n_subsamples"],
                        make_rng(index, window, period_index),
                    ),
                    **baselines,
                }
            )
    return rows


def run_experiment(config: dict, run_dir: Path, data_dir: Path, workers: int) -> None:
    """Download the records, then run every record that has not finished yet."""
    records = config["records"]
    download_records(config["database"], records, data_dir)
    start_run(
        config,
        run_dir,
        ["data.py", "method/*.py", "run_real.py"],
        RNG_PLAN,
        record_files(records, data_dir),
    )
    batches = {record: (index, record) for index, record in enumerate(records)}
    run_batches(partial(run_record, config, data_dir), batches, run_dir, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = args.output_dir or ROOT / "results" / args.config.stem
    data_dir = args.data_dir or ROOT / "data" / config["database"]
    started = perf_counter()
    run_experiment(config, run_dir, data_dir, args.workers)
    print(f"Wrote {run_dir / 'raw_results.csv'} in {(perf_counter() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
