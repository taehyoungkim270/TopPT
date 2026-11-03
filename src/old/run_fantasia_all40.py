"""Run the Fantasia real-data experiment on all 40 canonical records.

This is a convenience wrapper around ``simulation.real_experiments`` with the
same adaptive-window-period settings used in ``fantasia_workflow.ipynb``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from .real_experiments import (
    download_fantasia_records,
    fantasia_record_names,
    run_fantasia_record_grid,
    summarize_real_record_results,
    summarize_real_window_results,
    write_real_results,
)
from .run_real import default_fantasia_dir, parse_csv, parse_pair


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the all-40 Fantasia adaptive-window TDA experiment.")
    parser.add_argument("--data-dir", default=default_fantasia_dir())
    parser.add_argument("--output-dir", default="simulation/results/fantasia_all40_window_grid")
    parser.add_argument("--records", default=None, help="Optional comma-separated subset. Defaults to all 40.")
    parser.add_argument("--first-n-records", type=int, default=None, help="Debug option: run only the first N records.")
    parser.add_argument("--no-download", action="store_true", help="Use existing local WFDB files only.")
    parser.add_argument("--overwrite-download", action="store_true")
    parser.add_argument("--use-seconds", type=float, default=90.0)
    parser.add_argument("--window-start-grid", default="0,10,20,30,40,50,60")
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--downsample-to-hz", type=float, default=25.0)
    parser.add_argument("--period-bounds", default="2.0,8.0")
    parser.add_argument("--period-strategy", choices=["window", "full"], default="window")
    parser.add_argument("--top-periods", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=3)
    parser.add_argument("--grid-scale", type=float, default=0.10)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--b-fraction", type=float, default=0.99)
    parser.add_argument("--n-subsamples", type=int, default=300)
    parser.add_argument("--max-anchors", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--periodogram-permutations", type=int, default=500)
    parser.add_argument("--max-edge-length", type=float, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-bonferroni", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.records:
        records = [str(record) for record in parse_csv(args.records, str)]
    else:
        records = fantasia_record_names(first_n=args.first_n_records)

    window_starts = parse_csv(args.window_start_grid, float)
    if window_starts is None or len(window_starts) == 0:
        raise ValueError("--window-start-grid must contain at least one start time.")
    period_bounds = parse_pair(args.period_bounds)
    if period_bounds is None:
        raise ValueError("--period-bounds is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fantasia records: {len(records)}")
    print(", ".join(records))
    print(f"Output directory: {output_dir}")
    print(
        "Tests per record are adaptive: "
        f"{len(window_starts)} windows x about {args.grid_points} periods/window, "
        f"Bonferroni={'off' if args.no_bonferroni else 'on'}."
    )

    if not args.no_download:
        print("Downloading missing WFDB records through wfdb/PhysioNet...")
        download_fantasia_records(
            data_dir=args.data_dir,
            records=records,
            overwrite=args.overwrite_download,
        )

    started = time.perf_counter()
    df, metadata = run_fantasia_record_grid(
        data_dir=args.data_dir,
        records=records,
        use_seconds=args.use_seconds,
        window_starts=window_starts,
        window_seconds=args.window_seconds,
        downsample_to_hz=args.downsample_to_hz,
        linear_detrend=True,
        zscore=True,
        bandpass_hz=None,
        period_bounds=period_bounds,
        period_grid=None,
        period_strategy=args.period_strategy,
        top_periods=args.top_periods,
        grid_points=args.grid_points,
        grid_scale=args.grid_scale,
        m=args.m,
        alpha=args.alpha,
        bonferroni=not args.no_bonferroni,
        b=None,
        b_fraction=args.b_fraction,
        n_subsamples=args.n_subsamples,
        max_anchors=args.max_anchors,
        seed=args.seed,
        gls_ofac=10.0,
        gls_hifac=1.0,
        periodogram_permutations=args.periodogram_permutations,
        max_edge_length=args.max_edge_length,
        show_progress=not args.no_progress,
    )
    elapsed = time.perf_counter() - started

    config = {
        "records": records,
        "data_dir": str(args.data_dir),
        "output_dir": str(output_dir),
        "use_seconds": args.use_seconds,
        "window_starts": window_starts,
        "window_seconds": args.window_seconds,
        "downsample_to_hz": args.downsample_to_hz,
        "period_bounds": period_bounds,
        "period_grid": None,
        "period_strategy": args.period_strategy,
        "top_periods": args.top_periods,
        "grid_points": args.grid_points,
        "grid_scale": args.grid_scale,
        "m": args.m,
        "alpha": args.alpha,
        "bonferroni": not args.no_bonferroni,
        "b_fraction": args.b_fraction,
        "n_subsamples": args.n_subsamples,
        "max_anchors": args.max_anchors,
        "seed": args.seed,
        "periodogram_permutations": args.periodogram_permutations,
        "elapsed_seconds": elapsed,
    }
    write_real_results(df, output_dir=output_dir, config=config, metadata=metadata)

    record_summary = summarize_real_record_results(df)
    record_summary.to_csv(output_dir / "record_summary.csv", index=False)

    window_summary = summarize_real_window_results(df)
    if not window_summary.empty:
        window_summary.to_csv(output_dir / "window_summary.csv", index=False)

    print(f"Wrote raw results to {output_dir / 'real_results.csv'}")
    print(f"Wrote record summary to {output_dir / 'record_summary.csv'}")
    print(f"Elapsed minutes: {elapsed / 60.0:.2f}")
    print(record_summary.to_string(index=False))


if __name__ == "__main__":
    main()
