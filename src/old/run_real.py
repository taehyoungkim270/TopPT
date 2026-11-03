"""Command-line entry point for real-data experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .real_experiments import (
    download_fantasia_records,
    fantasia_record_names,
    list_local_wfdb_records,
    run_fantasia_experiment,
    run_fantasia_record_grid,
    run_fantasia_window_grid,
    write_real_results,
)


OLD_FANTASIA_DIR = Path(r"C:\Users\lion3\Desktop\PhD\Periodicity\simulation\data\fantasia")


def parse_csv(text: str | None, cast=float):
    if text is None or text == "":
        return None
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def parse_pair(text: str | None) -> tuple[float, float] | None:
    values = parse_csv(text, float)
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError(f"Expected two comma-separated values, got {text!r}.")
    return float(values[0]), float(values[1])


def default_fantasia_dir() -> str:
    if OLD_FANTASIA_DIR.exists():
        return str(OLD_FANTASIA_DIR)
    return "data/fantasia"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real-data periodicity experiments.")
    parser.add_argument("--dataset", choices=["fantasia"], default="fantasia")
    parser.add_argument("--record", default="f1y01")
    parser.add_argument("--records", default=None, help="Optional comma-separated record list for multi-record runs.")
    parser.add_argument("--first-n-records", type=int, default=None, help="Use the first N canonical Fantasia records.")
    parser.add_argument("--download-records", action="store_true", help="Download selected Fantasia records with wfdb.")
    parser.add_argument("--overwrite-download", action="store_true", help="Overwrite existing downloaded WFDB files.")
    parser.add_argument("--data-dir", default=default_fantasia_dir())
    parser.add_argument("--output-dir", default="simulation/results/fantasia")
    parser.add_argument("--use-seconds", type=float, default=90.0, help="Prefix length loaded from the record.")
    parser.add_argument("--window-start-seconds", type=float, default=0.0)
    parser.add_argument(
        "--window-start-grid",
        default=None,
        help="Optional comma-separated window starts. If set, runs the fixed-window grid.",
    )
    parser.add_argument("--window-seconds", type=float, default=60.0)
    parser.add_argument("--downsample-to-hz", type=float, default=12.0)
    parser.add_argument("--no-linear-detrend", action="store_true")
    parser.add_argument("--no-zscore", action="store_true")
    parser.add_argument("--bandpass-hz", default=None, help="Optional comma pair like 0.05,1.0.")
    parser.add_argument("--period-bounds", default="2.0,8.0", help="Comma pair in seconds.")
    parser.add_argument("--period-grid", default=None, help="Optional fixed comma-separated candidate periods in seconds.")
    parser.add_argument(
        "--period-strategy",
        choices=["window", "full"],
        default="window",
        help="When --period-grid is omitted, choose periods per window or from the full loaded interval.",
    )
    parser.add_argument("--top-periods", type=int, default=1)
    parser.add_argument("--grid-points", type=int, default=5)
    parser.add_argument("--grid-scale", type=float, default=0.10)
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--b", type=int, default=None)
    parser.add_argument("--b-fraction", type=float, default=0.75)
    parser.add_argument("--n-subsamples", type=int, default=300)
    parser.add_argument("--max-anchors", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gls-ofac", type=float, default=10.0)
    parser.add_argument("--gls-hifac", type=float, default=1.0)
    parser.add_argument("--periodogram-permutations", type=int, default=500)
    parser.add_argument("--max-edge-length", type=float, default=None)
    parser.add_argument("--show-progress", action="store_true", help="Show a tqdm progress bar while running.")
    parser.add_argument(
        "--no-bonferroni",
        action="store_true",
        help="Do not divide alpha by the number of window-period tests when using --window-start-grid.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    bandpass_hz = parse_pair(args.bandpass_hz)
    period_bounds = parse_pair(args.period_bounds)
    if period_bounds is None:
        raise ValueError("--period-bounds is required.")
    period_grid = parse_csv(args.period_grid, float)
    window_start_grid = parse_csv(args.window_start_grid, float)
    record_grid = parse_csv(args.records, str)
    if args.first_n_records is not None:
        record_grid = fantasia_record_names(first_n=args.first_n_records)
    if args.download_records:
        download_fantasia_records(
            data_dir=args.data_dir,
            records=record_grid if record_grid is not None else [args.record],
            overwrite=args.overwrite_download,
        )

    config = {
        "dataset": args.dataset,
        "record": args.record,
        "records": record_grid,
        "first_n_records": args.first_n_records,
        "download_records": args.download_records,
        "overwrite_download": args.overwrite_download,
        "data_dir": args.data_dir,
        "use_seconds": args.use_seconds,
        "window_start_seconds": args.window_start_seconds,
        "window_start_grid": window_start_grid,
        "window_seconds": args.window_seconds,
        "downsample_to_hz": args.downsample_to_hz,
        "linear_detrend": not args.no_linear_detrend,
        "zscore": not args.no_zscore,
        "bandpass_hz": bandpass_hz,
        "period_bounds": period_bounds,
        "period_grid": period_grid,
        "period_strategy": args.period_strategy,
        "top_periods": args.top_periods,
        "grid_points": args.grid_points,
        "grid_scale": args.grid_scale,
        "m": args.m,
        "alpha": args.alpha,
        "b": args.b,
        "b_fraction": args.b_fraction,
        "n_subsamples": args.n_subsamples,
        "max_anchors": args.max_anchors,
        "seed": args.seed,
        "gls_ofac": args.gls_ofac,
        "gls_hifac": args.gls_hifac,
        "periodogram_permutations": args.periodogram_permutations,
        "max_edge_length": args.max_edge_length,
        "show_progress": bool(args.show_progress),
        "bonferroni": not args.no_bonferroni,
        "output_dir": args.output_dir,
    }

    if args.dataset == "fantasia":
        common_kwargs = dict(
            data_dir=args.data_dir,
            use_seconds=args.use_seconds,
            window_seconds=args.window_seconds,
            downsample_to_hz=args.downsample_to_hz,
            linear_detrend=not args.no_linear_detrend,
            zscore=not args.no_zscore,
            bandpass_hz=bandpass_hz,
            period_bounds=period_bounds,
            period_grid=period_grid,
            period_strategy=args.period_strategy,
            top_periods=args.top_periods,
            grid_points=args.grid_points,
            grid_scale=args.grid_scale,
            m=args.m,
            alpha=args.alpha,
            b=args.b,
            b_fraction=args.b_fraction,
            n_subsamples=args.n_subsamples,
            max_anchors=args.max_anchors,
            seed=args.seed,
            gls_ofac=args.gls_ofac,
            gls_hifac=args.gls_hifac,
            periodogram_permutations=args.periodogram_permutations,
            max_edge_length=args.max_edge_length,
            show_progress=args.show_progress,
        )
        if record_grid is not None:
            df, metadata = run_fantasia_record_grid(
                records=record_grid,
                window_starts=window_start_grid if window_start_grid is not None else [args.window_start_seconds],
                bonferroni=not args.no_bonferroni,
                **common_kwargs,
            )
        elif window_start_grid is None:
            df, metadata = run_fantasia_experiment(
                record=args.record,
                window_start_seconds=args.window_start_seconds,
                **common_kwargs,
            )
        else:
            df, metadata = run_fantasia_window_grid(
                record=args.record,
                window_starts=window_start_grid,
                bonferroni=not args.no_bonferroni,
                **common_kwargs,
            )
    else:
        raise ValueError(f"Unsupported dataset {args.dataset!r}.")

    write_real_results(df, output_dir=args.output_dir, config=config, metadata=metadata)
    print(f"Wrote {len(df)} real-data rows to {Path(args.output_dir) / 'real_results.csv'}")
    print(f"Wrote metadata to {Path(args.output_dir) / 'metadata.json'}")
    if not df.empty:
        display_cols = [
            "window_start_seconds",
            "candidate_period_seconds",
            "alpha_per_test",
            "alpha_baseline",
            "reject",
            "c_alpha",
            "best_persistence",
            "gls_reject",
            "gls_fap",
            "fisher_reject",
            "fisher_p_value",
            "perm_reject",
            "perm_p_value",
        ]
        display_cols = [col for col in display_cols if col in df.columns]
        print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
