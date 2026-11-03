"""Command-line entry point for synthetic experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .synthetic_experiments import run_synthetic_grid, write_results


DEFAULT_MAIN_SIGNALS = [
    "f1_periodic",
    "f2_periodic",
    "h0_linear_trend",
    "h0_quadratic_trend",
    "h0_logistic_trend",
    "h0_aperiodic_trend",
]
DEFAULT_SIGNALS = DEFAULT_MAIN_SIGNALS
DEFAULT_BOUNDED_SIGNALS = DEFAULT_MAIN_SIGNALS
DEFAULT_NOISE_KINDS = ["none", "AG", "AL", "MG", "ML"]
DEFAULT_BOUNDED_NOISE_KINDS = ["none", "BU", "TG"]


def parse_csv(text: str, cast=str):
    if text is None or text == "":
        return []
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def parse_seed_range(text: str) -> list[int]:
    if ":" in text:
        start, stop = text.split(":", 1)
        return list(range(int(start), int(stop)))
    return parse_csv(text, int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic raw time-delay embedding simulations.")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--quick", action="store_true", help="Small smoke-test preset.")
    preset.add_argument("--paper", action="store_true", help="Larger paper-style preset.")
    preset.add_argument("--bounded-error", action="store_true", help="Bounded-error interpolation preset.")
    parser.add_argument("--signals", default=None)
    parser.add_argument("--noise-kinds", default=None)
    parser.add_argument("--noise-scales", default=None)
    parser.add_argument("--seeds", default=None, help="Either comma list like 0,1,2 or half-open range like 0:10.")
    parser.add_argument("--max-anchors", type=int, default=None, help="Optional cap on observed timestamps used as delay-vector anchors.")
    parser.add_argument("--n-observations", type=int, default=None)
    parser.add_argument("--n-observations-grid", default=None, help="Optional comma-separated scalar observation counts.")
    parser.add_argument("--m", type=int, default=20)
    parser.add_argument("--m-values", default=None, help="Optional comma-separated grid of embedding dimensions.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--b", type=int, default=None)
    parser.add_argument("--n-subsamples", type=int, default=None)
    parser.add_argument("--tau-multiplier", type=float, default=1.0)
    parser.add_argument("--tau-multipliers", default=None, help="Optional comma-separated known-period delay multipliers.")
    parser.add_argument("--tau-values", default=None, help="Optional comma-separated fixed tau grid.")
    parser.add_argument("--gls-ofac", type=float, default=10.0, help="GLS oversampling factor for the frequency grid.")
    parser.add_argument("--gls-hifac", type=float, default=1.0, help="GLS maximum frequency as a multiple of the average Nyquist frequency.")
    parser.add_argument("--gls-min-period", type=float, default=None, help="Optional shortest period searched by GLS.")
    parser.add_argument("--gls-max-period", type=float, default=None, help="Optional longest period searched by GLS.")
    parser.add_argument(
        "--periodogram-permutations",
        type=int,
        default=None,
        help="Number of permutations for the empirical max-periodogram test.",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show a tqdm progress bar while running.")
    parser.add_argument("--output-dir", default="simulation/results/synthetic")
    return parser


def resolve_args(args: argparse.Namespace) -> dict:
    if args.bounded_error:
        max_anchors = args.max_anchors
        b = 120 if args.b is None else args.b
        n_subsamples = 500 if args.n_subsamples is None else args.n_subsamples
        seeds = list(range(20)) if args.seeds is None else parse_seed_range(args.seeds)
        noise_scales = [0.02, 0.05, 0.10] if args.noise_scales is None else parse_csv(args.noise_scales, float)
        noise_kinds = DEFAULT_BOUNDED_NOISE_KINDS if args.noise_kinds is None else parse_csv(args.noise_kinds, str)
        if args.n_observations_grid is not None:
            n_observations_values = parse_csv(args.n_observations_grid, int)
        elif args.n_observations is None:
            n_observations_values = [200, 400]
        else:
            n_observations_values = None
        m_values = [12, 20] if args.m_values is None else parse_csv(args.m_values, int)
        periodogram_permutations = 500 if args.periodogram_permutations is None else args.periodogram_permutations
    elif args.paper:
        max_anchors = args.max_anchors
        b = 250 if args.b is None else args.b
        n_subsamples = 1000 if args.n_subsamples is None else args.n_subsamples
        seeds = list(range(100)) if args.seeds is None else parse_seed_range(args.seeds)
        noise_scales = [0.05 * i for i in range(7)] if args.noise_scales is None else parse_csv(args.noise_scales, float)
        noise_kinds = DEFAULT_NOISE_KINDS if args.noise_kinds is None else parse_csv(args.noise_kinds, str)
        n_observations_values = None if args.n_observations_grid is None else parse_csv(args.n_observations_grid, int)
        m_values = None if args.m_values is None else parse_csv(args.m_values, int)
        periodogram_permutations = 1000 if args.periodogram_permutations is None else args.periodogram_permutations
    else:
        max_anchors = args.max_anchors
        b = 80 if args.b is None else args.b
        n_subsamples = 60 if args.n_subsamples is None else args.n_subsamples
        seeds = list(range(3)) if args.seeds is None else parse_seed_range(args.seeds)
        noise_scales = [0.0, 0.05] if args.noise_scales is None else parse_csv(args.noise_scales, float)
        noise_kinds = DEFAULT_NOISE_KINDS if args.noise_kinds is None else parse_csv(args.noise_kinds, str)
        n_observations_values = None if args.n_observations_grid is None else parse_csv(args.n_observations_grid, int)
        m_values = None if args.m_values is None else parse_csv(args.m_values, int)
        periodogram_permutations = 100 if args.periodogram_permutations is None else args.periodogram_permutations

    tau_values = None if args.tau_values is None else parse_csv(args.tau_values, float)
    tau_multipliers = None if args.tau_multipliers is None else parse_csv(args.tau_multipliers, float)
    if args.signals is not None:
        signal_names = parse_csv(args.signals, str)
    elif args.bounded_error:
        signal_names = DEFAULT_BOUNDED_SIGNALS
    else:
        signal_names = DEFAULT_SIGNALS

    return {
        "signal_names": signal_names,
        "seeds": seeds,
        "noise_kinds": noise_kinds,
        "noise_scales": noise_scales,
        "max_anchors": max_anchors,
        "m": args.m,
        "alpha": args.alpha,
        "b": b,
        "n_subsamples": n_subsamples,
        "n_observations": args.n_observations,
        "n_observations_values": n_observations_values,
        "m_values": m_values,
        "tau_values": tau_values,
        "tau_multiplier": args.tau_multiplier,
        "tau_multipliers": tau_multipliers,
        "gls_ofac": args.gls_ofac,
        "gls_hifac": args.gls_hifac,
        "gls_min_period": args.gls_min_period,
        "gls_max_period": args.gls_max_period,
        "periodogram_permutations": periodogram_permutations,
        "show_progress": bool(args.show_progress),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = resolve_args(args)
    df = run_synthetic_grid(**config)
    output_dir = Path(args.output_dir)
    write_results(df, output_dir=output_dir, config={**config, "output_dir": str(output_dir)})
    print(f"Wrote {len(df)} raw rows to {output_dir / 'raw_results.csv'}")
    print(f"Wrote summary to {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
