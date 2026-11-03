"""Plot aggregate real-data experiment outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    ("TDA", "global_tda_reject", "tda_window_reject", "tab:blue"),
    ("GLS", "global_gls_reject", "gls_reject", "tab:orange"),
    ("Fisher g", "global_fisher_reject", "fisher_reject", "tab:green"),
    ("Permutation", "global_perm_reject", "perm_reject", "tab:red"),
]


def read_real_output(input_dir: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    record_path = input_dir / "record_summary.csv"
    window_path = input_dir / "window_summary.csv"
    if not record_path.exists():
        raise FileNotFoundError(f"Could not find {record_path}")
    if not window_path.exists():
        raise FileNotFoundError(f"Could not find {window_path}")
    records = pd.read_csv(record_path)
    windows = pd.read_csv(window_path)
    records["dataset"] = label
    windows["dataset"] = label
    return records, windows


def boolean_mean(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns or frame.empty:
        return float("nan")
    values = frame[col]
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype(str).str.strip().str.lower()
    text_bool = text.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )
    parsed = pd.Series(pd.NA, index=values.index, dtype="boolean")
    parsed[numeric.notna()] = numeric[numeric.notna()] != 0
    parsed[numeric.isna() & text_bool.notna()] = text_bool[numeric.isna() & text_bool.notna()]
    parsed = parsed.dropna()
    return float(parsed.mean()) if not parsed.empty else float("nan")


def plot_rejection_rates(records: pd.DataFrame, windows: pd.DataFrame, output_dir: Path) -> None:
    labels = list(records["dataset"].drop_duplicates())
    x = np.arange(len(labels))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for idx, (name, record_col, window_col, color) in enumerate(METHODS):
        offset = (idx - 1.5) * width
        record_rates = [boolean_mean(records[records["dataset"] == label], record_col) for label in labels]
        window_rates = [boolean_mean(windows[windows["dataset"] == label], window_col) for label in labels]
        axes[0].bar(x + offset, record_rates, width=width, label=name, color=color, alpha=0.9)
        axes[1].bar(x + offset, window_rates, width=width, label=name, color=color, alpha=0.9)

    axes[0].set_title("Record-level rejection")
    axes[1].set_title("Window-level rejection")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("rejection rate")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "real_rejection_rates.pdf")
    fig.savefig(output_dir / "real_rejection_rates.png", dpi=220)
    plt.close(fig)


def plot_tda_diagnostics(records: pd.DataFrame, windows: pd.DataFrame, output_dir: Path) -> None:
    labels = list(records["dataset"].drop_duplicates())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    margin_data = [
        pd.to_numeric(windows[windows["dataset"] == label]["best_tda_margin"], errors="coerce").dropna().to_numpy()
        for label in labels
    ]
    axes[0].boxplot(margin_data, tick_labels=labels, showmeans=True)
    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[0].set_title("Best TDA margin by window")
    axes[0].set_ylabel("best persistence - 2 c_alpha")
    axes[0].grid(axis="y", alpha=0.25)

    for label in labels:
        sub = records[records["dataset"] == label]
        axes[1].hist(
            pd.to_numeric(sub["tda_rejecting_windows"], errors="coerce").dropna(),
            bins=np.arange(-0.5, sub["n_windows"].max() + 1.5, 1.0),
            alpha=0.55,
            label=label,
        )
    axes[1].set_title("TDA rejecting windows per record")
    axes[1].set_xlabel("number of rejecting windows")
    axes[1].set_ylabel("records")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "real_tda_diagnostics.pdf")
    fig.savefig(output_dir / "real_tda_diagnostics.png", dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot aggregate real-data experiment results.")
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="Folders containing record_summary.csv and window_summary.csv.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels. Must match the number of input dirs if supplied.",
    )
    parser.add_argument("--output-dir", default="figures", help="Figure output folder.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dirs = [Path(item) for item in args.input_dirs]
    labels = args.labels if args.labels is not None else [path.name for path in input_dirs]
    if len(labels) != len(input_dirs):
        raise ValueError("--labels must have the same length as --input-dirs.")

    record_frames = []
    window_frames = []
    for input_dir, label in zip(input_dirs, labels):
        records, windows = read_real_output(input_dir, label)
        record_frames.append(records)
        window_frames.append(windows)
    records = pd.concat(record_frames, ignore_index=True)
    windows = pd.concat(window_frames, ignore_index=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_rejection_rates(records, windows, output_dir)
    plot_tda_diagnostics(records, windows, output_dir)
    print(f"Wrote real-data figures to {output_dir}")


if __name__ == "__main__":
    main()
