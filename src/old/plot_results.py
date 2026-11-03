"""Plot synthetic simulation outputs.

This module reads the CSV files produced by ``simulation.run_synthetic`` and
writes diagnostic figures.  Aggregate figures use the saved CSVs directly.
Representative PCA and persistence-diagram figures reconstruct selected point
clouds from the row metadata, because full clouds and diagrams are not stored
in the result CSVs.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import NoiseConfig
from .persistence import compute_rips_persistence, intervals_by_dim
from .signals import get_signal
from .synthetic_experiments import build_interpolated_embedding


RATE_COLORS = {
    "none": "black",
    "AG": "tab:blue",
    "AL": "tab:orange",
    "MG": "tab:green",
    "ML": "tab:red",
    "BU": "tab:purple",
    "TG": "tab:brown",
    "bounded": "tab:purple",
}


def safe_name(text: object) -> str:
    value = str(text)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "value"


def finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def read_results(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    raw_path = input_dir / "raw_results.csv"
    summary_path = input_dir / "summary.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Could not find {raw_path}")
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    return raw, summary


def prepare_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [
        "truth_periodic",
        "max_anchors",
        "n_points",
        "n_observations",
        "seed",
        "noise_scale",
        "m",
        "tau",
        "tau_multiplier",
        "alpha",
        "b",
        "n_subsamples",
        "reject",
        "robust_reject",
        "gls_reject",
        "gls_fap",
        "gls_max_power",
        "gls_power_threshold",
        "gls_best_frequency",
        "gls_best_period",
        "gls_n_frequencies",
        "fisher_reject",
        "fisher_p_value",
        "fisher_g",
        "fisher_best_frequency",
        "fisher_best_period",
        "perm_reject",
        "perm_p_value",
        "perm_statistic",
        "perm_best_frequency",
        "perm_best_period",
        "perm_n_permutations",
        "c_alpha",
        "zeta_mN",
        "c_alpha_err",
        "best_birth",
        "best_death",
        "best_persistence",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "reject" in out.columns:
        out["error"] = np.where(out["truth_periodic"].astype(bool), 1.0 - out["reject"], out["reject"])
    if "robust_reject" in out.columns:
        out["robust_error"] = np.where(
            out["truth_periodic"].astype(bool),
            1.0 - out["robust_reject"],
            out["robust_reject"],
        )
    if "gls_reject" in out.columns:
        out["gls_error"] = np.where(out["truth_periodic"].astype(bool), 1.0 - out["gls_reject"], out["gls_reject"])
    if "fisher_reject" in out.columns:
        out["fisher_error"] = np.where(
            out["truth_periodic"].astype(bool),
            1.0 - out["fisher_reject"],
            out["fisher_reject"],
        )
    if "perm_reject" in out.columns:
        out["perm_error"] = np.where(out["truth_periodic"].astype(bool), 1.0 - out["perm_reject"], out["perm_reject"])
    return out


def make_axes(n_items: int, width: float = 6.0, height: float = 3.8):
    n_cols = min(2, max(1, n_items))
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows), squeeze=False)
    return fig, axes.ravel()


def save_rate_vs_noise(df: pd.DataFrame, output_dir: Path, value_col: str, ylabel: str, filename: str) -> None:
    needed = {"signal", "noise_kind", "noise_scale", value_col}
    if not needed.issubset(df.columns):
        return
    data = df.dropna(subset=["noise_scale", value_col])
    if data.empty:
        return

    grouped = (
        data.groupby(["signal", "noise_kind", "noise_scale"], dropna=False)[value_col]
        .mean()
        .reset_index()
        .sort_values(["signal", "noise_kind", "noise_scale"])
    )
    signals = list(grouped["signal"].drop_duplicates())
    fig, axes = make_axes(len(signals))
    for ax, signal in zip(axes, signals):
        sub = grouped[grouped["signal"] == signal]
        for noise_kind, curve in sub.groupby("noise_kind", dropna=False):
            curve = curve.sort_values("noise_scale")
            ax.plot(
                curve["noise_scale"],
                curve[value_col],
                marker="o",
                label=str(noise_kind),
                color=RATE_COLORS.get(str(noise_kind), None),
            )
        ax.set_title(str(signal))
        ax.set_xlabel("noise scale")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[len(signals) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def save_raw_vs_robust(df: pd.DataFrame, output_dir: Path) -> None:
    if "robust_reject" not in df.columns or df["robust_reject"].dropna().empty:
        return
    needed = {"signal", "noise_kind", "noise_scale", "reject", "robust_reject"}
    if not needed.issubset(df.columns):
        return

    data = df.dropna(subset=["noise_scale"])
    if data.empty:
        return
    long = data.melt(
        id_vars=["signal", "noise_kind", "noise_scale"],
        value_vars=["reject", "robust_reject"],
        var_name="method",
        value_name="detection_rate",
    ).dropna(subset=["detection_rate"])
    grouped = (
        long.groupby(["signal", "noise_kind", "noise_scale", "method"], dropna=False)["detection_rate"]
        .mean()
        .reset_index()
        .sort_values(["signal", "noise_kind", "noise_scale", "method"])
    )
    signals = list(grouped["signal"].drop_duplicates())
    fig, axes = make_axes(len(signals))
    for ax, signal in zip(axes, signals):
        sub = grouped[grouped["signal"] == signal]
        for (noise_kind, method), curve in sub.groupby(["noise_kind", "method"], dropna=False):
            curve = curve.sort_values("noise_scale")
            style = "-" if method == "reject" else "--"
            label = f"{noise_kind}, {'raw' if method == 'reject' else 'robust'}"
            ax.plot(
                curve["noise_scale"],
                curve["detection_rate"],
                linestyle=style,
                marker="o",
                label=label,
                color=RATE_COLORS.get(str(noise_kind), None),
            )
        ax.set_title(str(signal))
        ax.set_xlabel("noise scale")
        ax.set_ylabel("detection rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes[len(signals) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "raw_vs_robust_detection.png", dpi=220)
    plt.close(fig)


def save_method_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    value_vars = [
        col
        for col in ["reject", "robust_reject", "gls_reject", "fisher_reject", "perm_reject"]
        if col in df.columns
    ]
    if len(value_vars) < 2 or not {"signal", "truth_periodic", "noise_kind", "noise_scale"}.issubset(df.columns):
        return
    data = df.dropna(subset=["noise_scale"])
    if data.empty:
        return

    method_labels = {
        "reject": "TDA raw",
        "robust_reject": "TDA robust",
        "gls_reject": "GLS",
        "fisher_reject": "Fisher g",
        "perm_reject": "Permutation",
    }
    long = data.melt(
        id_vars=["signal", "truth_periodic", "noise_kind", "noise_scale"],
        value_vars=value_vars,
        var_name="method",
        value_name="reject_value",
    ).dropna(subset=["reject_value"])
    if long.empty:
        return
    long["method"] = long["method"].map(method_labels).fillna(long["method"])
    long["error_value"] = np.where(
        long["truth_periodic"].astype(bool),
        1.0 - long["reject_value"],
        long["reject_value"],
    )

    for value_col, ylabel, filename in [
        ("reject_value", "detection rate", "method_comparison_detection.png"),
        ("error_value", "error rate", "method_comparison_error.png"),
    ]:
        grouped = (
            long.groupby(["signal", "noise_kind", "noise_scale", "method"], dropna=False)[value_col]
            .mean()
            .reset_index()
            .sort_values(["signal", "noise_kind", "noise_scale", "method"])
        )
        signals = list(grouped["signal"].drop_duplicates())
        fig, axes = make_axes(len(signals))
        for ax, signal in zip(axes, signals):
            sub = grouped[grouped["signal"] == signal]
            for (noise_kind, method), curve in sub.groupby(["noise_kind", "method"], dropna=False):
                curve = curve.sort_values("noise_scale")
                linestyle = {
                    "TDA raw": "-",
                    "TDA robust": "--",
                    "GLS": ":",
                    "Fisher g": "-.",
                    "Permutation": (0, (3, 1, 1, 1)),
                }.get(str(method), "-")
                marker = {
                    "TDA raw": "o",
                    "TDA robust": "s",
                    "GLS": "^",
                    "Fisher g": "D",
                    "Permutation": "v",
                }.get(str(method), "o")
                label = f"{noise_kind}, {method}"
                ax.plot(
                    curve["noise_scale"],
                    curve[value_col],
                    linestyle=linestyle,
                    marker=marker,
                    label=label,
                    color=RATE_COLORS.get(str(noise_kind), None),
                )
            ax.set_title(str(signal))
            ax.set_xlabel("noise scale")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
        for ax in axes[len(signals) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def save_radius_vs_grid(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [col for col in ["c_alpha", "zeta_mN", "c_alpha_err"] if col in df.columns]
    if "n_observations" not in df.columns or len(metrics) == 0:
        return
    data = df.dropna(subset=["n_observations"])
    if data.empty or data["n_observations"].nunique() < 2:
        return

    grouped = (
        data.groupby(["signal", "n_observations"], dropna=False)[metrics]
        .mean()
        .reset_index()
        .sort_values(["signal", "n_observations"])
    )
    signals = list(grouped["signal"].drop_duplicates())
    fig, axes = make_axes(len(signals))
    for ax, signal in zip(axes, signals):
        sub = grouped[grouped["signal"] == signal]
        for metric in metrics:
            if sub[metric].dropna().empty:
                continue
            ax.plot(sub["n_observations"], sub[metric], marker="o", label=metric)
        ax.set_title(str(signal))
        ax.set_xlabel("number of scalar observations")
        ax.set_ylabel("radius")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes[len(signals) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "radius_vs_observation_grid.png", dpi=220)
    plt.close(fig)


def save_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    if not {"signal", "m", "tau_multiplier", "reject"}.issubset(df.columns):
        return
    data = df.dropna(subset=["m", "tau_multiplier", "reject"])
    if data.empty:
        return
    if data["m"].nunique() < 2 and data["tau_multiplier"].nunique() < 2:
        return

    data = data.copy()
    data["error"] = np.where(data["truth_periodic"].astype(bool), 1.0 - data["reject"], data["reject"])
    for metric, title, filename_suffix in [
        ("reject", "Detection rate", "detection"),
        ("error", "Error rate", "error"),
    ]:
        for signal, sub in data.groupby("signal", dropna=False):
            pivot = sub.pivot_table(index="m", columns="tau_multiplier", values=metric, aggfunc="mean")
            if pivot.empty:
                continue
            fig, ax = plt.subplots(figsize=(1.1 * len(pivot.columns) + 3, 0.7 * len(pivot.index) + 2.5))
            image = ax.imshow(pivot.values, origin="lower", vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
            ax.set_title(f"{title}: {signal}")
            ax.set_xlabel("tau multiplier")
            ax.set_ylabel("m")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([str(int(x)) if float(x).is_integer() else f"{x:g}" for x in pivot.index])
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    value = pivot.values[i, j]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(image, ax=ax, label=metric)
            fig.tight_layout()
            fig.savefig(output_dir / f"heatmap_{filename_suffix}_{safe_name(signal)}.png", dpi=220)
            plt.close(fig)


def pca_two_dimensional(x: np.ndarray) -> np.ndarray:
    centered = np.asarray(x, dtype=float) - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    if vt.shape[0] == 1:
        return np.column_stack([centered @ vt[0], np.zeros(centered.shape[0])])
    return centered @ vt[:2].T


def reconstruct_cloud(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    signal = get_signal(str(row["signal"]))
    max_anchors_value = finite_or_none(row.get("max_anchors"))
    max_anchors = int(max_anchors_value) if max_anchors_value is not None else None
    m = int(row["m"])
    tau = float(row["tau"])
    seed = int(row["seed"])
    noise = NoiseConfig(kind=str(row["noise_kind"]), scale=float(row["noise_scale"]))
    n_observations = finite_or_none(row.get("n_observations"))
    anchors, x, _, _ = build_interpolated_embedding(
        signal,
        n_observations=int(n_observations) if n_observations is not None else max(4 * (m + 1), 1),
        m=m,
        tau=tau,
        seed=seed,
        noise=noise,
        max_anchors=max_anchors,
    )
    return anchors, x


def save_representative_plot(row: pd.Series, output_dir: Path, index: int) -> None:
    anchors, x = reconstruct_cloud(row)
    max_edge = finite_or_none(row.get("config_max_edge_length"))
    diagram = compute_rips_persistence(x, max_dimension=2, max_edge_length=max_edge)
    h0 = intervals_by_dim(diagram, dim=0, finite_only=True)
    h1 = intervals_by_dim(diagram, dim=1, finite_only=True)
    coords = pca_two_dimensional(x)

    c_alpha = finite_or_none(row.get("c_alpha"))
    c_alpha_err = finite_or_none(row.get("c_alpha_err"))
    signal = str(row["signal"])
    noise_label = f"{row['noise_kind']}_{float(row['noise_scale']):g}"
    title_bits = [
        signal,
        f"m={int(row['m'])}",
        f"tau={float(row['tau']):.4g}",
        f"seed={int(row['seed'])}",
        f"noise={noise_label}",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))

    scatter = axes[0].scatter(coords[:, 0], coords[:, 1], c=anchors, s=18, cmap="viridis", alpha=0.85)
    axes[0].set_title("PCA projection of delay cloud")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=axes[0], label="anchor time")

    for dim, intervals, color in [(0, h0, "tab:gray"), (1, h1, "tab:blue")]:
        if not intervals:
            continue
        births = np.array([item.birth for item in intervals], dtype=float)
        deaths = np.array([item.death for item in intervals], dtype=float)
        axes[1].scatter(births, deaths, s=22 if dim == 0 else 36, alpha=0.75, label=f"H{dim}", color=color)

    finite_values = [
        value
        for interval in h0 + h1
        for value in (interval.birth, interval.death)
        if np.isfinite(value)
    ]
    limit = max(finite_values) if finite_values else 1.0
    limit = max(limit * 1.08, 1e-8)
    axes[1].plot([0, limit], [0, limit], color="black", linewidth=1.0, alpha=0.55)
    if c_alpha is not None:
        xs = np.array([0.0, limit])
        axes[1].axvline(c_alpha, color="tab:red", linestyle="-", linewidth=1.2, label="c_alpha")
        axes[1].plot(xs, xs + 2.0 * c_alpha, color="tab:red", linestyle="-", linewidth=1.2)
    if c_alpha_err is not None:
        xs = np.array([0.0, limit])
        axes[1].axvline(c_alpha_err, color="tab:orange", linestyle="--", linewidth=1.2, label="c_alpha_err")
        axes[1].plot(xs, xs + 2.0 * c_alpha_err, color="tab:orange", linestyle="--", linewidth=1.2)
    axes[1].set_xlim(0.0, limit)
    axes[1].set_ylim(0.0, limit)
    axes[1].set_title("Persistence diagram")
    axes[1].set_xlabel("birth")
    axes[1].set_ylabel("death")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=8)

    fig.suptitle(" | ".join(title_bits), fontsize=10)
    fig.tight_layout()
    filename = f"representative_{index:02d}_{safe_name(signal)}_{safe_name(noise_label)}.png"
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def select_representatives(df: pd.DataFrame, max_representatives: int) -> pd.DataFrame:
    if df.empty or max_representatives <= 0:
        return df.iloc[0:0]
    sort_cols = [col for col in ["truth_periodic", "signal", "noise_kind", "noise_scale", "m", "seed"] if col in df.columns]
    data = df.sort_values(sort_cols)
    if "truth_periodic" in data.columns and max_representatives >= 2:
        periodic = data[data["truth_periodic"].astype(bool)].groupby("signal", dropna=False).head(1)
        nonperiodic = data[~data["truth_periodic"].astype(bool)].groupby("signal", dropna=False).head(1)
        n_periodic = min(len(periodic), max(1, max_representatives // 2))
        n_nonperiodic = min(len(nonperiodic), max_representatives - n_periodic)
        reps = pd.concat([periodic.head(n_periodic), nonperiodic.head(n_nonperiodic)], axis=0)
    else:
        reps = data.groupby("signal", dropna=False).head(1)
    if len(reps) < max_representatives:
        extra = data.drop(index=reps.index).head(max_representatives - len(reps))
        reps = pd.concat([reps, extra], axis=0)
    return reps.head(max_representatives)


def make_all_plots(input_dir: Path, output_dir: Path, max_representatives: int) -> None:
    raw, _ = read_results(input_dir)
    raw = prepare_raw(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_path in output_dir.glob("representative_*.png"):
        old_path.unlink()

    save_rate_vs_noise(raw, output_dir, value_col="reject", ylabel="detection rate", filename="detection_vs_noise.png")
    save_rate_vs_noise(raw, output_dir, value_col="error", ylabel="error rate", filename="error_vs_noise.png")
    save_raw_vs_robust(raw, output_dir)
    save_method_comparison(raw, output_dir)
    save_radius_vs_grid(raw, output_dir)
    save_heatmaps(raw, output_dir)

    reps = select_representatives(raw, max_representatives=max_representatives)
    for idx, (_, row) in enumerate(reps.iterrows(), start=1):
        save_representative_plot(row, output_dir, index=idx)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot synthetic simulation results.")
    parser.add_argument("--input-dir", default="simulation/results/synthetic", help="Folder containing raw_results.csv.")
    parser.add_argument("--output-dir", default=None, help="Figure output folder. Defaults to <input-dir>/figures.")
    parser.add_argument("--max-representatives", type=int, default=4, help="Number of reconstructed example figures.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir / "figures"
    make_all_plots(input_dir=input_dir, output_dir=output_dir, max_representatives=args.max_representatives)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
