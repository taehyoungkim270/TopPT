"""Make the paper's Tables 1-3 and Figure 4 from experiment results.

Usage: PYTHONPATH=src python -m new.report

Reads ``results/<experiment>/raw_results.csv`` and writes CSV and LaTeX tables, the
figure, and a comparison with the NeurIPS submission to ``results/paper/``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .runs import ROOT, read_csv  # noqa: E402

SYNTHETIC_COLUMNS = {
    "reject": "raw TopPT",
    "robust_reject": "robust TopPT",
    "gls_reject": "Lomb-Scargle",
    "fisher_reject": "Fisher g",
    "perm_reject": "permutation",
}
DATASETS = {"Fantasia": "real_fantasia", "BIDMC": "real_bidmc"}

# Names used by the submission's results, for the comparison only.
SUBMISSION = ROOT / "reference/neurips_submission"
SUBMISSION_RUNS = {
    "synthetic_main": "synthetic_full_main",
    "synthetic_stress": "synthetic_full_appendix_stress",
    "real_fantasia": "fantasia_all40_window_grid",
    "real_bidmc": "bidmc_all53_window_grid",
}
SUBMISSION_SIGNALS = {
    "f1_periodic": "f1_rational",
    "f2_periodic": "f2_log_exp",
    "h0_linear_trend": "h1_linear",
    "h0_quadratic_trend": "h2_quadratic",
    "h0_logistic_trend": "h3_logistic",
    "h0_aperiodic_trend": "h4_aperiodic",
    "f7_phase_drift": "g1_phase_drift",
    "f8_amplitude_modulated": "g2_amplitude_modulated",
}


def _load(experiment: str, submission: bool) -> pd.DataFrame:
    if not submission:
        return read_csv(ROOT / "results" / experiment / "raw_results.csv")
    filename = "real_results.csv" if experiment.startswith("real") else "raw_results.csv"
    raw = read_csv(SUBMISSION / SUBMISSION_RUNS[experiment] / filename)
    raw = raw.rename(columns={"window_start_seconds": "window_start"})
    if "signal" in raw:
        raw["signal"] = raw["signal"].map(SUBMISSION_SIGNALS)
    return raw


def synthetic_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Rejection rate of each method for each signal, averaged over noise settings."""
    table = raw.groupby("signal", sort=False)[list(SYNTHETIC_COLUMNS)].mean()
    table.index = [name.split("_")[0] for name in table.index]
    return table.rename(columns=SYNTHETIC_COLUMNS)


def window_results(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per window: TopPT rejects if any candidate period rejects."""
    margin = raw["best_persistence"] - 2.0 * raw["c_alpha"]
    return (
        raw.assign(margin=margin)
        .groupby(["record", "window_start"], sort=False)
        .agg(
            toppt=("reject", "any"),
            margin=("margin", "max"),
            gls=("gls_reject", "first"),
            fisher=("fisher_reject", "first"),
            perm=("perm_reject", "first"),
        )
    )


def real_table(windows_by_dataset: dict[str, pd.DataFrame], tests: dict[str, int]) -> pd.DataFrame:
    """Rejected records and windows per dataset, as in Table 2."""
    rows = {}
    for dataset, windows in windows_by_dataset.items():
        records = windows.groupby(level="record").any()
        n_records, n_windows = len(records), len(windows)
        row = {"records": n_records, "windows": n_windows, "TopPT tests": tests[dataset]}
        for method, label in [("toppt", "TopPT"), ("gls", "Lomb-Scargle"), ("fisher", "Fisher g"), ("perm", "permutation")]:
            row[f"{label} records"] = f"{int(records[method].sum())}/{n_records}"
            row[f"{label} windows"] = f"{int(windows[method].sum())}/{n_windows}"
        rows[dataset] = row
    return pd.DataFrame(rows).T


def figure_4(windows_by_dataset: dict[str, pd.DataFrame], path: Path) -> None:
    """Best TopPT margin per window, and rejecting windows per record."""
    labels = list(windows_by_dataset)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    margins = [windows["margin"].dropna().to_numpy() for windows in windows_by_dataset.values()]
    axes[0].boxplot(margins, tick_labels=labels, showmeans=True)
    axes[0].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[0].set_title("Best TDA margin by window")
    axes[0].set_ylabel("best persistence - 2 c_alpha")
    axes[0].grid(axis="y", alpha=0.25)
    for label, windows in windows_by_dataset.items():
        per_record = windows["toppt"].groupby(level="record").sum()
        n_windows = windows.groupby(level="record").size().max()
        axes[1].hist(per_record, bins=np.arange(-0.5, n_windows + 1.5, 1.0), alpha=0.55, label=label)
    axes[1].set_title("TDA rejecting windows per record")
    axes[1].set_xlabel("number of rejecting windows")
    axes[1].set_ylabel("records")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def make_report(submission: bool = False) -> dict[str, pd.DataFrame]:
    """Return Tables 1-3 and the per-window results, from new or submission results."""
    windows = {name: window_results(_load(run, submission)) for name, run in DATASETS.items()}
    tests = {name: len(_load(run, submission)) for name, run in DATASETS.items()}
    return {
        "table1": synthetic_table(_load("synthetic_main", submission)),
        "table2": real_table(windows, tests),
        "table3": synthetic_table(_load("synthetic_stress", submission)),
        "windows": windows,
    }


def _latex_rows(table: pd.DataFrame) -> str:
    def cell(value) -> str:
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    return "\n".join(
        " & ".join([str(index), *map(cell, row)]) + r" \\" for index, row in zip(table.index, table.values)
    )


def main() -> None:
    out = ROOT / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    new, old = make_report(), make_report(submission=True)
    lines = []
    for name in ["table1", "table2", "table3"]:
        table = new[name]
        table.to_csv(out / f"{name}.csv")
        (out / f"{name}.tex").write_text(_latex_rows(table) + "\n")
        comparison = pd.concat({"submission": old[name], "new": table}, axis=1)
        lines += [f"{name}", comparison.round(3).to_string(), ""]
    (out / "comparison.txt").write_text("\n".join(lines))
    figure_4(new["windows"], out / "figure4")
    print("\n".join(lines))
    print(f"Wrote tables, comparison, and Figure 4 to {out}")


if __name__ == "__main__":
    main()
