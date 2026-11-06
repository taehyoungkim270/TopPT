"""Run the paced-breathing experiment on the Canterbury CPAP dataset.

Usage: PYTHONPATH=src python -m new.run_cpap configs/real_cpap.toml

Subjects breathed at prescribed rates cued by a pacer, so each recording has a known
reference period. The delay is set from that period, with no period search, and each
recording gets one test on its cued segment: the first ``n_breaths`` periods.
"""

import argparse
from functools import partial
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import download_archive, load_cpap_flow, preprocess
from .method import delay_cloud, fisher_g_test, gls_test, permutation_test, toppt_test
from .runs import ROOT, load_config, make_rng, run_batches, start_run


def run_recording(config: dict, recording: tuple[Path, int, int]) -> list[dict]:
    """Test the cued segment of one recording and return one results row."""
    path, peep, rate = recording
    m, alpha, bounds = config["m"], config["alpha"], config["period_bounds"]
    period = 60.0 / rate
    times, flow = load_cpap_flow(path)
    cued = times <= times[0] + config["n_breaths"] * period
    fs = 1.0 / np.median(np.diff(times))
    times, values = preprocess(times[cued] - times[0], flow[cued], fs, config)
    cloud = delay_cloud(times, values, m, period / (m + 1), config["max_anchors"])
    b = int(np.floor(config["b_fraction"] * len(cloud)))
    return [
        {
            "recording": path.stem,
            "subject": int(path.stem.split("_")[1]),
            "peep_cmh2o": peep,
            "rate_per_min": rate,
            "period": period,
            "seconds": times[-1],
            "n_points": len(cloud),
            "b": b,
            **toppt_test(cloud, alpha, b, config["n_subsamples"], make_rng(config["seed"], path.stem, "subsampling")),
            **gls_test(times, values, alpha, min_period=bounds[0], max_period=bounds[1]),
            **fisher_g_test(times, values, alpha),
            **permutation_test(
                times, values, alpha, config["n_permutations"], make_rng(config["seed"], path.stem, "permutation")
            ),
        }
    ]


def run_experiment(config: dict, run_dir: Path, data_dir: Path, workers: int) -> None:
    """Download the dataset, then test every paced recording that has not finished yet."""
    archive = download_archive(config["url"], config["sha256"], data_dir)
    folder = data_dir / config["folder"] / "Processed_CSV_Data_Files"
    # Files are named "..._4cmH20_..." (zero, not O). The 4 cmH2O MediumShort folder
    # also holds copies of three 7 cmH2O files; matching the name skips them.
    recordings = {
        path.stem: (path, peep, rate)
        for peep in config["peep_cmh2o"]
        for name, rate in config["rates_per_min"].items()
        for path in sorted((folder / f"{peep}cmH2O_{name}_Breaths").glob(f"*_{peep}cmH20_*.csv"))
    }
    start_run(config, run_dir, ["data.py", "method/*.py", "run_cpap.py"], [archive])
    run_batches(partial(run_recording, config), recordings, run_dir, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    run_dir = args.output_dir or ROOT / "results" / args.config.stem
    data_dir = args.data_dir or ROOT / "data" / "cpap"
    started = perf_counter()
    run_experiment(load_config(args.config), run_dir, data_dir, args.workers)
    print(f"Wrote {run_dir / 'raw_results.csv'} in {(perf_counter() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
