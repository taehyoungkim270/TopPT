"""Run a synthetic experiment from a configuration file.

Usage: PYTHONPATH=src python -m new.run_synthetic configs/synthetic_main.toml

Results go to ``results/<config name>/`` unless ``--output-dir`` is given. Each
scenario (one signal and noise setting) is a batch, so rerunning the same command
after an interruption skips finished scenarios.
"""

import argparse
from functools import partial
import os
from pathlib import Path
from time import perf_counter

from .dgp import (
    PERIODIC,
    SIGNALS,
    NoiseModel,
    generate_observations,
    interpolation_error_bound,
    known_period_tau,
)
from .method import delay_cloud, fisher_g_test, gls_test, permutation_test, toppt_test
from .runs import ROOT, load_config, make_rng, run_batches, start_run


def run_replicate(config: dict, signal_name: str, noise: NoiseModel, replicate: int) -> dict:
    """Generate one dataset, apply every method to it, and return one results row.

    Every signal, noise setting, replicate, and random task has its own stream.
    """
    m, alpha = config["m"], config["alpha"]
    tau = known_period_tau(m)
    signal = SIGNALS[signal_name]

    def rng(task: str):
        return make_rng(config["seed"], signal_name, noise.kind, noise.scale, replicate, task)

    times, values = generate_observations(signal, noise, config["n_observations"], m * tau, rng("data"))
    cloud = delay_cloud(times, values, m, tau)
    zeta = interpolation_error_bound(signal, noise, times, m)
    return {
        "signal": signal_name,
        "is_periodic": signal_name in PERIODIC,
        "noise_kind": noise.kind,
        "noise_scale": noise.scale,
        "replicate": replicate,
        "n_points": len(cloud),
        "zeta": zeta,
        **toppt_test(cloud, alpha, config["b"], config["n_subsamples"], rng("subsampling"), zeta),
        **gls_test(times, values, alpha),
        **fisher_g_test(times, values, alpha),
        **permutation_test(times, values, alpha, config["n_permutations"], rng("permutation")),
    }


def _run_scenario(config: dict, scenario: tuple[str, NoiseModel]) -> list[dict]:
    signal_name, noise = scenario
    return [run_replicate(config, signal_name, noise, r) for r in range(config["n_replicates"])]


def run_experiment(config: dict, run_dir: Path, workers: int) -> None:
    """Run every scenario in the configuration that has not finished yet."""
    start_run(config, run_dir, ["dgp/*.py", "method/*.py", "run_synthetic.py"])
    scenarios = {
        f"{name}__{noise['kind']}_{noise['scale']}": (name, NoiseModel(noise["kind"], noise["scale"]))
        for name in config["signals"]
        for noise in config["noise"]
    }
    run_batches(partial(_run_scenario, config), scenarios, run_dir, workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    run_dir = args.output_dir or ROOT / "results" / args.config.stem
    started = perf_counter()
    run_experiment(load_config(args.config), run_dir, args.workers)
    print(f"Wrote {run_dir / 'raw_results.csv'} in {(perf_counter() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
