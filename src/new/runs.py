"""Shared machinery for experiment runs: configuration, provenance, and batches."""

from collections.abc import Callable, Iterable
from functools import partial
import hashlib
import json
from importlib import metadata
from multiprocessing import Pool
from pathlib import Path
import platform
import shutil
import subprocess
import tomllib

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("gudhi", "numpy", "pandas", "scipy", "wfdb")


def load_config(path: Path) -> dict:
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV through a temporary file, so a finished name means a finished file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV written by ``save_csv``, keeping every float exactly."""
    return pd.read_csv(path, float_precision="round_trip")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state() -> dict:
    if not shutil.which("git"):
        return {"commit": None, "uncommitted_changes": None}

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD") or None,
        "uncommitted_changes": bool(git("status", "--porcelain")),
    }


def start_run(
    config: dict,
    run_dir: Path,
    source_patterns: Iterable[str],
    rng_plan: str,
    data_files: Iterable[Path] = (),
) -> None:
    """Record what produces a run, and refuse to mix it with a different earlier run.

    The fingerprint covers the configuration, the source files matching
    ``source_patterns`` under ``src/new``, the data files, and package versions.
    Results already in ``run_dir`` can be reused only if it has not changed.
    """
    sources = sorted(p for pattern in source_patterns for p in (ROOT / "src/new").glob(pattern))
    identity = {
        "config": config,
        "rng_plan": rng_plan,
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        "data_sha256": {path.name: _sha256(path) for path in data_files},
        "python": platform.python_version(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = run_dir / "metadata.json"
    if path.exists():
        if json.loads(path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError(
                f"{run_dir} holds results from different settings, code, data, or package "
                "versions. Delete it or choose another output directory."
            )
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {"fingerprint": fingerprint, **identity, "git": _git_state(), "platform": platform.platform()}
    path.write_text(json.dumps(record, indent=2) + "\n")


def _save_batch(run_batch: Callable, folder: Path, item: tuple) -> str:
    name, batch = item
    save_csv(pd.DataFrame(run_batch(batch)), folder / f"{name}.csv")
    return name


def run_batches(run_batch: Callable, batches: dict, run_dir: Path, workers: int) -> None:
    """Run unfinished batches in parallel, then merge them into ``raw_results.csv``.

    ``batches`` maps a name to the argument of ``run_batch``, which returns a list of
    result rows. Each finished batch is saved, so a rerun after an interruption skips it.
    """
    folder = run_dir / "batches"
    folder.mkdir(exist_ok=True)
    todo = [item for item in batches.items() if not (folder / f"{item[0]}.csv").exists()]
    print(f"{len(batches) - len(todo)} of {len(batches)} batches already finished.", flush=True)
    with Pool(workers) as pool:
        job = partial(_save_batch, run_batch, folder)
        for count, name in enumerate(pool.imap_unordered(job, todo), start=1):
            print(f"[{count}/{len(todo)}] {name}", flush=True)
    merged = pd.concat(read_csv(folder / f"{name}.csv") for name in batches)
    save_csv(merged, run_dir / "raw_results.csv")
