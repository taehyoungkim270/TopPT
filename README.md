# Topological Periodicity Test (TopPT)

Code for the NeurIPS 2026 paper *Topological Periodicity Test (TopPT) via
Confidence Bound of Time-Delay Embeddings* by Donghyun Park, Junhyun An,
Taehyoung Kim, and Jisu Kim.

TopPT tests whether a time series is periodic by checking for a persistent loop
in its time-delay embedding, using a subsampling confidence bound on the
persistence diagram.

The code is being reorganized so that the paper's results can be regenerated
with one command. That command is not available yet.

## Setup

Requires Python 3.14. Tested with Python 3.14.7 on macOS (Apple silicon).

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Data

The real-data experiments use the respiration channels of two PhysioNet
databases, [Fantasia](https://physionet.org/content/fantasia/) and
[BIDMC PPG and Respiration](https://physionet.org/content/bidmc/). The data are
not stored in this repository. The runners download the records they need.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/old/` | Original code used for the NeurIPS submission (see its [README](src/old/README.md)) |
| `src/new/` | Reorganized code, in progress |
| `configs/` | Settings for each experiment |
| `tests/` | Test suite |
| `reference/neurips_submission/` | Saved results behind the submission's tables (see its [README](reference/neurips_submission/README.md)) |
| `data/` | Downloaded PhysioNet records (not tracked) |
| `results/` | Generated results (not tracked) |
| `requirements.txt` | Exact package versions |

## Tests

```bash
python -m pytest
```

## Running the experiments

```bash
PYTHONPATH=src python -m new.run_synthetic configs/synthetic_main.toml
PYTHONPATH=src python -m new.run_synthetic configs/synthetic_stress.toml
PYTHONPATH=src python -m new.run_real configs/real_fantasia.toml
PYTHONPATH=src python -m new.run_real configs/real_bidmc.toml
```

Results go to `results/<config name>/`, with `metadata.json` recording the
settings, code version, package versions, and data file hashes. If a run stops, the
same command resumes it. If the settings, code, or data change, the command stops
rather than mixing results, and you need to delete the folder or choose another
with `--output-dir`.

On an Apple silicon Mac with 8 worker processes, the synthetic experiments took
about 30 minutes and the real-data experiments about 9 minutes. The real-data
commands first download about 330 MB from PhysioNet. If PhysioNet returns an error,
run the command again.

## Running the original code

Run the original code from `src/`, where it imports as the package `old`. Its
own README shows commands using an earlier package name, `simulation`.

This command checks one Fantasia record and takes under a minute, plus the
download:

```bash
cd src
python -m old.run_fantasia_all40 --records f1y01 \
  --data-dir ../data/fantasia --output-dir ../results/fantasia_check --no-progress
```

The commit tagged `original` preserves the code as it was at submission.
