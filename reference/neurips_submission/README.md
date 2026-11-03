# NeurIPS submission results

Results from the runs behind the numbers in the NeurIPS 2026 submission, kept as
a fixed baseline for comparing later code. Do not edit these files. The figures
were not copied because the code can regenerate them.

| Folder | Paper results | Files |
| --- | --- | --- |
| `synthetic_full_main/` | Table 1 | `raw_results.csv`, `summary.csv`, `config.json` |
| `synthetic_full_appendix_stress/` | Table 3 | `raw_results.csv`, `summary.csv`, `config.json` |
| `fantasia_all40_window_grid/` | Table 2, Figure 4 | `real_results.csv`, `record_summary.csv`, `window_summary.csv`, `config.json`, `metadata.json` |
| `bidmc_all53_window_grid/` | Table 2, Figure 4 | `real_results.csv`, `record_summary.csv`, `window_summary.csv`, `config.json`, `metadata.json` |

The runs were made in May 2026 on Windows with Python 3.14.3, gudhi 3.12.0,
matplotlib 3.10.8, numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, tqdm 4.67.3, and
wfdb 4.3.1. Paths inside the JSON files refer to that machine.

The runs did not record which code version produced them. The evidence that they
came from the code at the Git tag `original` is a partial re-run on macOS with
the same package versions: 54 synthetic rows and two real records (39 rows) gave
identical reject decisions, with relative differences in the other numbers of
about 1e-10 or smaller.

These files reproduce every value in Tables 1, 2, and 3. Figure 4 has not been
compared against them.
