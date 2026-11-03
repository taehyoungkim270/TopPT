# Synthetic Simulation Code

This folder implements the simulation pipeline for the current paper version.
It uses raw time-delay embeddings only:

```text
(f(t), f(t+tau), ..., f(t+m tau))
```

There is no phase folding, centering, or row normalization in this package.

## Main Components

- `embedding.py`: raw sliding-window construction.
- `subsampling.py`: Hausdorff subsampling radii and `c_alpha`.
- `persistence.py`: GUDHI Vietoris-Rips persistence.
- `periodicity_test.py`: Section 6 rejection rule.
- `gls_baseline.py`: generalized Lomb-Scargle baseline on the scalar observations.
- `spectral_baselines.py`: Fisher g-test and permutation max-periodogram test.
- `synthetic_experiments.py`: synthetic signal experiments.
- `run_synthetic.py`: command-line runner.
- `real_experiments.py`: WFDB/Fantasia real-data loader and experiment helpers.
- `run_real.py`: command-line runner for real-data experiments.
- `run_bidmc_all53.py`: convenience runner for the BIDMC RESP all-record experiment.

## Quick Run

Use the Python environment that has `gudhi` installed. On this machine that was:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_synthetic --quick
```

Results are written to:

```text
simulation/results/synthetic/raw_results.csv
simulation/results/synthetic/summary.csv
simulation/results/synthetic/config.json
```

## Bounded-Error Synthetic Runs

For the theorem-style bounded interpolation error experiment, the code observes
scalar time-series values, builds a linear interpolant, and uses observed
timestamps as sliding-window anchors.  With bounded-uniform (`BU`) or truncated
Gaussian (`TG`) noise, `noise_scale` is interpreted as the deterministic scalar
bound `sigma_N`.

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_synthetic --bounded-error
```

The bounded-error preset uses clean smooth H0 signals for the main synthetic
table.  The near-periodic nonstationary signals `f7_phase_drift` and
`f8_amplitude_modulated` are available for an appendix stress test, but are not
part of the default main H0 set because they can still generate loop-like
sliding-window geometry.  The default bounded run uses:

```text
signals: f1_periodic, f2_periodic,
         h0_linear_trend, h0_quadratic_trend,
         h0_logistic_trend, h0_aperiodic_trend
seeds: 0,...,19
m: 12, 20
scalar observation grids: 200, 400
anchors: observed timestamps on the original signal interval
noise: none, BU, TG
noise scales: 0.02, 0.05, 0.10 for BU/TG, plus the separate noiseless case
```

`n_observations` is the number of scalar observed time-series values.  The
point-cloud size is then derived from the observed timestamps that can serve as
full sliding-window anchors.  Pass `--max-anchors` only if you want to cap the
number of anchor timestamps used.

Useful smaller check:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_synthetic --bounded-error --signals f1_periodic,h0_linear_trend --seeds 0:3 --m-values 12 --n-observations-grid 200 --n-subsamples 100
```

The raw output includes the interpolation-theorem bookkeeping columns
`delta_N`, `sigma_N`, `l_int_hat`, `eta_N`, `zeta_mN`, `c_alpha_err`, and
`robust_reject`, where `c_alpha_err = c_alpha + 5 zeta_mN`.  It also includes
statistical spectral-test columns:

```text
gls_reject, gls_fap, gls_max_power, gls_best_period
fisher_reject, fisher_p_value, fisher_g, fisher_best_period
perm_reject, perm_p_value, perm_statistic, perm_best_period
```

GLS is run on the same scalar observation grid used to build the interpolant.
By default it uses an oversampling factor `--gls-ofac 10` and searches up to
the average Nyquist frequency (`--gls-hifac 1`).  To restrict the searched
period range, pass `--gls-min-period` and/or `--gls-max-period`.

Fisher's g-test uses the classical exact p-value for the largest normalized
Fourier periodogram ordinate on the regular scalar observation grid.  The
permutation max-periodogram test uses the same statistic but obtains its p-value
by permuting scalar values over observed times.  Control its Monte Carlo budget
with `--periodogram-permutations`.

Delay choices use the known-period rule `tau = Xi / (m + 1)` by default.  To
scan nearby choices, pass for example:

```powershell
--tau-multipliers 0.5,0.75,1.0,1.25,1.5
```

To use an explicit fixed delay grid instead, pass `--tau-values`.

Appendix stress-test run:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_synthetic --bounded-error --signals f7_phase_drift,f8_amplitude_modulated --seeds 0:30 --m-values 20 --n-observations-grid 400 --noise-kinds none,BU,TG --noise-scales 0.02,0.05,0.10 --b 120 --n-subsamples 500 --periodogram-permutations 500 --show-progress --output-dir simulation/results/synthetic_appendix_stress
```

## Plot Results

After a run finishes, generate diagnostic figures from its result folder:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.plot_results --input-dir simulation/results/bounded_dev
```

Figures are written to `<input-dir>/figures` by default.  The plotting script
creates aggregate detection/error curves, raw-vs-robust detection plots,
GLS/TDA method-comparison plots, confidence-radius plots, delay/m heatmaps when
the grid is available, and a few representative PCA/persistence-diagram figures
reconstructed from saved row metadata.

## Vietoris-Rips Note

The paper's formal guarantees are stated for distance or Cech persistence.
For computation, this simulation code uses GUDHI's Vietoris-Rips complex by
default. This is computationally convenient, but it should be explained in the
paper as a practical approximation with the usual scale/interleaving caveat.

## Fantasia Real Data

The real-data runner currently supports the Fantasia respiration channel from
the local WFDB record `f1y01`.  It reuses the same no-folding pipeline:

```text
WFDB scalar observations
-> optional detrending / scalar z-scoring / downsampling
-> linear interpolation
-> raw sliding-window cloud
-> TDA test + GLS + Fisher g-test + permutation max-periodogram test
```

On this machine the old local data folder is detected automatically.  A small
check run is:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_real --dataset fantasia --record f1y01 --use-seconds 30 --window-seconds 20 --downsample-to-hz 12 --m 8 --grid-points 3 --n-subsamples 5 --periodogram-permutations 10 --max-anchors 80 --b 40 --output-dir simulation/results/fantasia_smoke
```

A more useful first Fantasia run is:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_real --dataset fantasia --record f1y01 --use-seconds 90 --window-seconds 20 --downsample-to-hz 25 --m 8 --period-grid 2.6667,2.8611,3.0 --n-subsamples 300 --periodogram-permutations 500 --max-anchors 300 --b-fraction 0.99 --show-progress --output-dir simulation/results/fantasia
```

This tuned setting loads a longer prefix for whole-interval inspection, then
tests a shorter 20-second window with more sliding-window anchors and a high
subsample ratio. The first 60-second notebook run produced a confidence radius
that was too conservative for the visible loop. Candidate breathing periods can
also be estimated from ACF/Welch hints within the period bounds
`--period-bounds 2.0,8.0`, then expanded into a local grid. For a window-grid
run, the default is `--period-strategy window`, meaning each window chooses its
own candidate period grid from its local Step-3 hints. To force specific
periods instead, pass for example:

```powershell
--period-grid 2.5,3.0,3.5,4.0
```

For a formal local-window analysis, scan a fixed window grid with Bonferroni
correction over all window-period pairs:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_real --dataset fantasia --record f1y01 --use-seconds 90 --window-start-grid 0,10,20,30,40,50,60 --window-seconds 20 --downsample-to-hz 25 --m 8 --period-grid 2.6667,2.8611,3.0 --alpha 0.05 --n-subsamples 300 --periodogram-permutations 500 --max-anchors 300 --b-fraction 0.99 --show-progress --output-dir simulation/results/fantasia_window_grid_bonferroni
```

With this command the per-test level is `0.05 / 21`, because there are seven
fixed windows and three fixed periods.

The public Fantasia database has 40 records in total: 20 young-subject records
(`f1y01`-`f1y10`, `f2y01`-`f2y10`) and 20 elderly-subject records
(`f1o01`-`f1o10`, `f2o01`-`f2o10`). To run the same local-window protocol on
the first five canonical Fantasia records, use:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_real --dataset fantasia --first-n-records 5 --download-records --use-seconds 90 --window-start-grid 0,10,20,30,40,50,60 --window-seconds 20 --downsample-to-hz 25 --m 8 --period-strategy window --grid-points 3 --grid-scale 0.10 --alpha 0.05 --n-subsamples 300 --periodogram-permutations 500 --max-anchors 300 --b-fraction 0.99 --show-progress --output-dir simulation/results/fantasia_first5_window_grid
```

With `--download-records`, the selected WFDB records are downloaded if they are
not already present. Without that flag, the selected records must already exist
under the Fantasia data directory.

## BIDMC Real Data

The BIDMC workflow uses the RESP channel from the 53 BIDMC records.  It follows
the same local-window adaptive-period protocol as the Fantasia workflow, with
longer 30-second windows and period bounds of 2 to 10 seconds:

```powershell
& 'C:\Users\lion3\Desktop\PhD\.venv\Scripts\python.exe' -m simulation.run_bidmc_all53 --use-seconds 180 --window-start-grid 0,30,60,90,120,150 --window-seconds 30 --downsample-to-hz 25 --bandpass-hz 0.05,0.8 --m 8 --period-strategy window --grid-points 3 --grid-scale 0.10 --period-bounds 2.0,10.0 --alpha 0.05 --n-subsamples 300 --periodogram-permutations 500 --max-anchors 300 --b-fraction 0.99 --output-dir simulation/results/bidmc_all53_window_grid
```

If the WFDB files are already present locally, add `--no-download`.  For a
notebook version with diagnostics and visualizations, open `bidmc_workflow.ipynb`.
