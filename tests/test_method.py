"""Tests for TopPT and the baselines."""

import numpy as np
import pandas as pd
import pytest

from toppt.dgp import (
    SIGNALS,
    NoiseModel,
    generate_observations,
    interpolation_error_bound,
    known_period_tau,
)
from toppt.method import (
    delay_cloud,
    fisher_g_test,
    gls_test,
    h1_intervals,
    permutation_test,
    strongest_period,
    toppt_test,
)
from test_dgp import OLD_NAMES


def test_saved_submission_results_are_reproduced():
    """Use the original seeds, so numbers match up to rounding across platforms."""
    saved = pd.read_csv(
        "reference/neurips_submission/synthetic_full_main/raw_results.csv",
        float_precision="round_trip",
    )
    m, alpha = 20, 0.05
    tau = known_period_tau(m)
    for name, kind, scale in [("f1_rational", "BU", 0.05), ("h4_aperiodic", "TG", 0.1)]:
        seed = 3
        noise = NoiseModel(kind, scale)
        times, values = generate_observations(
            SIGNALS[name], noise, 400, m * tau, np.random.default_rng(seed + 20_000)
        )
        zeta = interpolation_error_bound(SIGNALS[name], noise, times, m)
        result = {
            **toppt_test(delay_cloud(times, values, m, tau), alpha, 120, 500, np.random.default_rng(seed), zeta),
            **gls_test(times, values, alpha),
            **fisher_g_test(times, values, alpha),
            **permutation_test(times, values, alpha, 500, np.random.default_rng(seed + 30_000)),
        }
        row = saved[
            (saved.signal == OLD_NAMES[name])
            & (saved.noise_kind == kind)
            & (saved.noise_scale == scale)
            & (saved.seed == seed)
        ].iloc[0]
        for column, value in result.items():
            assert value == pytest.approx(row[column], rel=1e-8), column


def test_circle_has_one_loop_that_dies_at_the_known_scale():
    # The Rips H1 of a unit circle dies at sqrt(3), the side of an inscribed
    # equilateral triangle, which is present when the point count is divisible by 3.
    angles = 2.0 * np.pi * np.arange(120) / 120
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    intervals = h1_intervals(circle)
    assert intervals.shape == (1, 2)
    assert intervals[0] == pytest.approx([2.0 * np.sin(np.pi / 120), np.sqrt(3.0)])
    assert toppt_test(circle, 0.05, 60, 200, np.random.default_rng(0))["reject"]


def test_period_search_finds_the_period_of_a_sine():
    times = np.arange(0.0, 30.0, 1.0 / 25.0)
    values = np.sin(2.0 * np.pi * times / 3.0)
    # The Welch peak wins here, and its frequency grid has the nearby periods 3.2 and 2.67.
    assert strongest_period(values, 25.0, (2.0, 10.0)) == pytest.approx(3.2)
