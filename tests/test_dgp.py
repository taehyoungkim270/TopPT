"""Tests for the synthetic data-generating processes."""

import numpy as np

from toppt.dgp import (
    DOMAIN,
    PERIOD,
    PERIODIC,
    SIGNALS,
    NoiseModel,
    generate_observations,
    known_period_tau,
)
from old.config import NoiseConfig
from old.signals import get_signal as get_old_signal
from old.synthetic_experiments import build_interpolated_embedding
from old.synthetic_experiments import known_period_tau as old_known_period_tau

OLD_NAMES = {
    "f1_rational": "f1_periodic",
    "f2_log_exp": "f2_periodic",
    "h1_linear": "h0_linear_trend",
    "h2_quadratic": "h0_quadratic_trend",
    "h3_logistic": "h0_logistic_trend",
    "h4_aperiodic": "h0_aperiodic_trend",
    "g1_phase_drift": "f7_phase_drift",
    "g2_amplitude_modulated": "f8_amplitude_modulated",
}


def test_observations_match_the_original_code():
    m, seed = 20, 57
    for name, signal in SIGNALS.items():
        old_signal = get_old_signal(OLD_NAMES[name])
        tau = known_period_tau(m)
        assert tau == old_known_period_tau(old_signal, m)
        for kind, scale in [("none", 0.0), ("BU", 0.05), ("TG", 0.1)]:
            _, _, old_times, old_values = build_interpolated_embedding(
                old_signal,
                n_observations=400,
                m=m,
                tau=tau,
                seed=seed,
                noise=NoiseConfig(kind=kind, scale=scale),
            )
            times, values = generate_observations(
                signal,
                NoiseModel(kind, scale),
                400,
                m * tau,
                np.random.default_rng(seed + 20_000),
            )
            assert np.array_equal(times, old_times)
            assert np.array_equal(values, old_values)


def test_only_periodic_signals_repeat_after_one_period():
    t = np.linspace(*DOMAIN, 500)
    for name, signal in SIGNALS.items():
        repeats = np.allclose(signal(t + PERIOD), signal(t), atol=1e-9)
        assert repeats == (name in PERIODIC)


def test_noise_stays_within_its_bound():
    # The robust rejection rule relies on this deterministic bound.
    for kind in ["BU", "TG"]:
        noise = NoiseModel(kind, 0.1).sample((100_000,), np.random.default_rng(1))
        assert np.abs(noise).max() <= 0.1
