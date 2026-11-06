"""Test the experiment runner end to end on a tiny design."""

from new.run_synthetic import run_experiment


def test_results_do_not_depend_on_workers_or_on_resuming(tmp_path):
    config = {
        "seed": 0,
        "signals": ["f1_rational", "h1_linear"],
        "n_replicates": 2,
        "n_observations": 80,
        "m": 4,
        "alpha": 0.05,
        "b": 20,
        "n_subsamples": 20,
        "n_permutations": 20,
        "noise": [{"kind": "none", "scale": 0.0}, {"kind": "BU", "scale": 0.05}],
    }
    run_experiment(config, tmp_path / "one", workers=1)
    run_experiment(config, tmp_path / "two", workers=2)
    expected = (tmp_path / "one/raw_results.csv").read_bytes()
    assert len(expected.splitlines()) == 1 + 2 * 2 * 2
    assert (tmp_path / "two/raw_results.csv").read_bytes() == expected

    (tmp_path / "two/batches/h1_linear__BU_0.05.csv").unlink()
    run_experiment(config, tmp_path / "two", workers=2)
    assert (tmp_path / "two/raw_results.csv").read_bytes() == expected
