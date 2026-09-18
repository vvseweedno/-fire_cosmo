import numpy as np

from wildfire.ensemble import optimize_af_ensemble, optimize_bs_ensemble


def test_af_ensemble_never_worse_than_best_single_on_search_objective():
    target = np.array([0, 0, 1, 1, 1, 0], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)
    scores = {
        "a": np.array([0.1, 0.4, 0.8, 0.7, 0.3, 0.2], dtype=np.float32),
        "b": np.array([0.3, 0.1, 0.6, 0.9, 0.8, 0.5], dtype=np.float32),
    }

    result = optimize_af_ensemble(scores, target, valid, alpha_steps=10)
    assert result["f1"] >= result["best_single_f1"]
    assert np.isclose(sum(result["weights"].values()), 1.0)


def test_bs_ensemble_never_worse_than_best_single_on_search_objective():
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)
    scores = {
        "a": np.array([0.0, 0.05, 0.18, 0.30, 0.45, 0.50, 0.72, 0.80]),
        "b": np.array([0.02, 0.10, 0.22, 0.25, 0.40, 0.58, 0.75, 0.90]),
    }

    result = optimize_bs_ensemble(
        scores,
        target,
        valid,
        initial_thresholds=(0.10, 0.27, 0.44),
        alpha_steps=6,
        threshold_candidates=24,
        threshold_passes=2,
    )
    assert result["bs_subscore"] >= result["best_single_bs_subscore"]
    assert np.isclose(sum(result["weights"].values()), 1.0)


def test_single_model_ensemble_is_valid():
    target = np.array([0, 1], dtype=np.uint8)
    valid = np.ones_like(target, dtype=bool)
    scores = {"only": np.array([0.1, 0.9], dtype=np.float32)}
    result = optimize_af_ensemble(scores, target, valid)
    assert result["weights"] == {"only": 1.0}
    assert result["f1"] == 1.0
