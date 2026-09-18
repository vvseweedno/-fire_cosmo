from dataclasses import replace

from scripts.run_metric_candidates import candidate_configs, choose_winner
from wildfire.model_config import ModelConfig


def test_candidate_family_contains_baseline_sar_spectral_and_blends():
    configs = candidate_configs(ModelConfig())
    assert "baseline" in configs
    assert "sar_clear_pos_005" in configs
    assert "sar_cloud_pos_010" in configs
    assert "spectral_005" in configs
    assert "blend_sar_spectral_025_005" in configs
    assert configs["baseline"] == ModelConfig()
    assert configs["spectral_005"].bs.score_recipe == "spectral_consensus"
    assert configs["spectral_005"].bs.index_consensus_weight == 0.05


def test_choose_winner_never_promotes_non_improvement():
    results = {
        "baseline": {"score": 0.7000},
        "equal": {"score": 0.7000},
        "tiny": {"score": 0.7000001},
        "better": {"score": 0.7010},
    }
    winner, score = choose_winner(results, min_gain=0.0005)
    assert winner == "better"
    assert score == 0.7010


def test_choose_winner_keeps_baseline_when_gain_below_gate():
    results = {
        "baseline": {"score": 0.7000},
        "candidate": {"score": 0.7001},
    }
    winner, score = choose_winner(results, min_gain=0.0005)
    assert winner == "baseline"
    assert score == 0.7000
