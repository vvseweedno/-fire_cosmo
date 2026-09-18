from pathlib import Path

import pytest

from wildfire.model_config import (
    ModelConfig,
    load_model_config,
    model_config_from_dict,
    save_model_config,
)


def test_model_config_roundtrip(tmp_path: Path):
    source = ModelConfig()
    path = tmp_path / "config.json"
    save_model_config(source, path)
    loaded = load_model_config(path)
    assert loaded == source
    assert loaded.bs.default_thresholds == (0.10, 0.27, 0.44)
    assert loaded.bs.cloud_sar_weight == 0.0
    assert loaded.bs.score_recipe == "dnbr_sar"
    assert loaded.bs.index_consensus_weight == 0.0


def test_model_config_rejects_unknown_score_recipe():
    with pytest.raises(ValueError, match="score_recipe"):
        model_config_from_dict({"bs": {"score_recipe": "magic"}})


def test_model_config_rejects_negative_consensus_weight():
    with pytest.raises(ValueError, match="non-negative"):
        model_config_from_dict({"bs": {"index_consensus_weight": -0.1}})
