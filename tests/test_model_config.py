from pathlib import Path

from wildfire.model_config import ModelConfig, load_model_config, save_model_config


def test_model_config_roundtrip(tmp_path: Path):
    source = ModelConfig()
    path = tmp_path / "config.json"
    save_model_config(source, path)
    loaded = load_model_config(path)
    assert loaded == source
    assert loaded.bs.default_thresholds == (0.10, 0.27, 0.44)
    assert loaded.bs.cloud_sar_weight == 0.0
