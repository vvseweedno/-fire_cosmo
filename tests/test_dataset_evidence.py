from pathlib import Path

import numpy as np

from wildfire.dataset_evidence import dataset_fingerprint


def _write_dataset(root: Path) -> None:
    (root / "meta.csv").write_text(
        "chip_id,kind,width,height,gsd,fire_event_id\n"
        "af_001,af,2,2,375,event_1\n",
        encoding="utf-8",
    )
    chip = root / "af_001"
    chip.mkdir()
    np.save(chip / "I4.npy", np.ones((2, 2), dtype=np.float32))
    np.save(chip / "I5.npy", np.zeros((2, 2), dtype=np.float32))


def test_dataset_fingerprint_is_deterministic(tmp_path: Path):
    _write_dataset(tmp_path)

    first = dataset_fingerprint(tmp_path)
    second = dataset_fingerprint(tmp_path)

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["file_count"] == 3
    assert first["chip_count"] == 1
    assert first["chip_contract"] == [
        {"chip_id": "af_001", "task": "AF", "channels": ["I4", "I5"]}
    ]


def test_dataset_fingerprint_changes_when_metadata_changes(tmp_path: Path):
    _write_dataset(tmp_path)
    first = dataset_fingerprint(tmp_path)

    (tmp_path / "meta.csv").write_text(
        "chip_id,kind,width,height,gsd,fire_event_id\n"
        "af_001,af,2,2,375,event_changed\n",
        encoding="utf-8",
    )
    second = dataset_fingerprint(tmp_path)

    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_full_hash_detects_same_size_raster_content_change(tmp_path: Path):
    _write_dataset(tmp_path)
    first = dataset_fingerprint(tmp_path, full_hash=True)

    np.save(tmp_path / "af_001" / "I4.npy", np.full((2, 2), 2.0, dtype=np.float32))
    second = dataset_fingerprint(tmp_path, full_hash=True)

    assert first["manifest_sha256"] != second["manifest_sha256"]
