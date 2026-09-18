from pathlib import Path

import numpy as np

from scripts.profile_dataset import run


def test_profile_reports_tasks_channels_and_target_counts(tmp_path: Path):
    chip = tmp_path / "af_001"
    chip.mkdir()
    np.save(chip / "I4.npy", np.ones((2, 2), dtype=np.float32))
    np.save(chip / "I5.npy", np.ones((2, 2), dtype=np.float32))
    np.save(chip / "target.npy", np.array([[0, 0], [0, 1]], dtype=np.uint8))

    report = run(tmp_path)
    assert report["chips"] == 1
    assert report["tasks"] == {"AF": 1}
    assert report["target_class_pixels"]["AF"] == {"0": 3, "1": 1}
