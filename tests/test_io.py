from pathlib import Path

import numpy as np

from wildfire.io import discover_chips, infer_task, load_channels


def test_discover_and_load_af_chip(tmp_path: Path):
    chip = tmp_path / "af_001"
    chip.mkdir()
    np.save(chip / "I4.npy", np.ones((2, 2), dtype=np.float32))
    np.save(chip / "I5.npy", np.ones((2, 2), dtype=np.float32))

    chips = discover_chips(tmp_path)
    assert len(chips) == 1
    channels = load_channels(chips[0])
    assert infer_task(channels) == "AF"
    assert channels["I4"].shape == (2, 2)
