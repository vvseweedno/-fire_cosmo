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


def test_reader_recognises_official_context_and_red_edge_channels(tmp_path: Path):
    chip = tmp_path / "bs_001"
    chip.mkdir()
    for name in (
        "B5_PRE",
        "B6_PRE",
        "B7_PRE",
        "B8A_PRE",
        "B12_PRE",
        "B5_POST",
        "B6_POST",
        "B7_POST",
        "B8A_POST",
        "B12_POST",
        "SUN_ZENITH",
        "AIR_TEMPERATURE",
    ):
        np.save(chip / f"{name}.npy", np.ones((2, 2), dtype=np.float32))

    channels = load_channels(discover_chips(tmp_path)[0])
    assert {"B5_PRE", "B6_PRE", "B7_PRE", "B5_POST", "B6_POST", "B7_POST"} <= set(
        channels
    )
    assert {"SUN_ZENITH", "AIR_TEMPERATURE"} <= set(channels)
    assert infer_task(channels) == "BS"
