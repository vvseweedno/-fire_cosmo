import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from wildfire.io import discover_chips, infer_task, load_channels, read_array


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


def _write_stack(path: Path, arrays: list[np.ndarray], descriptions: list[str | None]):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arrays[0].shape[1],
        height=arrays[0].shape[0],
        count=len(arrays),
        dtype=str(arrays[0].dtype),
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        for index, (array, description) in enumerate(
            zip(arrays, descriptions, strict=True),
            start=1,
        ):
            dst.write(array, index)
            if description is not None:
                dst.set_band_description(index, description)


def test_multiband_geotiff_uses_band_descriptions(tmp_path: Path):
    path = tmp_path / "af_001.tif"
    arrays = [
        np.full((2, 3), index, dtype=np.float32)
        for index in range(1, 6)
    ]
    _write_stack(path, arrays, ["I1", "I2", "I3", "VIIRS_I4", "VIIRS_I5"])

    chips = discover_chips(tmp_path)
    assert [chip.chip_id for chip in chips] == ["af_001"]

    channels = load_channels(chips[0])
    assert infer_task(channels) == "AF"
    assert set(channels) == {"I1", "I2", "I3", "I4", "I5"}
    assert np.all(channels["I4"] == 4)
    assert np.all(channels["I5"] == 5)


def test_multiband_geotiff_accepts_explicit_sidecar_mapping(tmp_path: Path):
    path = tmp_path / "bs_007_stack.tif"
    names = ["B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST", "VH_PRE", "VH_POST"]
    arrays = [
        np.full((2, 2), index, dtype=np.float32)
        for index in range(1, len(names) + 1)
    ]
    _write_stack(path, arrays, [None] * len(names))
    path.with_suffix(".bands.json").write_text(
        json.dumps({"bands": names}),
        encoding="utf-8",
    )

    chips = discover_chips(tmp_path)
    assert [chip.chip_id for chip in chips] == ["bs_007"]

    channels = load_channels(chips[0])
    assert infer_task(channels) == "BS"
    assert np.all(channels["B8A_PRE"] == 1)
    assert np.all(channels["B12_POST"] == 4)
    assert np.all(channels["VH_POST"] == 6)


def test_read_array_rejects_ambiguous_multiband_tiff(tmp_path: Path):
    path = tmp_path / "ambiguous.tif"
    arrays = [np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]
    _write_stack(path, arrays, [None, None])

    try:
        read_array(path)
    except ValueError as exc:
        assert "multiband GeoTIFF requires" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ambiguous multiband raster must not silently use band 1")
