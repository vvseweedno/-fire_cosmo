from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from inference import run
from wildfire.submission import (
    read_submission_template,
    validate_submission_against_template,
)


def _write_stack(path: Path, arrays: list[np.ndarray], names: list[str]) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=arrays[0].shape[1],
        height=arrays[0].shape[0],
        count=len(arrays),
        dtype=str(arrays[0].dtype),
        transform=from_origin(0, 4, 1, 1),
    ) as dst:
        for index, (array, name) in enumerate(
            zip(arrays, names, strict=True),
            start=1,
        ):
            dst.write(array, index)
            dst.set_band_description(index, name)


def test_inference_end_to_end_on_stacked_geotiffs(tmp_path: Path):
    shape = (4, 4)

    i4 = np.full(shape, 300.0, dtype=np.float32)
    i5 = np.full(shape, 295.0, dtype=np.float32)
    i4[2, 2] = 380.0
    _write_stack(tmp_path / "af_001.tif", [i4, i5], ["I4", "I5"])

    _write_stack(
        tmp_path / "bs_001.tif",
        [
            np.full(shape, 0.7, dtype=np.float32),
            np.full(shape, 0.2, dtype=np.float32),
            np.full(shape, 0.3, dtype=np.float32),
            np.full(shape, 0.5, dtype=np.float32),
        ],
        ["B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST"],
    )

    (tmp_path / "meta.csv").write_text(
        "chip_id,kind,width,height,gsd\n"
        "af_001,af,4,4,375\n"
        "bs_001,bs,4,4,20\n",
        encoding="utf-8",
    )
    (tmp_path / "sample_submission.csv").write_text(
        'chip_id,class_id,rle\n'
        'af_001,1,""\n'
        'bs_001,1,""\n'
        'bs_001,2,""\n'
        'bs_001,3,""\n',
        encoding="utf-8",
    )

    output = tmp_path / "submission.csv"
    rows = run(tmp_path, output)

    assert rows == 4
    template = read_submission_template(tmp_path / "sample_submission.csv")
    errors = validate_submission_against_template(
        output,
        template,
        {"af_001": shape, "bs_001": shape},
    )
    assert errors == []
