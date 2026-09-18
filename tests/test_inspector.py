from pathlib import Path

import numpy as np

from scripts.inspect_dataset import probe_dataset


def test_probe_reports_unknown_stacked_file_and_csv_header(tmp_path: Path):
    np.save(tmp_path / "AF_tr_000001.npy", np.zeros((6, 2, 2), dtype=np.float32))
    (tmp_path / "meta.csv").write_text(
        "chip_id,kind,width,height,gsd\nAF_tr_000001,af,2,2,375\n",
        encoding="utf-8",
    )

    report = probe_dataset(tmp_path)
    assert report["raster_files_total"] == 1
    assert report["recognised_chips_total"] == 0
    assert report["raw_rasters_shown"][0]["shape"] == [6, 2, 2]
    assert report["csv_headers"]["meta.csv"] == [
        "chip_id",
        "kind",
        "width",
        "height",
        "gsd",
    ]
