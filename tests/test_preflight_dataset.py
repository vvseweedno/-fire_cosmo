from pathlib import Path

import numpy as np

from scripts.preflight_dataset import run


def _write_test_contract(root: Path, *, class_id: int = 1) -> None:
    chip = root / "af_001"
    chip.mkdir()
    for channel in ("I1", "I2", "I3", "I4", "I5"):
        np.save(chip / f"{channel}.npy", np.ones((2, 2), dtype=np.float32))
    (root / "meta.csv").write_text(
        "chip_id,kind,width,height,gsd\naf_001,af,2,2,375\n",
        encoding="utf-8",
    )
    (root / "sample_submission.csv").write_text(
        f'chip_id,class_id,rle\naf_001,{class_id},""\n',
        encoding="utf-8",
    )


def test_preflight_rejects_template_class_incompatible_with_meta(tmp_path: Path):
    _write_test_contract(tmp_path, class_id=2)

    report = run(tmp_path, mode="test", deep=True)

    assert report["ok"] is False
    assert any("template class 2" in error for error in report["errors"])


def test_preflight_reports_nonfinite_raster_values(tmp_path: Path):
    _write_test_contract(tmp_path)
    np.save(tmp_path / "af_001" / "I3.npy", np.full((2, 2), np.nan, dtype=np.float32))

    report = run(tmp_path, mode="test", deep=True)

    assert report["ok"] is False
    assert any("I3 contains no finite pixels" in error for error in report["errors"])
