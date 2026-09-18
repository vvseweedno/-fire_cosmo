import json
from pathlib import Path

import numpy as np

from scripts.generate_baseline_oof import run
from wildfire.oof import load_oof_directory


def test_generate_baseline_oof_covers_each_validation_chip_once(tmp_path: Path):
    af = tmp_path / "af_001"
    af.mkdir()
    i4 = np.full((7, 7), 300.0, dtype=np.float32)
    i5 = np.full((7, 7), 295.0, dtype=np.float32)
    i4[3, 3] = 380.0
    target_af = np.zeros((7, 7), dtype=np.uint8)
    target_af[3, 3] = 1
    np.save(af / "I4.npy", i4)
    np.save(af / "I5.npy", i5)
    np.save(af / "target.npy", target_af)

    bs = tmp_path / "bs_001"
    bs.mkdir()
    shape = (3, 3)
    np.save(bs / "B8A_PRE.npy", np.full(shape, 0.7, dtype=np.float32))
    np.save(bs / "B12_PRE.npy", np.full(shape, 0.2, dtype=np.float32))
    np.save(bs / "B8A_POST.npy", np.full(shape, 0.3, dtype=np.float32))
    np.save(bs / "B12_POST.npy", np.full(shape, 0.5, dtype=np.float32))
    np.save(bs / "target.npy", np.full(shape, 3, dtype=np.uint8))

    manifest = tmp_path / "folds.json"
    manifest.write_text(
        json.dumps(
            {
                "folds": [
                    {"fold": 0, "validation": ["af_001"], "train": ["bs_001"]},
                    {"fold": 1, "validation": ["bs_001"], "train": ["af_001"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "oof"
    report = run(tmp_path, manifest, output)
    assert report["records"] == 2
    assert report["by_task"] == {"AF": 1, "BS": 1}

    pool = load_oof_directory(output)
    assert {record.chip_id for record in pool.records} == {"af_001", "bs_001"}
    assert {record.task for record in pool.records} == {"AF", "BS"}
