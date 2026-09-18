from pathlib import Path

import numpy as np

from scripts.evaluate_baseline import run


def test_baseline_runner_evaluates_af_and_bs(tmp_path: Path):
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

    report = run(tmp_path)
    assert report["chips"] == {"AF": 1, "BS": 1}
    assert report["evaluated_chips"] == 2
    assert report["score"] is not None
