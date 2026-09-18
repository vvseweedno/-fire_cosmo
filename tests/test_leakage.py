import json
from pathlib import Path

import numpy as np

from wildfire.leakage import audit_leakage
from wildfire.metadata import read_meta_csv
from wildfire.split import build_group_folds


def _write_chip(root: Path, chip_id: str, kind: str) -> None:
    chip = root / chip_id
    chip.mkdir()
    if kind == "af":
        np.save(chip / "I4.npy", np.ones((2, 2), dtype=np.float32))
        np.save(chip / "I5.npy", np.zeros((2, 2), dtype=np.float32))
    else:
        for name in ("B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST"):
            np.save(chip / f"{name}.npy", np.ones((2, 2), dtype=np.float32))
    np.save(chip / "TARGET.npy", np.zeros((2, 2), dtype=np.uint8))


def _write_dataset(root: Path, missing_group: bool = False) -> None:
    rows = [
        "chip_id,kind,width,height,gsd,fire_event_id",
        "af_1,af,2,2,375,event_1",
        "bs_1,bs,2,2,20,event_1",
        "af_2,af,2,2,375,event_2",
        f"bs_2,bs,2,2,20,{'' if missing_group else 'event_3'}",
    ]
    (root / "meta.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for chip_id, kind in (("af_1", "af"), ("bs_1", "bs"), ("af_2", "af"), ("bs_2", "bs")):
        _write_chip(root, chip_id, kind)


def test_leakage_audit_passes_strict_group_folds(tmp_path: Path):
    _write_dataset(tmp_path)
    meta = read_meta_csv(tmp_path / "meta.csv")
    manifest = build_group_folds(meta, n_splits=2, seed=4, require_event_groups=True)

    report = audit_leakage(tmp_path, manifest)

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["failures"] == []
    assert report["blocked_checks"] == []


def test_leakage_audit_is_blocked_without_fold_manifest(tmp_path: Path):
    _write_dataset(tmp_path)

    report = audit_leakage(tmp_path)

    assert report["status"] == "BLOCKED"
    assert report["pass"] is False
    assert "validation_coverage_exact" in report["blocked_checks"]


def test_leakage_audit_rejects_missing_event_group(tmp_path: Path):
    _write_dataset(tmp_path, missing_group=True)
    meta = read_meta_csv(tmp_path / "meta.csv")
    manifest = build_group_folds(meta, n_splits=2, seed=4)

    report = audit_leakage(tmp_path, manifest)

    assert report["status"] == "FAIL"
    assert report["checks"]["strict_event_grouping"] is False
    assert "bs_2" in report["chips_missing_event_group"]


def test_leakage_audit_rejects_event_split_across_validation_folds(tmp_path: Path):
    _write_dataset(tmp_path)
    manifest = {
        "folds": [
            {
                "fold": 0,
                "train": ["bs_1", "af_2", "bs_2"],
                "validation": ["af_1"],
            },
            {
                "fold": 1,
                "train": ["af_1", "af_2", "bs_2"],
                "validation": ["bs_1"],
            },
            {
                "fold": 2,
                "train": ["af_1", "bs_1"],
                "validation": ["af_2", "bs_2"],
            },
        ]
    }

    report = audit_leakage(tmp_path, manifest)

    assert report["status"] == "FAIL"
    assert report["checks"]["no_event_cross_fold_leakage"] is False
    assert report["folds"]["events_spanning_validation_folds"]["event_1"] == [0, 1]


def test_leakage_audit_report_is_json_serializable(tmp_path: Path):
    _write_dataset(tmp_path)
    meta = read_meta_csv(tmp_path / "meta.csv")
    manifest = build_group_folds(meta, n_splits=2, seed=7, require_event_groups=True)

    report = audit_leakage(tmp_path, manifest)

    json.dumps(report)
