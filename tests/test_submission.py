from pathlib import Path

import numpy as np

from wildfire.submission import (
    Prediction,
    TemplateRow,
    read_submission_template,
    validate_submission_against_template,
    write_submission,
    write_submission_from_template,
)


def test_submission_rle_is_always_double_quoted(tmp_path: Path):
    predictions = [
        Prediction("af_001", "AF", np.zeros((2, 2), dtype=np.uint8)),
        Prediction("bs_001", "BS", np.array([[0, 1], [2, 3]], dtype=np.uint8)),
    ]
    output = tmp_path / "submission.csv"
    assert write_submission(predictions, output) == 4
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[1] == 'af_001,1,""'
    assert lines[2].endswith('"2 1"')


def test_template_drives_exact_pair_set_and_order(tmp_path: Path):
    template_path = tmp_path / "sample_submission.csv"
    template_path.write_text(
        'chip_id,class_id,rle\nbs_001,3,""\naf_001,1,""\nbs_001,1,""\nbs_001,2,""\n',
        encoding="utf-8",
    )
    template = read_submission_template(template_path)

    predictions = {
        "af_001": Prediction("af_001", "AF", np.zeros((2, 2), dtype=np.uint8)),
        "bs_001": Prediction("bs_001", "BS", np.array([[0, 1], [2, 3]], dtype=np.uint8)),
    }
    output = tmp_path / "submission.csv"
    assert write_submission_from_template(predictions, template, output) == 4

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[1].startswith("bs_001,3,")
    assert lines[2].startswith("af_001,1,")

    errors = validate_submission_against_template(
        output,
        template,
        {"af_001": (2, 2), "bs_001": (2, 2)},
    )
    assert errors == []


def test_validator_rejects_unquoted_rle(tmp_path: Path):
    output = tmp_path / "bad.csv"
    output.write_text("chip_id,class_id,rle\naf_001,1,\n", encoding="utf-8")
    errors = validate_submission_against_template(
        output,
        [TemplateRow("af_001", 1)],
        {"af_001": (2, 2)},
    )
    assert any("quoted RLE" in error for error in errors)


def test_validator_requires_template_order_and_canonical_rle(tmp_path: Path):
    output = tmp_path / "bad.csv"
    output.write_text(
        'chip_id,class_id,rle\n'
        'bs_001,1," 1 1 "\n'
        'af_001,1,""\n',
        encoding="utf-8",
    )
    template = [TemplateRow("af_001", 1), TemplateRow("bs_001", 1)]
    errors = validate_submission_against_template(
        output,
        template,
        {"af_001": (2, 2), "bs_001": (2, 2)},
        tasks={"af_001": "af", "bs_001": "bs"},
    )
    assert any("row order mismatch" in error for error in errors)
    assert any("non-canonical RLE" in error for error in errors)


def test_validator_rejects_class_mismatch_against_meta_task(tmp_path: Path):
    output = tmp_path / "bad.csv"
    output.write_text('chip_id,class_id,rle\naf_001,2,""\n', encoding="utf-8")
    errors = validate_submission_against_template(
        output,
        [TemplateRow("af_001", 2)],
        {"af_001": (2, 2)},
        tasks={"af_001": "af"},
    )
    assert any("invalid for af_001 task AF" in error for error in errors)


def test_validator_can_enforce_current_official_row_count(tmp_path: Path):
    output = tmp_path / "submission.csv"
    output.write_text('chip_id,class_id,rle\naf_001,1,""\n', encoding="utf-8")
    errors = validate_submission_against_template(
        output,
        [TemplateRow("af_001", 1)],
        {"af_001": (2, 2)},
        expected_row_count=447,
    )
    assert any("official 447" in error for error in errors)
