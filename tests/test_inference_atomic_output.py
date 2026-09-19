from types import SimpleNamespace

import numpy as np
import pytest

import inference


def _patch_minimal_run(monkeypatch, validation_errors):
    template = [SimpleNamespace(chip_id="AF_001")]
    meta = {"AF_001": SimpleNamespace(kind="AF", shape=(2, 2))}
    chip = SimpleNamespace(chip_id="AF_001", channels={})

    monkeypatch.setattr(inference, "read_submission_template", lambda _: template)
    monkeypatch.setattr(inference, "read_meta_csv", lambda _: meta)
    monkeypatch.setattr(inference, "validate_template_task_contract", lambda *_: [])
    monkeypatch.setattr(inference, "load_model_config", lambda _: object())
    monkeypatch.setattr(inference, "discover_chips", lambda _: [chip])
    monkeypatch.setattr(inference, "load_channels", lambda _: {})
    monkeypatch.setattr(
        inference, "predict", lambda *_: np.zeros((2, 2), dtype=np.uint8)
    )

    def write_submission(_predictions, _template, path):
        path.write_text("candidate\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(inference, "write_submission_from_template", write_submission)
    monkeypatch.setattr(
        inference,
        "validate_submission_against_template",
        lambda *_args, **_kwargs: validation_errors,
    )


def test_default_model_config_is_independent_of_cwd(tmp_path, monkeypatch):
    _patch_minimal_run(monkeypatch, [])
    loaded_configs = []
    monkeypatch.setattr(
        inference,
        "load_model_config",
        lambda path: loaded_configs.append(path) or object(),
    )
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)

    inference.run(tmp_path, tmp_path / "submission.csv")

    assert loaded_configs == [inference.DEFAULT_MODEL_CONFIG]
    assert inference.DEFAULT_MODEL_CONFIG.is_absolute()


def test_duplicate_discovered_chip_ids_fail_closed(tmp_path, monkeypatch):
    _patch_minimal_run(monkeypatch, [])
    duplicate_a = SimpleNamespace(chip_id="AF_001", channels={})
    duplicate_b = SimpleNamespace(chip_id="AF_001", channels={})
    monkeypatch.setattr(
        inference, "discover_chips", lambda _: [duplicate_a, duplicate_b]
    )
    monkeypatch.setattr(
        inference,
        "load_channels",
        lambda _: pytest.fail("ambiguous chip must not reach inference"),
    )

    with pytest.raises(RuntimeError, match="duplicate chip IDs: AF_001"):
        inference.run(tmp_path, tmp_path / "submission.csv")

    assert not (tmp_path / "submission.csv").exists()


def test_failed_validation_preserves_existing_submission(tmp_path, monkeypatch):
    _patch_minimal_run(monkeypatch, ["bad rle"])
    output = tmp_path / "submission.csv"
    output.write_text("known-good\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="strict validation"):
        inference.run(tmp_path, output)

    assert output.read_text(encoding="utf-8") == "known-good\n"
    assert list(tmp_path.glob(".submission.csv.*.tmp")) == []


def test_validated_submission_atomically_replaces_destination(tmp_path, monkeypatch):
    _patch_minimal_run(monkeypatch, [])
    output = tmp_path / "nested" / "submission.csv"

    rows = inference.run(tmp_path, output)

    assert rows == 1
    assert output.read_text(encoding="utf-8") == "candidate\n"
    assert list(output.parent.glob(".submission.csv.*.tmp")) == []
