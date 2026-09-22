from pathlib import Path

import pytest

from inference import run


def test_inference_refuses_to_overwrite_organizer_contract_inputs(tmp_path: Path):
    meta = tmp_path / "meta.csv"
    template = tmp_path / "sample_submission.csv"
    meta.write_text("meta-original\n", encoding="utf-8")
    template.write_text("template-original\n", encoding="utf-8")

    for protected in (meta, template):
        with pytest.raises(ValueError, match="must not overwrite organizer input"):
            run(tmp_path, protected)

    assert meta.read_text(encoding="utf-8") == "meta-original\n"
    assert template.read_text(encoding="utf-8") == "template-original\n"
