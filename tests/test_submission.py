from pathlib import Path

import numpy as np

from wildfire.submission import Prediction, validate_submission, write_submission


def test_submission_contains_all_required_rows(tmp_path: Path):
    predictions = [
        Prediction("af_001", "AF", np.zeros((2, 2), dtype=np.uint8)),
        Prediction("bs_001", "BS", np.array([[0, 1], [2, 3]], dtype=np.uint8)),
    ]
    output = tmp_path / "submission.csv"
    assert write_submission(predictions, output) == 4

    errors = validate_submission(
        output,
        {
            "af_001": ("AF", (2, 2)),
            "bs_001": ("BS", (2, 2)),
        },
    )
    assert errors == []
