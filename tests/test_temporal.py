import numpy as np
import pytest

from wildfire.temporal import persistent_heat_prior


def test_persistent_heat_prior_is_recurrence_fraction():
    history = np.array(
        [
            [[1, 0], [1, 0]],
            [[1, 0], [0, 0]],
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )

    prior = persistent_heat_prior(history, min_valid_observations=3)

    assert prior.dtype == np.float32
    assert prior[0, 0] == 1.0
    assert prior[0, 1] == 0.25
    assert prior[1, 0] == 0.25
    assert prior[1, 1] == 0.0


def test_persistent_heat_prior_ignores_invalid_observations():
    history = np.array(
        [
            [[1]],
            [[1]],
            [[0]],
            [[0]],
        ],
        dtype=np.uint8,
    )
    valid = np.array(
        [
            [[1]],
            [[1]],
            [[0]],
            [[0]],
        ],
        dtype=np.uint8,
    )

    blocked = persistent_heat_prior(
        history,
        valid=valid,
        min_valid_observations=3,
    )
    allowed = persistent_heat_prior(
        history,
        valid=valid,
        min_valid_observations=2,
    )

    assert blocked[0, 0] == 0.0
    assert allowed[0, 0] == 1.0


def test_persistent_heat_prior_rejects_unaligned_valid_history():
    history = np.zeros((3, 2, 2), dtype=np.uint8)
    valid = np.zeros((3, 3, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="shape differs"):
        persistent_heat_prior(history, valid=valid)
