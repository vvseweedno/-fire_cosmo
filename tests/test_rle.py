import numpy as np

from wildfire.rle import decode_binary_mask, encode_binary_mask, encode_class


def test_rle_roundtrip_row_major_one_based():
    mask = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
    encoded = encode_binary_mask(mask)
    assert encoded == "2 2 6 1"
    assert np.array_equal(decode_binary_mask(encoded, mask.shape), mask)


def test_empty_rle():
    mask = np.zeros((2, 3), dtype=np.uint8)
    assert encode_binary_mask(mask) == ""
    assert np.array_equal(decode_binary_mask("", mask.shape), mask)


def test_class_encoding():
    mask = np.array([[0, 2], [2, 3]], dtype=np.uint8)
    assert encode_class(mask, 2) == "2 2"
