"""Run-length encoding required by the competition submission."""

from __future__ import annotations

import numpy as np


def encode_binary_mask(mask: np.ndarray) -> str:
    """Encode a 2-D binary mask row-major with 1-based starts."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("RLE expects a 2-D mask")
    if not np.isfinite(array).all():
        raise ValueError("Mask contains NaN or infinity")

    flat = (array > 0).astype(np.uint8, copy=False).ravel(order="C")
    padded = np.concatenate(([0], flat, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[::2] + 1
    ends = changes[1::2] + 1
    lengths = ends - starts
    return " ".join(f"{start} {length}" for start, length in zip(starts, lengths, strict=True))


def decode_binary_mask(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode competition RLE into uint8 mask."""
    out = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    text = (rle or "").strip()
    if not text:
        return out.reshape(shape)

    tokens = text.split()
    if len(tokens) % 2:
        raise ValueError("RLE must contain start/length pairs")

    previous_end = 0
    for start_text, length_text in zip(tokens[::2], tokens[1::2], strict=True):
        start = int(start_text)
        length = int(length_text)
        if start < 1 or length < 1:
            raise ValueError("RLE starts and lengths must be positive")
        zero_start = start - 1
        end = zero_start + length
        if zero_start < previous_end:
            raise ValueError("RLE runs overlap or are not sorted")
        if end > out.size:
            raise ValueError("RLE run exceeds mask size")
        out[zero_start:end] = 1
        previous_end = end
    return out.reshape(shape)


def encode_class(mask: np.ndarray, class_id: int) -> str:
    """Encode pixels equal to one class id."""
    return encode_binary_mask(np.asarray(mask) == class_id)
