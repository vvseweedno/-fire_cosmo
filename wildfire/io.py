"""Dataset discovery and channel loading without assuming one unpublished layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


ALIASES: dict[str, tuple[str, ...]] = {
    "I1": ("i1", "viirs_i1"),
    "I2": ("i2", "viirs_i2"),
    "I3": ("i3", "viirs_i3"),
    "I4": ("i4", "viirs_i4"),
    "I5": ("i5", "viirs_i5"),
    "B2_PRE": ("b2_pre", "pre_b2"),
    "B3_PRE": ("b3_pre", "pre_b3"),
    "B4_PRE": ("b4_pre", "pre_b4"),
    "B8A_PRE": ("b8a_pre", "pre_b8a"),
    "B11_PRE": ("b11_pre", "pre_b11"),
    "B12_PRE": ("b12_pre", "pre_b12"),
    "SCL_PRE": ("scl_pre", "pre_scl"),
    "B2_POST": ("b2_post", "post_b2"),
    "B3_POST": ("b3_post", "post_b3"),
    "B4_POST": ("b4_post", "post_b4"),
    "B8A_POST": ("b8a_post", "post_b8a"),
    "B11_POST": ("b11_post", "post_b11"),
    "B12_POST": ("b12_post", "post_b12"),
    "SCL_POST": ("scl_post", "post_scl"),
    "VV_PRE": ("vv_pre", "pre_vv"),
    "VH_PRE": ("vh_pre", "pre_vh"),
    "VV_POST": ("vv_post", "post_vv"),
    "VH_POST": ("vh_post", "post_vh"),
    "LANDCOVER": ("landcover", "worldcover", "land_cover", "lc"),
    "DEM": ("dem", "elevation"),
    "SLOPE": ("slope",),
    "ASPECT": ("aspect",),
    "VALID_MASK": ("valid_mask", "valid", "mask_valid"),
    "TARGET": ("target", "label", "mask", "y"),
}

SUPPORTED_SUFFIXES = (".npy", ".npz", ".tif", ".tiff")


@dataclass(frozen=True)
class Chip:
    chip_id: str
    path: Path
    channels: dict[str, Path]


def _normalise_stem(path: Path) -> str:
    return path.stem.lower().replace("-", "_").replace(" ", "_")


def _canonical_name(path: Path) -> str | None:
    stem = _normalise_stem(path)
    for canonical, aliases in ALIASES.items():
        if stem in aliases:
            return canonical
        if any(stem.endswith(f"_{alias}") for alias in aliases):
            return canonical
    return None


def discover_chips(data_dir: str | Path) -> list[Chip]:
    """Treat each directory containing recognised channels as a chip."""
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    by_dir: dict[Path, dict[str, Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        canonical = _canonical_name(path)
        if canonical:
            by_dir.setdefault(path.parent, {})[canonical] = path

    chips = [
        Chip(chip_id=directory.name, path=directory, channels=channels)
        for directory, channels in by_dir.items()
        if channels
    ]
    return sorted(chips, key=lambda item: item.chip_id)


def read_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        payload = np.load(path)
        if len(payload.files) != 1:
            raise ValueError(f"{path}: NPZ must contain exactly one array")
        return payload[payload.files[0]]
    if suffix in {".tif", ".tiff"}:
        if rasterio is None:
            raise RuntimeError("rasterio is required for GeoTIFF input")
        with rasterio.open(path) as src:
            return src.read(1)
    raise ValueError(f"Unsupported raster format: {path}")


def load_channels(chip: Chip) -> dict[str, np.ndarray]:
    return {name: read_array(path) for name, path in chip.channels.items()}


def infer_task(channels: dict[str, np.ndarray] | dict[str, Path]) -> str:
    names = set(channels)
    if {"I4", "I5"} <= names:
        return "AF"
    if {"B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST"} <= names:
        return "BS"
    raise ValueError(f"Cannot infer task from channels: {sorted(names)}")
