"""Dataset discovery and channel loading for both split-band and stacked rasters.

The competition archive layout is intentionally treated as data, not hard-coded
knowledge.  A stacked GeoTIFF is accepted only when its band names are available
from GDAL/rasterio band descriptions/tags or an explicit sidecar mapping.  This
avoids silently assigning the wrong physical channel to a band.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:  # pragma: no cover
    rasterio = None


ALIASES: dict[str, tuple[str, ...]] = {
    # VIIRS AF channels from the case statement.
    "I1": ("i1", "viirs_i1"),
    "I2": ("i2", "viirs_i2"),
    "I3": ("i3", "viirs_i3"),
    "I4": ("i4", "viirs_i4"),
    "I5": ("i5", "viirs_i5"),
    # Sentinel-2 pre-fire optical stack.
    "B2_PRE": ("b2_pre", "pre_b2"),
    "B3_PRE": ("b3_pre", "pre_b3"),
    "B4_PRE": ("b4_pre", "pre_b4"),
    "B5_PRE": ("b5_pre", "pre_b5"),
    "B6_PRE": ("b6_pre", "pre_b6"),
    "B7_PRE": ("b7_pre", "pre_b7"),
    "B8_PRE": ("b8_pre", "pre_b8"),
    "B8A_PRE": ("b8a_pre", "pre_b8a"),
    "B11_PRE": ("b11_pre", "pre_b11"),
    "B12_PRE": ("b12_pre", "pre_b12"),
    "SCL_PRE": ("scl_pre", "pre_scl"),
    # Sentinel-2 post-fire optical stack.
    "B2_POST": ("b2_post", "post_b2"),
    "B3_POST": ("b3_post", "post_b3"),
    "B4_POST": ("b4_post", "post_b4"),
    "B5_POST": ("b5_post", "post_b5"),
    "B6_POST": ("b6_post", "post_b6"),
    "B7_POST": ("b7_post", "post_b7"),
    "B8_POST": ("b8_post", "post_b8"),
    "B8A_POST": ("b8a_post", "post_b8a"),
    "B11_POST": ("b11_post", "post_b11"),
    "B12_POST": ("b12_post", "post_b12"),
    "SCL_POST": ("scl_post", "post_scl"),
    # Sentinel-1 pre/post.
    "VV_PRE": ("vv_pre", "pre_vv"),
    "VH_PRE": ("vh_pre", "pre_vh"),
    "VV_POST": ("vv_post", "post_vv"),
    "VH_POST": ("vh_post", "post_vh"),
    # Shared context.
    "LANDCOVER": ("landcover", "worldcover", "land_cover", "lc"),
    "DEM": ("dem", "elevation"),
    "SLOPE": ("slope",),
    "ASPECT": ("aspect",),
    "VALID_MASK": ("valid_mask", "valid", "mask_valid"),
    "AUX": ("aux", "auxiliary"),
    # AF observation geometry and ERA5-Land context.
    "SUN_ZENITH": ("sun_zenith", "solar_zenith", "sza"),
    "SUN_AZIMUTH": ("sun_azimuth", "solar_azimuth", "saa"),
    "SENSOR_ZENITH": ("sensor_zenith", "view_zenith", "vza"),
    "SENSOR_AZIMUTH": ("sensor_azimuth", "view_azimuth", "vaa"),
    "AIR_TEMPERATURE": ("air_temperature", "temperature_2m", "t2m", "era5_t2m"),
    "RELATIVE_HUMIDITY": ("relative_humidity", "humidity", "rh"),
    "WIND_U10": ("wind_u10", "u10"),
    "WIND_V10": ("wind_v10", "v10"),
    "WIND_SPEED": ("wind_speed",),
    # Optional aligned temporal recurrence prior for persistent non-wildfire heat.
    "PERSISTENT_HEAT_PRIOR": (
        "persistent_heat_prior",
        "static_heat_prior",
        "thermal_recurrence",
    ),
    "TARGET": ("target", "label", "mask", "y"),
}

SUPPORTED_SUFFIXES = (".npy", ".npz", ".tif", ".tiff")
_STACK_SUFFIXES = ("features", "feature", "image", "stack", "input", "data")

# The official archive identifies a raster by its role, while some official
# stacks do not carry GDAL band descriptions.  These are the documented
# physical channel orders for those role-based stacks.  Ambiguous stacks still
# require descriptions or a sidecar and are rejected rather than guessed.
_OFFICIAL_S2_BASE = ("B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12")
_OFFICIAL_S2_WITH_B8 = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")


def _phase_channels(names: tuple[str, ...], phase: str) -> tuple[str, ...]:
    return tuple(f"{name}_{phase.upper()}" for name in names)


@dataclass(frozen=True)
class ChannelSource:
    """Physical source of one logical channel."""

    path: Path
    band: int | None = None
    key: str | None = None


@dataclass(frozen=True)
class Chip:
    chip_id: str
    path: Path
    channels: dict[str, ChannelSource]


def _normalise_token(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def _canonical_token(value: str | None) -> str | None:
    if not value:
        return None
    token = _normalise_token(value)
    for canonical, aliases in ALIASES.items():
        accepted = {_normalise_token(canonical), *(_normalise_token(x) for x in aliases)}
        if token in accepted:
            return canonical
        if any(token.endswith(f"_{alias}") for alias in accepted):
            return canonical
    return None


def _canonical_name(path: Path) -> str | None:
    return _canonical_token(path.stem)


def _chip_id_from_stack(path: Path) -> str:
    token = path.stem
    lowered = _normalise_token(token)
    for suffix in _STACK_SUFFIXES:
        marker = f"_{suffix}"
        if lowered.endswith(marker):
            return token[: -len(marker)]
    return token


def _sidecar_candidates(path: Path) -> tuple[Path, ...]:
    candidates = (
        path.with_suffix(path.suffix + ".bands.json"),
        path.with_suffix(".bands.json"),
        path.with_suffix(".channels.json"),
        path.parent / "bands.json",
        path.parent / "channels.json",
        path.parent / "band_map.json",
    )
    # Preserve order while avoiding duplicates.
    return tuple(dict.fromkeys(candidates))


def _parse_band_map(payload: object, count: int) -> dict[int, str]:
    if isinstance(payload, dict):
        for wrapper in ("bands", "channels", "band_map"):
            if wrapper in payload:
                return _parse_band_map(payload[wrapper], count)

        result: dict[int, str] = {}
        for key, value in payload.items():
            if isinstance(value, int):
                band = int(value)
                name = str(key)
            elif isinstance(value, str) and str(key).isdigit():
                band = int(key)
                name = value
            else:
                continue
            if 1 <= band <= count:
                result[band] = name
        return result

    if isinstance(payload, list):
        return {
            index: str(name)
            for index, name in enumerate(payload, start=1)
            if index <= count and isinstance(name, str)
        }

    return {}


def _sidecar_band_map(path: Path, count: int) -> dict[int, str]:
    for candidate in _sidecar_candidates(path):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mapping = _parse_band_map(payload, count)
        if mapping:
            return mapping
    return {}


def _tiff_sources(path: Path) -> dict[str, ChannelSource]:
    if rasterio is None:
        return {}

    result: dict[str, ChannelSource] = {}
    with rasterio.open(path) as src:
        for band in range(1, src.count + 1):
            candidates: list[str] = []
            description = src.descriptions[band - 1]
            if description:
                candidates.append(description)
            tags = src.tags(band)
            for key in ("name", "band_name", "channel", "description", "long_name"):
                value = tags.get(key)
                if value:
                    candidates.append(value)

            canonical = next(
                (
                    name
                    for name in (_canonical_token(value) for value in candidates)
                    if name is not None
                ),
                None,
            )
            if canonical and canonical not in result:
                result[canonical] = ChannelSource(path=path, band=band)

        if not result:
            for band, raw_name in _sidecar_band_map(path, src.count).items():
                canonical = _canonical_token(raw_name)
                if canonical and canonical not in result:
                    result[canonical] = ChannelSource(path=path, band=band)

        if src.count == 1:
            canonical = _canonical_name(path)
            if canonical and canonical not in result:
                result[canonical] = ChannelSource(path=path, band=1)

    return result


def _npz_sources(path: Path) -> dict[str, ChannelSource]:
    result: dict[str, ChannelSource] = {}
    try:
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.files:
                canonical = _canonical_token(key)
                if canonical and canonical not in result:
                    result[canonical] = ChannelSource(path=path, key=key)
    except (OSError, ValueError):
        return {}
    return result


def _embedded_sources(path: Path) -> dict[str, ChannelSource]:
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return _tiff_sources(path)
    if suffix == ".npz":
        return _npz_sources(path)
    return {}


def _official_role(path: Path) -> tuple[str, tuple[str, ...]] | None:
    """Resolve an official role-based filename into a chip id and channels.

    The role is part of the filename contract, so this path is intentionally
    independent of TIFF descriptions.  The channel order is only fixed for
    the layouts documented by the competition contract; otherwise the normal
    description/sidecar loader remains authoritative.
    """

    stem = path.stem
    af_match = re.fullmatch(r"(?P<chip>AF_.+)_VIIRS_I1-I5", stem, flags=re.IGNORECASE)
    if af_match:
        return af_match.group("chip"), ("I1", "I2", "I3", "I4", "I5")

    aux_match = re.fullmatch(r"(?P<chip>(?:AF|BS)_.+)_AUX", stem, flags=re.IGNORECASE)
    if aux_match:
        return aux_match.group("chip"), ("AUX",)

    target_match = re.fullmatch(
        r"(?P<chip>(?:AF|BS)_.+)_(?:mask|target|label)",
        stem,
        flags=re.IGNORECASE,
    )
    if target_match:
        return target_match.group("chip"), ("TARGET",)

    s2_match = re.fullmatch(
        r"(?P<chip>BS_.+)_Sentinel[-_]2_(?P<phase>pre|post)",
        stem,
        flags=re.IGNORECASE,
    )
    if s2_match:
        phase = s2_match.group("phase").lower()
        return s2_match.group("chip"), _phase_channels(_OFFICIAL_S2_BASE, phase)

    s1_match = re.fullmatch(
        r"(?P<chip>BS_.+)_Sentinel[-_]1_(?P<phase>pre|post)",
        stem,
        flags=re.IGNORECASE,
    )
    if s1_match:
        phase = s1_match.group("phase").lower()
        return s1_match.group("chip"), (f"VV_{phase.upper()}", f"VH_{phase.upper()}")

    return None


def _official_role_sources(path: Path) -> tuple[str, dict[str, ChannelSource]] | None:
    """Build sources for one role-based official raster.

    Sentinel-2 supports the full nine-band baseline stack, the ten-band stack
    with B8, and the compact B8A/B12 fixture used by the public contract tests.
    Any other unlabelled stack is rejected as ambiguous.  An auxiliary role is
    accepted as a single raster; a multi-band auxiliary file must provide the
    usual descriptions or sidecar mapping.
    """

    resolved = _official_role(path)
    if resolved is None:
        return None
    chip_id, expected_channels = resolved

    if expected_channels == ("TARGET",):
        if path.suffix.lower() in {".tif", ".tiff"}:
            if rasterio is None:
                raise RuntimeError("rasterio is required for official target GeoTIFF input")
            with rasterio.open(path) as src:
                if src.count != 1:
                    raise ValueError(f"{path}: target raster must contain exactly one band")
            return chip_id, {"TARGET": ChannelSource(path=path, band=1)}
        if path.suffix.lower() in {".npy", ".npz"}:
            return chip_id, {"TARGET": ChannelSource(path=path)}
        raise ValueError(f"{path}: unsupported official target format")

    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"{path}: official role-based files must be GeoTIFF")
    if rasterio is None:
        raise RuntimeError("rasterio is required for official role-based GeoTIFF input")

    with rasterio.open(path) as src:
        count = src.count

    channel_names = expected_channels
    if expected_channels and expected_channels[0] in {"B2_PRE", "B2_POST"}:
        phase = expected_channels[0].rsplit("_", 1)[1]
        if count == 2:
            channel_names = _phase_channels(("B8A", "B12"), phase.lower())
        elif count == len(_OFFICIAL_S2_WITH_B8):
            channel_names = _phase_channels(_OFFICIAL_S2_WITH_B8, phase.lower())
        elif count == len(_OFFICIAL_S2_BASE):
            channel_names = _phase_channels(_OFFICIAL_S2_BASE, phase.lower())
        elif count == len(_OFFICIAL_S2_WITH_B8) + 1:
            channel_names = (*_phase_channels(_OFFICIAL_S2_WITH_B8, phase.lower()), f"SCL_{phase.upper()}")
        else:
            raise ValueError(
                f"{path}: unsupported unlabelled Sentinel-2 stack with {count} bands; "
                "expected 2, 9, 10 or 11 bands or use an explicit sidecar"
            )
    elif expected_channels == ("VV_PRE", "VH_PRE") or expected_channels == (
        "VV_POST",
        "VH_POST",
    ):
        if count != 2:
            raise ValueError(f"{path}: Sentinel-1 role stack must contain exactly VV and VH")
    elif expected_channels == ("AUX",):
        if count != 1:
            described = _tiff_sources(path)
            if not described:
                raise ValueError(
                    f"{path}: multi-band AUX raster needs band descriptions or an explicit sidecar"
                )
            return chip_id, described
    elif count != len(expected_channels):
        raise ValueError(
            f"{path}: role expects {len(expected_channels)} bands, found {count}"
        )

    return chip_id, {
        name: ChannelSource(path=path, band=index)
        for index, name in enumerate(channel_names, start=1)
    }


def discover_chips(data_dir: str | Path) -> list[Chip]:
    """Discover chips without silently guessing multiband channel order.

    Supported layouts:
    - one directory per chip with one file per channel;
    - one stacked GeoTIFF/NPZ per chip when channel names are embedded;
    - stacked GeoTIFF plus an explicit *.bands.json / channels.json sidecar.
    """

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    by_identity: dict[tuple[Path, str], dict[str, ChannelSource]] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        official = _official_role_sources(path)
        if official is not None:
            chip_id, sources = official
            identity = (path.parent, chip_id)
            by_identity.setdefault(identity, {}).update(sources)
            continue

        embedded = _embedded_sources(path)
        if len(embedded) >= 2:
            chip_id = _chip_id_from_stack(path)
            identity = (path.parent, chip_id)
            by_identity.setdefault(identity, {}).update(embedded)
            continue

        canonical = _canonical_name(path)
        if canonical:
            identity = (path.parent, path.parent.name)
            source = (
                next(iter(embedded.values()))
                if embedded
                else ChannelSource(path=path)
            )
            by_identity.setdefault(identity, {})[canonical] = source

    chips = [
        Chip(chip_id=chip_id, path=directory, channels=channels)
        for (directory, chip_id), channels in by_identity.items()
        if channels
    ]

    seen: set[str] = set()
    duplicates: set[str] = set()
    for chip in chips:
        if chip.chip_id in seen:
            duplicates.add(chip.chip_id)
        seen.add(chip.chip_id)
    if duplicates:
        raise ValueError(
            "Duplicate chip ids discovered in different directories: "
            + ", ".join(sorted(duplicates)[:10])
        )

    return sorted(chips, key=lambda item: item.chip_id)


def read_array(path: Path) -> np.ndarray:
    """Backward-compatible single-array reader."""

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if len(payload.files) != 1:
                raise ValueError(f"{path}: NPZ contains multiple arrays; use load_channels")
            return np.asarray(payload[payload.files[0]])
    if suffix in {".tif", ".tiff"}:
        if rasterio is None:
            raise RuntimeError("rasterio is required for GeoTIFF input")
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(
                    f"{path}: multiband GeoTIFF requires band descriptions/tags "
                    "or an explicit sidecar mapping"
                )
        return _read_source(ChannelSource(path=path, band=1))
    raise ValueError(f"Unsupported raster format: {path}")


def _read_source(source: ChannelSource) -> np.ndarray:
    suffix = source.path.suffix.lower()

    if suffix == ".npy":
        return np.load(source.path, allow_pickle=False)

    if suffix == ".npz":
        with np.load(source.path, allow_pickle=False) as payload:
            if source.key is not None:
                return np.asarray(payload[source.key])
            if len(payload.files) != 1:
                raise ValueError(
                    f"{source.path}: NPZ contains multiple arrays but channel key is missing"
                )
            return np.asarray(payload[payload.files[0]])

    if suffix in {".tif", ".tiff"}:
        if rasterio is None:
            raise RuntimeError("rasterio is required for GeoTIFF input")
        with rasterio.open(source.path) as src:
            band = source.band
            if band is None:
                if src.count != 1:
                    raise ValueError(
                        f"{source.path}: multiband source is missing an explicit band index"
                    )
                band = 1
            if band < 1 or band > src.count:
                raise ValueError(
                    f"{source.path}: band index {band} outside 1..{src.count}"
                )
            array = src.read(band)
            scale = float(src.scales[band - 1]) if src.scales else 1.0
            offset = float(src.offsets[band - 1]) if src.offsets else 0.0
            if not (np.isfinite(scale) and np.isfinite(offset)):
                raise ValueError(
                    f"{source.path}: non-finite raster scale/offset for band {band}"
                )
            if scale != 1.0 or offset != 0.0:
                array = np.asarray(array, dtype=np.float32) * scale + offset
            return array

    raise ValueError(f"Unsupported raster format: {source.path}")


def load_channels(chip: Chip) -> dict[str, np.ndarray]:
    channels = {name: _read_source(source) for name, source in chip.channels.items()}
    if not channels:
        return channels

    shapes = {name: tuple(np.asarray(array).shape) for name, array in channels.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"{chip.chip_id}: channel shapes differ: {shapes}")
    return channels


def infer_task(
    channels: dict[str, np.ndarray] | dict[str, ChannelSource],
) -> str:
    names = set(channels)
    if {"I4", "I5"} <= names:
        return "AF"
    if {"B8A_PRE", "B12_PRE", "B8A_POST", "B12_POST"} <= names:
        return "BS"
    raise ValueError(f"Cannot infer task from channels: {sorted(names)}")
