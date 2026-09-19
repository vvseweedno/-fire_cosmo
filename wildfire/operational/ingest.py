"""Operational satellite-observation contracts.

This module is intentionally separate from competition scoring. It validates
explicit observation metadata and reports whether the currently implemented
inference adapters can consume an observation without pretending that every
sensor named in the public case is already supported end-to-end.
"""

from __future__ import annotations  # noqa: I001

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal


TaskStage = Literal["AF", "BS"]

_PUBLIC_AF_SENSORS = ("MODIS", "VIIRS", "LANDSAT")
_PUBLIC_BS_SENSORS = ("SENTINEL-2",)
_IMPLEMENTED_AF_SENSORS = ("VIIRS",)
_IMPLEMENTED_BS_SENSORS = ("SENTINEL-2",)
_VIIRS_REQUIRED_CHANNELS = frozenset({"I4", "I5"})
_SENTINEL2_REQUIRED_CHANNELS = frozenset({"B8A", "B12"})


def _normalise_sensor(value: str) -> str:
    token = value.strip().upper().replace("_", "-").replace(" ", "-")
    aliases = {
        "S2": "SENTINEL-2",
        "SENTINEL2": "SENTINEL-2",
        "LANDSAT-8": "LANDSAT",
        "LANDSAT-9": "LANDSAT",
    }
    return aliases.get(token, token)


def _normalise_channels(channels: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(str(channel).strip().upper() for channel in channels)
    if not normalized or any(not channel for channel in normalized):
        raise ValueError("observation channels must be non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("observation channels must be unique")
    return normalized


@dataclass(frozen=True)
class ObservationDescriptor:
    """Explicit provenance for one satellite observation."""

    observation_id: str
    sensor_family: str
    acquired_at: datetime
    crs: str
    bbox: tuple[float, float, float, float]
    channels: tuple[str, ...]
    source: str = ""

    @property
    def normalized_sensor(self) -> str:
        return _normalise_sensor(self.sensor_family)

    @property
    def normalized_channels(self) -> tuple[str, ...]:
        return _normalise_channels(self.channels)


@dataclass(frozen=True)
class BurnObservationPair:
    """Explicitly ordered, co-registered Sentinel-2 pre/post observation pair."""

    pre: ObservationDescriptor
    post: ObservationDescriptor


def validate_observation(observation: ObservationDescriptor) -> None:
    """Validate provenance/geometry without guessing missing metadata."""

    if not observation.observation_id.strip():
        raise ValueError("observation_id must not be empty")
    if not observation.source.strip():
        raise ValueError("source must not be empty; observation provenance is required")
    if observation.acquired_at.tzinfo is None or observation.acquired_at.utcoffset() is None:
        raise ValueError("acquired_at must be timezone-aware with a valid UTC offset")
    if not observation.crs.strip():
        raise ValueError("crs must not be empty")

    min_x, min_y, max_x, max_y = observation.bbox
    if not all(math.isfinite(value) for value in observation.bbox):
        raise ValueError("bbox coordinates must be finite")
    if not (min_x < max_x and min_y < max_y):
        raise ValueError("bbox must satisfy min_x < max_x and min_y < max_y")

    if observation.crs.strip().upper() == "EPSG:4326":
        if not (-180.0 <= min_x <= 180.0 and -180.0 <= max_x <= 180.0):
            raise ValueError("EPSG:4326 longitude is outside [-180, 180]")
        if not (-90.0 <= min_y <= 90.0 and -90.0 <= max_y <= 90.0):
            raise ValueError("EPSG:4326 latitude is outside [-90, 90]")

    _normalise_channels(observation.channels)


def operational_capabilities() -> dict[str, object]:
    """Return public-case sensor families separately from implemented adapters."""

    return {
        "active_fire": {
            "public_sensor_families": list(_PUBLIC_AF_SENSORS),
            "implemented_inference_adapters": list(_IMPLEMENTED_AF_SENSORS),
            "implemented_viirs_minimum_channels": sorted(_VIIRS_REQUIRED_CHANNELS),
        },
        "burn_assessment": {
            "public_sensor_families": list(_PUBLIC_BS_SENSORS),
            "implemented_inference_adapters": list(_IMPLEMENTED_BS_SENSORS),
            "implemented_sentinel2_minimum_channels": sorted(_SENTINEL2_REQUIRED_CHANNELS),
            "pairing": "explicit temporally ordered pre/post observations",
        },
    }


def observation_readiness(
    observation: ObservationDescriptor,
    stage: TaskStage,
) -> dict[str, object]:
    """Report whether the current repository can run this observation.

    A public-case sensor can be structurally valid while still lacking an
    implemented inference adapter. That state is reported rather than hidden.
    """

    validate_observation(observation)
    resolved_stage = stage.upper()
    if resolved_stage not in {"AF", "BS"}:
        raise ValueError("stage must be AF or BS")

    sensor = observation.normalized_sensor
    channels = set(observation.normalized_channels)

    if resolved_stage == "AF":
        public = sensor in _PUBLIC_AF_SENSORS
        implemented = sensor in _IMPLEMENTED_AF_SENSORS
        required = _VIIRS_REQUIRED_CHANNELS if sensor == "VIIRS" else frozenset()
    else:
        public = sensor in _PUBLIC_BS_SENSORS
        implemented = sensor in _IMPLEMENTED_BS_SENSORS
        required = _SENTINEL2_REQUIRED_CHANNELS if sensor == "SENTINEL-2" else frozenset()

    missing = sorted(required - channels)
    ready = bool(public and implemented and not missing)
    return {
        "stage": resolved_stage,
        "sensor_family": sensor,
        "public_contract_sensor": public,
        "inference_adapter_implemented": implemented,
        "required_channels": sorted(required),
        "missing_required_channels": missing,
        "ready": ready,
    }


def build_burn_pair(
    pre: ObservationDescriptor,
    post: ObservationDescriptor,
) -> BurnObservationPair:
    """Create a strict Sentinel-2 pre/post pair for burned-area assessment."""

    validate_observation(pre)
    validate_observation(post)

    if pre.normalized_sensor != "SENTINEL-2" or post.normalized_sensor != "SENTINEL-2":
        raise ValueError("burn observation pair currently requires Sentinel-2")
    if pre.acquired_at >= post.acquired_at:
        raise ValueError("pre observation must be earlier than post observation")
    if pre.crs.strip().upper() != post.crs.strip().upper():
        raise ValueError("pre/post CRS differ")
    if pre.bbox != post.bbox:
        raise ValueError("pre/post bounding boxes differ; observations must be co-registered")

    for label, observation in (("pre", pre), ("post", post)):
        readiness = observation_readiness(observation, "BS")
        if not readiness["ready"]:
            missing = readiness["missing_required_channels"]
            raise ValueError(
                f"{label} Sentinel-2 observation is not inference-ready; "
                f"missing channels={missing}"
            )

    return BurnObservationPair(pre=pre, post=post)
