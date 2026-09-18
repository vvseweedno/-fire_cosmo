"""Versioned, serialisable baseline configuration used by train/eval/inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AFConfig:
    threshold: float = 4.0
    z4_weight: float = 1.10
    z5_weight: float = -0.30
    local_anomaly_weight: float = 0.85
    i3_sunglint_penalty: float = 0.10
    water_snow_penalty: float = 4.0
    built_penalty: float = 2.0
    bare_penalty: float = 0.5


@dataclass(frozen=True)
class BSConfig:
    default_thresholds: tuple[float, float, float] = (0.10, 0.27, 0.44)
    natural_open_thresholds: tuple[float, float, float] = (0.08, 0.20, 0.35)
    crop_thresholds: tuple[float, float, float] = (0.18, 0.32, 0.48)
    forest_thresholds: tuple[float, float, float] = (0.10, 0.27, 0.44)
    sar_weight: float = 0.025
    cloud_sar_weight: float = 0.0
    sar_clip: float = 3.0


@dataclass(frozen=True)
class ModelConfig:
    version: int = 2
    af: AFConfig = field(default_factory=AFConfig)
    bs: BSConfig = field(default_factory=BSConfig)
    training: dict[str, Any] = field(default_factory=dict)


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three thresholds")
    result = tuple(float(item) for item in value)
    if not result[0] < result[1] < result[2]:
        raise ValueError(f"{name} thresholds must be strictly increasing")
    return result  # type: ignore[return-value]


def model_config_from_dict(payload: dict[str, Any]) -> ModelConfig:
    af_payload = dict(payload.get("af") or {})
    bs_payload = dict(payload.get("bs") or {})

    af = AFConfig(
        threshold=float(af_payload.get("threshold", 4.0)),
        z4_weight=float(af_payload.get("z4_weight", 1.10)),
        z5_weight=float(af_payload.get("z5_weight", -0.30)),
        local_anomaly_weight=float(af_payload.get("local_anomaly_weight", 0.85)),
        i3_sunglint_penalty=float(af_payload.get("i3_sunglint_penalty", 0.10)),
        water_snow_penalty=float(af_payload.get("water_snow_penalty", 4.0)),
        built_penalty=float(af_payload.get("built_penalty", 2.0)),
        bare_penalty=float(af_payload.get("bare_penalty", 0.5)),
    )
    bs = BSConfig(
        default_thresholds=_triple(
            bs_payload.get("default_thresholds", (0.10, 0.27, 0.44)),
            "default_thresholds",
        ),
        natural_open_thresholds=_triple(
            bs_payload.get("natural_open_thresholds", (0.08, 0.20, 0.35)),
            "natural_open_thresholds",
        ),
        crop_thresholds=_triple(
            bs_payload.get("crop_thresholds", (0.18, 0.32, 0.48)),
            "crop_thresholds",
        ),
        forest_thresholds=_triple(
            bs_payload.get("forest_thresholds", (0.10, 0.27, 0.44)),
            "forest_thresholds",
        ),
        sar_weight=float(bs_payload.get("sar_weight", 0.025)),
        cloud_sar_weight=float(bs_payload.get("cloud_sar_weight", 0.0)),
        sar_clip=float(bs_payload.get("sar_clip", 3.0)),
    )
    return ModelConfig(
        version=int(payload.get("version", 1)),
        af=af,
        bs=bs,
        training=dict(payload.get("training") or {}),
    )


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    if path is None:
        return ModelConfig()
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Model config root must be an object")
    return model_config_from_dict(payload)


def save_model_config(config: ModelConfig, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
