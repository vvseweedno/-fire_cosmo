"""Small deterministic training utilities used before heavier ML models are justified."""

import numpy as np

from wildfire.evaluation import BinaryAccumulator
from wildfire.model_config import ModelConfig


AFTrainingSample = tuple[np.ndarray, np.ndarray, np.ndarray]


def threshold_grid(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold step must be positive")
    if maximum < minimum:
        raise ValueError("threshold maximum must be >= minimum")
    count = int(np.floor((maximum - minimum) / step + 1e-9)) + 1
    values = [minimum + index * step for index in range(count)]
    if values[-1] < maximum - 1e-9:
        values.append(maximum)
    return [round(float(value), 10) for value in values]


def calibrate_af_threshold(
    samples,
    thresholds: list[float],
    base_config: ModelConfig,
) -> tuple[ModelConfig, list[dict[str, float | int]]]:
    """Maximise micro-F1 on the supplied training partition.

    Ties prefer a threshold closest to the baseline default and then the higher
    threshold, which is conservative under the extreme AF class imbalance.
    """
    candidates = [float(value) for value in thresholds]
    if not candidates:
        raise ValueError("threshold grid is empty")
    accumulators = {value: BinaryAccumulator() for value in candidates}
    sample_count = 0

    for score, target, model_valid in samples:
        score_array = np.asarray(score, dtype=np.float32)
        target_array = np.asarray(target)
        valid_array = np.asarray(model_valid, dtype=bool)
        if score_array.shape != target_array.shape or score_array.shape != valid_array.shape:
            raise ValueError("score, target and model_valid shapes must match")
        for value, accumulator in accumulators.items():
            pred = (score_array >= value) & valid_array
            accumulator.update(pred, target_array > 0)
        sample_count += 1

    if sample_count == 0:
        raise ValueError("no AF training samples were supplied")

    trace: list[dict[str, float | int]] = []
    for value in candidates:
        metric = accumulators[value]
        trace.append(
            {
                "threshold": value,
                "f1": metric.f1,
                "tp": metric.tp,
                "fp": metric.fp,
                "fn": metric.fn,
            }
        )

    default_threshold = base_config.af.threshold
    best = max(
        trace,
        key=lambda row: (
            float(row["f1"]),
            -abs(float(row["threshold"]) - default_threshold),
            float(row["threshold"]),
        ),
    )
    best_threshold = float(best["threshold"])

    training_metadata = dict(base_config.training)
    training_metadata["af_threshold_calibration"] = {
        "method": "deterministic_micro_f1_grid_search_on_train_partition",
        "samples": sample_count,
        "best_threshold": best_threshold,
        "best_f1_train": float(best["f1"]),
        "trace": trace,
    }
    calibrated_af = type(base_config.af)(
        threshold=best_threshold,
        z4_weight=base_config.af.z4_weight,
        z5_weight=base_config.af.z5_weight,
        local_anomaly_weight=base_config.af.local_anomaly_weight,
        i3_sunglint_penalty=base_config.af.i3_sunglint_penalty,
        water_snow_penalty=base_config.af.water_snow_penalty,
        built_penalty=base_config.af.built_penalty,
        bare_penalty=base_config.af.bare_penalty,
    )
    calibrated = ModelConfig(
        version=base_config.version,
        af=calibrated_af,
        bs=base_config.bs,
        training=training_metadata,
    )
    return calibrated, trace
