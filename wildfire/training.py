"""Small deterministic training utilities used before heavier ML models are justified."""

import numpy as np


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
    base_config,
):
    """Maximise official micro-F1 on the supplied training partition.

    The model-side valid mask constrains predictions, but it never removes GT
    pixels from scoring: official AF evaluation pools all pixels.
    """
    candidates = [float(value) for value in thresholds]
    if not candidates:
        raise ValueError("threshold grid is empty")

    counts = {value: {"tp": 0, "fp": 0, "fn": 0} for value in candidates}
    sample_count = 0

    for score, target, model_valid in samples:
        score_array = np.asarray(score, dtype=np.float32)
        target_array = np.asarray(target) > 0
        valid_array = np.asarray(model_valid, dtype=bool)
        if score_array.shape != target_array.shape or score_array.shape != valid_array.shape:
            raise ValueError("score, target and model_valid shapes must match")

        for value in candidates:
            pred = (score_array >= value) & valid_array
            truth = target_array
            counts[value]["tp"] += int(np.count_nonzero(pred & truth))
            counts[value]["fp"] += int(np.count_nonzero(pred & ~truth))
            counts[value]["fn"] += int(np.count_nonzero(~pred & truth))
        sample_count += 1

    if sample_count == 0:
        raise ValueError("no AF training samples were supplied")

    trace: list[dict[str, float | int]] = []
    for value in candidates:
        tp = counts[value]["tp"]
        fp = counts[value]["fp"]
        fn = counts[value]["fn"]
        denom = 2 * tp + fp + fn
        f1 = 1.0 if denom == 0 else (2.0 * tp) / denom
        trace.append(
            {
                "threshold": value,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
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
    calibrated = type(base_config)(
        version=base_config.version,
        af=calibrated_af,
        bs=base_config.bs,
        training=training_metadata,
    )
    return calibrated, trace
