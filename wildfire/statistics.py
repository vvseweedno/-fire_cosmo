"""Paired group bootstrap for OOF experiment comparison.

The sampling unit is a fire-event group, not an individual pixel. This preserves
within-event correlation and avoids pretending millions of adjacent pixels are
independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wildfire.evaluation import CompetitionEvaluator


@dataclass(frozen=True)
class GroupedPrediction:
    chip_id: str
    group_id: str
    task: str
    prediction: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        if not self.chip_id:
            raise ValueError("chip_id must not be empty")
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if self.task.upper() not in {"AF", "BS"}:
            raise ValueError(f"unsupported task: {self.task}")
        if self.prediction.shape != self.target.shape:
            raise ValueError(f"{self.chip_id}: prediction/target shapes differ")


def _score_records(records: list[GroupedPrediction]) -> float:
    evaluator = CompetitionEvaluator()
    for record in records:
        if record.task.upper() == "AF":
            evaluator.update_af(record.prediction, record.target)
        else:
            evaluator.update_bs(record.prediction, record.target)
    score = evaluator.summary()["score"]
    if score is None:
        raise ValueError("bootstrap sample must contain both AF and BS records")
    return float(score)


def _index_by_chip(
    records: list[GroupedPrediction],
) -> dict[str, GroupedPrediction]:
    indexed: dict[str, GroupedPrediction] = {}
    for record in records:
        if record.chip_id in indexed:
            raise ValueError(f"duplicate chip_id: {record.chip_id}")
        indexed[record.chip_id] = record
    return indexed


def paired_group_bootstrap(
    experiment_a: list[GroupedPrediction],
    experiment_b: list[GroupedPrediction],
    *,
    n_boot: int = 5000,
    seed: int = 99173,
) -> dict[str, object]:
    """Compare two calibrated experiments using paired event-level bootstrap.

    The returned delta is Score(A) - Score(B). Positive values favor A.
    """
    if n_boot < 100:
        raise ValueError("n_boot must be at least 100")

    a_by_chip = _index_by_chip(experiment_a)
    b_by_chip = _index_by_chip(experiment_b)
    if set(a_by_chip) != set(b_by_chip):
        raise ValueError("experiments must contain exactly the same chip ids")

    for chip_id in sorted(a_by_chip):
        left = a_by_chip[chip_id]
        right = b_by_chip[chip_id]
        if left.task.upper() != right.task.upper():
            raise ValueError(f"{chip_id}: task mismatch between experiments")
        if left.group_id != right.group_id:
            raise ValueError(f"{chip_id}: group mismatch between experiments")
        if not np.array_equal(left.target, right.target):
            raise ValueError(f"{chip_id}: target mismatch between experiments")

    groups: dict[str, list[str]] = {}
    for chip_id, record in a_by_chip.items():
        groups.setdefault(record.group_id, []).append(chip_id)

    group_ids = sorted(groups)
    if len(group_ids) < 2:
        raise ValueError("at least two groups are required for bootstrap")

    full_a = _score_records(experiment_a)
    full_b = _score_records(experiment_b)
    full_delta = full_a - full_b

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    skipped = 0

    for _ in range(n_boot):
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True)
        sampled_a: list[GroupedPrediction] = []
        sampled_b: list[GroupedPrediction] = []

        for group_id in sampled_groups:
            for chip_id in groups[str(group_id)]:
                sampled_a.append(a_by_chip[chip_id])
                sampled_b.append(b_by_chip[chip_id])

        try:
            diffs.append(_score_records(sampled_a) - _score_records(sampled_b))
        except ValueError:
            skipped += 1

    if not diffs:
        raise ValueError("all bootstrap replicates lacked both AF and BS samples")

    arr = np.asarray(diffs, dtype=np.float64)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {
        "metric": "official_competition_score",
        "definition": "experiment_a_minus_experiment_b",
        "groups": len(group_ids),
        "chips": len(a_by_chip),
        "n_boot_requested": n_boot,
        "n_boot_used": int(arr.size),
        "n_boot_skipped": skipped,
        "score_a": full_a,
        "score_b": full_b,
        "delta_score": full_delta,
        "bootstrap_mean_delta": float(arr.mean()),
        "bootstrap_median_delta": float(np.median(arr)),
        "bootstrap_95_ci": [float(lo), float(hi)],
        "probability_delta_positive": float(np.mean(arr > 0)),
        "interpretation": (
            "supports_a"
            if lo > 0
            else "supports_b"
            if hi < 0
            else "inconclusive"
        ),
        "caution": (
            "This is a cluster bootstrap over fire-event groups. It quantifies "
            "OOF sampling uncertainty; it does not replace external test evaluation."
        ),
    }
