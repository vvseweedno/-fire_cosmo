"""Machine-readable experiment records for reproducible model selection."""

from __future__ import annotations  # noqa: I001

import json
import os
import resource
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal


ExperimentState = Literal[
    "EXPERIMENTAL",
    "CANDIDATE",
    "PROMOTED",
    "REJECTED",
    "FINAL",
]


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    git_commit_sha: str
    dataset_fingerprint: str
    task: str
    candidate_name: str
    candidate_version: str
    configuration: dict[str, Any]
    seed: int
    fold_manifest_sha256: str
    train_events: list[str]
    holdout_events: list[str]
    per_fold_metrics: Any
    aggregate_metrics: dict[str, float]
    official_score: float | None
    reference_score: float | None
    delta_score: float | None
    runtime_seconds: float
    peak_ram_mb: float | None
    gpu_info: dict[str, Any]
    bootstrap_result: dict[str, Any] | None
    state: ExperimentState
    promoted: bool
    promotion_reason: str | None
    rejected_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def resolve_git_commit_sha(repo_root: str | Path = ".") -> str:
    """Resolve source SHA without inventing one when git metadata is unavailable."""

    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "UNRESOLVED"
    sha = completed.stdout.strip()
    return sha or "UNRESOLVED"


def peak_ram_mb() -> float | None:
    """Best-effort process peak RSS; Linux reports ru_maxrss in KiB."""

    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    if value <= 0:
        return None
    # Competition CI/runtime is Linux; keep the evidence unit explicit here.
    return value / 1024.0


def validate_experiment_record(record: ExperimentRecord) -> None:
    if not record.experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    if not record.git_commit_sha.strip():
        raise ValueError("git_commit_sha must not be empty")
    if not record.dataset_fingerprint.strip():
        raise ValueError("dataset_fingerprint must not be empty")
    if not record.fold_manifest_sha256.strip():
        raise ValueError("fold_manifest_sha256 must not be empty")
    if record.runtime_seconds < 0:
        raise ValueError("runtime_seconds must be non-negative")
    if record.peak_ram_mb is not None and record.peak_ram_mb < 0:
        raise ValueError("peak_ram_mb must be non-negative")

    if record.promoted and record.state == "REJECTED":
        raise ValueError("rejected experiment cannot be promoted")
    if record.state == "REJECTED" and not record.rejected_reason:
        raise ValueError("rejected experiment requires rejected_reason")
    if record.promoted and not record.promotion_reason:
        raise ValueError("promoted experiment requires promotion_reason")

    if (
        record.official_score is not None
        and record.reference_score is not None
        and record.delta_score is not None
    ):
        expected = record.official_score - record.reference_score
        if abs(expected - record.delta_score) > 1e-10:
            raise ValueError("delta_score does not match score-reference_score")


def write_experiment_record(
    output_dir: str | Path,
    record: ExperimentRecord,
) -> Path:
    """Validate and atomically write artifacts/experiments/<id>.json."""

    validate_experiment_record(record)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{record.experiment_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
