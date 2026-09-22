"""Build a release manifest from measured final-run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wildfire.experiment_registry import resolve_git_commit_sha
from wildfire.release_gate import evaluate_proven_release


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def build_manifest(
    work_dir: str | Path,
    *,
    output: str | Path | None = None,
    hardware: str | None = None,
) -> dict[str, Any]:
    work = Path(work_dir)
    repo_root = Path(__file__).resolve().parents[1]
    freeze = _read_json(work / "artifacts" / "freeze_manifest.json")
    evidence = _read_json(work / "artifacts" / "release_evidence.json")
    benchmark = _read_json(work / "artifacts" / "benchmark_inference.json")

    proof = evaluate_proven_release(evidence) if evidence is not None else None
    proven = bool(proof and proof.get("proven") is True)
    final_metrics = None
    dataset_fingerprint = None
    submission_sha = None
    config_sha = None
    experiment_ids: list[str] = []
    if freeze is not None:
        final_metrics = freeze.get("final_crossfit_metrics")
        fingerprints = freeze.get("dataset_fingerprint")
        if isinstance(fingerprints, dict):
            dataset_fingerprint = fingerprints
        artifacts = freeze.get("artifacts")
        if isinstance(artifacts, dict):
            config_path = Path(str(artifacts.get("model_config", "")))
            submission_path = Path(str(artifacts.get("submission", "")))
            if not config_path.is_absolute():
                config_path = repo_root / config_path
            if not submission_path.is_absolute():
                submission_path = repo_root / submission_path
            config_sha = _sha256(config_path)
            submission_sha = _sha256(submission_path)
            experiment_path = artifacts.get("experiment_record")
            if experiment_path:
                experiment_ids.append(Path(str(experiment_path)).stem)

    dependency_lock = next(
        (
            candidate
            for candidate in ("requirements.lock", "uv.lock", "poetry.lock", "Pipfile.lock")
            if (repo_root / candidate).is_file()
        ),
        None,
    )
    runtime = benchmark.get("max_seconds") if isinstance(benchmark, dict) else None
    manifest = {
        "status": "READY" if proven else "PENDING",
        "release_ready": proven,
        "git_sha": resolve_git_commit_sha(),
        "weights_sha256": {},
        "config_sha256": {"deployment_config": config_sha} if config_sha else {},
        "dataset_fingerprint": dataset_fingerprint,
        "python_version": sys.version,
        "dependency_lock_sha256": _sha256(repo_root / dependency_lock) if dependency_lock else None,
        "hardware": hardware or platform.platform(),
        "runtime_seconds": runtime,
        "submission_sha256": submission_sha,
        "cv_metrics": final_metrics,
        "experiment_ids": experiment_ids,
        "timestamp": datetime.now(UTC).isoformat(),
        "proof_status": proof,
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="final_run")
    parser.add_argument("--output", default="release/final_manifest.json")
    parser.add_argument("--hardware")
    args = parser.parse_args()
    manifest = build_manifest(
        args.work_dir,
        output=args.output,
        hardware=args.hardware,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if manifest["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
