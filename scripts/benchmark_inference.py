"""Measure cold-start end-to-end inference time for the production command."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_tier(seconds: float | None) -> str:
    """Return the rubric tier for one complete cold-start run."""

    if seconds is None:
        return "PENDING"
    if seconds < 30.0:
        return "8/8"
    if seconds <= 300.0:
        return "6/8"
    if seconds <= 720.0:
        return "4/8"
    if seconds <= 1800.0:
        return "2/8"
    return "0/8"


def run_benchmark(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    runs: int = 2,
    model_config: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be at least one")

    root = (Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]).resolve()
    data_path = Path(data_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    command_base = [
        sys.executable,
        "inference.py",
        "--data-dir",
        str(data_path),
    ]
    if model_config is not None:
        command_base.extend(["--model-config", str(Path(model_config).resolve())])

    records: list[dict[str, Any]] = []
    for index in range(1, runs + 1):
        output = output_root / f"submission_benchmark_run_{index}.csv"
        output.unlink(missing_ok=True)
        command = [*command_base, "--output", str(output)]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - started
        record: dict[str, Any] = {
            "run": index,
            "command": command,
            "returncode": completed.returncode,
            "seconds": elapsed,
            "output": str(output),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        if completed.returncode == 0 and output.is_file():
            record.update(
                {
                    "status": "PASS",
                    "bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                }
            )
        else:
            record["status"] = "FAIL"
        records.append(record)

    successful = [
        float(record["seconds"])
        for record in records
        if record.get("status") == "PASS"
    ]
    result: dict[str, Any] = {
        "status": "PASS" if len(successful) == runs else "FAIL",
        "data_dir": str(data_path),
        "repo_root": str(root),
        "runs": records,
        "successful_run_count": len(successful),
        "min_seconds": min(successful) if successful else None,
        "max_seconds": max(successful) if successful else None,
        "p95_seconds": max(successful) if successful else None,
        "expected_official_runtime_tier": runtime_tier(max(successful) if successful else None),
        "cold_start": True,
    }
    if successful:
        result["submission_sha256_equal"] = len(
            {str(record.get("sha256")) for record in records if record.get("sha256")}
        ) == 1
    else:
        result["submission_sha256_equal"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/benchmark_inference")
    parser.add_argument("--model-config")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--repo-root")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = run_benchmark(
        args.data_dir,
        args.output_dir,
        runs=args.runs,
        model_config=args.model_config,
        repo_root=args.repo_root,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
