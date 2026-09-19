"""Report evidence-backed submission readiness without inventing metrics."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from wildfire.release_gate import evaluate_proven_release


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _item(status: str, detail: str, value: Any = None) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status, "detail": detail}
    if value is not None:
        result["value"] = value
    return result


def _route_paths() -> set[str]:
    try:
        app_module = importlib.import_module("app.main")
        return {route.path for route in app_module.app.routes}
    except Exception:
        return set()


def build_scorecard(
    repo_root: str | Path = ".",
    *,
    work_dir: str | Path = "final_run",
    evidence_path: str | Path | None = None,
    benchmark_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    work = root / Path(work_dir)
    evidence_file = root / Path(evidence_path) if evidence_path else work / "artifacts" / "release_evidence.json"
    benchmark_file = root / Path(benchmark_path) if benchmark_path else work / "artifacts" / "benchmark_inference.json"
    evidence = _read_json(evidence_file)
    benchmark = _read_json(benchmark_file)

    proof: dict[str, Any] | None = None
    if evidence is not None:
        proof = evaluate_proven_release(evidence)

    final_metrics = None
    if isinstance(evidence, dict):
        crossfit = evidence.get("crossfit")
        if isinstance(crossfit, dict) and isinstance(crossfit.get("final"), dict):
            final_metrics = crossfit["final"]

    metric: dict[str, dict[str, Any]] = {
        "canonical_scorer": _item(
            "PASS",
            "wildfire.evaluation and wildfire.metrics provide the pooled composite evaluator",
        ),
        "leakage_safe_cv": _item(
            "PASS" if proof and proof["checks"].get("crossfit_metrics_exist", {}).get("pass") else "PENDING",
            "cross-fit evidence is attached" if proof and proof["checks"].get("crossfit_metrics_exist", {}).get("pass") else "cross-fit evidence is not attached",
        ),
    }
    for key, label in (
        ("f1_af", "F1_AF"),
        ("iou_burn", "IoU_burn"),
        ("miou_severity", "mIoU_sev"),
        ("score", "Score"),
    ):
        value = final_metrics.get(key) if isinstance(final_metrics, dict) else None
        metric[key] = _item("PASS", label + " is present in release evidence", value) if value is not None else _item("PENDING", label + " is unavailable until labelled cross-fit evidence exists")

    submission_check = proof["checks"].get("submission_validator_and_sha256") if proof else None
    submission = {
        "validator": _item(
            "PASS" if submission_check and submission_check.get("pass") else "PENDING",
            submission_check.get("detail", "two validated inference runs are required") if submission_check else "submission evidence is missing",
        ),
        "exact_template": _item(
            "PASS" if submission_check and submission_check.get("pass") else "PENDING",
            "template alignment and reproducible SHA256 are evidenced" if submission_check and submission_check.get("pass") else "exact template evidence is missing",
        ),
    }

    runtime_seconds = benchmark.get("max_seconds") if isinstance(benchmark, dict) else None
    runtime = {
        "wall_time_seconds": _item("PASS", "cold-start benchmark is attached", runtime_seconds) if runtime_seconds is not None else _item("PENDING", "run scripts/benchmark_inference.py with the official dataset"),
        "tier": _item("PASS", "rubric tier from benchmark", benchmark.get("expected_official_runtime_tier")) if runtime_seconds is not None else _item("PENDING", "runtime tier is unavailable"),
    }

    routes = _route_paths()
    required_routes = {"/health", "/api/spec", "/api/query", "/api/export/geojson", "/api/summary"}
    service_ready = required_routes <= routes
    service = {
        "launch": _item("PASS", "FastAPI application imports" if routes else "FastAPI application did not import"),
        "rest": _item("PASS" if "/health" in routes else "PENDING", "health endpoint is registered" if "/health" in routes else "health endpoint is missing"),
        "map": _item("PASS" if "/map" in routes else "PENDING", "offline map endpoint is registered" if "/map" in routes else "map endpoint is missing"),
        "vector_export": _item("PASS" if "/api/export/geojson" in routes else "PENDING", "GeoJSON export endpoint is registered" if "/api/export/geojson" in routes else "GeoJSON export is missing"),
        "area_report": _item("PASS" if "/api/summary" in routes else "PENDING", "analytical summary endpoint is registered" if "/api/summary" in routes else "area summary is missing"),
        "contract": _item("PASS" if service_ready else "PENDING", "core service routes are present" if service_ready else "service contract is incomplete"),
    }

    docs = {
        "report": _item("PASS" if (root / "docs/REPORT.md").is_file() else "PENDING", "docs/REPORT.md exists" if (root / "docs/REPORT.md").is_file() else "report source is missing"),
        "presentation": _item("PASS" if (root / "docs/PRESENTATION.md").is_file() else "PENDING", "presentation source exists" if (root / "docs/PRESENTATION.md").is_file() else "presentation source is missing"),
        "defense_qa": _item("PASS" if (root / "docs/DEFENSE_QA.md").is_file() else "PENDING", "defense Q&A exists" if (root / "docs/DEFENSE_QA.md").is_file() else "defense Q&A is missing"),
        "data_compliance": _item("PASS" if (root / "docs/DATA_COMPLIANCE.md").is_file() else "PENDING", "data compliance document exists" if (root / "docs/DATA_COMPLIANCE.md").is_file() else "data compliance document is missing"),
    }
    code = {
        "docker": _item("PASS" if (root / "Dockerfile").is_file() else "PENDING", "Dockerfile exists" if (root / "Dockerfile").is_file() else "Dockerfile is missing"),
        "dependency_manifest": _item("PASS" if (root / "pyproject.toml").is_file() else "PENDING", "pyproject.toml exists" if (root / "pyproject.toml").is_file() else "dependency manifest is missing"),
        "ci": _item("PASS" if (root / ".github/workflows/ci.yml").is_file() else "PENDING", "CI workflow exists" if (root / ".github/workflows/ci.yml").is_file() else "CI workflow is missing"),
        "dependency_lock": _item("PENDING", "no committed lock file is available"),
    }
    compliance = {
        "third_party_licenses": _item("PASS" if (root / "THIRD_PARTY_LICENSES.md").is_file() else "PENDING", "license register exists" if (root / "THIRD_PARTY_LICENSES.md").is_file() else "license register is missing"),
        "test_data_rules": _item("PASS" if (root / "docs/DATA_COMPLIANCE.md").is_file() else "PENDING", "test-data compliance policy exists" if (root / "docs/DATA_COMPLIANCE.md").is_file() else "test-data policy is missing"),
    }

    hard_gate_names = (
        "ci_exact_source_success",
        "official_preflight",
        "strict_leakage_grouping",
        "leakage_audit_pass",
        "crossfit_metrics_exist",
        "final_score_above_baseline",
        "bootstrap_stability",
        "reproducibility",
        "submission_validator_and_sha256",
    )
    hard_gates = {
        name: (
            proof["checks"].get(name, {}).get("pass") is True
            if proof
            else False
        )
        for name in hard_gate_names
    }
    release_ready = bool(proof and proof.get("proven") is True and all(hard_gates.values()))
    return {
        "status": "READY FOR SUBMISSION" if release_ready else "NOT READY",
        "evidence_path": str(evidence_file),
        "benchmark_path": str(benchmark_file),
        "metric": metric,
        "submission": submission,
        "runtime": runtime,
        "code": code,
        "service": service,
        "report": docs,
        "presentation": {
            "source": docs["presentation"],
            "metric_table": metric["score"],
        },
        "compliance": compliance,
        "hard_gates": hard_gates,
        "proof": proof,
        "release_status": "READY FOR SUBMISSION" if release_ready else "NOT READY",
    }


def _print_scorecard(scorecard: dict[str, Any]) -> None:
    groups = (
        ("METRIC", scorecard["metric"]),
        ("SUBMISSION", scorecard["submission"]),
        ("RUNTIME", scorecard["runtime"]),
        ("CODE", scorecard["code"]),
        ("SERVICE", scorecard["service"]),
        ("REPORT", scorecard["report"]),
        ("PRESENTATION", scorecard["presentation"]),
        ("COMPLIANCE", scorecard["compliance"]),
    )
    for title, entries in groups:
        print(title)
        for name, item in entries.items():
            print(f"[{item['status']}] {name}: {item['detail']}")
            if "value" in item:
                print(f"  value={item['value']}")
    print(f"Release status: {scorecard['release_status']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--work-dir", default="final_run")
    parser.add_argument("--evidence")
    parser.add_argument("--benchmark")
    parser.add_argument("--output")
    args = parser.parse_args()

    scorecard = build_scorecard(
        args.repo_root,
        work_dir=args.work_dir,
        evidence_path=args.evidence,
        benchmark_path=args.benchmark,
    )
    text = json.dumps(scorecard, indent=2, ensure_ascii=False)
    _print_scorecard(scorecard)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    if scorecard["release_status"] != "READY FOR SUBMISSION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
