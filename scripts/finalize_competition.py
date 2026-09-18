"""One-command final competition gate.

Implements the nine-step release path:
1) strict train/test reading preflight;
2) deterministic group-aware folds;
3) cross-fitted AF/BS metrics;
4) SAR + land-cover + spectral candidate sweep;
5) baseline-preserving promotion gate;
6) full OOF/inference end-to-end run;
7) two-run reproducibility check;
8) strict submission validation;
9) immutable final config + SHA256 freeze manifest.

The script never queries external fire products or reconstructs private-test
geolocation/dates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from inference import run as run_inference
from scripts.generate_baseline_oof import run as generate_oof
from scripts.preflight_dataset import run as preflight
from scripts.run_metric_candidates import run as run_candidates
from wildfire.crossfit import crossfit_calibrate_and_evaluate
from wildfire.metadata import read_meta_csv
from wildfire.model_config import load_model_config, save_model_config
from wildfire.oof import load_oof_directory
from wildfire.split import build_group_folds, write_split_manifest
from wildfire.submission import (
    read_submission_template,
    validate_submission_against_template,
)


METRIC_KEYS = ("f1_af", "iou_burn", "miou_severity", "score")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_submission(data_dir: Path, submission: Path) -> dict[str, object]:
    template = read_submission_template(data_dir / "sample_submission.csv")
    meta = read_meta_csv(data_dir / "meta.csv")
    shapes = {chip_id: item.shape for chip_id, item in meta.items()}
    errors = validate_submission_against_template(submission, template, shapes)
    if errors:
        raise RuntimeError("invalid submission: " + "; ".join(errors[:10]))
    return {
        "rows": len(template),
        "sha256": _sha256(submission),
        "errors": [],
    }


def _reproduce_crossfit(
    train_dir: Path,
    fold_manifest: Path,
    model_config: Path,
    output_dir: Path,
    *,
    bs_max_candidates: int,
    bs_passes: int,
    bs_landcover_passes: int,
) -> dict[str, object]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_dir = output_dir / "oof"
    oof_summary = generate_oof(
        train_dir,
        fold_manifest,
        oof_dir,
        model_config=model_config,
    )
    manifest = json.loads(fold_manifest.read_text(encoding="utf-8"))
    pool = load_oof_directory(oof_dir)
    base = load_model_config(model_config)
    deployment, report = crossfit_calibrate_and_evaluate(
        pool,
        manifest,
        base,
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
    )
    config_path = output_dir / "deployment_config.json"
    report_path = output_dir / "crossfit_report.json"
    save_model_config(deployment, config_path)
    _write_json(report_path, report)
    return {
        "oof_summary": oof_summary,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "metrics": report["crossfit"],
    }


def _assert_reproducible(
    first: dict[str, object],
    second: dict[str, object],
    *,
    tolerance: float,
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    left = first["metrics"]
    right = second["metrics"]
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise RuntimeError("reproduction reports do not contain metric dictionaries")

    for key in METRIC_KEYS:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value is None or right_value is None:
            raise RuntimeError(f"reproduction metric {key} is unavailable")
        delta = abs(float(left_value) - float(right_value))
        deltas[key] = delta
        if delta > tolerance:
            raise RuntimeError(
                f"reproducibility failed for {key}: delta={delta:.8f} > {tolerance:.8f}"
            )

    if first["config_sha256"] != second["config_sha256"]:
        raise RuntimeError("repeated cross-fit produced different deployment configs")
    return deltas


def run(
    train_dir: str | Path,
    test_dir: str | Path,
    work_dir: str | Path,
    *,
    folds: int = 5,
    seed: int = 42,
    allow_chip_fallback: bool = False,
    tolerance: float = 0.005,
    min_gain: float = 1e-6,
    bs_max_candidates: int = 128,
    bs_passes: int = 4,
    bs_landcover_passes: int = 2,
) -> dict[str, object]:
    train_root = Path(train_dir)
    test_root = Path(test_dir)
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_preflight = preflight(train_root, mode="train", deep=True)
    test_preflight = preflight(test_root, mode="test", deep=True)
    _write_json(root / "train_preflight.json", train_preflight)
    _write_json(root / "test_preflight.json", test_preflight)
    if not train_preflight["ok"] or not test_preflight["ok"]:
        raise RuntimeError("dataset preflight failed")

    meta = read_meta_csv(train_root / "meta.csv")
    fold_manifest = root / f"folds_seed{seed}.json"
    manifest = build_group_folds(
        meta,
        n_splits=folds,
        seed=seed,
        require_event_groups=not allow_chip_fallback,
    )
    write_split_manifest(manifest, fold_manifest)

    candidate_report = run_candidates(
        train_root,
        fold_manifest,
        root / "candidates",
        min_gain=min_gain,
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
    )
    selected = Path(str(candidate_report["selected_deployment_config"]))

    repro1 = _reproduce_crossfit(
        train_root,
        fold_manifest,
        selected,
        root / "repro_run_1",
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
    )
    repro2 = _reproduce_crossfit(
        train_root,
        fold_manifest,
        selected,
        root / "repro_run_2",
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
    )
    metric_deltas = _assert_reproducible(repro1, repro2, tolerance=tolerance)

    final_config = root / "artifacts" / "final_model_config.json"
    final_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(str(repro1["config"])), final_config)

    inference_runs: list[dict[str, object]] = []
    for index in (1, 2):
        submission = root / f"submission_run_{index}.csv"
        started = time.perf_counter()
        rows = run_inference(test_root, submission, final_config)
        elapsed = time.perf_counter() - started
        validation = _validate_submission(test_root, submission)
        validation.update(
            {
                "run": index,
                "rows_written": rows,
                "seconds": elapsed,
                "path": str(submission),
            }
        )
        inference_runs.append(validation)

    if inference_runs[0]["sha256"] != inference_runs[1]["sha256"]:
        raise RuntimeError("repeated inference produced different submission.csv bytes")

    final_submission = root / "submission.csv"
    shutil.copyfile(Path(str(inference_runs[0]["path"])), final_submission)

    freeze = {
        "status": "frozen",
        "selection": {
            "winner": candidate_report["winner"],
            "baseline_score": candidate_report["baseline_score"],
            "winner_score": candidate_report["winner_score"],
            "winner_delta": candidate_report["winner_delta"],
        },
        "crossfit_metrics": repro1["metrics"],
        "reproducibility": {
            "tolerance": tolerance,
            "metric_deltas": metric_deltas,
            "config_sha256_equal": repro1["config_sha256"] == repro2["config_sha256"],
            "submission_sha256_equal": (
                inference_runs[0]["sha256"] == inference_runs[1]["sha256"]
            ),
        },
        "leakage_audit": manifest.get("leakage_audit"),
        "artifacts": {
            "model_config": str(final_config),
            "model_config_sha256": _sha256(final_config),
            "submission": str(final_submission),
            "submission_sha256": _sha256(final_submission),
            "fold_manifest": str(fold_manifest),
            "fold_manifest_sha256": _sha256(fold_manifest),
        },
        "inference_runs": inference_runs,
    }
    freeze_path = root / "artifacts" / "freeze_manifest.json"
    _write_json(freeze_path, freeze)

    return {
        "ok": True,
        "winner": candidate_report["winner"],
        "metrics": repro1["metrics"],
        "submission": str(final_submission),
        "final_model_config": str(final_config),
        "freeze_manifest": str(freeze_path),
        "reproducibility": freeze["reproducibility"],
        "leakage_audit": freeze["leakage_audit"],
        "inference_runs": inference_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--work-dir", default="final_run")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-chip-fallback",
        action="store_true",
        help=(
            "Allow chips without organiser event/group ids. This weakens the "
            "leakage-safety claim and is intentionally opt-in."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--bs-max-candidates", type=int, default=128)
    parser.add_argument("--bs-passes", type=int, default=4)
    parser.add_argument("--bs-landcover-passes", type=int, default=2)
    args = parser.parse_args()

    report = run(
        args.train_dir,
        args.test_dir,
        args.work_dir,
        folds=args.folds,
        seed=args.seed,
        allow_chip_fallback=args.allow_chip_fallback,
        tolerance=args.tolerance,
        min_gain=args.min_gain,
        bs_max_candidates=args.bs_max_candidates,
        bs_passes=args.bs_passes,
        bs_landcover_passes=args.bs_landcover_passes,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
