"""One-command final competition gate for the official AF/BS task.

The gate executes the requested release protocol without private-test lookup:
1. deep train/test input preflight;
2. deterministic organiser-group folds with an explicit leakage audit;
3. cross-fitted F1_AF / IoU_burn / mIoU_severity;
4. BASE-anchored SAR/spectral/land-cover candidate ensemble;
5. promotion only after positive fold-wise cross-fitted Score delta;
6. full OOF -> deployment -> test inference path;
7. two independent deterministic reproductions;
8. strict template/RLE submission validation twice;
9. frozen model/submission/fold hashes in a release manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from inference import run as run_inference
from scripts.generate_baseline_oof import run as generate_baseline_oof
from scripts.generate_bs_candidate_oof import run as generate_bs_candidate_oof
from scripts.optimize_bs_candidate_ensemble import run as optimize_bs_candidate_ensemble
from scripts.preflight_dataset import run as preflight

from wildfire.bs_ensemble_validation import crossfit_bs_candidate_ensemble
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


def _deployment_signature(config_path: Path) -> dict[str, object]:
    """Compare deployable parameters while ignoring path-bearing training notes."""
    payload = asdict(load_model_config(config_path))
    payload.pop("training", None)
    return payload


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


def _selected_metrics(
    baseline_crossfit: dict[str, object],
    candidate_crossfit: dict[str, object],
) -> tuple[bool, dict[str, float]]:
    base = baseline_crossfit["crossfit"]
    if not isinstance(base, dict):
        raise RuntimeError("baseline crossfit report has no summary")
    f1 = base.get("f1_af")
    if f1 is None:
        raise RuntimeError("baseline crossfit F1_AF is unavailable")

    promoted = bool(candidate_crossfit.get("promotion_allowed"))
    if promoted:
        ensemble = candidate_crossfit.get("ensemble")
        if not isinstance(ensemble, dict):
            raise RuntimeError("candidate report has no ensemble summary")
        iou = ensemble.get("iou_burn")
        miou = ensemble.get("miou_severity")
        if iou is None or miou is None:
            raise RuntimeError("candidate BS metrics are unavailable")
        metrics = {
            "f1_af": float(f1),
            "iou_burn": float(iou),
            "miou_severity": float(miou),
        }
        metrics["score"] = (
            0.35 * metrics["f1_af"]
            + 0.35 * metrics["iou_burn"]
            + 0.30 * metrics["miou_severity"]
        )
        return True, metrics

    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = base.get(key)
        if value is None:
            raise RuntimeError(f"baseline crossfit metric {key} is unavailable")
        result[key] = float(value)
    return False, result


def _fit_once(
    train_dir: Path,
    fold_manifest: Path,
    run_dir: Path,
    *,
    base_config_path: Path,
    bs_max_candidates: int,
    bs_passes: int,
    bs_landcover_passes: int,
    alpha_steps: int,
    ensemble_threshold_candidates: int,
    ensemble_threshold_passes: int,
) -> dict[str, object]:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(fold_manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("fold manifest root must be an object")

    baseline_oof = run_dir / "oof_baseline"
    baseline_oof_summary = generate_baseline_oof(
        train_dir,
        fold_manifest,
        baseline_oof,
        model_config=base_config_path,
    )
    pool = load_oof_directory(baseline_oof)
    base_config = load_model_config(base_config_path)
    baseline_deployment, baseline_report = crossfit_calibrate_and_evaluate(
        pool,
        manifest,
        base_config,
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
    )
    baseline_config_path = run_dir / "baseline_deployment_config.json"
    save_model_config(baseline_deployment, baseline_config_path)
    _write_json(run_dir / "baseline_crossfit.json", baseline_report)

    candidate_root = run_dir / "oof_bs_candidates"
    candidate_oof_summary = generate_bs_candidate_oof(
        train_dir,
        fold_manifest,
        candidate_root,
        model_config=baseline_config_path,
    )
    candidate_report = crossfit_bs_candidate_ensemble(
        candidate_root,
        manifest,
        baseline_deployment,
        alpha_steps=alpha_steps,
        threshold_candidates=ensemble_threshold_candidates,
        threshold_passes=ensemble_threshold_passes,
        landcover_passes=bs_landcover_passes,
    )
    _write_json(run_dir / "bs_candidate_crossfit.json", candidate_report)

    promoted, metrics = _selected_metrics(baseline_report, candidate_report)
    final_config_path = run_dir / "deployment_config.json"
    pooled_ensemble_report: dict[str, object] | None = None
    if promoted:
        pooled_ensemble_report = optimize_bs_candidate_ensemble(
            candidate_root,
            base_config_path=baseline_config_path,
            output_config=final_config_path,
            output_report=run_dir / "bs_ensemble_pooled.json",
            alpha_steps=max(alpha_steps, 16),
            threshold_candidates=max(ensemble_threshold_candidates, 96),
            threshold_passes=max(ensemble_threshold_passes, 3),
            landcover_passes=bs_landcover_passes,
        )
    else:
        shutil.copyfile(baseline_config_path, final_config_path)

    return {
        "promoted": promoted,
        "metrics": metrics,
        "deployment_config": str(final_config_path),
        "deployment_signature": _deployment_signature(final_config_path),
        "baseline_oof": baseline_oof_summary,
        "candidate_oof": candidate_oof_summary,
        "baseline_crossfit": baseline_report,
        "candidate_crossfit": candidate_report,
        "pooled_ensemble": pooled_ensemble_report,
    }


def _assert_reproducible(
    first: dict[str, object],
    second: dict[str, object],
    *,
    tolerance: float,
) -> dict[str, float]:
    if bool(first["promoted"]) != bool(second["promoted"]):
        raise RuntimeError("repeated fitting disagreed on ensemble promotion")
    if first["deployment_signature"] != second["deployment_signature"]:
        raise RuntimeError("repeated fitting produced different deployable parameters")

    left = first["metrics"]
    right = second["metrics"]
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise RuntimeError("reproduction metrics are missing")

    deltas: dict[str, float] = {}
    for key in METRIC_KEYS:
        delta = abs(float(left[key]) - float(right[key]))
        deltas[key] = delta
        if delta > tolerance:
            raise RuntimeError(
                f"reproducibility failed for {key}: "
                f"delta={delta:.8f} > {tolerance:.8f}"
            )
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
    bs_max_candidates: int = 128,
    bs_passes: int = 4,
    bs_landcover_passes: int = 2,
    alpha_steps: int = 16,
    ensemble_threshold_candidates: int = 96,
    ensemble_threshold_passes: int = 3,
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

    base_config_path = Path("configs/baseline.json")
    first = _fit_once(
        train_root,
        fold_manifest,
        root / "repro_run_1",
        base_config_path=base_config_path,
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
        alpha_steps=alpha_steps,
        ensemble_threshold_candidates=ensemble_threshold_candidates,
        ensemble_threshold_passes=ensemble_threshold_passes,
    )
    second = _fit_once(
        train_root,
        fold_manifest,
        root / "repro_run_2",
        base_config_path=base_config_path,
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
        alpha_steps=alpha_steps,
        ensemble_threshold_candidates=ensemble_threshold_candidates,
        ensemble_threshold_passes=ensemble_threshold_passes,
    )
    metric_deltas = _assert_reproducible(first, second, tolerance=tolerance)

    final_config = root / "artifacts" / "final_model_config.json"
    final_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(str(first["deployment_config"])), final_config)

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
        "validation": "cross_fitted_oof",
        "promoted_bs_candidate_ensemble": bool(first["promoted"]),
        "crossfit_metrics": first["metrics"],
        "candidate_delta_total_score": first["candidate_crossfit"].get(
            "delta_total_score"
        ),
        "reproducibility": {
            "tolerance": tolerance,
            "metric_deltas": metric_deltas,
            "deployment_signature_equal": (
                first["deployment_signature"] == second["deployment_signature"]
            ),
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
        "promoted_bs_candidate_ensemble": bool(first["promoted"]),
        "metrics": first["metrics"],
        "candidate_delta_total_score": first["candidate_crossfit"].get(
            "delta_total_score"
        ),
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
            "Allow chips without organiser event/group ids. This is explicit "
            "because it weakens the leakage-safety guarantee."
        ),
    )
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--bs-max-candidates", type=int, default=128)
    parser.add_argument("--bs-passes", type=int, default=4)
    parser.add_argument("--bs-landcover-passes", type=int, default=2)
    parser.add_argument("--alpha-steps", type=int, default=16)
    parser.add_argument("--ensemble-threshold-candidates", type=int, default=96)
    parser.add_argument("--ensemble-threshold-passes", type=int, default=3)
    args = parser.parse_args()

    report = run(
        args.train_dir,
        args.test_dir,
        args.work_dir,
        folds=args.folds,
        seed=args.seed,
        allow_chip_fallback=args.allow_chip_fallback,
        tolerance=args.tolerance,
        bs_max_candidates=args.bs_max_candidates,
        bs_passes=args.bs_passes,
        bs_landcover_passes=args.bs_landcover_passes,
        alpha_steps=args.alpha_steps,
        ensemble_threshold_candidates=args.ensemble_threshold_candidates,
        ensemble_threshold_passes=args.ensemble_threshold_passes,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
