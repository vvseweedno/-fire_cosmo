"""One-command final competition pipeline for the official AF/BS task.

This script executes the deterministic competition pipeline and freezes artifacts.
It deliberately does *not* label the release PROVEN by itself.  The authoritative
engineering proof status is produced by scripts/verify_proven_release.py only
after CI and bootstrap evidence are attached.

Pipeline:
1. deep train/test input preflight;
2. deterministic organiser-group folds with an explicit leakage audit;
3. baseline cross-fitted F1_AF / IoU_burn / mIoU_severity;
4. independently cross-fitted AF and BS candidate ensembles;
5. promotion only after non-trivial fold-wise gain;
6. pooled OOF deployment fitting only after promotion;
7. two independent deterministic reproductions;
8. strict template/RLE submission validation twice;
9. frozen model/submission/fold hashes plus release-evidence skeleton.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from inference import run as run_inference
from scripts.generate_af_candidate_oof import run as generate_af_candidate_oof
from scripts.generate_baseline_oof import run as generate_baseline_oof
from scripts.generate_bs_candidate_oof import run as generate_bs_candidate_oof
from scripts.optimize_af_candidate_ensemble import (
    run as optimize_af_candidate_ensemble,
)
from scripts.optimize_bs_candidate_ensemble import (
    run as optimize_bs_candidate_ensemble,
)
from scripts.preflight_dataset import run as preflight
from wildfire.af_ensemble_validation import crossfit_af_candidate_ensemble
from wildfire.bs_ensemble_validation import crossfit_bs_candidate_ensemble
from wildfire.crossfit import crossfit_calibrate_and_evaluate
from wildfire.metadata import read_meta_csv
from wildfire.model_config import load_model_config, save_model_config
from wildfire.oof import load_oof_directory
from wildfire.release_gate import official_score
from wildfire.split import build_group_folds, write_split_manifest
from wildfire.statistics import GroupedPrediction, paired_group_bootstrap
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
        "valid": True,
        "rows": len(template),
        "sha256": _sha256(submission),
        "errors": [],
    }


def _load_grouped_predictions(
    directory: Path,
    *,
    task: str,
    meta: dict[str, object],
) -> list[GroupedPrediction]:
    if not directory.exists():
        raise RuntimeError(f"missing cross-fit prediction directory: {directory}")
    records: list[GroupedPrediction] = []
    for path in sorted(directory.glob("*.npz")):
        chip_id = path.stem
        chip_meta = meta.get(chip_id)
        if chip_meta is None:
            raise RuntimeError(f"{chip_id}: missing from organiser meta.csv")
        with np.load(path, allow_pickle=False) as payload:
            prediction = np.asarray(payload["prediction"])
            target = np.asarray(payload["target"])
        group_id = str(getattr(chip_meta, "split_group"))
        records.append(
            GroupedPrediction(
                chip_id=chip_id,
                group_id=group_id,
                task=task,
                prediction=prediction,
                target=target,
            )
        )
    if not records:
        raise RuntimeError(f"no saved {task} cross-fit predictions in {directory}")
    return records


def _bootstrap_final_vs_baseline(
    run_dir: Path,
    meta: dict[str, object],
    promotions: dict[str, bool],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    root = run_dir / "crossfit_predictions"
    final_records: list[GroupedPrediction] = []
    baseline_records: list[GroupedPrediction] = []

    for task, key in (("AF", "af"), ("BS", "bs")):
        task_root = root / task
        baseline = _load_grouped_predictions(
            task_root / "baseline",
            task=task,
            meta=meta,
        )
        final_variant = "ensemble" if promotions[key] else "baseline"
        selected = _load_grouped_predictions(
            task_root / final_variant,
            task=task,
            meta=meta,
        )
        baseline_records.extend(baseline)
        final_records.extend(selected)

    return paired_group_bootstrap(
        final_records,
        baseline_records,
        n_boot=n_boot,
        seed=seed,
    )


def _selected_metrics(
    baseline_crossfit: dict[str, object],
    af_candidate_crossfit: dict[str, object],
    bs_candidate_crossfit: dict[str, object],
) -> tuple[dict[str, bool], dict[str, float]]:
    base = baseline_crossfit.get("crossfit")
    if not isinstance(base, dict):
        raise RuntimeError("baseline crossfit report has no summary")

    for key in METRIC_KEYS:
        if base.get(key) is None:
            raise RuntimeError(f"baseline crossfit metric {key} is unavailable")

    af_promoted = bool(af_candidate_crossfit.get("promotion_allowed"))
    bs_promoted = bool(bs_candidate_crossfit.get("promotion_allowed"))

    f1 = float(base["f1_af"])
    if af_promoted:
        value = af_candidate_crossfit.get("ensemble_f1")
        if value is None:
            raise RuntimeError("promoted AF report has no ensemble_f1")
        f1 = float(value)

    iou = float(base["iou_burn"])
    miou = float(base["miou_severity"])
    if bs_promoted:
        ensemble = bs_candidate_crossfit.get("ensemble")
        if not isinstance(ensemble, dict):
            raise RuntimeError("promoted BS report has no ensemble summary")
        if ensemble.get("iou_burn") is None or ensemble.get("miou_severity") is None:
            raise RuntimeError("promoted BS metrics are unavailable")
        iou = float(ensemble["iou_burn"])
        miou = float(ensemble["miou_severity"])

    metrics = {
        "f1_af": f1,
        "iou_burn": iou,
        "miou_severity": miou,
    }
    metrics["score"] = official_score(metrics)
    return {"af": af_promoted, "bs": bs_promoted}, metrics


def _fit_once(
    train_dir: Path,
    fold_manifest: Path,
    run_dir: Path,
    *,
    base_config_path: Path,
    af_alpha_steps: int,
    af_epsilon: float,
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

    af_candidate_root = run_dir / "oof_af_candidates"
    af_candidate_oof_summary = generate_af_candidate_oof(
        train_dir,
        fold_manifest,
        af_candidate_root,
        model_config=base_config_path,
    )
    af_candidate_report = crossfit_af_candidate_ensemble(
        af_candidate_root,
        manifest,
        alpha_steps=af_alpha_steps,
        epsilon=af_epsilon,
        prediction_output=run_dir / "crossfit_predictions" / "AF",
    )
    _write_json(run_dir / "af_candidate_crossfit.json", af_candidate_report)

    bs_candidate_root = run_dir / "oof_bs_candidates"
    bs_candidate_oof_summary = generate_bs_candidate_oof(
        train_dir,
        fold_manifest,
        bs_candidate_root,
        model_config=base_config_path,
    )
    # Cross-fitted candidate comparison must start from the frozen pre-OOF
    # baseline config. Passing pooled deployment thresholds here would leak
    # holdout-label information through optimizer initialization.
    bs_candidate_report = crossfit_bs_candidate_ensemble(
        bs_candidate_root,
        manifest,
        base_config,
        alpha_steps=alpha_steps,
        threshold_candidates=ensemble_threshold_candidates,
        threshold_passes=ensemble_threshold_passes,
        landcover_passes=bs_landcover_passes,
        prediction_output=run_dir / "crossfit_predictions" / "BS",
    )
    _write_json(run_dir / "bs_candidate_crossfit.json", bs_candidate_report)

    promotions, metrics = _selected_metrics(
        baseline_report,
        af_candidate_report,
        bs_candidate_report,
    )

    bs_config_path = run_dir / "bs_deployment_config.json"
    pooled_bs_report: dict[str, object] | None = None
    if promotions["bs"]:
        pooled_bs_report = optimize_bs_candidate_ensemble(
            bs_candidate_root,
            base_config_path=baseline_config_path,
            output_config=bs_config_path,
            output_report=run_dir / "bs_ensemble_pooled.json",
            alpha_steps=max(alpha_steps, 16),
            threshold_candidates=max(ensemble_threshold_candidates, 96),
            threshold_passes=max(ensemble_threshold_passes, 3),
            landcover_passes=bs_landcover_passes,
        )
    else:
        shutil.copyfile(baseline_config_path, bs_config_path)

    final_config_path = run_dir / "deployment_config.json"
    pooled_af_report: dict[str, object] | None = None
    if promotions["af"]:
        pooled_af_report = optimize_af_candidate_ensemble(
            af_candidate_root,
            base_config_path=bs_config_path,
            output_config=final_config_path,
            output_report=run_dir / "af_ensemble_pooled.json",
            alpha_steps=max(af_alpha_steps, 24),
        )
    else:
        shutil.copyfile(bs_config_path, final_config_path)

    base_summary = baseline_report["crossfit"]
    if not isinstance(base_summary, dict):
        raise RuntimeError("baseline crossfit summary is missing")
    delta_score = float(metrics["score"]) - float(base_summary["score"])

    return {
        "promotions": promotions,
        "metrics": metrics,
        "baseline_metrics": {key: float(base_summary[key]) for key in METRIC_KEYS},
        "delta_score": delta_score,
        "deployment_config": str(final_config_path),
        "deployment_signature": _deployment_signature(final_config_path),
        "baseline_oof": baseline_oof_summary,
        "af_candidate_oof": af_candidate_oof_summary,
        "bs_candidate_oof": bs_candidate_oof_summary,
        "baseline_crossfit": baseline_report,
        "af_candidate_crossfit": af_candidate_report,
        "bs_candidate_crossfit": bs_candidate_report,
        "pooled_af_ensemble": pooled_af_report,
        "pooled_bs_ensemble": pooled_bs_report,
    }


def _assert_reproducible(
    first: dict[str, object],
    second: dict[str, object],
    *,
    tolerance: float,
) -> dict[str, float]:
    if first["promotions"] != second["promotions"]:
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
                f"delta={delta:.12f} > {tolerance:.12f}"
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
    tolerance: float = 1e-8,
    score_epsilon: float = 1e-4,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 99173,
    af_alpha_steps: int = 20,
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
        af_alpha_steps=af_alpha_steps,
        af_epsilon=score_epsilon / 0.35,
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
        af_alpha_steps=af_alpha_steps,
        af_epsilon=score_epsilon / 0.35,
        bs_max_candidates=bs_max_candidates,
        bs_passes=bs_passes,
        bs_landcover_passes=bs_landcover_passes,
        alpha_steps=alpha_steps,
        ensemble_threshold_candidates=ensemble_threshold_candidates,
        ensemble_threshold_passes=ensemble_threshold_passes,
    )
    metric_deltas = _assert_reproducible(first, second, tolerance=tolerance)

    if float(first["delta_score"]) <= score_epsilon:
        raise RuntimeError(
            "technical proof gate failed: final cross-fitted Score did not beat "
            f"baseline by epsilon={score_epsilon:g}; delta={first['delta_score']}"
        )

    bootstrap = _bootstrap_final_vs_baseline(
        root / "repro_run_1",
        meta,
        first["promotions"],
        n_boot=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    _write_json(root / "bootstrap_final_vs_baseline.json", bootstrap)
    ci95 = bootstrap.get("bootstrap_95_ci")
    if not isinstance(ci95, list) or len(ci95) != 2:
        raise RuntimeError("bootstrap report is missing a valid 95% confidence interval")
    bootstrap_ok = (
        int(bootstrap.get("n_boot_used", 0)) >= 1000
        and float(ci95[0]) > 0.0
        and float(bootstrap.get("probability_delta_positive", 0.0)) >= 0.95
    )
    if not bootstrap_ok:
        raise RuntimeError(
            "technical proof gate failed: event-level bootstrap stability is "
            f"not acceptable: {bootstrap}"
        )

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
        "status": "frozen_candidate",
        "proof_status": "PENDING_CI",
        "validation": "cross_fitted_oof",
        "promotions": first["promotions"],
        "baseline_crossfit_metrics": first["baseline_metrics"],
        "final_crossfit_metrics": first["metrics"],
        "delta_score_vs_baseline": first["delta_score"],
        "score_epsilon": score_epsilon,
        "bootstrap": bootstrap,
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

    evidence = {
        "ci": {
            "green": False,
            "note": "Set from the actual GitHub Actions result before PROVEN verification.",
        },
        "preflight": {
            "train": train_preflight,
            "test": test_preflight,
        },
        "crossfit": {
            "baseline": first["baseline_metrics"],
            "final": first["metrics"],
        },
        "bootstrap": bootstrap,
        "reproducibility": {
            "run_1": first["metrics"],
            "run_2": second["metrics"],
            "deployment_signature_equal": (
                first["deployment_signature"] == second["deployment_signature"]
            ),
        },
        "submission": {
            "run_1": inference_runs[0],
            "run_2": inference_runs[1],
        },
    }
    evidence_path = root / "artifacts" / "release_evidence.json"
    _write_json(evidence_path, evidence)

    return {
        "pipeline_ok": True,
        "proven": False,
        "proof_status": "PENDING_CI",
        "promotions": first["promotions"],
        "baseline_metrics": first["baseline_metrics"],
        "metrics": first["metrics"],
        "delta_score_vs_baseline": first["delta_score"],
        "submission": str(final_submission),
        "final_model_config": str(final_config),
        "freeze_manifest": str(freeze_path),
        "release_evidence": str(evidence_path),
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
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--score-epsilon", type=float, default=1e-4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=99173)
    parser.add_argument("--af-alpha-steps", type=int, default=20)
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
        score_epsilon=args.score_epsilon,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        af_alpha_steps=args.af_alpha_steps,
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
