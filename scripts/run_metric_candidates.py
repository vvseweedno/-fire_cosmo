"""Run deterministic metric candidates and promote only cross-fit improvements.

This is the competition-safe experiment gate for the inexpensive physics stack.
Every candidate is evaluated on exactly the same fold manifest. The baseline is
always present, and a candidate can be selected only when its cross-fitted
official Score exceeds the baseline by at least --min-gain.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

from scripts.generate_baseline_oof import run as generate_oof
from wildfire.crossfit import crossfit_calibrate_and_evaluate
from wildfire.model_config import ModelConfig, load_model_config, save_model_config
from wildfire.oof import load_oof_directory


def candidate_configs(base: ModelConfig) -> dict[str, ModelConfig]:
    """Return auditable low-cost candidates; no candidate sees validation labels."""
    candidates = {
        "baseline": base,
        "optical_only": replace(
            base,
            bs=replace(
                base.bs,
                sar_weight=0.0,
                cloud_sar_weight=0.0,
                score_recipe="dnbr_sar",
                index_consensus_weight=0.0,
            ),
        ),
        "sar_clear_pos_005": replace(
            base,
            bs=replace(
                base.bs,
                sar_weight=0.05,
                cloud_sar_weight=0.0,
                score_recipe="dnbr_sar",
                index_consensus_weight=0.0,
            ),
        ),
        "sar_clear_neg_005": replace(
            base,
            bs=replace(
                base.bs,
                sar_weight=-0.05,
                cloud_sar_weight=0.0,
                score_recipe="dnbr_sar",
                index_consensus_weight=0.0,
            ),
        ),
        "sar_cloud_pos_010": replace(
            base,
            bs=replace(
                base.bs,
                cloud_sar_weight=0.10,
                score_recipe="dnbr_sar",
                index_consensus_weight=0.0,
            ),
        ),
        "sar_cloud_neg_010": replace(
            base,
            bs=replace(
                base.bs,
                cloud_sar_weight=-0.10,
                score_recipe="dnbr_sar",
                index_consensus_weight=0.0,
            ),
        ),
        "spectral_0025": replace(
            base,
            bs=replace(
                base.bs,
                score_recipe="spectral_consensus",
                index_consensus_weight=0.025,
                cloud_sar_weight=0.0,
            ),
        ),
        "spectral_005": replace(
            base,
            bs=replace(
                base.bs,
                score_recipe="spectral_consensus",
                index_consensus_weight=0.05,
                cloud_sar_weight=0.0,
            ),
        ),
        "spectral_010": replace(
            base,
            bs=replace(
                base.bs,
                score_recipe="spectral_consensus",
                index_consensus_weight=0.10,
                cloud_sar_weight=0.0,
            ),
        ),
        # Exact deployable convex score blends. Within this physics family
        # score = dNBR + w_sar*SAR + w_idx*consensus, so blending two model
        # scores is equivalent to blending these weights and needs no special
        # inference runtime.
        "blend_sar_spectral_025_005": replace(
            base,
            bs=replace(
                base.bs,
                sar_weight=0.025,
                cloud_sar_weight=0.0,
                score_recipe="spectral_consensus",
                index_consensus_weight=0.05,
            ),
        ),
        "blend_sar_spectral_050_005": replace(
            base,
            bs=replace(
                base.bs,
                sar_weight=0.05,
                cloud_sar_weight=0.0,
                score_recipe="spectral_consensus",
                index_consensus_weight=0.05,
            ),
        ),
        "blend_sar_spectral_neg050_005": replace(
            base,
            bs=replace(
                base.bs,
                sar_weight=-0.05,
                cloud_sar_weight=0.0,
                score_recipe="spectral_consensus",
                index_consensus_weight=0.05,
            ),
        ),
    }
    return candidates


def choose_winner(
    results: dict[str, dict[str, object]],
    *,
    min_gain: float,
) -> tuple[str, float]:
    baseline = results.get("baseline")
    if baseline is None:
        raise ValueError("results must contain baseline")
    baseline_score = float(baseline["score"])

    winner = "baseline"
    winner_score = baseline_score
    for name in sorted(results):
        if name == "baseline":
            continue
        score = float(results[name]["score"])
        if score > baseline_score + float(min_gain) and score > winner_score:
            winner = name
            winner_score = score
    return winner, winner_score


def run(
    data_dir: str | Path,
    fold_manifest: str | Path,
    work_dir: str | Path,
    *,
    base_config: str | Path = "configs/baseline.json",
    min_gain: float = 1e-6,
    bs_max_candidates: int = 128,
    bs_passes: int = 4,
    bs_landcover_passes: int = 2,
) -> dict[str, object]:
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(fold_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("fold manifest root must be an object")

    base = load_model_config(base_config)
    results: dict[str, dict[str, object]] = {}

    for name, candidate in candidate_configs(base).items():
        candidate_dir = root / name
        oof_dir = candidate_dir / "oof"
        if oof_dir.exists():
            shutil.rmtree(oof_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        candidate_config_path = candidate_dir / "candidate_config.json"
        deployment_path = candidate_dir / "deployment_config.json"
        report_path = candidate_dir / "crossfit_report.json"
        save_model_config(candidate, candidate_config_path)

        oof_summary = generate_oof(
            data_dir,
            fold_manifest,
            oof_dir,
            model_config=candidate_config_path,
        )
        pool = load_oof_directory(oof_dir)
        deployment, report = crossfit_calibrate_and_evaluate(
            pool,
            manifest,
            candidate,
            bs_max_candidates=bs_max_candidates,
            bs_passes=bs_passes,
            bs_landcover_passes=bs_landcover_passes,
        )
        save_model_config(deployment, deployment_path)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        summary = report["crossfit"]
        score = summary["score"]
        if score is None:
            raise RuntimeError(f"{name}: cross-fitted score is unavailable")
        results[name] = {
            "score": float(score),
            "f1_af": summary["f1_af"],
            "iou_burn": summary["iou_burn"],
            "miou_severity": summary["miou_severity"],
            "candidate_config": str(candidate_config_path),
            "deployment_config": str(deployment_path),
            "crossfit_report": str(report_path),
            "oof_dir": str(oof_dir),
            "oof_summary": oof_summary,
            "bs_candidate": asdict(candidate.bs),
        }

    winner, winner_score = choose_winner(results, min_gain=min_gain)
    baseline_score = float(results["baseline"]["score"])
    selected_source = Path(str(results[winner]["deployment_config"]))
    selected = root / "selected_deployment_config.json"
    shutil.copyfile(selected_source, selected)

    report = {
        "selection_rule": "crossfit_score_above_baseline_only",
        "min_gain": float(min_gain),
        "baseline_score": baseline_score,
        "winner": winner,
        "winner_score": winner_score,
        "winner_delta": winner_score - baseline_score,
        "selected_deployment_config": str(selected),
        "candidates": results,
        "note": (
            "Land-cover thresholds are calibrated inside cross-fit. "
            "Optional spectral indices fall back to the dNBR/SAR score when "
            "their source bands are absent. Convex SAR/spectral score blends "
            "are represented exactly as deployable blended weights and pass "
            "through the same cross-fitted promotion gate."
        ),
    }
    (root / "candidate_sweep.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--work-dir", default="outputs/metric_candidates")
    parser.add_argument("--base-config", default="configs/baseline.json")
    parser.add_argument("--min-gain", type=float, default=1e-6)
    parser.add_argument("--bs-max-candidates", type=int, default=128)
    parser.add_argument("--bs-passes", type=int, default=4)
    parser.add_argument("--bs-landcover-passes", type=int, default=2)
    args = parser.parse_args()

    report = run(
        args.data_dir,
        args.fold_manifest,
        args.work_dir,
        base_config=args.base_config,
        min_gain=args.min_gain,
        bs_max_candidates=args.bs_max_candidates,
        bs_passes=args.bs_passes,
        bs_landcover_passes=args.bs_landcover_passes,
    )
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "baseline_score": report["baseline_score"],
                "winner_score": report["winner_score"],
                "winner_delta": report["winner_delta"],
                "selected_deployment_config": report["selected_deployment_config"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
