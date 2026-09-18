"""Generate deterministic baseline OOF score records for every validation fold.

The baseline score functions contain no fitted fold-specific parameters, so the
same frozen physics score can be emitted for each chip exactly once according
to the fold manifest. Thresholds are intentionally *not* fitted here; they are
cross-fitted later by scripts/evaluate_crossfit_oof.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.baselines import active_fire_score, burn_severity_score
from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.model_config import load_model_config
from wildfire.oof import OOFRecord, save_oof_record


def _validation_assignment(manifest: dict[str, object]) -> dict[str, int]:
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("fold manifest must contain at least two folds")

    assignment: dict[str, int] = {}
    for fallback_index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError("fold entries must be objects")
        fold_index = int(fold.get("fold", fallback_index))
        validation = fold.get("validation")
        if not isinstance(validation, list) or not all(
            isinstance(item, str) for item in validation
        ):
            raise ValueError("each fold must contain a validation chip list")
        for chip_id in validation:
            if chip_id in assignment:
                raise ValueError(f"chip appears in multiple validation folds: {chip_id}")
            assignment[chip_id] = fold_index
    if not assignment:
        raise ValueError("fold manifest contains no validation chips")
    return assignment


def run(
    data_dir: str | Path,
    fold_manifest: str | Path,
    output_dir: str | Path,
    *,
    model_config: str | Path = "configs/baseline.json",
) -> dict[str, object]:
    root = Path(data_dir)
    manifest = json.loads(Path(fold_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("fold manifest root must be an object")
    assignment = _validation_assignment(manifest)

    discovered = {chip.chip_id: chip for chip in discover_chips(root)}
    missing = sorted(set(assignment) - set(discovered))
    if missing:
        raise RuntimeError(
            "fold manifest references undiscovered chips: " + ", ".join(missing[:10])
        )

    config = load_model_config(model_config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written = 0
    by_task = {"AF": 0, "BS": 0}
    for chip_id in sorted(assignment):
        channels = load_channels(discovered[chip_id])
        if "TARGET" not in channels:
            raise RuntimeError(f"{chip_id}: TARGET is missing")
        task = infer_task(channels)
        target = np.asarray(channels["TARGET"])

        if task == "AF":
            score, valid = active_fire_score(channels, config)
        else:
            score, valid = burn_severity_score(channels, config)

        landcover = (
            np.asarray(channels["LANDCOVER"])
            if task == "BS" and "LANDCOVER" in channels
            else None
        )
        record = OOFRecord(
            chip_id=chip_id,
            task=task,
            score=np.asarray(score, dtype=np.float32),
            target=target,
            valid=np.asarray(valid, dtype=bool),
            landcover=landcover,
        )
        save_oof_record(record, output / f"{chip_id}.npz")
        by_task[task] += 1
        written += 1

    return {
        "records": written,
        "by_task": by_task,
        "folds": len(set(assignment.values())),
        "output_dir": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-dir", default="outputs/oof_baseline")
    parser.add_argument("--model-config", default="configs/baseline.json")
    args = parser.parse_args()

    report = run(
        args.data_dir,
        args.fold_manifest,
        args.output_dir,
        model_config=args.model_config,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
