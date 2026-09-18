"""Generate aligned AF OOF score candidates from organiser-provided channels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.af_candidates import AF_CANDIDATE_NAMES, active_fire_score_candidates
from wildfire.io import discover_chips, infer_task, load_channels
from wildfire.model_config import load_model_config
from wildfire.oof import OOFRecord, save_oof_record


def _validation_chip_ids(manifest: dict[str, object]) -> set[str]:
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("fold manifest must contain at least two folds")

    result: set[str] = set()
    for fold in folds:
        if not isinstance(fold, dict):
            raise ValueError("fold entries must be objects")
        validation = fold.get("validation")
        if not isinstance(validation, list) or not all(
            isinstance(item, str) for item in validation
        ):
            raise ValueError("each fold must contain validation chip ids")
        for chip_id in validation:
            if chip_id in result:
                raise ValueError(f"chip appears in multiple validation folds: {chip_id}")
            result.add(chip_id)
    return result


def run(
    data_dir: str | Path,
    fold_manifest: str | Path,
    output_root: str | Path,
    *,
    model_config: str | Path = "configs/baseline.json",
) -> dict[str, object]:
    root = Path(data_dir)
    manifest_payload = json.loads(Path(fold_manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ValueError("fold manifest root must be an object")
    validation_ids = _validation_chip_ids(manifest_payload)

    discovered = {chip.chip_id: chip for chip in discover_chips(root)}
    missing = sorted(validation_ids - set(discovered))
    if missing:
        raise RuntimeError(
            "fold manifest references undiscovered chips: " + ", ".join(missing[:10])
        )

    config = load_model_config(model_config)
    output = Path(output_root)
    counts = {name: 0 for name in AF_CANDIDATE_NAMES}
    af_chips = 0

    for chip_id in sorted(validation_ids):
        channels = load_channels(discovered[chip_id])
        if infer_task(channels) != "AF":
            continue
        if "TARGET" not in channels:
            raise RuntimeError(f"{chip_id}: TARGET is missing")

        candidates, valid = active_fire_score_candidates(channels, config)
        target = np.asarray(channels["TARGET"])

        for name in AF_CANDIDATE_NAMES:
            record = OOFRecord(
                chip_id=chip_id,
                task="AF",
                score=np.asarray(candidates[name], dtype=np.float32),
                target=target,
                valid=np.asarray(valid, dtype=bool),
            )
            save_oof_record(record, output / name / f"{chip_id}.npz")
            counts[name] += 1
        af_chips += 1

    if af_chips == 0:
        raise RuntimeError("no AF chips were found in the validation folds")

    return {
        "af_chips": af_chips,
        "candidate_names": list(AF_CANDIDATE_NAMES),
        "records_by_candidate": counts,
        "output_root": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-root", default="outputs/oof_af_candidates")
    parser.add_argument("--model-config", default="configs/baseline.json")
    args = parser.parse_args()

    report = run(
        args.data_dir,
        args.fold_manifest,
        args.output_root,
        model_config=args.model_config,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
