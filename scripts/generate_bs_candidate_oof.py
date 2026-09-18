"""Generate aligned BS OOF score candidates from organiser-provided channels.

Each candidate is deterministic and label-free. Labels are stored only so the
separate OOF optimizer can compare candidates under the official metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wildfire.bs_candidates import BS_CANDIDATE_NAMES, burn_score_candidates
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
    counts = {name: 0 for name in BS_CANDIDATE_NAMES}
    bs_chips = 0

    for chip_id in sorted(validation_ids):
        channels = load_channels(discovered[chip_id])
        if infer_task(channels) != "BS":
            continue
        if "TARGET" not in channels:
            raise RuntimeError(f"{chip_id}: TARGET is missing")

        candidates, valid = burn_score_candidates(channels, config)
        target = np.asarray(channels["TARGET"])
        landcover = (
            np.asarray(channels["LANDCOVER"])
            if "LANDCOVER" in channels
            else None
        )

        for name in BS_CANDIDATE_NAMES:
            record = OOFRecord(
                chip_id=chip_id,
                task="BS",
                score=np.asarray(candidates[name], dtype=np.float32),
                target=target,
                valid=np.asarray(valid, dtype=bool),
                landcover=landcover,
            )
            save_oof_record(record, output / name / f"{chip_id}.npz")
            counts[name] += 1
        bs_chips += 1

    if bs_chips == 0:
        raise RuntimeError("no BS chips were found in the validation folds")

    return {
        "bs_chips": bs_chips,
        "candidate_names": list(BS_CANDIDATE_NAMES),
        "records_by_candidate": counts,
        "output_root": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--fold-manifest", required=True)
    parser.add_argument("--output-root", default="outputs/oof_bs_candidates")
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
