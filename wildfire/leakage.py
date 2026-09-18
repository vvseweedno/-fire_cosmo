"""Explicit leakage checks for official-data cross-validation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from wildfire.io import ChannelSource, discover_chips
from wildfire.metadata import ChipMeta, read_meta_csv


def _source_identity(source: ChannelSource, root: Path) -> str:
    try:
        path = source.path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        path = source.path.resolve().as_posix()
    return "|".join(
        (
            path,
            f"band={source.band if source.band is not None else ''}",
            f"key={source.key if source.key is not None else ''}",
        )
    )


def _audit_folds(
    meta: dict[str, ChipMeta],
    fold_manifest: dict[str, object],
) -> dict[str, object]:
    folds = fold_manifest.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("fold manifest must contain at least two folds")

    validation_owner: dict[str, int] = {}
    duplicate_validation: set[str] = set()
    overlaps: list[dict[str, object]] = []
    event_folds: dict[str, set[int]] = defaultdict(set)

    for fallback_index, fold in enumerate(folds):
        if not isinstance(fold, dict):
            raise ValueError("fold entries must be objects")
        fold_id = int(fold.get("fold", fallback_index))
        train = fold.get("train")
        validation = fold.get("validation")
        if not isinstance(train, list) or not isinstance(validation, list):
            raise ValueError("fold train and validation entries must be lists")
        if not all(isinstance(item, str) for item in train + validation):
            raise ValueError("fold chip ids must be strings")

        overlap = sorted(set(train) & set(validation))
        if overlap:
            overlaps.append({"fold": fold_id, "chips": overlap})

        for chip_id in validation:
            if chip_id in validation_owner:
                duplicate_validation.add(chip_id)
            validation_owner[chip_id] = fold_id
            item = meta.get(chip_id)
            if item is None:
                continue
            if item.fire_event_id:
                event_folds[item.fire_event_id].add(fold_id)

    expected = set(meta)
    validation_ids = set(validation_owner)
    missing_validation = sorted(expected - validation_ids)
    unexpected_validation = sorted(validation_ids - expected)
    spanning = {
        event_id: sorted(fold_ids)
        for event_id, fold_ids in sorted(event_folds.items())
        if len(fold_ids) > 1
    }

    return {
        "duplicate_validation_chips": sorted(duplicate_validation),
        "train_validation_overlaps": overlaps,
        "missing_validation_chips": missing_validation,
        "unexpected_validation_chips": unexpected_validation,
        "events_spanning_validation_folds": spanning,
    }


def audit_leakage(
    data_dir: str | Path,
    fold_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    """Audit grouping and source-level contamination risks.

    Without a fold manifest the audit cannot establish cross-fold safety and
    therefore reports BLOCKED rather than PASS.
    """

    root = Path(data_dir)
    meta = read_meta_csv(root / "meta.csv")
    chips = discover_chips(root)
    discovered = {chip.chip_id: chip for chip in chips}

    missing_rasters = sorted(set(meta) - set(discovered))
    unexpected_rasters = sorted(set(discovered) - set(meta))
    missing_groups = sorted(
        item.chip_id for item in meta.values() if not item.fire_event_id
    )

    source_owners: dict[str, set[str]] = defaultdict(set)
    for chip in chips:
        for source in chip.channels.values():
            source_owners[_source_identity(source, root)].add(chip.chip_id)
    reused_sources = {
        source: sorted(owners)
        for source, owners in sorted(source_owners.items())
        if len(owners) > 1
    }

    fold_report: dict[str, object] | None = None
    if fold_manifest is not None:
        fold_report = _audit_folds(meta, fold_manifest)

    checks: dict[str, bool | None] = {
        "all_metadata_chips_discovered": not missing_rasters,
        "no_unexpected_discovered_chips": not unexpected_rasters,
        "strict_event_grouping": not missing_groups,
        "no_cross_chip_source_reuse": not reused_sources,
        "validation_coverage_exact": None,
        "no_duplicate_validation_assignment": None,
        "no_train_validation_overlap": None,
        "no_event_cross_fold_leakage": None,
    }

    if fold_report is not None:
        checks.update(
            {
                "validation_coverage_exact": (
                    not fold_report["missing_validation_chips"]
                    and not fold_report["unexpected_validation_chips"]
                ),
                "no_duplicate_validation_assignment": (
                    not fold_report["duplicate_validation_chips"]
                ),
                "no_train_validation_overlap": (
                    not fold_report["train_validation_overlaps"]
                ),
                "no_event_cross_fold_leakage": (
                    not fold_report["events_spanning_validation_folds"]
                ),
            }
        )

    definite_failures = [key for key, value in checks.items() if value is False]
    blocked = [key for key, value in checks.items() if value is None]
    status = "FAIL" if definite_failures else ("BLOCKED" if blocked else "PASS")

    return {
        "status": status,
        "pass": status == "PASS",
        "checks": checks,
        "failures": definite_failures,
        "blocked_checks": blocked,
        "chips": len(chips),
        "metadata_chips": len(meta),
        "missing_rasters": missing_rasters,
        "unexpected_rasters": unexpected_rasters,
        "chips_missing_event_group": missing_groups,
        "reused_channel_sources": reused_sources,
        "folds": fold_report,
    }
