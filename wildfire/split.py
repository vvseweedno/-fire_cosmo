"""Deterministic group-aware splitting and balanced OOF folds."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from wildfire.metadata import ChipMeta


def _unit_hash(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{seed}|{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64)


def _grouped(meta: Mapping[str, ChipMeta]) -> dict[str, list[ChipMeta]]:
    groups: dict[str, list[ChipMeta]] = defaultdict(list)
    for item in meta.values():
        groups[item.split_group].append(item)
    return groups


def _leakage_audit(meta: Mapping[str, ChipMeta]) -> dict[str, object]:
    grouped = sum(item.fire_event_id is not None for item in meta.values())
    fallback = len(meta) - grouped
    return {
        "event_grouped_chips": grouped,
        "chip_fallbacks": fallback,
        "event_group_coverage": grouped / len(meta) if meta else 0.0,
        "strict_event_grouping": fallback == 0,
    }


def _require_grouping(meta: Mapping[str, ChipMeta]) -> None:
    missing = sorted(item.chip_id for item in meta.values() if not item.fire_event_id)
    if missing:
        raise ValueError(
            "strict leakage-safe split requested, but organiser metadata has no "
            f"event/group id for {len(missing)} chips; examples={missing[:10]}. "
            "Do not silently call chip-level fallback leakage-safe."
        )


def build_group_split(
    meta: Mapping[str, ChipMeta],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    require_event_groups: bool = False,
) -> dict[str, object]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not meta:
        raise ValueError("meta is empty")
    if require_event_groups:
        _require_grouping(meta)

    groups = _grouped(meta)
    group_partition: dict[str, str] = {}
    for group in sorted(groups):
        group_partition[group] = (
            "validation"
            if _unit_hash(seed, group) < validation_fraction
            else "train"
        )

    if len(groups) > 1:
        if all(partition == "train" for partition in group_partition.values()):
            group = min(groups, key=lambda key: _unit_hash(seed, key))
            group_partition[group] = "validation"
        if all(partition == "validation" for partition in group_partition.values()):
            group = max(groups, key=lambda key: _unit_hash(seed, key))
            group_partition[group] = "train"

    train: list[str] = []
    validation: list[str] = []
    for group, items in groups.items():
        target = validation if group_partition[group] == "validation" else train
        target.extend(item.chip_id for item in items)

    train.sort()
    validation.sort()

    def _kind_counts(chip_ids: list[str]) -> dict[str, int]:
        counts = Counter(meta[chip_id].kind for chip_id in chip_ids)
        return {key: counts.get(key, 0) for key in ("af", "bs")}

    return {
        "version": 2,
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "group_key": "organiser event/group id; chip_id fallback when missing",
        "leakage_audit": _leakage_audit(meta),
        "train": train,
        "validation": validation,
        "summary": {
            "chips_total": len(meta),
            "groups_total": len(groups),
            "train_chips": len(train),
            "validation_chips": len(validation),
            "train_by_kind": _kind_counts(train),
            "validation_by_kind": _kind_counts(validation),
        },
    }


def build_group_folds(
    meta: Mapping[str, ChipMeta],
    *,
    n_splits: int = 5,
    seed: int = 42,
    require_event_groups: bool = False,
) -> dict[str, object]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if not meta:
        raise ValueError("meta is empty")
    if require_event_groups:
        _require_grouping(meta)

    groups = _grouped(meta)
    if len(groups) < n_splits:
        raise ValueError("number of groups must be >= n_splits")

    total = len(meta)
    total_af = sum(item.kind == "af" for item in meta.values())
    total_bs = total - total_af
    target_total = total / n_splits
    target_af = max(total_af / n_splits, 1.0)
    target_bs = max(total_bs / n_splits, 1.0)

    group_stats: list[tuple[str, int, int, int]] = []
    for group, items in groups.items():
        af = sum(item.kind == "af" for item in items)
        bs = len(items) - af
        group_stats.append((group, len(items), af, bs))

    group_stats.sort(
        key=lambda row: (
            -row[1],
            -max(row[2], row[3]),
            _unit_hash(seed, row[0]),
        )
    )

    fold_state = [
        {"groups": [], "chips": 0, "af": 0, "bs": 0}
        for _ in range(n_splits)
    ]

    for group, chips, af, bs in group_stats:
        def _cost(index: int) -> tuple[float, int, int]:
            fold = fold_state[index]
            imbalance = (
                ((fold["chips"] + chips) / target_total) ** 2
                + ((fold["af"] + af) / target_af) ** 2
                + ((fold["bs"] + bs) / target_bs) ** 2
            )
            return (imbalance, int(fold["chips"]), index)

        chosen = min(range(n_splits), key=_cost)
        fold = fold_state[chosen]
        fold["groups"].append(group)
        fold["chips"] += chips
        fold["af"] += af
        fold["bs"] += bs

    all_chip_ids = set(meta)
    folds: list[dict[str, object]] = []
    for index, state in enumerate(fold_state):
        validation = sorted(
            item.chip_id
            for group in state["groups"]
            for item in groups[group]
        )
        train = sorted(all_chip_ids - set(validation))
        folds.append(
            {
                "fold": index,
                "train": train,
                "validation": validation,
                "summary": {
                    "validation_chips": len(validation),
                    "validation_af": int(state["af"]),
                    "validation_bs": int(state["bs"]),
                    "validation_groups": len(state["groups"]),
                },
            }
        )

    return {
        "version": 2,
        "seed": seed,
        "n_splits": n_splits,
        "group_key": "organiser event/group id; chip_id fallback when missing",
        "leakage_audit": _leakage_audit(meta),
        "folds": folds,
    }


def write_split_manifest(manifest: Mapping[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_split_manifest(path: str | Path) -> dict[str, object]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    for key in ("train", "validation"):
        value = payload.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Split manifest field {key!r} must be a list of chip ids")
    overlap = set(payload["train"]) & set(payload["validation"])
    if overlap:
        raise ValueError(f"Split manifest has overlapping chips: {sorted(overlap)[:5]}")
    return payload
