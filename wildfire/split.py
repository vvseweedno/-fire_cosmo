"""Deterministic group-aware train/validation splitting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from wildfire.metadata import ChipMeta


def _unit_hash(seed: int, group: str) -> float:
    digest = hashlib.sha256(f"{seed}|{group}".encode()).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64)


def build_group_split(
    meta: Mapping[str, ChipMeta],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, object]:
    """Split whole fire-event groups deterministically.

    fire_event_id is the organiser-provided anti-leakage group. If a row has no
    event id (for example an isolated negative chip), that chip becomes a group
    of its own.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not meta:
        raise ValueError("meta is empty")

    groups: dict[str, list[ChipMeta]] = defaultdict(list)
    for item in meta.values():
        groups[item.split_group].append(item)

    group_partition: dict[str, str] = {}
    for group in sorted(groups):
        group_partition[group] = (
            "validation"
            if _unit_hash(seed, group) < validation_fraction
            else "train"
        )

    # Degenerate safeguards while still moving whole groups only.
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
        "version": 1,
        "seed": seed,
        "validation_fraction_requested": validation_fraction,
        "group_key": "fire_event_id (fallback: chip_id when missing)",
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
