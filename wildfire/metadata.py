"""Read official meta.csv while keeping train-only fields separate from test usage."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChipMeta:
    chip_id: str
    kind: str
    width: int
    height: int
    gsd: float
    fire_event_id: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def split_group(self) -> str:
        """Keep known fire/event groups intact; isolated chips remain identifiable."""
        if self.fire_event_id:
            return f"event:{self.fire_event_id}"
        return f"chip:{self.chip_id}"


def _first_group_id(row: dict[str, str | None]) -> str | None:
    """Accept common organiser-provided grouping column names without guessing."""
    for key in ("fire_event_id", "event_id", "incident_id", "group_id"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return None


def read_meta_csv(path: str | Path) -> dict[str, ChipMeta]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    result: dict[str, ChipMeta] = {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chip_id", "kind", "width", "height", "gsd"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"meta.csv missing columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            chip_id = (row.get("chip_id") or "").strip()
            kind = (row.get("kind") or "").strip().lower()
            if not chip_id:
                raise ValueError(f"meta.csv line {line_no}: empty chip_id")
            if chip_id in result:
                raise ValueError(f"meta.csv line {line_no}: duplicate chip_id {chip_id}")
            if kind not in {"af", "bs"}:
                raise ValueError(f"meta.csv line {line_no}: invalid kind {kind!r}")

            result[chip_id] = ChipMeta(
                chip_id=chip_id,
                kind=kind,
                width=int(row["width"]),
                height=int(row["height"]),
                gsd=float(row["gsd"]),
                fire_event_id=_first_group_id(row),
            )
    return result
