"""Stable evidence describing exactly which dataset tree produced an experiment."""

from __future__ import annotations  # noqa: I001

import hashlib
import json
from pathlib import Path

from wildfire.io import discover_chips, infer_task


_HASHED_METADATA_SUFFIXES = {".csv", ".json"}


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(
    data_dir: str | Path,
    *,
    full_hash: bool = False,
) -> dict[str, object]:
    """Return a deterministic file/chip manifest and its SHA256.

    The default avoids hashing every raster byte. Metadata/template/sidecar files
    are content-hashed, while raster identity is represented by relative path and
    size. Set full_hash=True when a full byte-level fingerprint is required.
    """

    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    files: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        entry: dict[str, object] = {
            "path": relative,
            "size": int(path.stat().st_size),
        }
        if full_hash or path.suffix.lower() in _HASHED_METADATA_SUFFIXES:
            entry["sha256"] = sha256_file(path)
        files.append(entry)

    chips = discover_chips(root)
    chip_contract: list[dict[str, object]] = []
    for chip in chips:
        try:
            task = infer_task(chip.channels)
        except ValueError:
            task = "UNKNOWN"
        chip_contract.append(
            {
                "chip_id": chip.chip_id,
                "task": task,
                "channels": sorted(chip.channels),
            }
        )

    canonical = {
        "version": 1,
        "files": files,
        "chip_contract": chip_contract,
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        **canonical,
        "root": str(root),
        "full_hash": full_hash,
        "file_count": len(files),
        "total_bytes": int(sum(int(item["size"]) for item in files)),
        "chip_count": len(chips),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }
