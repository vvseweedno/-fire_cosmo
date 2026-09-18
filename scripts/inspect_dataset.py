"""Inspect official files without assuming an unpublished archive layout."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

from wildfire.io import SUPPORTED_SUFFIXES, discover_chips, infer_task, load_channels


def _probe_raster(path: Path) -> dict[str, object]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            return {
                "format": "npy",
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                return {
                    "format": "npz",
                    "arrays": {
                        key: {
                            "shape": list(payload[key].shape),
                            "dtype": str(payload[key].dtype),
                        }
                        for key in payload.files
                    },
                }
        if suffix in {".tif", ".tiff"}:
            with rasterio.open(path) as src:
                return {
                    "format": "geotiff",
                    "shape": [src.count, src.height, src.width],
                    "dtypes": list(src.dtypes),
                }
    except Exception as exc:  # diagnostic script: report, do not hide the file
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"format": suffix.lstrip(".") or "unknown"}


def _csv_header(path: Path) -> list[str] | dict[str, str]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return next(csv.reader(handle), [])
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def probe_dataset(data_dir: str | Path, max_files: int = 100) -> dict[str, object]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(root)

    files = sorted(path for path in root.rglob("*") if path.is_file())
    suffix_counts = Counter(path.suffix.lower() or "<none>" for path in files)

    raster_files = [
        path for path in files if path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    csv_files = [path for path in files if path.suffix.lower() == ".csv"]

    raw_rasters = []
    for path in raster_files[:max_files]:
        raw_rasters.append(
            {
                "path": str(path.relative_to(root)),
                **_probe_raster(path),
            }
        )

    csv_headers = {
        str(path.relative_to(root)): _csv_header(path)
        for path in csv_files[:max_files]
    }

    chips = discover_chips(root)
    recognised: list[dict[str, object]] = []
    task_counts: Counter[str] = Counter()
    for chip in chips[:max_files]:
        channels = load_channels(chip)
        try:
            task = infer_task(channels)
        except ValueError:
            task = "UNKNOWN"
        task_counts[task] += 1
        recognised.append(
            {
                "chip_id": chip.chip_id,
                "task": task,
                "channels": sorted(channels),
                "shapes": {
                    name: list(array.shape) for name, array in channels.items()
                },
            }
        )

    return {
        "root": str(root),
        "files_total": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "raster_files_total": len(raster_files),
        "csv_files_total": len(csv_files),
        "raw_rasters_shown": raw_rasters,
        "csv_headers": csv_headers,
        "recognised_chips_total": len(chips),
        "recognised_chips_shown": recognised,
        "recognised_task_counts_shown": dict(task_counts),
        "truncated": len(raster_files) > max_files or len(csv_files) > max_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = probe_dataset(args.data_dir, max_files=args.max_files)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved inspection report to {output}")

    if report["recognised_chips_total"] == 0 and report["raster_files_total"]:
        print(
            "No current channel aliases matched the archive. "
            "Use raw_rasters_shown shapes/keys to adapt wildfire/io.py before inference."
        )


if __name__ == "__main__":
    main()
