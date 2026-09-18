"""Stream a compact EDA profile without loading the full dataset into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wildfire.io import discover_chips, infer_task, load_channels


def _finite_stats(array: np.ndarray) -> dict[str, object]:
    arr = np.asarray(array)
    finite = arr[np.isfinite(arr)]
    result: dict[str, object] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "pixels": int(arr.size),
        "finite_pixels": int(finite.size),
        "nan_or_inf_pixels": int(arr.size - finite.size),
    }
    if finite.size:
        result.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite, dtype=np.float64)),
            }
        )
    return result


def run(data_dir: str | Path) -> dict[str, object]:
    chips = discover_chips(data_dir)
    task_counts: Counter[str] = Counter()
    channel_presence: Counter[str] = Counter()
    shapes: dict[str, Counter[str]] = defaultdict(Counter)
    target_counts: dict[str, Counter[int]] = defaultdict(Counter)
    sample_stats: dict[str, dict[str, object]] = {}

    for chip in chips:
        channels = load_channels(chip)
        try:
            task = infer_task(channels)
        except ValueError:
            task = "UNKNOWN"
        task_counts[task] += 1

        for name, array in channels.items():
            channel_presence[name] += 1
            shapes[name][str(tuple(array.shape))] += 1
            sample_stats.setdefault(name, _finite_stats(array))

        if "TARGET" in channels and task != "UNKNOWN":
            target = np.asarray(channels["TARGET"])
            values, counts = np.unique(target[np.isfinite(target)], return_counts=True)
            for value, count in zip(values, counts, strict=True):
                if float(value).is_integer():
                    target_counts[task][int(value)] += int(count)

    return {
        "chips": len(chips),
        "tasks": dict(task_counts),
        "channel_presence": dict(sorted(channel_presence.items())),
        "channel_shapes": {
            name: dict(counter) for name, counter in sorted(shapes.items())
        },
        "sample_channel_stats": dict(sorted(sample_stats.items())),
        "target_class_pixels": {
            task: {str(key): value for key, value in sorted(counter.items())}
            for task, counter in sorted(target_counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="outputs/dataset_profile.json")
    args = parser.parse_args()

    report = run(args.data_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved profile to {output}")


if __name__ == "__main__":
    main()
