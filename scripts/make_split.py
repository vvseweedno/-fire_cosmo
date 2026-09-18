"""Create a deterministic fire_event_id-aware split manifest from train/meta.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.metadata import read_meta_csv
from wildfire.split import build_group_split, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to training meta.csv")
    parser.add_argument("--output", default="splits/seed42.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = read_meta_csv(Path(args.meta))
    manifest = build_group_split(
        meta,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    write_split_manifest(manifest, args.output)
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    print(f"Saved fixed split to {args.output}")


if __name__ == "__main__":
    main()
