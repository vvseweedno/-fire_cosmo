"""Create deterministic fire_event_id-aware OOF folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.metadata import read_meta_csv
from wildfire.split import build_group_folds, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to training meta.csv")
    parser.add_argument("--output", default="splits/folds_seed42.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = read_meta_csv(Path(args.meta))
    manifest = build_group_folds(meta, n_splits=args.folds, seed=args.seed)
    write_split_manifest(manifest, args.output)
    print(
        json.dumps(
            [fold["summary"] for fold in manifest["folds"]],
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved {args.folds} leakage-safe folds to {args.output}")


if __name__ == "__main__":
    main()
