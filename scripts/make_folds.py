"""Create deterministic organiser-group-aware OOF folds."""

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
    parser.add_argument(
        "--require-event-groups",
        action="store_true",
        help=(
            "Fail if any chip lacks an organiser-provided fire/event/group id. "
            "Use this for a strict leakage-safe final validation claim."
        ),
    )
    args = parser.parse_args()

    meta = read_meta_csv(Path(args.meta))
    manifest = build_group_folds(
        meta,
        n_splits=args.folds,
        seed=args.seed,
        require_event_groups=args.require_event_groups,
    )
    write_split_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "leakage_audit": manifest["leakage_audit"],
                "folds": [fold["summary"] for fold in manifest["folds"]],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"Saved {args.folds} group-aware folds to {args.output}")


if __name__ == "__main__":
    main()
