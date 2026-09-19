"""Inspect organiser monitoring AOI without geospatial heavyweight dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wildfire.aoi import load_monitoring_aoi, monitoring_aoi_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--feature-id", default="aoi")
    parser.add_argument("--output")
    args = parser.parse_args()

    aoi = load_monitoring_aoi(args.geojson, feature_id=args.feature_id)
    summary = monitoring_aoi_summary(aoi)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
