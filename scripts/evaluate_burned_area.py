#!/usr/bin/env python3
"""Evaluate predicted burned-area GeoJSON against an independent reference.

Usage:
    python scripts/evaluate_burned_area.py predicted.geojson reference.geojson

Both inputs are reprojected to EPSG:6933 before area metrics are computed.
The script reports IoU, Dice, predicted/reference area and absolute/relative
area error. It does not assume the reference is perfect ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd


def _load_union(path: str):
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No geometries in {path}")
    if gdf.crs is None:
        # GeoJSON coordinates are conventionally WGS84.
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].to_crs("EPSG:6933")
    if gdf.empty:
        raise ValueError(f"No valid geometries in {path}")
    return gdf.geometry.union_all()


def evaluate(predicted_path: str, reference_path: str) -> dict:
    predicted = _load_union(predicted_path)
    reference = _load_union(reference_path)

    intersection_m2 = predicted.intersection(reference).area
    union_m2 = predicted.union(reference).area
    predicted_m2 = predicted.area
    reference_m2 = reference.area

    iou = intersection_m2 / union_m2 if union_m2 else 1.0
    dice_denominator = predicted_m2 + reference_m2
    dice = 2 * intersection_m2 / dice_denominator if dice_denominator else 1.0
    error_ha = abs(predicted_m2 - reference_m2) / 10000.0
    relative_error_pct = (
        abs(predicted_m2 - reference_m2) / reference_m2 * 100.0
        if reference_m2
        else None
    )

    return {
        "predicted_area_ha": round(predicted_m2 / 10000.0, 3),
        "reference_area_ha": round(reference_m2 / 10000.0, 3),
        "intersection_area_ha": round(intersection_m2 / 10000.0, 3),
        "iou": round(iou, 4),
        "dice": round(dice, 4),
        "absolute_area_error_ha": round(error_ha, 3),
        "relative_area_error_pct": round(relative_error_pct, 2) if relative_error_pct is not None else None,
        "projection": "EPSG:6933",
        "note": "Metrics compare prediction with the supplied independent reference; reference quality must be documented separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predicted", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(str(args.predicted), str(args.reference))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
