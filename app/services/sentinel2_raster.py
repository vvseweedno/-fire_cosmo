"""Real Sentinel-2 L2A raster processing for burned-area mapping.

The processor works directly with STAC COG asset URLs. B12 (20 m) is the
reference grid; B08 is reprojected to it and SCL is resampled with nearest
neighbour. Cloud/shadow/snow/no-data pixels are excluded before dNBR.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.schemas import BurnedAreaResult, Sentinel2Scene, SeveritySummary

logger = logging.getLogger(__name__)


@dataclass
class RasterAnalysis:
    geojson: dict[str, Any]
    result: BurnedAreaResult
    provenance: dict[str, Any]


class Sentinel2RasterProcessor:
    """Compute dNBR and burn severity from real Sentinel-2 L2A STAC assets."""

    # Sentinel-2 Scene Classification Layer values to exclude:
    # 0 no-data, 1 saturated/defective, 3 cloud shadow, 8/9 clouds,
    # 10 cirrus, 11 snow/ice.
    INVALID_SCL = {0, 1, 3, 8, 9, 10, 11}

    def __init__(self, low: float = 0.10, moderate: float = 0.27, high: float = 0.44):
        self.low = low
        self.moderate = moderate
        self.high = high

    @staticmethod
    def _nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        nir = nir.astype("float32")
        swir = swir.astype("float32")
        denominator = nir + swir
        out = np.full(nir.shape, np.nan, dtype="float32")
        valid = denominator != 0
        out[valid] = (nir[valid] - swir[valid]) / denominator[valid]
        return np.clip(out, -1.0, 1.0)

    @staticmethod
    def _require_assets(scene: Sentinel2Scene) -> None:
        missing = {"B08", "B12"} - set(scene.assets)
        if missing:
            raise ValueError(f"Scene {scene.scene_id} misses required assets: {sorted(missing)}")

    @staticmethod
    def _event_bounds(center_lon: float, center_lat: float, half_size_deg: float) -> tuple[float, float, float, float]:
        return (
            center_lon - half_size_deg,
            center_lat - half_size_deg,
            center_lon + half_size_deg,
            center_lat + half_size_deg,
        )

    def _read_scene(
        self,
        scene: Sentinel2Scene,
        bounds_wgs84: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any, Any]:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, transform_bounds
        from rasterio.windows import from_bounds

        self._require_assets(scene)

        # B12 is native 20 m and is used as the common grid. This avoids
        # implying 10 m information that is not present in SWIR2.
        with rasterio.open(scene.assets["B12"]) as swir_src:
            projected_bounds = transform_bounds("EPSG:4326", swir_src.crs, *bounds_wgs84, densify_pts=21)
            window = from_bounds(*projected_bounds, transform=swir_src.transform).round_offsets().round_lengths()
            window = window.intersection(rasterio.windows.Window(0, 0, swir_src.width, swir_src.height))
            if window.width <= 0 or window.height <= 0:
                raise ValueError(f"Event window does not intersect scene {scene.scene_id}")
            swir = swir_src.read(1, window=window, masked=True).astype("float32")
            target_transform = swir_src.window_transform(window)
            target_crs = swir_src.crs
            target_shape = swir.shape

        nir = np.full(target_shape, np.nan, dtype="float32")
        with rasterio.open(scene.assets["B08"]) as nir_src:
            reproject(
                source=rasterio.band(nir_src, 1),
                destination=nir,
                src_transform=nir_src.transform,
                src_crs=nir_src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )

        scl: np.ndarray | None = None
        if scene.assets.get("SCL"):
            scl = np.full(target_shape, 0, dtype="uint8")
            with rasterio.open(scene.assets["SCL"]) as scl_src:
                reproject(
                    source=rasterio.band(scl_src, 1),
                    destination=scl,
                    src_transform=scl_src.transform,
                    src_crs=scl_src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0,
                )

        swir_data = np.asarray(swir.filled(np.nan), dtype="float32") if np.ma.isMaskedArray(swir) else swir
        return nir, swir_data, scl, target_transform, target_crs

    def _valid_mask(self, nir: np.ndarray, swir: np.ndarray, scl: np.ndarray | None) -> np.ndarray:
        valid = np.isfinite(nir) & np.isfinite(swir) & ((nir + swir) != 0)
        if scl is not None:
            invalid = np.isin(scl, list(self.INVALID_SCL))
            valid &= ~invalid
        return valid

    def _classify(self, dnbr: np.ndarray, valid: np.ndarray) -> np.ndarray:
        severity = np.full(dnbr.shape, 255, dtype="uint8")
        severity[valid & (dnbr < self.low)] = 0
        severity[valid & (dnbr >= self.low) & (dnbr < self.moderate)] = 1
        severity[valid & (dnbr >= self.moderate) & (dnbr < self.high)] = 2
        severity[valid & (dnbr >= self.high)] = 3
        return severity

    @staticmethod
    def _pixel_area_m2(transform: Any) -> float:
        # Sentinel-2 tiles are projected; determinant handles north-up and
        # rotated affine transforms.
        return abs(transform.a * transform.e - transform.b * transform.d)

    def _polygonize(self, severity: np.ndarray, transform: Any, crs: Any, event_id: str) -> dict[str, Any]:
        from rasterio.features import shapes
        from rasterio.warp import transform_geom

        labels = {1: "low_severity", 2: "moderate_severity", 3: "high_severity"}
        features: list[dict[str, Any]] = []
        for geom, value in shapes(severity, mask=(severity > 0) & (severity < 255), transform=transform, connectivity=8):
            class_value = int(value)
            if class_value not in labels:
                continue
            geom_wgs84 = transform_geom(crs, "EPSG:4326", geom, precision=6)
            features.append({
                "type": "Feature",
                "geometry": geom_wgs84,
                "properties": {
                    "event_id": event_id,
                    "class_value": class_value,
                    "severity": labels[class_value],
                    "method": "sentinel2_l2a_dnbr",
                },
            })
        return {"type": "FeatureCollection", "features": features}

    def process_pair(
        self,
        pre_scene: Sentinel2Scene,
        post_scene: Sentinel2Scene,
        event_id: str,
        center_lon: float,
        center_lat: float,
        half_size_deg: float = 0.15,
    ) -> RasterAnalysis:
        """Process one real pre/post pair. A pre-fire scene is mandatory for dNBR."""
        bounds = self._event_bounds(center_lon, center_lat, half_size_deg)
        pre_nir, pre_swir, pre_scl, transform, crs = self._read_scene(pre_scene, bounds)
        post_nir, post_swir, post_scl, post_transform, post_crs = self._read_scene(post_scene, bounds)

        if pre_nir.shape != post_nir.shape or str(crs) != str(post_crs) or transform != post_transform:
            raise ValueError("Pre/post Sentinel-2 windows are not aligned to the same tile/grid")

        valid = self._valid_mask(pre_nir, pre_swir, pre_scl) & self._valid_mask(post_nir, post_swir, post_scl)
        pre_nbr = self._nbr(pre_nir, pre_swir)
        post_nbr = self._nbr(post_nir, post_swir)
        dnbr = pre_nbr - post_nbr
        valid &= np.isfinite(dnbr)
        severity = self._classify(dnbr, valid)

        pixel_area_m2 = self._pixel_area_m2(transform)
        area_for = lambda value: float(np.count_nonzero(severity == value) * pixel_area_m2 / 10000.0)
        low_ha, moderate_ha, high_ha = area_for(1), area_for(2), area_for(3)
        total_ha = low_ha + moderate_ha + high_ha
        unburned_ha = area_for(0)

        geojson = self._polygonize(severity, transform, crs, event_id)
        result = BurnedAreaResult(
            event_id=event_id,
            area_ha=round(total_ha, 2),
            area_m2=round(total_ha * 10000.0, 2),
            projection_used=str(crs),
            method="sentinel2_l2a_dnbr_pixel_count",
            uncertainty="medium",
            severity_summary=SeveritySummary(
                unburned_ha=round(unburned_ha, 2),
                low_severity_ha=round(low_ha, 2),
                moderate_severity_ha=round(moderate_ha, 2),
                high_severity_ha=round(high_ha, 2),
                total_affected_ha=round(total_ha, 2),
            ),
        )
        provenance = {
            "event_id": event_id,
            "method": "dNBR = NBR(pre) - NBR(post)",
            "common_grid": "Sentinel-2 B12 native grid (~20 m)",
            "cloud_mask": "SCL classes 0,1,3,8,9,10,11 excluded when available",
            "pre_scene": pre_scene.model_dump(mode="json"),
            "post_scene": post_scene.model_dump(mode="json"),
            "analysis_bounds_wgs84": list(bounds),
            "valid_pixels": int(np.count_nonzero(valid)),
            "affected_pixels": int(np.count_nonzero((severity >= 1) & (severity <= 3))),
        }
        return RasterAnalysis(geojson=geojson, result=result, provenance=provenance)
