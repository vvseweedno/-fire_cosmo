"""Real Sentinel-2 L2A raster processing for burned-area mapping.

The processor works directly with STAC COG asset URLs. B12 (20 m) is the
reference grid. B8A (20 m) is preferred for NIR; B08 (10 m) is a fallback.
STAC/GDAL raster scale and offset are applied before NBR. SCL excludes invalid,
shadow, water, cloud, cirrus and snow/ice pixels before dNBR.
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

    # Sentinel-2 L2A Scene Classification Layer classes excluded from burn mapping:
    # 0 NO_DATA, 1 SATURATED/DEFECTIVE, 2 CAST_SHADOWS, 3 CLOUD_SHADOWS,
    # 6 WATER, 8/9 CLOUDS, 10 CIRRUS, 11 SNOW/ICE.
    INVALID_SCL = {0, 1, 2, 3, 6, 8, 9, 10, 11}

    def __init__(self, low: float = 0.10, moderate: float = 0.27, high: float = 0.44):
        if not (low < moderate < high):
            raise ValueError("Severity thresholds must be strictly increasing")
        self.low = low
        self.moderate = moderate
        self.high = high

    @staticmethod
    def _nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        nir = nir.astype("float32")
        swir = swir.astype("float32")
        denominator = nir + swir
        out = np.full(nir.shape, np.nan, dtype="float32")
        valid = np.isfinite(nir) & np.isfinite(swir) & (denominator != 0)
        out[valid] = (nir[valid] - swir[valid]) / denominator[valid]
        return np.clip(out, -1.0, 1.0)

    @staticmethod
    def _nir_key(scene: Sentinel2Scene) -> str:
        if "B8A" in scene.assets:
            return "B8A"
        if "B08" in scene.assets:
            return "B08"
        raise ValueError(f"Scene {scene.scene_id} misses B8A/B08 NIR asset")

    @classmethod
    def _require_assets(cls, scene: Sentinel2Scene) -> None:
        if "B12" not in scene.assets:
            raise ValueError(f"Scene {scene.scene_id} misses required B12 asset")
        cls._nir_key(scene)

    @staticmethod
    def _event_bounds(
        center_lon: float, center_lat: float, half_size_deg: float
    ) -> tuple[float, float, float, float]:
        return (
            center_lon - half_size_deg,
            center_lat - half_size_deg,
            center_lon + half_size_deg,
            center_lat + half_size_deg,
        )

    @staticmethod
    def _calibration(scene: Sentinel2Scene, key: str, src: Any) -> tuple[float, float, float | None]:
        metadata = scene.asset_metadata.get(key, {})
        scale = metadata.get("scale")
        offset = metadata.get("offset")
        nodata = metadata.get("nodata")

        if scale is None:
            scale = src.scales[0] if getattr(src, "scales", None) else 1.0
        if offset is None:
            offset = src.offsets[0] if getattr(src, "offsets", None) else 0.0
        if nodata is None:
            nodata = src.nodata

        return float(scale or 1.0), float(offset or 0.0), None if nodata is None else float(nodata)

    @staticmethod
    def _apply_calibration(
        raw: np.ndarray, scale: float, offset: float, nodata: float | None
    ) -> np.ndarray:
        data = np.asarray(raw, dtype="float32")
        invalid = ~np.isfinite(data) | (data == 0)  # Sentinel-2 DN=0 is NO_DATA.
        if nodata is not None and np.isfinite(nodata):
            invalid |= data == nodata
        data = data * scale + offset
        data[invalid] = np.nan
        return data

    def _read_scene(
        self,
        scene: Sentinel2Scene,
        bounds_wgs84: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any, Any, str]:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject, transform_bounds
        from rasterio.windows import from_bounds

        self._require_assets(scene)
        nir_key = self._nir_key(scene)

        # B12 is native 20 m and defines the analysis grid.
        with rasterio.open(scene.assets["B12"]) as swir_src:
            projected_bounds = transform_bounds(
                "EPSG:4326", swir_src.crs, *bounds_wgs84, densify_pts=21
            )
            window = (
                from_bounds(*projected_bounds, transform=swir_src.transform)
                .round_offsets()
                .round_lengths()
            )
            full_window = rasterio.windows.Window(0, 0, swir_src.width, swir_src.height)
            try:
                window = window.intersection(full_window)
            except rasterio.errors.WindowError as exc:
                raise ValueError(f"Event window does not intersect scene {scene.scene_id}") from exc
            if window.width <= 0 or window.height <= 0:
                raise ValueError(f"Event window does not intersect scene {scene.scene_id}")

            swir_raw = swir_src.read(1, window=window, masked=False).astype("float32")
            swir_scale, swir_offset, swir_nodata = self._calibration(scene, "B12", swir_src)
            swir = self._apply_calibration(swir_raw, swir_scale, swir_offset, swir_nodata)
            target_transform = swir_src.window_transform(window)
            target_crs = swir_src.crs
            target_shape = swir.shape

        nir_raw = np.full(target_shape, np.nan, dtype="float32")
        with rasterio.open(scene.assets[nir_key]) as nir_src:
            nir_scale, nir_offset, nir_nodata = self._calibration(scene, nir_key, nir_src)
            # B8A is 20 m; B08 fallback is 10 m. Bilinear is appropriate for reflectance.
            reproject(
                source=rasterio.band(nir_src, 1),
                destination=nir_raw,
                src_transform=nir_src.transform,
                src_crs=nir_src.crs,
                src_nodata=nir_nodata if nir_nodata is not None else 0,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan,
            )
        nir = self._apply_calibration(nir_raw, nir_scale, nir_offset, nir_nodata)

        scl: np.ndarray | None = None
        if scene.assets.get("SCL"):
            scl = np.full(target_shape, 0, dtype="uint8")
            with rasterio.open(scene.assets["SCL"]) as scl_src:
                reproject(
                    source=rasterio.band(scl_src, 1),
                    destination=scl,
                    src_transform=scl_src.transform,
                    src_crs=scl_src.crs,
                    src_nodata=scl_src.nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=0,
                )

        return nir, swir, scl, target_transform, target_crs, nir_key

    def _valid_mask(
        self, nir: np.ndarray, swir: np.ndarray, scl: np.ndarray | None
    ) -> np.ndarray:
        valid = np.isfinite(nir) & np.isfinite(swir) & ((nir + swir) != 0)
        if scl is not None:
            valid &= ~np.isin(scl, list(self.INVALID_SCL))
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
        return abs(transform.a * transform.e - transform.b * transform.d)

    def _polygonize(
        self,
        severity: np.ndarray,
        transform: Any,
        crs: Any,
        event_id: str,
        pre_scene_id: str,
        post_scene_id: str,
    ) -> dict[str, Any]:
        from rasterio.features import shapes
        from rasterio.warp import transform_geom
        from shapely.geometry import shape

        labels = {1: "low_severity", 2: "moderate_severity", 3: "high_severity"}
        features: list[dict[str, Any]] = []
        affected_mask = (severity >= 1) & (severity <= 3)
        for geom, value in shapes(
            severity,
            mask=affected_mask,
            transform=transform,
            connectivity=8,
        ):
            class_value = int(value)
            if class_value not in labels:
                continue
            native_area_ha = float(shape(geom).area / 10000.0)
            geom_wgs84 = transform_geom(crs, "EPSG:4326", geom, precision=6)
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom_wgs84,
                    "properties": {
                        "event_id": event_id,
                        "class_value": class_value,
                        "severity": labels[class_value],
                        "area_ha": round(native_area_ha, 3),
                        "method": "sentinel2_l2a_dnbr",
                        "pre_scene_id": pre_scene_id,
                        "post_scene_id": post_scene_id,
                    },
                }
            )
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
        pre_nir, pre_swir, pre_scl, transform, crs, pre_nir_key = self._read_scene(
            pre_scene, bounds
        )
        post_nir, post_swir, post_scl, post_transform, post_crs, post_nir_key = self._read_scene(
            post_scene, bounds
        )

        if (
            pre_nir.shape != post_nir.shape
            or str(crs) != str(post_crs)
            or transform != post_transform
        ):
            raise ValueError("Pre/post Sentinel-2 windows are not aligned to the same tile/grid")

        valid = self._valid_mask(pre_nir, pre_swir, pre_scl) & self._valid_mask(
            post_nir, post_swir, post_scl
        )
        pre_nbr = self._nbr(pre_nir, pre_swir)
        post_nbr = self._nbr(post_nir, post_swir)
        dnbr = pre_nbr - post_nbr
        valid &= np.isfinite(dnbr)
        severity = self._classify(dnbr, valid)

        pixel_area_m2 = self._pixel_area_m2(transform)

        def area_for(value: int) -> float:
            return float(np.count_nonzero(severity == value) * pixel_area_m2 / 10000.0)

        low_ha, moderate_ha, high_ha = area_for(1), area_for(2), area_for(3)
        total_ha = low_ha + moderate_ha + high_ha
        unburned_ha = area_for(0)

        geojson = self._polygonize(
            severity,
            transform,
            crs,
            event_id,
            pre_scene.scene_id,
            post_scene.scene_id,
        )
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

        total_pixels = int(severity.size)
        valid_pixels = int(np.count_nonzero(valid))
        affected_pixels = int(np.count_nonzero((severity >= 1) & (severity <= 3)))
        total_area_ha = total_pixels * pixel_area_m2 / 10000.0
        observed_area_ha = valid_pixels * pixel_area_m2 / 10000.0
        provenance = {
            "event_id": event_id,
            "method": "dNBR = NBR(pre) - NBR(post)",
            "nbr_formula": "(NIR - SWIR2) / (NIR + SWIR2)",
            "nir_band_pre": pre_nir_key,
            "nir_band_post": post_nir_key,
            "common_grid": "Sentinel-2 B12 native grid (~20 m)",
            "radiometry": "STAC/GDAL scale and offset applied before NBR when supplied",
            "cloud_surface_mask": "SCL classes 0,1,2,3,6,8,9,10,11 excluded when available",
            "pre_scene": pre_scene.model_dump(mode="json"),
            "post_scene": post_scene.model_dump(mode="json"),
            "analysis_bounds_wgs84": list(bounds),
            "projection": str(crs),
            "pixel_area_m2": pixel_area_m2,
            "total_pixels": total_pixels,
            "valid_pixels": valid_pixels,
            "masked_pixels": total_pixels - valid_pixels,
            "affected_pixels": affected_pixels,
            "total_window_area_ha": round(total_area_ha, 3),
            "observed_area_ha": round(observed_area_ha, 3),
            "unobserved_area_ha": round(total_area_ha - observed_area_ha, 3),
        }
        return RasterAnalysis(geojson=geojson, result=result, provenance=provenance)
