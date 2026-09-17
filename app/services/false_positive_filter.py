# False Positive Filter Service
import logging
from collections import Counter
from typing import List, Tuple

from app.core.schemas import FireCandidate


logger = logging.getLogger(__name__)


class FalsePositiveFilter:
    """Auditable heuristic filter for thermal-anomaly candidates.

    Implemented checks are deliberately narrower than a production classifier:
    coordinates, FIRMS confidence, optional industrial-source whitelist, and
    simple thermal/FRP sanity thresholds. Water/cloud/land-cover filtering is
    not claimed here until a validated contextual data source is connected.
    """

    def __init__(self, whitelist_path: str | None = None):
        self.whitelist = self._load_whitelist(whitelist_path) if whitelist_path else []
        self.last_rejected: List[FireCandidate] = []
        self.last_accepted: List[FireCandidate] = []

    def _load_whitelist(self, path: str) -> List[Tuple[float, float, float]]:
        """Загрузить список известных промышленных источников (lat, lon, radius_km)."""
        import json
        from pathlib import Path

        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                return [(item["lat"], item["lon"], item.get("radius_km", 2.0)) for item in data]
        return []

    async def filter_points(self, points: List[FireCandidate]) -> List[FireCandidate]:
        """Return accepted points and retain rejected candidates for audit."""
        filtered: List[FireCandidate] = []
        self.last_rejected = []
        self.last_accepted = []

        for point in points:
            is_valid, reason = await self._validate_point(point)
            if is_valid:
                point.is_valid = True
                for name in ("coordinates", "confidence", "industrial_whitelist", "thermal_properties"):
                    if name not in point.filters_passed:
                        point.filters_passed.append(name)
                filtered.append(point)
                self.last_accepted.append(point)
            else:
                point.is_valid = False
                if reason not in point.filters_failed:
                    point.filters_failed.append(reason)
                point.reason = reason
                self.last_rejected.append(point)
                logger.debug("Filtered out point %s: %s", point.id, reason)
        return filtered

    def audit_summary(self) -> dict:
        """Machine-readable explanation of the latest filtering pass."""
        reasons = Counter(point.reason or "unknown" for point in self.last_rejected)
        return {
            "accepted": len(self.last_accepted),
            "rejected": len(self.last_rejected),
            "rejections_by_reason": dict(sorted(reasons.items())),
            "implemented_checks": [
                "coordinates",
                "confidence",
                "industrial_whitelist_if_configured",
                "thermal_frp_sanity",
            ],
            "not_yet_claimed": ["water_mask", "cloud_mask_for_thermal_points", "forest_landcover_mask"],
        }

    async def _validate_point(self, point: FireCandidate) -> Tuple[bool, str]:
        if not self._check_coordinates(point):
            return False, "invalid_coordinates"
        if not self._check_confidence(point):
            return False, "low_confidence"
        if not self._check_whitelist(point):
            return False, "industrial_source"
        if not self._check_thermal_properties(point):
            return False, "suspicious_thermal_properties"
        return True, ""

    def _check_coordinates(self, point: FireCandidate) -> bool:
        return -90 <= point.latitude <= 90 and -180 <= point.longitude <= 180

    def _check_confidence(self, point: FireCandidate) -> bool:
        return point.confidence.value != "low"

    def _check_whitelist(self, point: FireCandidate) -> bool:
        from math import radians, cos, sin, asin, sqrt

        for lat, lon, radius in self.whitelist:
            dlat = radians(lat - point.latitude)
            dlon = radians(lon - point.longitude)
            a = sin(dlat / 2) ** 2 + cos(radians(lat)) * cos(radians(point.latitude)) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            if c * 6371 <= radius:
                return False
        return True

    def _check_thermal_properties(self, point: FireCandidate) -> bool:
        if point.brightness_temp_k < 310:
            return False
        if point.frp_mw is not None and point.frp_mw < 1.0 and point.brightness_temp_k < 320:
            return False
        return True

    def mark_as_low_confidence(self, point: FireCandidate, reason: str) -> FireCandidate:
        point.false_positive_score += 0.3
        point.filters_failed.append(reason)
        point.reason = reason
        return point
