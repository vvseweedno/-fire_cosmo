# False Positive Filter Service
import logging
from collections import Counter
from typing import List, Tuple

from app.core.schemas import ConfidenceLevel, FireCandidate


logger = logging.getLogger(__name__)


class FalsePositiveFilter:
    """Auditable heuristic filter for thermal-anomaly candidates.

    Implemented checks are deliberately narrower than a production classifier:
    coordinates, configurable confidence, optional industrial-source whitelist,
    and simple thermal/FRP sanity thresholds. Water/cloud/land-cover filtering
    is not claimed until a validated contextual data source is connected.
    """

    def __init__(
        self,
        whitelist_path: str | None = None,
        min_confidence: ConfidenceLevel = ConfidenceLevel.NOMINAL,
    ):
        self.whitelist = self._load_whitelist(whitelist_path) if whitelist_path else []
        self.whitelist_path = whitelist_path
        self.min_confidence = min_confidence
        self.last_rejected: List[FireCandidate] = []
        self.last_accepted: List[FireCandidate] = []
        self.last_total = 0

    def _load_whitelist(self, path: str) -> List[Tuple[float, float, float]]:
        """Load known industrial sources as (lat, lon, radius_km)."""
        import json
        from pathlib import Path

        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
                return [
                    (float(item["lat"]), float(item["lon"]), float(item.get("radius_km", 2.0)))
                    for item in data
                ]
        logger.warning("Industrial whitelist path does not exist: %s", path)
        return []

    async def filter_points(self, points: List[FireCandidate]) -> List[FireCandidate]:
        """Return accepted points and retain all rejected candidates for audit."""
        filtered: List[FireCandidate] = []
        self.last_rejected = []
        self.last_accepted = []
        self.last_total = len(points)

        for point in points:
            # Make repeated analyses deterministic from an audit perspective.
            point.filters_passed = []
            point.filters_failed = []
            point.reason = None
            point.is_valid = True

            is_valid, reason = await self._validate_point(point)
            if is_valid:
                point.is_valid = True
                for name in (
                    "coordinates",
                    "confidence",
                    "industrial_whitelist",
                    "thermal_properties",
                ):
                    point.filters_passed.append(name)
                filtered.append(point)
                self.last_accepted.append(point)
            else:
                point.is_valid = False
                point.filters_failed.append(reason)
                point.reason = reason
                self.last_rejected.append(point)
                logger.debug("Filtered out point %s: %s", point.id, reason)
        return filtered

    def audit_summary(self) -> dict:
        """Machine-readable explanation of the latest filtering pass."""
        reasons = Counter(point.reason or "unknown" for point in self.last_rejected)
        return {
            "candidates_total": self.last_total,
            "accepted": len(self.last_accepted),
            "rejected": len(self.last_rejected),
            "rejected_by_reason": dict(sorted(reasons.items())),
            "min_confidence": self.min_confidence.value,
            "industrial_whitelist_configured": bool(self.whitelist_path),
            "industrial_whitelist_entries": len(self.whitelist),
            "implemented_checks": [
                "coordinates",
                "confidence",
                "industrial_whitelist_if_configured",
                "thermal_frp_sanity",
            ],
            "not_yet_claimed": [
                "water_mask_for_thermal_points",
                "cloud_mask_for_thermal_points",
                "forest_landcover_mask",
            ],
        }

    async def _validate_point(self, point: FireCandidate) -> Tuple[bool, str]:
        if not self._check_coordinates(point):
            return False, "invalid_coordinates"
        if not self._check_confidence(point):
            return False, "below_min_confidence"
        if not self._check_whitelist(point):
            return False, "industrial_source"
        if not self._check_thermal_properties(point):
            return False, "suspicious_thermal_properties"
        return True, ""

    @staticmethod
    def _check_coordinates(point: FireCandidate) -> bool:
        return -90 <= point.latitude <= 90 and -180 <= point.longitude <= 180

    def _check_confidence(self, point: FireCandidate) -> bool:
        order = {
            ConfidenceLevel.LOW: 0,
            ConfidenceLevel.NOMINAL: 1,
            ConfidenceLevel.HIGH: 2,
        }
        return order[point.confidence] >= order[self.min_confidence]

    def _check_whitelist(self, point: FireCandidate) -> bool:
        from math import asin, cos, radians, sin, sqrt

        for lat, lon, radius in self.whitelist:
            dlat = radians(lat - point.latitude)
            dlon = radians(lon - point.longitude)
            a = (
                sin(dlat / 2) ** 2
                + cos(radians(lat))
                * cos(radians(point.latitude))
                * sin(dlon / 2) ** 2
            )
            c = 2 * asin(sqrt(a))
            if c * 6371 <= radius:
                return False
        return True

    @staticmethod
    def _check_thermal_properties(point: FireCandidate) -> bool:
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
