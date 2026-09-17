# Fire Detection Service
import logging
from typing import Dict, List

from app.adapters.base import BaseFireAdapter
from app.adapters.modis_adapter import ModisAdapter
from app.adapters.viirs_adapter import ViirsAdapter
from app.core.schemas import ConfidenceLevel, FireCandidate


logger = logging.getLogger(__name__)


class FireDetectionService:
    """Load and normalize thermal-anomaly candidates from configured sensors."""

    def __init__(
        self,
        adapters: Dict[str, BaseFireAdapter],
        min_confidence: ConfidenceLevel = ConfidenceLevel.NOMINAL,
    ):
        self.adapters = adapters
        # Retained for backwards compatibility; confidence decisions belong to
        # FalsePositiveFilter so rejected candidates remain visible in audit.
        self.min_confidence = min_confidence

    @classmethod
    def create_default(
        cls,
        firms_api_key: str | None = None,
        offline_mode: bool = False,
        cache_dir: str = "./data/cache",
    ) -> "FireDetectionService":
        """Create the default production-backed detector set.

        Landsat is intentionally not enabled by default until its thermal-pixel
        download/detection path is implemented. The adapter remains available as
        an experimental component and can be wired explicitly later.
        """
        adapters: Dict[str, BaseFireAdapter] = {}
        if firms_api_key or offline_mode:
            adapters["MODIS"] = ModisAdapter(
                api_key=firms_api_key,
                offline_mode=offline_mode,
                cache_dir=cache_dir,
            )
            adapters["VIIRS"] = ViirsAdapter(
                api_key=firms_api_key,
                offline_mode=offline_mode,
                cache_dir=cache_dir,
            )
        else:
            logger.warning("FIRMS MAP_KEY not set and offline mode disabled; no thermal adapters enabled")
        return cls(adapters=adapters, min_confidence=ConfidenceLevel.NOMINAL)

    async def detect_fires(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        sensors: List[str] | None = None,
    ) -> List[FireCandidate]:
        """Load normalized candidates without silently discarding low-confidence rows."""
        if sensors is None:
            sensors = list(self.adapters.keys())

        all_points: List[FireCandidate] = []
        for sensor_name in sensors:
            adapter = self.adapters.get(sensor_name)
            if not adapter:
                logger.warning("Adapter for %s is not enabled; skipping", sensor_name)
                continue
            try:
                points = await adapter.fetch_fire_points(bbox, start_date, end_date)
                if not points and getattr(adapter, "offline_mode", False) and hasattr(
                    adapter, "fetch_from_cache_or_fixture"
                ):
                    points = await adapter.fetch_from_cache_or_fixture(bbox, start_date, end_date)
                all_points.extend(points)
                logger.info("Loaded %d candidates from %s", len(points), sensor_name)
            except Exception as exc:
                logger.error("Error detecting fires with %s: %s", sensor_name, exc)

        # Deterministic source IDs allow de-duplication across repeated/chunked requests.
        unique = {point.id: point for point in all_points}
        result = list(unique.values())
        logger.info("Total normalized thermal candidates: %d", len(result))
        return result

    def _filter_by_confidence(self, points: List[FireCandidate]) -> List[FireCandidate]:
        """Legacy helper retained for callers/tests; main pipeline uses FalsePositiveFilter."""
        confidence_order = [ConfidenceLevel.LOW, ConfidenceLevel.NOMINAL, ConfidenceLevel.HIGH]
        min_idx = confidence_order.index(self.min_confidence)
        return [p for p in points if confidence_order.index(p.confidence) >= min_idx]

    async def get_all_available_sensors(self) -> List[str]:
        return list(self.adapters.keys())
