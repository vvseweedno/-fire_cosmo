# MODIS Adapter for NASA FIRMS data
import asyncio
import csv
import hashlib
import logging
from datetime import date, datetime, timedelta
from io import StringIO
from typing import List

import httpx

from app.adapters.base import BaseFireAdapter
from app.core.schemas import ConfidenceLevel, FireCandidate


logger = logging.getLogger(__name__)


class ModisAdapter(BaseFireAdapter):
    """Adapter for NASA FIRMS MODIS active-fire detections."""

    MAX_DAY_RANGE = 5  # FIRMS Area API contract: 1..5 days per request.

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
        self.source = "MODIS_NRT"

    def get_sensor_type(self) -> str:
        return "MODIS"

    @staticmethod
    def _iter_date_chunks(start_date: str, end_date: str):
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        if end < start:
            raise ValueError("end_date must be on or after start_date")
        cursor = start
        while cursor <= end:
            chunk_end = min(end, cursor + timedelta(days=ModisAdapter.MAX_DAY_RANGE - 1))
            yield cursor, (chunk_end - cursor).days + 1, chunk_end
            cursor = chunk_end + timedelta(days=1)

    async def _request_chunk(self, client: httpx.AsyncClient, url: str) -> str:
        for attempt in range(3):
            response = await client.get(url)
            if response.status_code != 429:
                response.raise_for_status()
                return response.text
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else float(2**attempt)
            except ValueError:
                delay = float(2**attempt)
            delay = min(max(delay, 0.5), 10.0)
            logger.warning("FIRMS MODIS rate limited; retrying in %.1fs", delay)
            await asyncio.sleep(delay)
        logger.error("FIRMS MODIS remained rate limited after retries")
        return ""

    async def fetch_fire_points(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        country: str = "USA",
    ) -> List[FireCandidate]:
        """Fetch MODIS detections, chunking long ranges to FIRMS' 5-day maximum."""
        if self.offline_mode:
            return await self._load_from_fixture(bbox, start_date, end_date)
        if not self.api_key:
            logger.warning("MODIS FIRMS MAP_KEY not set; returning no online detections")
            return []

        points: List[FireCandidate] = []
        area = ",".join(str(x) for x in bbox)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                for chunk_start, day_range, chunk_end in self._iter_date_chunks(start_date, end_date):
                    url = (
                        f"{self.base_url}/{self.api_key}/{self.source}/{area}/"
                        f"{day_range}/{chunk_start.isoformat()}"
                    )
                    csv_text = await self._request_chunk(client, url)
                    if csv_text:
                        points.extend(
                            self._parse_csv_response(
                                csv_text,
                                bbox,
                                chunk_start.isoformat(),
                                chunk_end.isoformat(),
                            )
                        )
            return points
        except (httpx.HTTPError, ValueError) as exc:
            self._log_error("Error fetching MODIS FIRMS data", exc)
            return []
        except Exception as exc:
            self._log_error("Unexpected error fetching MODIS FIRMS data", exc)
            return []

    @staticmethod
    def _stable_id(row: dict, lat: float, lon: float, dt: datetime) -> str:
        identity = "|".join(
            ["MODIS", f"{lat:.6f}", f"{lon:.6f}", dt.isoformat(), str(row.get("satellite", ""))]
        )
        return "modis_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]

    async def _load_from_fixture(
        self, bbox: List[float], start_date: str, end_date: str
    ) -> List[FireCandidate]:
        fixture_path = self._get_fixture_path("modis_demo.csv")
        if not fixture_path.exists():
            raise FileNotFoundError(
                f"MODIS fixture not found at {fixture_path}; run scripts/download_demo_data.py --offline"
            )
        with open(fixture_path, "r", encoding="utf-8") as f:
            return self._parse_csv_response(f.read(), bbox, start_date, end_date)

    def _parse_csv_response(
        self,
        csv_data: str,
        bbox: List[float],
        start_date: str = "0001-01-01",
        end_date: str = "9999-12-31",
    ) -> List[FireCandidate]:
        points: List[FireCandidate] = []
        reader = csv.DictReader(StringIO(csv_data))
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                if not (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]):
                    continue
                brightness = float(row.get("brightness") or row.get("bright") or 0)
                confidence = self._parse_confidence(row.get("confidence", "nominal"))
                acq_date = row.get("acq_date", "2024-01-01")
                acq_time = str(row.get("acq_time", "1200")).zfill(4)
                dt = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M")
                if not self._date_in_range(dt, start_date, end_date):
                    continue
                points.append(
                    FireCandidate(
                        id=self._stable_id(row, lat, lon, dt),
                        sensor="MODIS",
                        source=self.source,
                        datetime=dt,
                        latitude=lat,
                        longitude=lon,
                        brightness_temp_k=brightness,
                        frp_mw=float(row["frp"]) if row.get("frp") else None,
                        confidence=confidence,
                        daynight=row.get("daynight", "D"),
                        satellite=row.get("satellite", "Terra"),
                        raw=dict(row),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("Skipping invalid MODIS row: %s", exc)
        return points

    def _parse_confidence(self, value: str | int | float | None) -> ConfidenceLevel:
        text = str(value or "").strip().lower()
        if text in {"h", "high"}:
            return ConfidenceLevel.HIGH
        if text in {"n", "nominal"}:
            return ConfidenceLevel.NOMINAL
        if text in {"l", "low"}:
            return ConfidenceLevel.LOW
        try:
            numeric = float(text)
        except ValueError:
            return ConfidenceLevel.NOMINAL
        score = numeric * 100.0 if 0.0 <= numeric <= 1.0 else numeric
        if score >= 80.0:
            return ConfidenceLevel.HIGH
        if score >= 30.0:
            return ConfidenceLevel.NOMINAL
        return ConfidenceLevel.LOW

    @staticmethod
    def _date_in_range(dt: datetime, start_date: str, end_date: str) -> bool:
        return date.fromisoformat(start_date) <= dt.date() <= date.fromisoformat(end_date)
