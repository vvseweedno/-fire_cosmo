"""
Base types and utilities for ingest module.
"""

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Geographic bounding box."""

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def to_firms_format(self) -> str:
        """Convert to FIRMS API format string."""
        return f"{self.min_lat},{self.min_lon},{self.max_lat},{self.max_lon}"

    def contains(self, lat: float, lon: float) -> bool:
        """Check if point is within bounds."""
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )
