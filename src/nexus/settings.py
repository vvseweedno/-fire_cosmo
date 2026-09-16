"""Configuration and settings management."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_base_dir() -> Path:
    """Get the base directory of the project."""
    return Path(__file__).parent.parent.parent


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    config_path = get_base_dir() / "config" / filename
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # FIRMS API
    firms_map_key: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/wildfire.db"

    # Detection settings from YAML
    min_brightness_viirs: float = 320.0
    min_brightness_modis: float = 310.0
    min_confidence: float = 0.4
    min_frp: float = 5.0
    tracking_radius_km: float = 3.0
    event_decay_hours: int = 12
    dedup_interval_hours: int = 6

    # Alert settings
    critical_distance_km: float = 10.0
    warning_distance_km: float = 15.0
    min_population_critical: int = 1000

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Regions
    regions: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Load YAML configs
        settings_config = load_yaml_config("settings.yaml")
        regions_config = load_yaml_config("regions.yaml")

        # Override with YAML values
        detect_cfg = settings_config.get("detect", {})
        alert_cfg = settings_config.get("alert", {})
        api_cfg = settings_config.get("api", {})

        self.min_brightness_viirs = detect_cfg.get("min_brightness_viirs", self.min_brightness_viirs)
        self.min_brightness_modis = detect_cfg.get("min_brightness_modis", self.min_brightness_modis)
        self.min_confidence = detect_cfg.get("min_confidence", self.min_confidence)
        self.min_frp = detect_cfg.get("min_frp", self.min_frp)
        self.tracking_radius_km = detect_cfg.get("tracking_radius_km", self.tracking_radius_km)
        self.event_decay_hours = detect_cfg.get("event_decay_hours", self.event_decay_hours)
        self.dedup_interval_hours = detect_cfg.get("dedup_interval_hours", self.dedup_interval_hours)

        self.critical_distance_km = alert_cfg.get("critical_distance_km", self.critical_distance_km)
        self.warning_distance_km = alert_cfg.get("warning_distance_km", self.warning_distance_km)
        self.min_population_critical = alert_cfg.get("min_population_critical", self.min_population_critical)

        self.api_host = api_cfg.get("host", self.api_host)
        self.api_port = api_cfg.get("port", self.api_port)

        self.regions = regions_config.get("regions", [])


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
