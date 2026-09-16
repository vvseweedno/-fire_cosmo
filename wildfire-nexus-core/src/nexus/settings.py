"""
Configuration loading and settings management.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


def _load_yaml_config() -> dict[str, Any]:
    """Load configuration from YAML files."""
    config_dir = Path(__file__).parent.parent / "config"
    
    settings_path = config_dir / "settings.yaml"
    regions_path = config_dir / "regions.yaml"
    
    config: dict[str, Any] = {}
    
    if settings_path.exists():
        with open(settings_path) as f:
            config.update(yaml.safe_load(f) or {})
    
    if regions_path.exists():
        with open(regions_path) as f:
            regions_config = yaml.safe_load(f) or {}
            if "regions" in regions_config:
                config["regions"] = regions_config["regions"]
    
    return config


class Settings(BaseSettings):
    """Application settings."""
    
    # FIRMS API
    firms_map_key: str = ""
    
    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # Configuration from YAML
    regions: list[dict[str, Any]] = []
    ingest: dict[str, Any] = {}
    detect: dict[str, Any] = {}
    alert: dict[str, Any] = {}
    api: dict[str, Any] = {}
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }
    
    def __init__(self, **kwargs: Any) -> None:
        # Load environment variables first
        load_dotenv()
        
        # Load YAML config
        yaml_config = _load_yaml_config()
        
        # Merge YAML config with kwargs
        merged_kwargs = {**yaml_config, **kwargs}
        
        # Extract environment variables
        env_settings = {
            "firms_map_key": os.getenv("FIRMS_MAP_KEY", ""),
            "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        }
        
        super().__init__(**{**merged_kwargs, **env_settings})


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
