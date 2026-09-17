# App Core Configuration
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.schemas import ConfidenceLevel


class Settings(BaseSettings):
    """Настройки приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    offline_mode: bool = False

    # Data Directories
    data_dir: str = "./data"
    cache_dir: str = "./data/cache"
    output_dir: str = "./data/outputs"
    fixtures_dir: str = "./data/fixtures"

    # External data / API keys
    firms_map_key: Optional[str] = None
    earthdata_token: Optional[str] = None
    sentinel2_stac_api: str = "https://earth-search.aws.element84.com/v1"
    industrial_whitelist_path: Optional[str] = None

    # Detection Thresholds
    default_min_confidence: ConfidenceLevel = ConfidenceLevel.NOMINAL
    max_cloud_cover: float = 20.0
    pre_image_days_before: int = 90
    post_image_days_after: int = 30
    min_fire_point_cluster_size: int = 2

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/wildfire.db"

    # Severity Thresholds (dNBR)
    severity_unburned_max: float = 0.10
    severity_low_max: float = 0.27
    severity_moderate_max: float = 0.44

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    def get_fixtures_path(self) -> Path:
        return Path(self.fixtures_dir)

    def ensure_dirs_exist(self):
        for dir_path in [self.data_dir, self.cache_dir, self.output_dir, self.fixtures_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


settings = Settings()
