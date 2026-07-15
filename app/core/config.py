from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "HeatGuard AI Uzbekistan"
    open_meteo_base_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    request_timeout_seconds: float = 12.0
    cache_ttl_seconds: int = 900
    gee_project: str | None = None
    gee_service_account: str | None = None
    gee_private_key_file: str | None = None
    model_path: str = "models/best_model.joblib"
    metrics_path: str = "models/metrics.json"
    model_config = SettingsConfigDict(env_file=".env", env_prefix="HEATGUARD_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
