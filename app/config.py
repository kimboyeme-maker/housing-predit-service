"""Application settings loaded from environment (12-factor)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_APP_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Environment-backed runtime configuration for API and model artifacts."""
    model_config = SettingsConfigDict(env_prefix="HPP_", env_file=".env", extra="ignore")

    app_name: str = "Housing Price Prediction API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # ML artifacts (committed to git, produced by train.py). Overridable so users can
    # point the API at their own model.pkl + metadata.json without touching code.
    model_path: Path = _APP_DIR / "ml" / "artifacts" / "model.pkl"
    metadata_path: Path = _APP_DIR / "ml" / "artifacts" / "metadata.json"

    # CORS — comma separated origins. "*" allows all (dev default).
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings instance for the process lifetime."""
    return Settings()
