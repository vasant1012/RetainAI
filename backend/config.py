"""Environment-backed configuration and deterministic project paths."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """RetainAI settings. Environment variables use the ``RETAINAI_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RETAINAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    PROJECT_NAME: str = "RetainAI API"
    ENVIRONMENT: Literal["development", "test", "staging", "production"] = "development"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    LOG_LEVEL: str = "INFO"
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:19006",
        "http://localhost:3000",
    ]
    REQUIRE_MODEL_ON_STARTUP: bool = False
    MODEL_ARTIFACT_MIN_VERSION: str = "1.1.0"
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        """Accept a JSON list or a comma-separated origin string."""
        if isinstance(value, str) and not value.lstrip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> str:
        return str(value).upper()

    @property
    def BASE_DIR(self) -> str:
        """Backward-compatible string form of the project root."""
        return str(self.PROJECT_ROOT)

    @property
    def RAW_DATA_PATH(self) -> Path:
        return self.PROJECT_ROOT / "data" / "raw" / "eCommerce_dataset.csv"

    @property
    def PROCESSED_DATA_PATH(self) -> Path:
        return self.PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

    @property
    def ARTIFACTS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "backend" / "ml_pipeline" / "artifacts"

    @property
    def MODEL_PATH(self) -> Path:
        return self.ARTIFACTS_DIR / "xgboost_model.pkl"

    @property
    def METRICS_PATH(self) -> Path:
        return self.ARTIFACTS_DIR / "metrics.json"

    @property
    def CHURN_PLOT_PATH(self) -> Path:
        return self.ARTIFACTS_DIR / "churn_distribution.png"

    @property
    def LOGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "logs"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    def ensure_output_directories(self) -> None:
        for directory in (
            self.PROCESSED_DATA_PATH.parent,
            self.ARTIFACTS_DIR,
            self.LOGS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    configured = Settings()
    configured.ensure_output_directories()
    return configured


settings = get_settings()
