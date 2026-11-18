"""Central application configuration for RetainAI."""

from pathlib import Path

from pydantic import BaseModel

# backend/config.py -> backend -> RetainAI project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    """Application settings and canonical project paths."""

    PROJECT_NAME: str = "RetainAI API"
    ENVIRONMENT: Literal["development", "test", "staging", "production"] = "development"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Permissive only for local development; use explicit origins in production.
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    PROJECT_ROOT: Path = PROJECT_ROOT
    BACKEND_DIR: Path = PROJECT_ROOT / "backend"
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "eCommerce_dataset.csv"
    PROCESSED_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

    ML_PIPELINE_DIR: Path = PROJECT_ROOT / "backend" / "ml_pipeline"
    ML_ARTIFACTS_DIR: Path = PROJECT_ROOT / "backend" / "ml_pipeline" / "artifacts"
    MODEL_PATH: Path = ML_ARTIFACTS_DIR / "xgboost_model.pkl"
    METRICS_PATH: Path = ML_ARTIFACTS_DIR / "metrics.json"
    CHURN_DISTRIBUTION_PATH: Path = ML_ARTIFACTS_DIR / "churn_distribution.png"

    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    @property
    def BASE_DIR(self) -> str:
        """Backward-compatible string form of the project root."""
        return str(self.PROJECT_ROOT)

    def ensure_directories(self) -> None:
        """Create runtime output directories when they do not exist."""
        for directory in (
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.ML_ARTIFACTS_DIR,
            self.LOGS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
