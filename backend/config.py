"""Small, explicit configuration module for RetainAI."""
from pathlib import Path

APP_NAME = "RetainAI API"
APP_VERSION = "0.0.1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "eCommerce_dataset.csv"
ARTIFACT_DIR = PROJECT_ROOT / "backend" / "ml_pipeline" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
DATA_SUMMARY_PATH = ARTIFACT_DIR / "data_summary.json"

# Keep this open for local/mobile development. Replace with explicit origins in production.
CORS_ORIGINS = ["*"]


def ensure_directories() -> None:
    """Create directories used by generated model files."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
