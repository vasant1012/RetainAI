from pathlib import Path

from backend.config import Settings


def test_paths_derive_from_project_root(tmp_path: Path) -> None:
    configured = Settings(PROJECT_ROOT=tmp_path, _env_file=None)
    assert configured.RAW_DATA_PATH == tmp_path / "data/raw/eCommerce_dataset.csv"
    assert configured.MODEL_PATH == tmp_path / "backend/ml_pipeline/artifacts/xgboost_model.pkl"


def test_comma_separated_cors_origins() -> None:
    configured = Settings(
        BACKEND_CORS_ORIGINS="https://one.example, https://two.example",
        _env_file=None,
    )
    assert configured.BACKEND_CORS_ORIGINS == [
        "https://one.example",
        "https://two.example",
    ]
