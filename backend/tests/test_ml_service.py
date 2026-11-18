import pickle
from pathlib import Path

import pytest

from backend.core.exceptions import ArtifactError, InvalidFeatureError
from backend.core.ml_service import MLService


def test_artifact_and_metrics_load(loaded_service: MLService) -> None:
    assert loaded_service.loaded is True
    assert loaded_service.artifact_version == "1.2.0"
    assert loaded_service.get_metrics()["f1_score"] == 0.81


def test_missing_numeric_uses_training_median(loaded_service: MLService) -> None:
    processed = loaded_service.preprocess_customer_features({"Gender": "Female"})
    assert float(processed.iloc[0]["Age"]) == pytest.approx(40.0)


def test_known_category_is_encoded(loaded_service: MLService) -> None:
    processed = loaded_service.preprocess_customer_features({"Age": 30, "Gender": "Male"})
    assert int(processed.iloc[0]["Gender_Male"]) == 1


def test_unseen_category_is_safe(loaded_service: MLService) -> None:
    processed = loaded_service.preprocess_customer_features({"Age": 30, "Gender": "Other"})
    assert int(processed.iloc[0]["Gender_Male"]) == 0


def test_prediction_is_percentage(loaded_service: MLService) -> None:
    score = loaded_service.predict_churn({"Age": 40, "Gender": "Male"})
    assert score == pytest.approx(60.0)


def test_unknown_feature_is_rejected(loaded_service: MLService) -> None:
    with pytest.raises(InvalidFeatureError, match="Unknown customer feature"):
        loaded_service.predict_churn({"Age": 30, "Typo": 1})


def test_missing_model_artifact_is_explicit(tmp_path: Path) -> None:
    service = MLService(tmp_path / "missing.pkl", tmp_path / "metrics.json")
    with pytest.raises(ArtifactError, match="not found"):
        service.load()


def test_corrupt_model_artifact_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "bad.pkl"
    model_path.write_bytes(b"not a pickle")
    service = MLService(model_path, tmp_path / "metrics.json")
    with pytest.raises(ArtifactError, match="invalid, corrupt, or incompatible"):
        service.load()


def test_incomplete_artifact_is_rejected(tmp_path: Path) -> None:
    model_path = tmp_path / "incomplete.pkl"
    with model_path.open("wb") as file:
        pickle.dump({"feature_names": ["Age"]}, file)
    service = MLService(model_path, tmp_path / "metrics.json")
    with pytest.raises(ArtifactError, match="missing 'model' or 'pipeline'"):
        service.load()
