from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.core.ml_service import MLService
from backend.ml_pipeline.src.train_model import train_and_save


@pytest.fixture(scope="module")
def trained_service(tmp_path_factory: pytest.TempPathFactory) -> MLService:
    rng = np.random.default_rng(42)
    rows = 180
    age = rng.integers(18, 75, rows)
    inactivity = rng.integers(0, 150, rows)
    returns = rng.uniform(0, 1, rows)
    churn = ((inactivity > 65) | ((returns > 0.55) & (age < 45))).astype(int)
    dataframe = pd.DataFrame(
        {
            "Customer_ID": [f"CUST-{i:04d}" for i in range(rows)],
            "Age": age,
            "Gender": rng.choice(["Female", "Male"], rows),
            "Country": rng.choice(["India", "USA", "UK"], rows),
            "Membership_Years": rng.uniform(0, 10, rows),
            "Days_Since_Last_Purchase": inactivity,
            "Returns_Rate": returns,
            "Churned": churn,
        }
    )
    folder = tmp_path_factory.mktemp("model")
    model_path = folder / "model.joblib"
    metrics_path = folder / "metrics.json"
    train_and_save(dataframe, model_path, metrics_path)
    service = MLService(model_path, metrics_path)
    service.load()
    return service


def test_training_and_partial_prediction(trained_service: MLService) -> None:
    score = trained_service.predict_churn({"Age": 45, "Gender": "Female"})
    assert 0.0 <= score <= 1.0
    assert "f1_score" in trained_service.get_metrics()


def test_unseen_category_is_handled(trained_service: MLService) -> None:
    score = trained_service.predict_churn({"Age": 35, "Country": "NewCountry"})
    assert 0.0 <= score <= 1.0


def test_unknown_feature_is_rejected(trained_service: MLService) -> None:
    with pytest.raises(ValueError, match="Unknown customer feature"):
        trained_service.predict_churn({"Typo_Field": 10})


def test_api_predict(monkeypatch: pytest.MonkeyPatch, trained_service: MLService) -> None:
    monkeypatch.setattr(main_module, "ml_service", trained_service)
    with TestClient(main_module.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is True

        response = client.post(
            "/api/predict",
            json={
                "customer_id": "CUST-TEST",
                "features": {"Age": 45, "Gender": "Female"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["churn_risk_score"] <= 100
        assert body["customer_id"] == "CUST-TEST"
