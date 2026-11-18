from fastapi.testclient import TestClient

from backend.api.dependencies import get_ml_service
from backend.core.exceptions import InvalidFeatureError, ModelUnavailableError
from backend.main import create_app


class FakeAPIService:
    loaded = True
    artifact_version = "test-1.0"

    def predict_churn(self, values):
        unknown = set(values) - {"Age", "Gender"}
        if unknown:
            raise InvalidFeatureError(f"Unknown customer feature(s): {sorted(unknown)}")
        return 85.25

    def get_metrics(self):
        return {"f1_score": 0.9}

    def get_model_info(self):
        return {
            "ready": True,
            "artifact_version": self.artifact_version,
            "model_type": "Fake",
            "raw_feature_count": 2,
            "encoded_feature_count": 2,
            "raw_features": ["Age", "Gender"],
        }


class UnavailableService(FakeAPIService):
    loaded = False

    def predict_churn(self, values):
        raise ModelUnavailableError("private startup detail")


def test_prediction_contract_and_policy() -> None:
    app = create_app()
    fake = FakeAPIService()
    app.dependency_overrides[get_ml_service] = lambda: fake
    with TestClient(app) as client:
        response = client.post(
            "/api/predict",
            json={
                "customer_id": "C-42",
                "features": {"Age": 36, "Gender": "Male"},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["churn_risk_score"] == 85.25
    assert body["risk_level"] == "High"
    assert body["customer_id"] == "C-42"
    assert body["model_version"] == "test-1.0"
    assert response.headers["X-Request-ID"]


def test_schema_rejects_invalid_and_unknown_fields() -> None:
    app = create_app()
    app.dependency_overrides[get_ml_service] = lambda: FakeAPIService()
    with TestClient(app) as client:
        negative_age = client.post(
            "/api/predict", json={"features": {"Age": -1}}
        )
        unknown = client.post(
            "/api/predict", json={"features": {"Age": 30, "Typo": 1}}
        )
    assert negative_age.status_code == 422
    assert unknown.status_code == 422


def test_model_unavailable_returns_safe_503() -> None:
    app = create_app()
    app.dependency_overrides[get_ml_service] = lambda: UnavailableService()
    with TestClient(app) as client:
        response = client.post("/api/predict", json={"features": {"Age": 30}})
    assert response.status_code == 503
    assert "private startup detail" not in response.text


def test_metrics_and_model_info() -> None:
    app = create_app()
    app.dependency_overrides[get_ml_service] = lambda: FakeAPIService()
    with TestClient(app) as client:
        metrics = client.get("/api/metrics")
        info = client.get("/api/model/info")
    assert metrics.status_code == 200
    assert metrics.json()["metrics"]["f1_score"] == 0.9
    assert info.status_code == 200
    assert info.json()["raw_features"] == ["Age", "Gender"]


def test_liveness_and_degraded_readiness() -> None:
    app = create_app()
    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json()["model_ready"] is False
