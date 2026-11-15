from fastapi import APIRouter, HTTPException

from app.core.ml_service import ml_service
from app.schemas.pydantic_models import (
    CustomerData,
    MetricsResponse,
    PredictionResponse,
)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict_customer_churn(data: CustomerData):
    try:
        # Calculate risk score
        risk_score = ml_service.predict_churn(data.features)

        # Determine risk level and automated action
        if risk_score >= 80:
            level = "High"
            action = "Trigger 20% Win-Back Discount Webhook"
        elif risk_score >= 50:
            level = "Medium"
            action = "Add to Nurture Email Sequence"
        else:
            level = "Low"
            action = "Monitor"

        return PredictionResponse(
            churn_risk_score=risk_score,
            risk_level=level,
            action_recommended=action
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=MetricsResponse)
async def get_model_metrics():
    if not ml_service.metrics:
        raise HTTPException(status_code=404, detail="Metrics not found.")

    return MetricsResponse(
        f1_score=ml_service.metrics.get("f1_score", 0.0),
        classification_report=ml_service.metrics.get(
            "classification_report", {}),
    )
