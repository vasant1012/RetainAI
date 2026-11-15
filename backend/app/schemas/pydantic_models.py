from typing import Any, dict

from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    # For the MVP, we accept a dictionary of features.
    # In a production app, you would define each
    # specific field (e.g., age: float, city: str)
    features: dict[str, Any] = Field(
        ...,
        description=(
            "Dictionary of preprocessed customer features matching the model's"
            "expected columns"
        ),
    )


class PredictionResponse(BaseModel):
    churn_risk_score: float
    risk_level: str
    action_recommended: str


class MetricsResponse(BaseModel):
    f1_score: float
    classification_report: dict[str, Any]
