"""RetainAI API v1 endpoints."""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.dependencies import get_ml_service
from backend.core.exceptions import InvalidFeatureError, ModelUnavailableError
from backend.core.ml_service import MLService
from backend.core.retention_service import retention_service
from backend.logger import logger
from backend.schemas.pydantic_models import (
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn risk",
    responses={422: {"description": "Invalid customer features"}, 503: {"description": "Model unavailable"}},
)
def predict_churn(
    request: PredictionRequest,
    ml_service: MLService = Depends(get_ml_service),
) -> PredictionResponse:
    try:
        risk_score = ml_service.predict_churn(request.features.to_model_payload())
        assessment = retention_service.assess(risk_score)
        return PredictionResponse(
            prediction_id=uuid4(),
            customer_id=request.customer_id,
            churn_risk_score=risk_score,
            risk_level=assessment.risk_level,
            recommended_action=assessment.recommended_action,
            model_version=ml_service.artifact_version or "unknown",
        )
    except InvalidFeatureError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is temporarily unavailable.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction endpoint error.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred.",
        ) from exc


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(
    ml_service: MLService = Depends(get_ml_service),
) -> MetricsResponse:
    try:
        return MetricsResponse(
            model_version=ml_service.artifact_version or "unknown",
            metrics=ml_service.get_metrics(),
        )
    except ModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metrics are temporarily unavailable.",
        ) from exc


@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(
    ml_service: MLService = Depends(get_ml_service),
) -> ModelInfoResponse:
    return ModelInfoResponse.model_validate(ml_service.get_model_info())
