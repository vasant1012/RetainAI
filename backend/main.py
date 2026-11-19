"""FastAPI application for RetainAI."""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import APP_NAME, APP_VERSION, CORS_ORIGINS
from backend.core.ml_service import ml_service
from backend.schemas.pydantic_models import (
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retainai")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        ml_service.load()
        logger.info("Model loaded: version %s", ml_service.model_version)
    except Exception as exc:  # noqa: BLE001
        # The API can still expose docs and health before the first training run.
        logger.warning("Model is not ready: %s", exc)
    yield


app = FastAPI(title=APP_NAME, version=APP_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "docs": "to test API, visit /docs"
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=APP_NAME,
        version=APP_VERSION,
        model_ready=ml_service.loaded,
    )


def _risk_guidance(score: float) -> tuple[str, str]:
    if score >= 70:
        return "high", "Contact the customer now with a targeted retention offer."
    if score >= 40:
        return "medium", "Add the customer to a proactive engagement campaign."
    return "low", "Continue normal engagement and monitor behavior."


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        probability = ml_service.predict_churn(
            request.features.to_model_payload())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    score = round(probability * 100, 2)
    risk_level, action = _risk_guidance(score)
    return PredictionResponse(
        prediction_id=uuid.uuid4(),
        customer_id=request.customer_id,
        churn_risk_score=score,
        risk_level=risk_level,
        recommended_action=action,
        model_version=ml_service.model_version,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    try:
        values = ml_service.get_metrics()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return MetricsResponse(model_version=ml_service.model_version, metrics=values)


@app.get("/modelInfo", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(**ml_service.get_model_info())
