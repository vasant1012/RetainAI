"""FastAPI application factory and model lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from backend.api.endpoints import router as api_router
from backend.config import Settings, settings
from backend.core.exceptions import ArtifactError
from backend.core.ml_service import MLService
from backend.logger import logger
from backend.schemas.pydantic_models import HealthResponse


def create_app(app_settings: Settings | None = None) -> FastAPI:
    active_settings = app_settings or settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = MLService(
            model_path=active_settings.MODEL_PATH,
            metrics_path=active_settings.METRICS_PATH,
        )
        app.state.ml_service = service
        app.state.model_startup_error = None
        try:
            service.load()
        except ArtifactError as exc:
            app.state.model_startup_error = str(exc)
            logger.error("Model startup validation failed: %s", exc)
            if active_settings.REQUIRE_MODEL_ON_STARTUP:
                raise
        yield

    application = FastAPI(
        title=active_settings.PROJECT_NAME,
        version=active_settings.VERSION,
        description="Customer churn risk and retention recommendation API.",
        lifespan=lifespan,
    )
    application.state.settings = active_settings

    origins = active_settings.BACKEND_CORS_ORIGINS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials="*" not in origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.get("/", tags=["system"])
    def root() -> dict[str, str]:
        return {
            "service": active_settings.PROJECT_NAME,
            "version": active_settings.VERSION,
            "docs": "/docs",
        }

    @application.get("/health/live", response_model=HealthResponse, tags=["system"])
    def liveness(request: Request) -> HealthResponse:
        service = request.app.state.ml_service
        return HealthResponse(
            status="ok",
            service=active_settings.PROJECT_NAME,
            version=active_settings.VERSION,
            model_ready=service.loaded,
        )

    @application.get(
        "/health/ready",
        response_model=HealthResponse,
        tags=["system"],
        responses={503: {"model": HealthResponse}},
    )
    def readiness(request: Request, response: Response) -> HealthResponse:
        service = request.app.state.ml_service
        if not service.loaded:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="ready" if service.loaded else "not_ready",
            service=active_settings.PROJECT_NAME,
            version=active_settings.VERSION,
            model_ready=service.loaded,
        )

    application.include_router(
        api_router,
        prefix=active_settings.API_V1_STR,
        tags=["prediction"],
    )
    return application


app = create_app()
