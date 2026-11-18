"""FastAPI dependencies."""

from fastapi import Request

from backend.core.ml_service import MLService


def get_ml_service(request: Request) -> MLService:
    return request.app.state.ml_service
