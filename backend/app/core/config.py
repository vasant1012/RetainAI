import os
from typing import list

from pydantic import BaseModel


class Settings(BaseModel):
    """
    Application settings and configuration parameters.
    Centralizing these allows for easy environment
    variable overrides in production.
    """

    PROJECT_NAME: str = "RetainAI API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"

    # Configure CORS origins so the React Native
    # frontend can securely access the API
    # In production, replace "*" with your explicit frontend URL/IP
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    # Resolve the absolute path to the backend directory
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Instantiate the settings object to be imported across the application
settings = Settings()
