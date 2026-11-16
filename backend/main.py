from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.endpoints import router as api_router
from backend.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Machine Learning backend serving XGBoost churn predictions.",
)

# Configure CORS using the settings list
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints router with the dynamic prefix
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {
        "status": f"{settings.PROJECT_NAME} is live! with {settings.VERSION} version in {settings.ENVIRONMENT} environment",
    }
