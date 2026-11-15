from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Machine Learning backend serving XGBoost churn predictions."
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
    message = (
        f"{settings.PROJECT_NAME} is running. "
        "Access /docs for Swagger UI."
    )
    return {"message": message}
