from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router as api_router

app = FastAPI(
    title="RetainAI API",
    description="Machine Learning backend serving XGBoost churn predictions.",
    version="1.0.0",
)

# Configure CORS for React Native / Web frontend
app.add_middleware(
    CORSMiddleware,
    # Adjust this in production (e.g., ["http://localhost:3000"])
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints router
app.include_router(api_router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "RetainAI API is running. Access /docs for Swagger UI."}
