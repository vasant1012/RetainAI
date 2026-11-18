# RetainAI FastAPI Backend

This backend accepts raw customer characteristics, reproduces the model's saved training transformations, predicts churn probability, and applies retention policy separately from machine learning.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health/live` | Process liveness |
| `GET` | `/health/ready` | Model readiness (`503` until artifacts load) |
| `POST` | `/api/predict` | Churn score and retention action |
| `GET` | `/api/metrics` | Saved model metrics |
| `GET` | `/api/model/info` | Non-sensitive model contract metadata |
| `GET` | `/docs` | OpenAPI/Swagger UI |

## Local setup

Run all commands from the `RetainAI` root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cp .env.example .env
```

Place the raw dataset at `data/raw/eCommerce_dataset.csv`, then run:

```bash
python -m backend.ml_pipeline.src.data_prep
python -m backend.ml_pipeline.src.train_model
pytest backend/tests -v
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API can start in degraded mode when the model is absent: liveness remains healthy, readiness returns `503`, and prediction returns `503`. Set `RETAINAI_REQUIRE_MODEL_ON_STARTUP=true` in production to fail startup instead.

## Example prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customer_id": "C-1001",
    "features": {
      "Age": 36,
      "Gender": "Male",
      "Country": "India",
      "City": "Pune",
      "Signup_Quarter": "Q2",
      "Session_Duration_Avg": 14.2,
      "Pages_Per_Session": 6.5,
      "Discount_Usage_Rate": 0.25
    }
  }'
```

## Production notes

- Only load model pickle files produced and controlled by this project; pickle is unsafe for untrusted input.
- Configure explicit CORS origins. Never use `*` with credentialed browser requests.
- Mount `backend/ml_pipeline/artifacts` read-only in deployment.
- Keep `RETAINAI_REQUIRE_MODEL_ON_STARTUP=true` for fail-fast production releases.
- Put authentication, rate limiting, and TLS at an API gateway before exposing prediction endpoints publicly.
