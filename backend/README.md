# RetainAI Simple Backend

This version deliberately removes the earlier preprocessing metadata and configuration layers.
One saved pipeline owns numeric imputation, categorical imputation, one-hot encoding, and XGBoost prediction. The exact same pipeline is used during training and inference.

## Replace the old backend

1. Back up the current `RetainAI/backend/` folder.
2. Replace it with the `backend/` folder from this package instead of merging file by file.
3. Keep the raw dataset here:

```text
data/raw/eCommerce_dataset.csv
```

Do not reuse the old `.pkl` model. This version writes `churn_pipeline.joblib`, so retrain once after replacement. The old `api/`, `logger.py`, `retention_service.py`, manual preprocessing metadata, and exception layers are not required by this version.

The training script accepts `Churned` or `Churn` as the target and automatically removes common customer ID columns.

## Setup

Run from the `RetainAI` project root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt
```

## Optional data check

```bash
python -m backend.ml_pipeline.src.data_prep
```

This writes `backend/ml_pipeline/artifacts/data_summary.json`. It does not transform the training data; the saved model pipeline owns preprocessing.

## Train

```bash
python -m backend.ml_pipeline.src.train_model
```

Generated files:

```text
backend/ml_pipeline/artifacts/churn_pipeline.joblib
backend/ml_pipeline/artifacts/metrics.json
```

## Test

```bash
pytest backend/tests -v --tb=short
```

## Run

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`.

## Endpoints

- `GET /`
- `GET /health`
- `POST /api/predict`
- `GET /api/metrics`
- `GET /api/model/info`

Use `example_request.json` as a Swagger request. Fields may be omitted; the pipeline uses training medians for numeric values and the most frequent training value for categorical values. Unexpected request fields are rejected unless they exist in the trained artifact.

## Why this is simpler

- No environment-settings framework.
- No current-working-directory path logic.
- No manual list of encoded columns.
- No manual preprocessing metadata.
- No separate SMOTE step.
- No duplicated training and prediction transforms.
- One model file contains the complete fitted pipeline and basic metadata.

For production later, add authentication, persistent prediction audit storage, rate limiting, and restrictive CORS only when they are actually needed.
