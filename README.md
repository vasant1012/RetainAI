# RetainAI: Customer Churn and Win-Back Prediction Engine

RetainAI is a machine-learning-based B2B SaaS MVP designed to identify e-commerce customers who are at risk of churning. It analyzes behavioral and transactional signals, calculates a 0–100 churn-risk score, and maps that score to a retention recommendation that marketing teams can act on.

This monorepo contains:

- the raw and processed datasets;
- an exploratory data analysis and visualization workflow;
- an XGBoost training pipeline with SMOTE-based class balancing;
- versioned model, preprocessing, and evaluation artifacts;
- a FastAPI prediction backend;
- a React Native/Expo frontend that is planned for Phase 4.

> **Current state:** Phases 1 and 2 are complete. Phase 3A backend implementation is complete. Phase 3B backend hardening and validation is in progress. Frontend integration should begin only after the backend API contract and automated tests are stable.

---

## Product Goal

RetainAI converts customer data into an actionable retention signal:

```text
Raw customer behavior and transaction data
                    ↓
         Validation and preprocessing
                    ↓
          XGBoost churn classifier
                    ↓
          Churn probability (0–100)
                    ↓
        Risk level and retention action
```

The ML model is responsible for estimating churn probability. Risk bands and marketing recommendations are business rules layered on top of that probability and should remain independently configurable.

---

## Technology Stack

- **Development environment:** Ubuntu 22.04 LTS and Visual Studio Code
- **Local coding assistance:** Ollama with the Continue extension
- **Language:** Python 3.10+
- **Data processing:** Pandas and NumPy
- **Visualization:** Matplotlib and Seaborn
- **Machine learning:** Scikit-learn, Imbalanced-learn/SMOTE, and XGBoost
- **Backend API:** FastAPI, Uvicorn, and Pydantic
- **Testing:** Pytest
- **Frontend:** React Native, Expo, TypeScript, and NativeWind, planned for Phase 4

---

## Repository Structure

```text
RetainAI/
├── .gitignore
├── LICENSE
├── README.md
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── main.py
│   ├── requirements.txt
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── ml_service.py
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── artifacts/
│   │   │   ├── metrics.json
│   │   │   ├── xgboost_model.pkl
│   │   │   └── churn_distribution.png
│   │   ├── notebooks/
│   │   │   └── analysis.ipynb
│   │   └── src/
│   │       ├── __init__.py
│   │       ├── data_prep.py
│   │       └── train_model.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── pydantic_models.py
│   └── tests/
│       ├── __init__.py
│       └── test_ml_services.py
├── data/
│   ├── raw/
│   │   └── eCommerce_dataset.csv
│   └── processed/
│       └── processed_data.csv
├── frontend/
│   ├── App.tsx
│   ├── app.json
│   ├── index.ts
│   ├── package.json
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── assets/
└── logs/
```

Generated `__pycache__` directories and `.pyc` files are intentionally omitted from this structure and should not be committed.

---

### Model Artifact Contract

The model artifact currently contains:

```python
{
    "model": ...,                     # fitted XGBClassifier
    "feature_names": [...],          # exact encoded training feature order
    "metrics": {...},                # evaluation metrics
    "preprocessing": {
        "numeric_medians": {...},
        "categorical_levels": {...},
        "raw_feature_names": [...]
    },
    "target_column": "Churn",
    "model_type": "XGBClassifier",
    "artifact_version": "1.1.0"
}
```

This contract allows the inference service to reuse the preprocessing information produced during training.

## Local Setup

### 1. Clone and enter the repository

```bash
git clone <your-repository-url>
cd RetainAI
```

All Python commands in this README assume the current directory is the `RetainAI` project root.

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install backend and ML dependencies

```bash
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

Ensure `pytest` is present in the requirements file or install it for local validation:

```bash
pip install pytest
```

---

## Data Inspection

Run from the project root:

```bash
python -m backend.ml_pipeline.src.data_prep
```

Expected input:

```text
data/raw/eCommerce_dataset.csv
```

Expected output:

```text
backend/ml_pipeline/artifacts/churn_distribution.png
```

The command also logs dataset shape, deterministic sample records, missing-value counts, and target distribution.

---

## Model Training

Run the complete preprocessing and training workflow from the project root:

```bash
python -m backend.ml_pipeline.src.train_model
```

The command performs:

```text
Raw CSV
  → schema and target validation
  → numeric conversion and median imputation
  → categorical encoding
  → processed CSV export
  → stratified train/test split
  → SMOTE on training data only
  → XGBoost training
  → test-set evaluation
  → artifact export
```

Expected outputs:

```text
data/processed/processed_data.csv
backend/ml_pipeline/artifacts/xgboost_model.pkl
backend/ml_pipeline/artifacts/metrics.json
```

Whenever preprocessing or artifact structure changes, retrain the model before starting the backend.

---

## Running Tests

Run the MLService integration tests from the project root:

```bash
pytest backend/tests/test_ml_services.py -v --tb=short
```

For the complete backend test suite, once endpoint tests are added:

```bash
pytest backend/tests -v --tb=short
```

The MLService test suite reads the real trained model artifact and raw dataset. Run the training command before running these tests.

---

## Starting the FastAPI Backend

Start Uvicorn from the project root:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Prediction Flow

The intended API flow is:

```text
Raw customer JSON
        ↓
Pydantic request validation
        ↓
MLService raw-field validation
        ↓
Saved median imputation
        ↓
Categorical encoding
        ↓
Exact trained-feature alignment
        ↓
XGBoost predict_proba()
        ↓
Churn-risk score (0–100)
        ↓
Retention policy and recommendation
        ↓
API response
```

The frontend must send raw customer fields. It must not send pre-encoded columns such as city or country dummy variables.

---

## License

See `LICENSE` for licensing information.
