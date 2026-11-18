# RetainAI: Customer Churn & Win-Back Prediction Engine

RetainAI is a machine learning-based B2B SaaS MVP designed to identify at-risk e-commerce customers before they churn. By analyzing behavioral and transactional data, it calculates a real-time "Flight Risk Score," allowing marketing teams to automate targeted retention incentives. 

This monorepo contains the machine learning pipeline, the RESTful prediction API, and the cross-platform mobile frontend.

---

## 🛠 Tech Stack & Environment
* **OS & IDE**: Ubuntu 22.04 LTS, Visual Studio Code (augmented with Ollama and the Continue extension for local AI coding assistance).
* **Data & ML**: Python (Pandas, Scikit-learn, Imbalanced-learn/SMOTE, XGBoost).
* **Backend Serving**: FastAPI, Uvicorn, Pydantic.
* **Frontend Application**: React Native, Expo, NativeWind (Tailwind CSS) — *In Progress*.

---

## 🚀 Project Status: Phases 1–3 Completed

### Phase 1: Data Strategy (Completed)
* Loaded the raw `eCommerce-dataset.csv`.
* Handled widespread missing values across 14 numeric columns using median imputation to maintain robust distributions.
* Performed one-hot encoding on categorical text fields (Gender, Country, City, Signup_Quarter).
* Exported the cleaned, ML-ready dataset to `data/processed/processed_data.csv`.

### Phase 2: Machine Learning Pipeline (Completed)
* Handled the moderate class imbalance (approx. 35,000 retained vs. 14,500 churned) using SMOTE on the training split.
* Trained an optimized XGBoost classifier (`max_depth=6`, `learning_rate=0.05`).
* Generated evaluation metrics (F1 Score, Classification Report) on the test split.
* Exported the trained model, feature names array, and `metrics.json` directly into the `ml_pipeline/artifacts/` directory for backend consumption.

### Phase 3: Backend API (Completed)
* Built a high-performance REST API using FastAPI.
* Created the `MLService` singleton to load the `.pkl` and `.json` artifacts into memory securely on application startup.
* Configured CORS middleware for seamless integration with the upcoming mobile client.
* **Endpoints Configured:**
  * `POST /api/predict`: Accepts a JSON payload of a customer's behavioral features, dynamically shapes it to the exact training columns, and returns a 0-100 `churn_risk_score` with an automated marketing action.
  * `GET /api/metrics`: Returns the JSON metadata payload containing the model's F1-score and classification report.

---

## 💻 Local Setup & Execution Guide

### 1. Repository Initialization
Clone the repository and initialize the Python virtual environment:
```bash
git clone <your-repo-url>
cd retain-ai-mvp
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Run the ML Pipeline
To retrain the model and regenerate the serialized artifacts:
```bash
cd ml_pipeline/src
# 1. Clean data and engineer features
python data_prep.py
# 2. Apply SMOTE, train XGBoost, and export artifacts
python train_model.py
```

### 3. Start the FastAPI Backend
Serve the trained model locally:
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
*Access the auto-generated API documentation and test the endpoints at:* `http://localhost:8000/docs`

---

## 🗺 Roadmap: Upcoming Phases

### Phase 4: Frontend GUI (React Native + NativeWind)
* **Goal**: Build a cross-platform mobile dashboard to consume the FastAPI endpoints.
* **Tasks**:
  * Scaffold custom UI components (Risk Score Cards, User List, Metrics Dashboard).
  * Configure Axios to fetch real-time predictions from `POST /api/predict` and render "High Risk" flags.
  * Implement Tailwind utility classes via NativeWind for a clean, modern SaaS aesthetic.

### Phase 5: Refinement & Polish
* **Goal**: End-to-end local testing and preparation for deployment.
* **Tasks**:
  * Ensure API edge-case handling (e.g., missing keys in the JSON payload gracefully defaulting to medians/zeros).
  * Tune mobile UI responsiveness across different simulated device screens.
  * Containerize the backend with Docker for seamless production deployment.