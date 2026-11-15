import json
import pickle
from pathlib import Path

import pandas as pd


class MLService:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.metrics = None

        # Resolve paths relative to the backend directory
        self.project_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.artifacts_dir = self.project_dir / "ml_pipeline" / "artifacts"

        self._load_artifacts()

    def _load_artifacts(self):
        model_path = self.artifacts_dir / "xgboost_model.pkl"
        metrics_path = self.artifacts_dir / "metrics.json"

        try:
            # Load Model
            with open(model_path, "rb") as f:
                artifacts = pickle.load(f)
                self.model = artifacts["model"]
                self.feature_names = artifacts["feature_names"]

            # Load Metrics
            with open(metrics_path, "r") as f:
                self.metrics = json.load(f)

            print(
                "Successfully loaded XGBoost model and metrics from "
                f"{self.artifacts_dir}"
            )
        except FileNotFoundError as e:
            print(
                "Error loading artifacts: "
                f"{e}. Ensure train_model.py has been run."
            )

    def predict_churn(self, customer_features: dict) -> float:
        if not self.model:
            raise RuntimeError("Model is not loaded.")

        # Convert dictionary to DataFrame
        df = pd.DataFrame([customer_features])

        # Ensure all required features are present and in the exact order
        # Missing columns will be filled with 0 (or you could use medians)
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_names]

        # Get probability of class 1 (Churn)
        churn_prob = self.model.predict_proba(df)[0][1]

        # Return as a percentage score (0-100)
        return round(float(churn_prob) * 100, 2)


# Instantiate as a singleton to be used across endpoints
ml_service = MLService()
