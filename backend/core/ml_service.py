"""Load the saved training pipeline and make predictions."""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.config import METRICS_PATH, MODEL_PATH


class MLService:
    """Thin wrapper around one saved scikit-learn/XGBoost pipeline."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metrics_path: Path = METRICS_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.artifact: dict[str, Any] | None = None

    @property
    def loaded(self) -> bool:
        return self.artifact is not None

    @property
    def model_version(self) -> str:
        if not self.artifact:
            return "not-loaded"
        return str(self.artifact.get("version", "unknown"))

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Run the training command first."
            )

        artifact = joblib.load(self.model_path)
        if not isinstance(artifact, dict):
            raise ValueError("Model artifact must be a dictionary.")  # noqa: TRY004

        required = {"pipeline", "raw_features", "version"}
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"Model artifact is missing: {sorted(missing)}")
        if not hasattr(artifact["pipeline"], "predict_proba"):
            raise ValueError("Saved pipeline does not support predict_proba().")

        self.artifact = artifact

    def predict_churn(self, features: dict[str, Any]) -> float:
        """Return churn probability from 0.0 to 1.0."""
        if not self.artifact:
            raise RuntimeError("Model is not loaded. Train or load the model first.")
        if not features:
            raise ValueError("At least one customer feature is required.")

        raw_features = list(self.artifact["raw_features"])
        unknown = sorted(set(features) - set(raw_features))
        if unknown:
            raise ValueError(f"Unknown customer feature(s): {unknown}")

        # Missing values are intentionally left as NaN. The saved pipeline applies
        # exactly the same imputers and encoders used during training.
        row = {name: features.get(name, np.nan) for name in raw_features}
        frame = pd.DataFrame([row], columns=raw_features)
        probability = float(self.artifact["pipeline"].predict_proba(frame)[0, 1])
        return max(0.0, min(1.0, probability))

    def get_metrics(self) -> dict[str, Any]:
        if not self.artifact:
            raise RuntimeError("Model is not loaded.")
        return dict(self.artifact.get("metrics", {}))

    def get_model_info(self) -> dict[str, Any]:
        if not self.artifact:
            return {
                "ready": False,
                "model_version": None,
                "trained_at_utc": None,
                "raw_features": [],
            }
        return {
            "ready": True,
            "model_version": self.model_version,
            "trained_at_utc": self.artifact.get("trained_at_utc"),
            "raw_features": list(self.artifact["raw_features"]),
        }


ml_service = MLService()
