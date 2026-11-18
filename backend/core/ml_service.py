"""Artifact loading, inference preprocessing, and churn prediction."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import settings
from backend.core.exceptions import ArtifactError, InvalidFeatureError, ModelUnavailableError
from backend.logger import logger


class MLService:
    """Load a trusted RetainAI artifact and serve deterministic predictions."""

    def __init__(
        self,
        model_path: Path | None = None,
        metrics_path: Path | None = None,
    ) -> None:
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.metrics_path = Path(metrics_path or settings.METRICS_PATH)
        self.model: Any | None = None
        self.pipeline: Any | None = None
        self.feature_names: list[str] = []
        self.raw_feature_names: list[str] = []
        self.metrics: dict[str, Any] = {}
        self.preprocessing: dict[str, Any] = {}
        self.numeric_medians: dict[str, float] = {}
        self.categorical_levels: dict[str, list[str]] = {}
        self.artifact_version: str | None = None
        self.model_type: str | None = None
        self.loaded = False
        self.load_error: str | None = None

    def load(self) -> None:
        """Load and validate artifacts. Only load pickle files you trust."""
        self.loaded = False
        self.load_error = None
        if not self.model_path.is_file():
            self._fail(f"Model artifact not found: {self.model_path}")

        try:
            with self.model_path.open("rb") as file:
                artifact = pickle.load(file)  # nosec B301: local trusted training artifact
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as exc:
            self._fail("Model artifact is invalid, corrupt, or incompatible.", exc)
        except OSError as exc:
            self._fail("Model artifact could not be read.", exc)

        if not isinstance(artifact, dict):
            self._fail("Model artifact must be a dictionary.")

        self.pipeline = artifact.get("pipeline")
        self.model = artifact.get("model")
        if self.pipeline is None and self.model is None:
            self._fail("Model artifact is missing 'model' or 'pipeline'.")

        self.feature_names = [str(item) for item in artifact.get("feature_names", [])]
        self.preprocessing = artifact.get("preprocessing") or {}
        self.artifact_version = str(artifact.get("artifact_version") or "unknown")
        self.model_type = str(artifact.get("model_type") or type(self.model).__name__)
        self.raw_feature_names = [
            str(item)
            for item in (
                self.preprocessing.get("raw_feature_names")
                or artifact.get("raw_feature_names")
                or []
            )
        ]
        self.numeric_medians = {
            str(name): float(value)
            for name, value in (self.preprocessing.get("numeric_medians") or {}).items()
        }
        self.categorical_levels = {
            str(name): [str(value) for value in values]
            for name, values in (self.preprocessing.get("categorical_levels") or {}).items()
        }

        if not self.raw_feature_names:
            self._fail("Artifact preprocessing metadata has no raw_feature_names.")
        if self.pipeline is None:
            if not self.feature_names:
                self._fail("Legacy artifact has no encoded feature_names.")
            if not self.preprocessing:
                self._fail("Legacy artifact has no preprocessing metadata.")

        self.metrics = self._load_metrics(artifact)
        self.loaded = True
        logger.info(
            "Model loaded: version=%s type=%s raw_features=%d model_features=%d",
            self.artifact_version,
            self.model_type,
            len(self.raw_feature_names),
            len(self.feature_names),
        )

    def _fail(self, message: str, cause: Exception | None = None) -> None:
        self.load_error = message
        if cause is None:
            raise ArtifactError(message)
        raise ArtifactError(message) from cause

    def _load_metrics(self, artifact: dict[str, Any]) -> dict[str, Any]:
        if self.metrics_path.is_file():
            try:
                value = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                self._fail("Metrics artifact is invalid or unreadable.", exc)
            if not isinstance(value, dict):
                self._fail("Metrics artifact must contain a JSON object.")
            return value
        embedded = artifact.get("metrics")
        if isinstance(embedded, dict):
            logger.warning("metrics.json missing; using metrics embedded in model artifact.")
            return dict(embedded)
        self._fail("No valid metrics artifact is available.")
        return {}

    def _validate_customer_features(self, values: dict[str, Any]) -> None:
        if not isinstance(values, dict):
            raise InvalidFeatureError("customer features must be a dictionary.")
        if not values:
            raise InvalidFeatureError("customer features cannot be empty.")
        unknown = sorted(set(values) - set(self.raw_feature_names))
        if unknown:
            raise InvalidFeatureError(f"Unknown customer feature(s): {unknown}")

    def _raw_frame(self, values: dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(
            [{name: values.get(name, pd.NA) for name in self.raw_feature_names}]
        )

    def preprocess_customer_features(self, values: dict[str, Any]) -> pd.DataFrame:
        """Reproduce legacy 1.1 training transformations exactly."""
        self._validate_customer_features(values)
        frame = self._raw_frame(values)

        for column, median in self.numeric_medians.items():
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(median)

        categorical_columns = list(self.categorical_levels)
        for column, levels in self.categorical_levels.items():
            # Fixed categories are essential: one-row get_dummies(drop_first=True)
            # would otherwise erase every supplied category.
            normalized = frame[column].astype("string")
            frame[column] = pd.Categorical(normalized, categories=levels)

        if categorical_columns:
            frame = pd.get_dummies(
                frame,
                columns=categorical_columns,
                drop_first=True,
                dtype=int,
            )

        frame = frame.reindex(columns=self.feature_names, fill_value=0)
        non_numeric = [
            column
            for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column])
        ]
        if non_numeric:
            raise InvalidFeatureError(
                f"Preprocessing left non-numeric model features: {non_numeric}"
            )
        return frame

    def predict_churn(self, values: dict[str, Any]) -> float:
        if not self.loaded:
            raise ModelUnavailableError("The churn model is not ready.")
        self._validate_customer_features(values)
        try:
            if self.pipeline is not None:
                probabilities = self.pipeline.predict_proba(self._raw_frame(values))
            else:
                probabilities = self.model.predict_proba(
                    self.preprocess_customer_features(values)
                )
            probability = float(probabilities[0][1])
        except InvalidFeatureError:
            raise
        except (ValueError, TypeError, IndexError, AttributeError) as exc:
            logger.exception("Model prediction failed.")
            raise InvalidFeatureError(
                "The supplied customer features could not be processed."
            ) from exc

        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ModelUnavailableError("The model returned an invalid probability.")
        return round(probability * 100, 2)

    def get_metrics(self) -> dict[str, Any]:
        if not self.loaded:
            raise ModelUnavailableError("Model metrics are not ready.")
        return dict(self.metrics)

    def get_model_info(self) -> dict[str, Any]:
        return {
            "ready": self.loaded,
            "artifact_version": self.artifact_version,
            "model_type": self.model_type,
            "raw_feature_count": len(self.raw_feature_names),
            "encoded_feature_count": len(self.feature_names),
            "raw_features": list(self.raw_feature_names),
        }
