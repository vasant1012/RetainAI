"""
Model loading, inference preprocessing, and churn prediction for RetainAI.
"""

import json
import pickle
from typing import Any

import pandas as pd

from backend.config import settings
from backend.logger import logger


class MLService:
    """
    Load RetainAI model artifacts and provide churn predictions.

    The service uses preprocessing metadata saved by train_model.py to
    reproduce training-time preprocessing during inference.
    """

    def __init__(self) -> None:
        self.model = None

        self.feature_names: list[str] = []
        self.raw_feature_names: list[str] = []

        self.metrics: dict[str, Any] = {}
        self.preprocessing: dict[str, Any] = {}

        self.numeric_medians: dict[str, float] = {}
        self.categorical_levels: dict[str, list[str]] = {}

        self.artifact_version: str | None = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Load the trained model, preprocessing metadata, and metrics.

        Raises:
            FileNotFoundError:
                If required model or metrics artifacts don't exist.

            RuntimeError:
                If the model artifact is corrupt, incomplete, or was
                generated using the previous training implementation.
        """

        model_path = settings.MODEL_PATH
        metrics_path = settings.METRICS_PATH

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}. "
                "Run backend.ml_pipeline.src.train_model first."
            )

        try:
            with model_path.open("rb") as model_file:
                artifact = pickle.load(model_file)

        except (pickle.UnpicklingError, EOFError) as exc:
            raise RuntimeError(
                f"Model artifact is invalid or corrupted: {model_path}"
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"Could not read model artifact: {model_path}") from exc

        if not isinstance(artifact, dict):
            raise RuntimeError(  # noqa: TRY004
                "Model artifact must be a dictionary containing the model, "
                "feature names, and preprocessing metadata."
            )

        required_keys = {
            "model",
            "feature_names",
        }

        missing_keys = required_keys.difference(artifact.keys())

        if missing_keys:
            raise RuntimeError(
                f"Model artifact is missing required keys: {sorted(missing_keys)}"
            )

        self.model = artifact["model"]
        self.feature_names = list(artifact["feature_names"])

        self.preprocessing = artifact.get("preprocessing") or {}
        self.artifact_version = artifact.get("artifact_version")

        if not self.feature_names:
            raise RuntimeError(
                "Model artifact contains an empty feature_names list.")

        if not self.preprocessing:
            raise RuntimeError(
                "Model artifact does not contain preprocessing metadata. "
                "Retrain the model using the updated train_model.py before "
                "starting the API."
            )

        self.raw_feature_names = list(
            self.preprocessing.get(
                "raw_feature_names",
                [],
            )
        )

        numeric_medians = self.preprocessing.get(
            "numeric_medians",
            {},
        )

        self.numeric_medians = {
            str(column): float(median) for column, median in numeric_medians.items()
        }

        categorical_levels = self.preprocessing.get(
            "categorical_levels",
            {},
        )

        self.categorical_levels = {
            str(column): [str(category) for category in categories]
            for column, categories in categorical_levels.items()
        }

        if not self.raw_feature_names:
            raise RuntimeError(
                "Preprocessing metadata does not contain raw_feature_names."
            )

        self._load_metrics(
            artifact=artifact,
            metrics_path=metrics_path,
        )

        logger.info(
            "RetainAI model loaded successfully. Artifact version: %s. Model path: %s",
            self.artifact_version or "unknown",
            model_path,
        )

        logger.info(
            "Loaded %s raw features and %s encoded model features.",
            len(self.raw_feature_names),
            len(self.feature_names),
        )

    def _load_metrics(
        self,
        artifact: dict[str, Any],
        metrics_path,
    ) -> None:
        """
        Load metrics from metrics.json.

        If metrics.json is missing, use metrics embedded in the model
        artifact as a fallback.
        """

        if metrics_path.is_file():
            try:
                with metrics_path.open(
                    "r",
                    encoding="utf-8",
                ) as metrics_file:
                    metrics = json.load(metrics_file)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Metrics artifact contains invalid JSON: {metrics_path}"
                ) from exc

            except OSError as exc:
                raise RuntimeError(
                    f"Could not read metrics artifact: {metrics_path}"
                ) from exc

            if not isinstance(metrics, dict):
                raise RuntimeError(
                    "Metrics artifact must contain a JSON object.")

            self.metrics = metrics
            return

        embedded_metrics = artifact.get("metrics")

        if not isinstance(embedded_metrics, dict):
            raise FileNotFoundError(
                f"Metrics artifact not found: {metrics_path}")

        self.metrics = embedded_metrics

        logger.warning(
            "Metrics file not found at %s. Using metrics embedded "
            "inside the model artifact.",
            metrics_path,
        )

    def _validate_customer_features(
        self,
        customer_features: dict[str, Any],
    ) -> None:
        """
        Validate the raw customer payload.

        Unknown fields are rejected to prevent spelling mistakes or
        unsupported fields from being silently ignored.
        """

        if not isinstance(customer_features, dict):
            raise ValueError("customer_features must be a dictionary.")  # noqa: TRY004

        if not customer_features:
            raise ValueError("customer_features cannot be empty.")

        unknown_fields = sorted(
            set(customer_features.keys()).difference(self.raw_feature_names)
        )

        if unknown_fields:
            raise ValueError(
                "Unknown customer feature(s): "
                f"{unknown_fields}. "
                "Expected raw customer features are: "
                f"{self.raw_feature_names}"
            )

    def _create_raw_dataframe(
        self,
        customer_features: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Create a one-row DataFrame using the saved raw feature contract.

        Features omitted by the caller are represented as missing values.
        Numeric missing values are subsequently median-imputed.
        """

        raw_record = {
            feature_name: customer_features.get(
                feature_name,
                pd.NA,
            )
            for feature_name in self.raw_feature_names
        }

        return pd.DataFrame([raw_record])

    def _process_numeric_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply numeric conversion and saved median imputation.

        This reproduces the numeric preprocessing performed by
        train_model.py.
        """

        for column, median in self.numeric_medians.items():
            if column not in dataframe.columns:
                dataframe[column] = pd.NA

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            dataframe[column] = dataframe[column].fillna(median)

        return dataframe

    def _process_categorical_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply categorical missing-value handling and dummy encoding.

        Categories not seen during training are allowed. Their generated
        columns are removed during final feature alignment because the
        model was not trained on those columns.
        """

        categorical_columns = list(self.categorical_levels.keys())

        for column in categorical_columns:
            if column not in dataframe.columns:
                dataframe[column] = "Unknown"

            dataframe[column] = dataframe[column].astype(
                "string").fillna("Unknown")

        if categorical_columns:
            dataframe = pd.get_dummies(
                dataframe,
                columns=categorical_columns,
                drop_first=True,
                dtype=int,
            )

        return dataframe

    def _align_model_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align inference columns with the exact training feature contract.

        This operation:

        1. Adds training features that are absent from the current record.
        2. Sets those missing encoded features to zero.
        3. Removes columns created for unseen inference categories.
        4. Reorders features to match the model's expected input order.
        """

        dataframe = dataframe.reindex(
            columns=self.feature_names,
            fill_value=0,
        )

        non_numeric_columns = [
            column
            for column in dataframe.columns
            if not pd.api.types.is_numeric_dtype(dataframe[column])
        ]

        if non_numeric_columns:
            raise ValueError(
                "Inference preprocessing left non-numeric "
                "model features: "
                f"{non_numeric_columns}"
            )

        return dataframe

    def preprocess_customer_features(
        self,
        customer_features: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert one raw customer record into model-ready features.

        This method is public so it can be tested independently from
        prediction.
        """

        self._validate_customer_features(customer_features)

        dataframe = self._create_raw_dataframe(customer_features)

        dataframe = self._process_numeric_features(dataframe)

        dataframe = self._process_categorical_features(dataframe)

        dataframe = self._align_model_features(dataframe)

        return dataframe

    def predict_churn(
        self,
        customer_features: dict[str, Any],
    ) -> float:
        """
        Predict customer churn probability.

        Args:
            customer_features:
                Raw customer feature names and values.

        Returns:
            Churn probability as a percentage from 0 to 100,
            rounded to two decimal places.
        """

        if self.model is None:
            raise RuntimeError("The churn model is not loaded.")

        model_input = self.preprocess_customer_features(customer_features)

        try:
            prediction_probabilities = self.model.predict_proba(model_input)

        except (ValueError, TypeError) as exc:
            logger.exception(
                "Model prediction failed due to invalid inference data.")

            raise ValueError(
                "The supplied customer features could not be "
                "processed by the trained model."
            ) from exc

        if (
            prediction_probabilities.ndim != 2
            or prediction_probabilities.shape[0] != 1
            or prediction_probabilities.shape[1] < 2
        ):
            raise RuntimeError(
                "The model returned an unexpected probability output shape."
            )

        churn_probability = float(prediction_probabilities[0][1])

        churn_percentage = round(
            churn_probability * 100,
            2,
        )

        logger.info(
            "Churn prediction completed successfully. Predicted probability: %.2f%%",
            churn_percentage,
        )

        return churn_percentage

    def get_metrics(self) -> dict[str, Any]:
        """Return a copy of the saved model metrics."""

        return dict(self.metrics)


# Shared singleton used by the API endpoints.
#
# Importing this module loads and validates the model artifacts. Therefore,
# train_model.py must be executed before starting FastAPI.
ml_service = MLService()
