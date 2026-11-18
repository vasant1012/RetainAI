"""Train and save one pipeline that includes preprocessing and XGBoost."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from backend.config import (
    APP_VERSION,
    METRICS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    ensure_directories,
)
from backend.logger import logger

TARGET_CANDIDATES = ("Churned", "Churn")
ID_CANDIDATES = ("Customer_ID", "CustomerID", "customer_id", "ID", "Id")


def _target_name(dataframe: pd.DataFrame) -> str:
    target = next((name for name in TARGET_CANDIDATES if name in dataframe), None)
    if not target:
        logger.error("Dataset must contain a 'Churned' or 'Churn' column.")
        raise ValueError("Dataset must contain a 'Churned' or 'Churn' column.")
    return target


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def train_and_save(
    dataframe: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, Any]:
    """Train from a raw dataframe and save a reusable inference pipeline."""
    if dataframe.empty:
        logger.error("Training dataset is empty.")
        raise ValueError("Training dataset is empty.")

    target = _target_name(dataframe)
    y = pd.to_numeric(dataframe[target], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        logger.error("Target must contain both classes 0 and 1.")
        raise ValueError("Target must contain both classes 0 and 1.")

    drop_columns = [target] + [name for name in ID_CANDIDATES if name in dataframe]
    X = dataframe.drop(columns=drop_columns).copy()
    if X.empty:
        logger.error("No model features remain after removing target and ID columns.")
        raise ValueError("No model features remain after removing target and ID columns.")

    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [name for name in X.columns if name not in categorical]
    if not numeric and not categorical:
        logger.error("No usable model features were found.")
        raise ValueError("No usable model features were found.")

    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            )
        )

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())
    scale_pos_weight = negative_count / positive_count

    pipeline = Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers=transformers)),
            (
                "model",
                XGBClassifier(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    logger.info(f"Training with {len(X_train)} samples and testing with {len(X_test)} samples.")
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = _json_safe(
        {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
    )

    artifact = {
        "pipeline": pipeline,
        "raw_features": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target_column": target,
        "threshold": 0.5,
        "version": APP_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "metrics": metrics,
    }

    model_path = Path(model_path)
    metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return artifact


def main() -> None:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
    ensure_directories()
    artifact = train_and_save(pd.read_csv(RAW_DATA_PATH))
    logger.info(f"Saved model to {MODEL_PATH}")
    logger.info(f"Saved metrics to {METRICS_PATH}")
    logger.info(f"Model features: {len(artifact['raw_features'])}")


if __name__ == "__main__":
    main()
