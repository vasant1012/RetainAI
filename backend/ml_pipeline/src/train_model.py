"""Train XGBoost and export a versioned inference-compatible artifact."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from backend.config import settings
from backend.logger import logger

ARTIFACT_VERSION = "1.2.0"
TARGET_CANDIDATES = ("Churned", "Churn")
CATEGORICAL_CANDIDATES = ("Gender", "Country", "City", "Signup_Quarter")
MISSING_CATEGORY = "__MISSING__"


def _target_name(dataframe: pd.DataFrame) -> str:
    for name in TARGET_CANDIDATES:
        if name in dataframe.columns:
            return name
    raise ValueError("Dataset must contain 'Churned' or 'Churn'.")


def _validate_target(target: pd.Series) -> pd.Series:
    if target.isna().any():
        raise ValueError("Target column contains missing values.")
    numeric = pd.to_numeric(target, errors="raise").astype(int)
    if set(numeric.unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes 0 and 1.")
    return numeric


def _fit_preprocessing(
    train: pd.DataFrame,
) -> tuple[list[str], list[str], dict[str, float], dict[str, list[str]]]:
    categorical = [name for name in CATEGORICAL_CANDIDATES if name in train.columns]
    unsupported = [
        name
        for name in train.select_dtypes(exclude=[np.number]).columns
        if name not in categorical
    ]
    if unsupported:
        raise ValueError(
            "Unsupported non-numeric features. Add an explicit encoding policy for: "
            f"{unsupported}"
        )
    numeric = [name for name in train.columns if name not in categorical]
    medians: dict[str, float] = {}
    for name in numeric:
        converted = pd.to_numeric(train[name], errors="coerce")
        median = converted.median()
        if pd.isna(median):
            raise ValueError(f"Numeric feature '{name}' has no usable values.")
        medians[name] = float(median)

    levels: dict[str, list[str]] = {}
    for name in categorical:
        values = train[name].astype("string").fillna(MISSING_CATEGORY)
        levels[name] = sorted(str(value) for value in values.unique())
    return numeric, categorical, medians, levels


def _transform(
    dataframe: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    medians: dict[str, float],
    levels: dict[str, list[str]],
) -> pd.DataFrame:
    transformed = dataframe.copy()
    for name in numeric:
        transformed[name] = pd.to_numeric(transformed[name], errors="coerce").fillna(medians[name])
    for name in categorical:
        values = transformed[name].astype("string").fillna(MISSING_CATEGORY)
        transformed[name] = pd.Categorical(values, categories=levels[name])
    transformed = pd.get_dummies(
        transformed,
        columns=categorical,
        drop_first=True,
        dtype=int,
    )
    if transformed.select_dtypes(exclude=[np.number]).columns.any():
        raise ValueError("Preprocessing left non-numeric model features.")
    return transformed


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


def train_and_export(raw: pd.DataFrame) -> dict[str, Any]:
    if raw.empty:
        raise ValueError("Training dataset is empty.")
    target_name = _target_name(raw)
    y = _validate_target(raw[target_name])
    X = raw.drop(columns=[target_name]).copy()
    if X.empty:
        raise ValueError("No model features are available.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    numeric, categorical, medians, levels = _fit_preprocessing(X_train_raw)
    X_train = _transform(X_train_raw, numeric, categorical, medians, levels)
    X_test = _transform(X_test_raw, numeric, categorical, medians, levels).reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    minority_count = int(y_train.value_counts().min())
    if minority_count < 2:
        raise ValueError("At least two minority-class training rows are required for SMOTE.")
    smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_resampled, y_resampled)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "resampled_train_samples": len(X_resampled),
    }
    metrics = _json_safe(metrics)

    preprocessing = {
        "numeric_medians": medians,
        "categorical_levels": levels,
        "raw_feature_names": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "drop_first": True,
        "unseen_category_policy": "all-zero encoded categorical group",
    }
    artifact = {
        "model": model,
        "feature_names": X_train.columns.tolist(),
        "metrics": metrics,
        "preprocessing": preprocessing,
        "target_column": "Churn",
        "model_type": "XGBClassifier",
        "artifact_version": ARTIFACT_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    settings.ensure_output_directories()
    with settings.MODEL_PATH.open("wb") as file:
        pickle.dump(artifact, file, protocol=pickle.HIGHEST_PROTOCOL)
    settings.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    full_processed = _transform(X, numeric, categorical, medians, levels).reindex(
        columns=X_train.columns,
        fill_value=0,
    )
    full_processed["Churn"] = y.to_numpy()
    full_processed.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    logger.info("Saved model artifact to %s", settings.MODEL_PATH)
    logger.info("Saved metrics to %s", settings.METRICS_PATH)
    return artifact


def main() -> None:
    if not settings.RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {settings.RAW_DATA_PATH}")
    train_and_export(pd.read_csv(settings.RAW_DATA_PATH))


if __name__ == "__main__":
    main()
