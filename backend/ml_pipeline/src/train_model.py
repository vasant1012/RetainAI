"""Data preprocessing, XGBoost training, evaluation, and artifact export."""

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Allow both module execution and direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.logger import logger

TARGET_SOURCE_COLUMN = "Churned"
TARGET_COLUMN = "Churn"
RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "Age",
    "Session_Duration_Avg",
    "Pages_Per_Session",
    "Wishlist_Items",
    "Days_Since_Last_Purchase",
    "Discount_Usage_Rate",
    "Returns_Rate",
    "Email_Open_Rate",
    "Customer_Service_Calls",
    "Product_Reviews_Written",
    "Social_Media_Engagement_Score",
    "Mobile_App_Usage",
    "Payment_Method_Diversity",
    "Credit_Balance",
]

CATEGORICAL_COLUMNS = ["Gender", "Country", "City", "Signup_Quarter"]


def _to_builtin(value: Any) -> Any:
    """Convert NumPy values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate the source dataset before preprocessing."""
    if df.empty:
        raise ValueError("The raw dataset is empty.")

    target_candidates = {TARGET_SOURCE_COLUMN, TARGET_COLUMN}
    if not target_candidates.intersection(df.columns):
        raise ValueError(
            f"Target column '{TARGET_SOURCE_COLUMN}' or '{TARGET_COLUMN}' "
            f"was not found. Available columns: {df.columns.tolist()}"
        )


def load_and_clean_data(df: pd.DataFrame) -> dict[str, Any]:
    """Clean, encode, and persist the dataset used for model training."""
    validate_raw_data(df)
    cleaned_df = df.copy()

    if TARGET_SOURCE_COLUMN in cleaned_df.columns:
        cleaned_df = cleaned_df.rename(
            columns={TARGET_SOURCE_COLUMN: TARGET_COLUMN})

    if cleaned_df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")

    target_values = set(cleaned_df[TARGET_COLUMN].dropna().unique().tolist())
    if not target_values.issubset({0, 1, False, True}):  # noqa: B033
        raise ValueError(
            f"Target column must be binary 0/1. Found values: {sorted(target_values)}"
        )
    cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].astype(int)

    numeric_medians: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        if column not in cleaned_df.columns:
            logger.warning("Expected numeric column is absent: %s", column)
            continue

        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
        median = cleaned_df[column].median()
        if pd.isna(median):
            raise ValueError(
                f"Numeric column '{column}' contains no usable numeric values."
            )
        numeric_medians[column] = float(median)
        cleaned_df[column] = cleaned_df[column].fillna(median)

    existing_categorical_columns = [
        column for column in CATEGORICAL_COLUMNS if column in cleaned_df.columns
    ]
    categorical_levels: dict[str, list[str]] = {}
    for column in existing_categorical_columns:
        cleaned_df[column] = cleaned_df[column].astype(
            "string").fillna("Unknown")
        categorical_levels[column] = sorted(
            cleaned_df[column].dropna().astype(str).unique().tolist()
        )

    encoded_df = pd.get_dummies(
        cleaned_df,
        columns=existing_categorical_columns,
        drop_first=True,
        dtype=int,
    )

    feature_columns = [
        column for column in encoded_df.columns if column != TARGET_COLUMN
    ]
    non_numeric_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(encoded_df[column])
    ]
    if non_numeric_features:
        raise ValueError(
            "Unencoded non-numeric feature columns remain: "
            f"{non_numeric_features}. Add them to CATEGORICAL_COLUMNS or remove them."
        )

    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    encoded_df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    logger.info("Processed data saved to: %s", settings.PROCESSED_DATA_PATH)

    return {
        "df_encoded": encoded_df,
        "numeric_medians": numeric_medians,
        "categorical_levels": categorical_levels,
        "raw_feature_names": [
            column for column in cleaned_df.columns if column != TARGET_COLUMN
        ],
    }


def train_and_export(
    df: pd.DataFrame,
    preprocessing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the churn model and save model and metrics artifacts."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Processed dataset must contain '{TARGET_COLUMN}'.")

    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]

    if features.empty:
        raise ValueError(
            "No model features are available after preprocessing.")
    if target.nunique() != 2:
        raise ValueError("Target column must contain exactly two classes.")

    minimum_class_count = int(target.value_counts().min())
    if minimum_class_count < 2:
        raise ValueError(
            "Each target class requires at least two records for a stratified split."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    logger.info(
        "Original training target distribution:\n%s",
        y_train.value_counts().sort_index().to_string(),
    )

    minority_train_count = int(y_train.value_counts().min())
    if minority_train_count < 2:
        raise ValueError(
            "SMOTE requires at least two minority-class training records.")
    smote_neighbors = min(5, minority_train_count - 1)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_neighbors)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    logger.info(
        "Resampled training target distribution:\n%s",
        y_train_resampled.value_counts().sort_index().to_string(),
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    logger.info("Training XGBoost model...")
    model.fit(x_train_resampled, y_train_resampled)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    metrics_data = {
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": report,
        "test_size": int(len(y_test)),  # noqa: RUF046
        "training_size_before_smote": int(len(y_train)),  # noqa: RUF046
        "training_size_after_smote": int(len(y_train_resampled)),  # noqa: RUF046
        "random_state": RANDOM_STATE,
    }
    metrics_data = _to_builtin(metrics_data)

    logger.info(
        "Classification report (test data):\n%s",
        classification_report(y_test, predictions, zero_division=0),
    )
    logger.info("F1 score: %.4f", metrics_data["f1_score"])
    logger.info("ROC-AUC: %.4f", metrics_data["roc_auc"])

    artifact = {
        "model": model,
        "feature_names": features.columns.tolist(),
        "metrics": metrics_data,
        "preprocessing": preprocessing_metadata or {},
        "target_column": TARGET_COLUMN,
        "model_type": "XGBClassifier",
        "artifact_version": "1.1.0",
    }

    settings.ML_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with settings.MODEL_PATH.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    with settings.METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_data, metrics_file, indent=4)

    logger.info("Model artifact exported to: %s", settings.MODEL_PATH)
    logger.info("Metrics exported to: %s", settings.METRICS_PATH)
    return metrics_data


def main() -> None:
    """Run preprocessing and model training from the raw dataset."""
    if not settings.RAW_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found: {settings.RAW_DATA_PATH}")

    logger.info("Loading raw dataset from: %s", settings.RAW_DATA_PATH)
    raw_df = pd.read_csv(settings.RAW_DATA_PATH)

    preprocessing_result = load_and_clean_data(raw_df)
    encoded_df = preprocessing_result.pop("df_encoded")
    train_and_export(encoded_df, preprocessing_result)


if __name__ == "__main__":
    try:
        main()
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        pickle.PickleError,
        pd.errors.ParserError,
    ) as exc:
        logger.exception("Model training failed: %s", exc)
        raise SystemExit(1) from exc
