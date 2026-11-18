"""
Integration tests for RetainAI MLService.

These tests validate:

1. Model artifact loading
2. Preprocessing metadata loading
3. Raw customer preprocessing
4. Training/inference feature alignment
5. Churn prediction
6. Numeric median imputation
7. Unseen categorical values
8. Unknown feature rejection
9. Metrics loading

Run from the RetainAI project root:

    pytest backend/tests/test_ml_services.py -v
"""

from typing import Any

import pandas as pd
import pytest

from backend.config import settings
from backend.core.ml_service import ml_service

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def raw_dataframe() -> pd.DataFrame:
    """
    Load the RetainAI raw dataset once for the complete test session.
    """

    assert settings.RAW_DATA_PATH.is_file(), (
        f"Raw dataset not found: {settings.RAW_DATA_PATH}"
    )

    df = pd.read_csv(settings.RAW_DATA_PATH)

    assert not df.empty, "Raw dataset is empty."

    return df


@pytest.fixture()
def customer_features(
    raw_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build one valid raw inference record using the actual training dataset.

    The target variable is removed because it must never be passed
    to the inference service.
    """

    customer = raw_dataframe.iloc[0].to_dict()

    customer.pop("Churned", None)
    customer.pop("Churn", None)

    return customer


# ---------------------------------------------------------------------------
# Artifact loading tests
# ---------------------------------------------------------------------------


def test_model_artifact_exists() -> None:
    """
    Verify that the trained XGBoost artifact exists.
    """

    assert settings.MODEL_PATH.is_file(), (
        f"Model artifact not found: {settings.MODEL_PATH}. "
        "Run the training pipeline first."
    )


def test_metrics_artifact_exists() -> None:
    """
    Verify that the model metrics artifact exists.
    """

    assert settings.METRICS_PATH.is_file(), (
        f"Metrics artifact not found: {settings.METRICS_PATH}. "
        "Run the training pipeline first."
    )


def test_model_loaded() -> None:
    """
    Verify that MLService successfully loaded the trained model.
    """

    assert ml_service.model is not None


def test_artifact_version() -> None:
    """
    Verify that the model was generated using the updated training pipeline.
    """

    assert ml_service.artifact_version == "1.1.0"


# ---------------------------------------------------------------------------
# Preprocessing metadata tests
# ---------------------------------------------------------------------------


def test_preprocessing_metadata_loaded() -> None:
    """
    Verify that preprocessing metadata exists in the model artifact.
    """

    assert ml_service.preprocessing

    assert isinstance(
        ml_service.preprocessing,
        dict,
    )


def test_raw_feature_names_loaded() -> None:
    """
    Verify that the original raw model features were stored.
    """

    assert ml_service.raw_feature_names

    assert isinstance(
        ml_service.raw_feature_names,
        list,
    )


def test_numeric_medians_loaded() -> None:
    """
    Verify that numerical imputation values were loaded.
    """

    assert ml_service.numeric_medians

    assert isinstance(
        ml_service.numeric_medians,
        dict,
    )


def test_expected_categorical_features_loaded() -> None:
    """
    Verify the categorical encoding metadata generated during training.
    """

    expected_columns = {
        "Gender",
        "Country",
        "City",
        "Signup_Quarter",
    }

    actual_columns = set(
        ml_service.categorical_levels.keys()
    )

    missing_columns = expected_columns - actual_columns

    assert not missing_columns, (
        "Expected categorical preprocessing metadata is missing "
        f"for: {sorted(missing_columns)}"
    )


# ---------------------------------------------------------------------------
# Basic preprocessing tests
# ---------------------------------------------------------------------------


def test_preprocess_customer_features(
    customer_features: dict[str, Any],
) -> None:
    """
    Verify that a real raw customer can be converted into
    model-ready features.
    """

    processed = ml_service.preprocess_customer_features(
        customer_features
    )

    assert isinstance(
        processed,
        pd.DataFrame,
    )

    assert processed.shape[0] == 1


def test_processed_feature_count_matches_model(
    customer_features: dict[str, Any],
) -> None:
    """
    The inference feature count must exactly match the feature
    count used while training XGBoost.
    """

    processed = ml_service.preprocess_customer_features(
        customer_features
    )

    assert processed.shape[1] == len(
        ml_service.feature_names
    )


def test_processed_feature_order_matches_model(
    customer_features: dict[str, Any],
) -> None:
    """
    Feature order is important for model inference.

    Verify that inference features use the exact order stored
    during training.
    """

    processed = ml_service.preprocess_customer_features(
        customer_features
    )

    assert processed.columns.tolist() == (
        ml_service.feature_names
    )


def test_all_processed_features_are_numeric(
    customer_features: dict[str, Any],
) -> None:
    """
    XGBoost must receive only numerical feature columns.
    """

    processed = ml_service.preprocess_customer_features(
        customer_features
    )

    non_numeric_columns = [
        column
        for column in processed.columns
        if not pd.api.types.is_numeric_dtype(
            processed[column]
        )
    ]

    assert non_numeric_columns == []


# ---------------------------------------------------------------------------
# Prediction tests
# ---------------------------------------------------------------------------


def test_predict_churn(
    customer_features: dict[str, Any],
) -> None:
    """
    Verify that MLService can successfully generate
    a churn probability.
    """

    risk_score = ml_service.predict_churn(
        customer_features
    )

    assert isinstance(
        risk_score,
        float,
    )


def test_prediction_is_valid_percentage(
    customer_features: dict[str, Any],
) -> None:
    """
    Churn prediction must remain within the API's 0-100 scale.
    """

    risk_score = ml_service.predict_churn(
        customer_features
    )

    assert 0.0 <= risk_score <= 100.0


# ---------------------------------------------------------------------------
# Missing-value tests
# ---------------------------------------------------------------------------


def test_missing_numeric_features_use_saved_medians(
    customer_features: dict[str, Any],
) -> None:
    """
    Verify that missing numerical inputs use the exact medians
    saved during model training.
    """

    test_customer = customer_features.copy()

    columns_to_test = [
        column
        for column in (
            "Age",
            "Session_Duration_Avg",
        )
        if column in ml_service.numeric_medians
    ]

    assert columns_to_test, (
        "No expected numeric columns were found in "
        "preprocessing metadata."
    )

    for column in columns_to_test:
        test_customer[column] = None

    processed = ml_service.preprocess_customer_features(
        test_customer
    )

    for column in columns_to_test:

        expected_median = (
            ml_service.numeric_medians[column]
        )

        actual_value = float(
            processed.iloc[0][column]
        )

        assert actual_value == pytest.approx(
            expected_median
        )


def test_prediction_with_missing_numeric_values(
    customer_features: dict[str, Any],
) -> None:
    """
    Prediction should still succeed when supported numeric
    features are missing.
    """

    test_customer = customer_features.copy()

    if "Age" in test_customer:
        test_customer["Age"] = None

    if "Session_Duration_Avg" in test_customer:
        test_customer["Session_Duration_Avg"] = None

    risk_score = ml_service.predict_churn(
        test_customer
    )

    assert 0.0 <= risk_score <= 100.0


# ---------------------------------------------------------------------------
# Categorical-value tests
# ---------------------------------------------------------------------------


def test_unseen_city_does_not_break_preprocessing(
    customer_features: dict[str, Any],
) -> None:
    """
    A city that was never present during training must not break
    feature transformation.
    """

    if "City" not in ml_service.raw_feature_names:
        pytest.skip(
            "City is not part of the trained feature contract."
        )

    test_customer = customer_features.copy()

    test_customer["City"] = (
        "__RETAINAI_PREVIOUSLY_UNSEEN_CITY__"
    )

    processed = ml_service.preprocess_customer_features(
        test_customer
    )

    assert processed.shape[1] == len(
        ml_service.feature_names
    )


def test_prediction_with_unseen_city(
    customer_features: dict[str, Any],
) -> None:
    """
    Prediction should succeed for a categorical value not
    observed during training.
    """

    if "City" not in ml_service.raw_feature_names:
        pytest.skip(
            "City is not part of the trained feature contract."
        )

    test_customer = customer_features.copy()

    test_customer["City"] = (
        "__RETAINAI_PREVIOUSLY_UNSEEN_CITY__"
    )

    risk_score = ml_service.predict_churn(
        test_customer
    )

    assert 0.0 <= risk_score <= 100.0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


def test_unknown_feature_is_rejected(
    customer_features: dict[str, Any],
) -> None:
    """
    Unknown feature names must fail instead of being silently ignored.
    """

    test_customer = customer_features.copy()

    test_customer["Wrong_Feature_Name"] = 123

    with pytest.raises(
        ValueError,
        match="Unknown customer feature",
    ):
        ml_service.predict_churn(
            test_customer
        )


def test_empty_customer_payload_is_rejected() -> None:
    """
    Empty inference requests must not reach XGBoost.
    """

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        ml_service.predict_churn({})


def test_non_dictionary_payload_is_rejected() -> None:
    """
    MLService must reject payloads that are not dictionaries.
    """

    with pytest.raises(
        ValueError,
        match="must be a dictionary",
    ):
        ml_service.predict_churn(
            None  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_metrics_loaded() -> None:
    """
    Verify that model metrics are available through MLService.
    """

    metrics = ml_service.get_metrics()

    assert isinstance(
        metrics,
        dict,
    )

    assert metrics


def test_expected_metrics_present() -> None:
    """
    Verify the main metrics saved by the updated training pipeline.
    """

    metrics = ml_service.get_metrics()

    expected_metrics = {
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
        "confusion_matrix",
    }

    missing_metrics = (
        expected_metrics - set(metrics.keys())
    )

    assert not missing_metrics, (
        f"Missing model metrics: {sorted(missing_metrics)}"
    )


def test_metric_ranges() -> None:
    """
    Classification metrics should be bounded between 0 and 1.
    """

    metrics = ml_service.get_metrics()

    bounded_metrics = [
        "f1_score",
        "precision",
        "recall",
        "roc_auc",
    ]

    for metric_name in bounded_metrics:

        metric_value = float(
            metrics[metric_name]
        )

        assert 0.0 <= metric_value <= 1.0, (
            f"{metric_name} is outside expected "
            f"range: {metric_value}"
        )