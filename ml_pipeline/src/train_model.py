import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path.cwd()))

from ml_pipeline.src.logger import logger  # NOQA E402

project_dir = str(Path.cwd())
Path(__file__).parent.parent.joinpath(
    project_dir).mkdir(parents=True, exist_ok=True)
# Ensure artifacts directory exists
Path(f"{project_dir}/ml_pipeline/artifacts").mkdir(parents=True, exist_ok=True)


def load_and_clean_data(df, project_dir):
    """
    Load and clean the dataset.
    Parameters:
        df (pd.DataFrame): The Processed DataFrame for feature engineering.
        project_dir (str): The project directory path.

    Returns:
        df_encoded (pd.DataFrame): The cleaned and encoded DataFrame ready
        for model training.
    """
    na_cols = [
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

    for col in na_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 2. Encode Categoricals
    # Assuming Gender, Country, City, and Signup_Quarter are text/categorical
    cat_cols = ["Gender", "Country", "City", "Signup_Quarter"]
    existing_cat_cols = [col for col in cat_cols if col in df.columns]

    df_encoded = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True)

    # Ensure target variable is correctly named for downstream tasks
    if "Churned" in df_encoded.columns:
        df_encoded = df_encoded.rename(
            columns={"Churned": "Churn"})

    # Ensure processed directory exists before saving
    Path(f"{project_dir}/data/processed").mkdir(parents=True, exist_ok=True)
    df_encoded.to_csv(
        f"{project_dir}/data/processed/processed_data.csv", index=False
    )
    logger.info("Preprocessed data saved to data/processed/")

    return df_encoded


def train_and_export(df, project_dir):
    # 2. Define Features (X) and Target (y)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 3. Train/Test Split (Stratified to maintain class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Apply SMOTE to training data only
    # This synthesizes minority class examples to balance the dataset
    print(f"Original Training Target Distribution:\n{y_train.value_counts()}")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(
        f"Resampled Target Distribution:\n{y_train_resampled.value_counts()}")

    # 5. Initialize and Train XGBoost
    # Using parameters optimized for tabular behavioral data
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    print("\nTraining XGBoost model...")
    model.fit(X_train_resampled, y_train_resampled)

    # 6. Basic Evaluation
    y_pred = model.predict(X_test)

    # Get metrics directly as a dictionary
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    f1 = f1_score(y_test, y_pred)

    print("\nClassification Report (Test Data):")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score: {f1:.4f}")

    # 7. Export Model and Feature Columns
    # We save feature_names to ensure the FastAPI backend
    # enforces the exact column order
    artifacts = {"model": model, "feature_names": X.columns.tolist()}
    artifacts_dir = f"{project_dir}/ml_pipeline/artifacts/xgboost_model.pkl"

    with open(artifacts_dir, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"\nModel artifacts successfully exported to {artifacts_dir}")

    # 8. Export Metrics as JSON
    metrics_data = {
        "f1_score": f1,
        "classification_report": report_dict
    }

    metrics_path = f"{project_dir}/ml_pipeline/artifacts/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)

    print(f"Evaluation metrics successfully exported to {metrics_path}")


if __name__ == "__main__":
    # Load the processed data
    try:
        df = pd.read_csv(f"{project_dir}/data/raw/eCommerce_dataset.csv")
        df_cleaned = load_and_clean_data(df, project_dir)
        train_and_export(df_cleaned, project_dir)
    except FileNotFoundError:
        print("Processed data not found. Ensure raw data exists.")
