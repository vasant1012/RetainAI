"""Deterministic data-quality inspection for the RetainAI raw dataset."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.config import settings
from backend.logger import logger


def inspect_data(dataframe: pd.DataFrame) -> dict[str, object]:
    if dataframe.empty:
        raise ValueError("Raw dataset is empty.")
    if "Churned" not in dataframe.columns and "Churn" not in dataframe.columns:
        raise ValueError("Raw dataset must contain 'Churned' or 'Churn'.")

    target = "Churned" if "Churned" in dataframe.columns else "Churn"
    logger.info("Dataset shape: %s", dataframe.shape)
    logger.info("Sample rows:\n%s", dataframe.sample(min(5, len(dataframe)), random_state=42))
    logger.info("Missing values:\n%s", dataframe.isnull().sum())

    settings.CHURN_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=dataframe, x=target, hue=target, palette="Set2", ax=axis, legend=False)
    axis.set_title("Distribution of Churn (Target Variable)")
    figure.tight_layout()
    figure.savefig(settings.CHURN_PLOT_PATH)
    plt.close(figure)

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "target": target,
        "missing_values": int(dataframe.isna().sum().sum()),
        "plot_path": str(settings.CHURN_PLOT_PATH),
    }


def main() -> None:
    if not settings.RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {settings.RAW_DATA_PATH}")
    summary = inspect_data(pd.read_csv(settings.RAW_DATA_PATH))
    logger.info("Data inspection completed: %s", summary)


if __name__ == "__main__":
    main()
