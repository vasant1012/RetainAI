"""Dataset inspection and exploratory visualization for RetainAI."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Allow both module execution and direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings
from backend.logger import logger

TARGET_COLUMN = "Churned"


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate the minimum schema required for data inspection."""
    if df.empty:
        raise ValueError("The raw dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Required target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {df.columns.tolist()}"
        )


def inspect_data(
    df: pd.DataFrame,
    plot_path: Path | None = None,
) -> Path:
    """Inspect raw data and save the churn-distribution visualization."""
    validate_raw_data(df)

    output_path = plot_path or settings.CHURN_DISTRIBUTION_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset shape: %s", df.shape)
    logger.info("Dataset columns: %s", df.columns.tolist())

    sample_size = min(5, len(df))
    logger.info(
        "Sample records:\n%s",
        df.sample(n=sample_size, random_state=42).to_string(index=False),
    )
    logger.info("Missing values per feature:\n%s", df.isna().sum().to_string())
    logger.info(
        "Target distribution:\n%s",
        df[TARGET_COLUMN].value_counts(dropna=False).sort_index().to_string(),
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=df,
        x=TARGET_COLUMN,
        hue=TARGET_COLUMN,
        palette="Set2",
        legend=False,
        ax=axis,
    )
    axis.set_title("Distribution of Churn (Target Variable)")
    axis.set_xlabel("Churned")
    axis.set_ylabel("Customer Count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    logger.info("Churn distribution plot saved to: %s", output_path)
    return output_path


def main() -> None:
    """Load the raw dataset and run inspection."""
    raw_data_path = settings.RAW_DATA_PATH

    if not raw_data_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")

    logger.info("Loading raw dataset from: %s", raw_data_path)
    dataframe = pd.read_csv(raw_data_path)
    inspect_data(dataframe)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        logger.exception("Data inspection failed: %s", exc)
        raise SystemExit(1) from exc
