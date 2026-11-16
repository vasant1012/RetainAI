from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.logger import logger

project_dir = str(Path.cwd())  # Adjust the path as needed
Path(__file__).parent.parent.joinpath(
    project_dir).mkdir(parents=True, exist_ok=True)


def inspect_data(df, project_dir):
    """
    Perform basic data inspection and visualization.

    Parameters:
    df (pd.DataFrame): The DataFrame to inspect.
    project_dir (str): The project directory path.

    Returns:
    None
    """
    # 1. Basic Inspection
    logger.info("Dataset Shape: %s", df.shape)
    logger.info("\nFirst 5 rows:")
    logger.info(df.sample(5))

    # 2. Check for Missing Values
    logger.info("\nMissing Values per Feature:")
    logger.info(df.isnull().sum())

    # 3. Target Variable Distribution (Churn)
    # Assuming 'Churn' is the target column (1 = Churned, 0 = Retained)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churned", hue="Churned", palette="Set2")
    plt.title("Distribution of Churn (Target Variable)")
    save_path = Path(project_dir) / "ml_pipeline" / \
        "artifacts" / "churn_distribution.png"
    plt.savefig(save_path)
    logger.info("Churn distribution plot saved to: %s", save_path)


if __name__ == "__main__":
    # Load the processed data
    try:
        df = pd.read_csv(f"{project_dir}/data/raw/eCommerce_dataset.csv")
        inspect_data(df, project_dir)
    except FileNotFoundError:
        print("Processed data not found. Run data_prep.py first.")
