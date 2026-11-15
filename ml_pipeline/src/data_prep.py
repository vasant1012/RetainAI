from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    print("Dataset Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.sample(5))

    # 2. Check for Missing Values
    print("\nMissing Values per Feature:")
    print(df.isnull().sum())

    # 3. Target Variable Distribution (Churn)
    # Assuming 'Churn' is the target column (1 = Churned, 0 = Retained)
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="Churned", hue="Churned", palette="Set2")
    plt.title("Distribution of Churn (Target Variable)")
    save_path = Path(project_dir) / "artifacts" / "churn_distribution.png"
    plt.savefig(save_path)


if __name__ == "__main__":
    # Load the processed data
    project_dir = str(Path.cwd().parent.parent)  # Adjust the path as needed
    try:
        df = pd.read_csv(f"{project_dir}/data/processed/processed_data.csv")
        inspect_data(df, project_dir)
    except FileNotFoundError:
        print("Processed data not found. Run data_prep.py first.")
