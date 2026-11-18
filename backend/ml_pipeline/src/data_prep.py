"""Inspect the raw dataset without duplicating model preprocessing."""
import json
from typing import Any

import pandas as pd

from backend.config import DATA_SUMMARY_PATH, RAW_DATA_PATH, ensure_directories

TARGET_CANDIDATES = ("Churned", "Churn")


def build_summary(dataframe: pd.DataFrame) -> dict[str, Any]:
    target = next((name for name in TARGET_CANDIDATES if name in dataframe), None)
    return {
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "column_names": dataframe.columns.tolist(),
        "data_types": {name: str(dtype) for name, dtype in dataframe.dtypes.items()},
        "missing_values": {
            name: int(count) for name, count in dataframe.isna().sum().items()
        },
        "target_column": target,
        "target_distribution": (
            {
                str(label): int(count)
                for label, count in dataframe[target].value_counts(dropna=False).items()
            }
            if target
            else {}
        ),
    }


def main() -> None:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
    ensure_directories()
    summary = build_summary(pd.read_csv(RAW_DATA_PATH))
    DATA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved dataset summary to {DATA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
