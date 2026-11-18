from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from backend.core.ml_service import MLService


class FakeModel:
    def predict_proba(self, dataframe):
        age = float(dataframe.iloc[0]["Age"])
        male = float(dataframe.iloc[0].get("Gender_Male", 0))
        probability = min(0.99, max(0.01, 0.20 + age / 200 + male * 0.20))
        return np.array([[1 - probability, probability]])


@pytest.fixture()
def artifact_paths(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "xgboost_model.pkl"
    metrics_path = tmp_path / "metrics.json"
    artifact = {
        "model": FakeModel(),
        "feature_names": ["Age", "Gender_Male"],
        "metrics": {"f1_score": 0.81},
        "preprocessing": {
            "numeric_medians": {"Age": 40.0},
            "categorical_levels": {"Gender": ["Female", "Male"]},
            "raw_feature_names": ["Age", "Gender"],
        },
        "artifact_version": "1.2.0",
        "model_type": "FakeModel",
    }
    with model_path.open("wb") as file:
        pickle.dump(artifact, file)
    metrics_path.write_text(json.dumps({"f1_score": 0.81}), encoding="utf-8")
    return model_path, metrics_path


@pytest.fixture()
def loaded_service(artifact_paths: tuple[Path, Path]) -> MLService:
    model_path, metrics_path = artifact_paths
    service = MLService(model_path, metrics_path)
    service.load()
    return service
