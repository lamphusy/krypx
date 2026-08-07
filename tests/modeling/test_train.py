"""Tests for model evaluation, baselines, and serialization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_ai.artifacts.registry import load_xgboost_model, save_xgboost_model
from crypto_ai.modeling.metrics import classification_metrics
from crypto_ai.modeling.train import (
    evaluate_logistic_walk_forward,
    evaluate_walk_forward,
    train_evaluation_model,
)


@pytest.fixture
def modeling_data() -> pd.DataFrame:
    n = 100
    timestamps = pd.date_range("2025-01-01", periods=n + 5, freq="h", tz="UTC")
    x = np.arange(n, dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": timestamps[:n],
            "exit_timestamp": timestamps[5:],
            "f1": np.sin(x / 3),
            "f2": np.cos(x / 5),
            "label": (x.astype(int) % 2).astype("int8"),
        },
        index=np.arange(500, 500 + n),
    )


def test_new_model_is_created_for_each_fold(modeling_data: pd.DataFrame) -> None:
    instances: list[object] = []

    class FakeModel:
        def __init__(self) -> None:
            instances.append(self)

        def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
            self.rate = float(labels.mean())

        def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
            probability = np.full(len(features), self.rate)
            return np.column_stack([1 - probability, probability])

    result = evaluate_walk_forward(
        modeling_data,
        ["f1", "f2"],
        {},
        3,
        10,
        5,
        model_factory=FakeModel,
    )

    assert len(instances) == 3
    assert len(result.predictions) == 30
    assert result.predictions.index.tolist() == list(range(570, 600))
    for fold_number, indexes in enumerate(result.fitted_training_indexes, start=1):
        validation_index = result.predictions.loc[
            result.predictions["fold_number"] == fold_number
        ].index
        assert set(indexes).isdisjoint(validation_index)


def test_logistic_baseline_uses_identical_oof_rows(modeling_data: pd.DataFrame) -> None:
    result = evaluate_logistic_walk_forward(modeling_data, ["f1", "f2"], 3, 10, 5)
    assert result.predictions.index.tolist() == list(range(570, 600))
    assert result.predictions["timestamp"].is_monotonic_increasing


def test_model_serialization_preserves_predictions(
    tmp_path: Path, modeling_data: pd.DataFrame
) -> None:
    params = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": 1,
        "eval_metric": "logloss",
    }
    model = train_evaluation_model(modeling_data, ["f1", "f2"], params)
    before = model.predict_proba(modeling_data[["f1", "f2"]])
    path = tmp_path / "model.json"
    save_xgboost_model(model, path)
    restored = load_xgboost_model(path)
    np.testing.assert_allclose(restored.predict_proba(modeling_data[["f1", "f2"]]), before)


def test_single_class_metrics_are_defined_without_infinity() -> None:
    with pytest.warns(UserWarning, match="undefined"):
        metrics = classification_metrics(
            np.zeros(4, dtype="int8"),
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.zeros(4, dtype="int8"),
        )
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert np.isfinite(metrics["log_loss"])


def test_single_class_training_folds_use_only_fold_local_information(
    modeling_data: pd.DataFrame,
) -> None:
    data = modeling_data.copy()
    data["label"] = 1
    instances: list[object] = []

    class ModelThatCannotFitOneClass:
        def __init__(self) -> None:
            instances.append(self)

        def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
            raise AssertionError("single-class folds must not be passed to the estimator")

    with pytest.warns(UserWarning, match="undefined"):
        result = evaluate_walk_forward(
            data,
            ["f1", "f2"],
            {},
            3,
            10,
            5,
            model_factory=ModelThatCannotFitOneClass,
        )
    assert len(instances) == 3
    assert (result.predictions["probability_score"] == 1.0).all()
    assert len(result.warnings) == 3


def test_evaluation_model_receives_saved_feature_order_and_development_rows(
    modeling_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeModel:
        def fit(self, features: pd.DataFrame, labels: pd.Series) -> None:
            captured["columns"] = features.columns.tolist()
            captured["index"] = features.index.tolist()
            captured["labels"] = labels.index.tolist()

    monkeypatch.setattr(
        "crypto_ai.modeling.train._xgboost_classifier", lambda **params: FakeModel()
    )
    train_evaluation_model(modeling_data.iloc[:70], ["f2", "f1"], {})
    assert captured == {
        "columns": ["f2", "f1"],
        "index": modeling_data.index[:70].tolist(),
        "labels": modeling_data.index[:70].tolist(),
    }


def test_fixed_random_seed_is_reproducible(modeling_data: pd.DataFrame) -> None:
    params = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": 1,
        "eval_metric": "logloss",
    }
    first = evaluate_walk_forward(modeling_data, ["f1", "f2"], params, 3, 10, 5)
    second = evaluate_walk_forward(modeling_data, ["f1", "f2"], params, 3, 10, 5)
    np.testing.assert_allclose(
        first.predictions["probability_score"], second.predictions["probability_score"]
    )
