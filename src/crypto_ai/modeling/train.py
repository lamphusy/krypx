"""Walk-forward, evaluation, and production model training."""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from crypto_ai.config import settings
from crypto_ai.exceptions import ModelTrainingError
from crypto_ai.modeling.baselines import create_logistic_pipeline
from crypto_ai.modeling.metrics import classification_metrics
from crypto_ai.modeling.splits import walk_forward_splits

logger = logging.getLogger(__name__)
ModelFactory = Callable[[], Any]

if TYPE_CHECKING:
    from xgboost import XGBClassifier


def _xgboost_classifier(**model_params: object) -> Any:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ModelTrainingError(
            "XGBoost could not be loaded; ensure its native OpenMP runtime is installed"
        ) from exc
    return XGBClassifier(**model_params)


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-fold predictions, metrics, and fold training boundaries."""

    predictions: pd.DataFrame
    fold_metrics: tuple[dict[str, Any], ...]
    aggregate_metrics: dict[str, Any]
    fitted_training_indexes: tuple[tuple[Any, ...], ...]
    warnings: tuple[str, ...]


def feature_schema_hash(feature_columns: list[str]) -> str:
    """Hash the authoritative ordered model feature schema."""
    payload = json.dumps(feature_columns, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_training_data(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate ordered, complete features and binary labels before model fitting."""
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ModelTrainingError("Feature columns must be non-empty and unique")
    missing = [
        column
        for column in [*feature_columns, "timestamp", "exit_timestamp", "label"]
        if column not in df
    ]
    if missing:
        raise ModelTrainingError(f"Training data is missing required columns: {missing}")
    matrix = df[feature_columns].to_numpy(dtype="float64")
    if not np.isfinite(matrix).all():
        raise ModelTrainingError("Training feature matrix contains missing or infinite values")
    if not df["label"].isin([0, 1]).all():
        raise ModelTrainingError("Training labels must contain only 0 and 1")


def _evaluate_with_factory(
    development_df: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    test_size_rows: int,
    gap_rows: int,
    model_factory: ModelFactory,
    signal_threshold: float,
) -> WalkForwardResult:
    validate_training_data(development_df, feature_columns)
    splits = walk_forward_splits(development_df, n_splits, test_size_rows, gap_rows)
    prediction_frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    fitted_indexes: list[tuple[Any, ...]] = []
    evaluation_warnings: list[str] = []

    for fold_number, (train_positions, validation_positions) in enumerate(splits, start=1):
        train = development_df.iloc[train_positions]
        validation = development_df.iloc[validation_positions]
        try:
            model = model_factory()
        except ModelTrainingError:
            raise
        except Exception as exc:
            raise ModelTrainingError(
                f"Unable to create model for walk-forward fold {fold_number}"
            ) from exc
        training_classes = np.sort(train["label"].unique())
        if len(training_classes) == 1:
            # Very short early folds can legitimately contain a single class.
            # XGBoost and logistic regression cannot fit those folds, so use the
            # only probability that can be estimated without leaking future rows.
            constant_probability = float(training_classes[0])
            probabilities = np.full(len(validation), constant_probability, dtype="float64")
            warning = (
                f"Fold {fold_number} contains only training class "
                f"{int(training_classes[0])}; constant probabilities were used"
            )
            evaluation_warnings.append(warning)
            logger.warning(warning)
        else:
            try:
                model.fit(train[feature_columns], train["label"])
                probabilities = np.asarray(
                    model.predict_proba(validation[feature_columns])[:, 1], dtype="float64"
                )
            except Exception as exc:
                raise ModelTrainingError(
                    f"Model fitting or prediction failed in walk-forward fold {fold_number}"
                ) from exc
        predicted = (probabilities >= signal_threshold).astype("int8")
        fold_predictions = pd.DataFrame(
            {
                "timestamp": validation["timestamp"],
                "fold_number": fold_number,
                "actual_label": validation["label"].astype("int8"),
                "probability_score": probabilities,
                "predicted_label": predicted,
                "signal": predicted,
            },
            index=validation.index,
        )
        fold_metric = classification_metrics(
            fold_predictions["actual_label"].to_numpy(),
            fold_predictions["probability_score"].to_numpy(),
            fold_predictions["predicted_label"].to_numpy(),
        )
        fold_metric["fold_number"] = fold_number
        fold_metric["training_label_classes"] = training_classes.tolist()
        metrics.append(fold_metric)
        prediction_frames.append(fold_predictions)
        fitted_indexes.append(tuple(train.index.tolist()))

    predictions = pd.concat(prediction_frames).sort_values("timestamp", kind="stable")
    aggregate = classification_metrics(
        predictions["actual_label"].to_numpy(),
        predictions["probability_score"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
    )
    return WalkForwardResult(
        predictions,
        tuple(metrics),
        aggregate,
        tuple(fitted_indexes),
        tuple(evaluation_warnings),
    )


def evaluate_walk_forward(
    development_df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, object],
    n_splits: int,
    test_size_rows: int,
    gap_rows: int,
    *,
    signal_threshold: float = settings.SIGNAL_THRESHOLD,
    model_factory: ModelFactory | None = None,
) -> WalkForwardResult:
    """Evaluate a fixed XGBoost configuration using purged walk-forward splits."""
    factory = model_factory or (lambda: _xgboost_classifier(**model_params))
    return _evaluate_with_factory(
        development_df,
        feature_columns,
        n_splits,
        test_size_rows,
        gap_rows,
        factory,
        signal_threshold,
    )


def evaluate_logistic_walk_forward(
    development_df: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    test_size_rows: int,
    gap_rows: int,
    *,
    model_params: dict[str, object] | None = None,
    signal_threshold: float = settings.SIGNAL_THRESHOLD,
) -> WalkForwardResult:
    """Evaluate fold-local scaled logistic regression on identical splits."""
    return _evaluate_with_factory(
        development_df,
        feature_columns,
        n_splits,
        test_size_rows,
        gap_rows,
        lambda: create_logistic_pipeline(model_params),
        signal_threshold,
    )


def train_evaluation_model(
    development_df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, object],
) -> "XGBClassifier":
    """Train the model used for untouched holdout evaluation."""
    validate_training_data(development_df, feature_columns)
    if development_df["label"].nunique() < 2:
        raise ModelTrainingError("Evaluation-model training requires both label classes")
    model = _xgboost_classifier(**model_params)
    try:
        model.fit(development_df[feature_columns], development_df["label"])
    except Exception as exc:
        raise ModelTrainingError("Evaluation-model fitting failed") from exc
    logger.info("Trained evaluation model on %s development rows", len(development_df))
    return model


def train_production_model(
    full_labeled_df: pd.DataFrame,
    feature_columns: list[str],
    model_params: dict[str, object],
) -> "XGBClassifier":
    """Train a production model using all currently labeled history."""
    validate_training_data(full_labeled_df, feature_columns)
    if full_labeled_df["label"].nunique() < 2:
        raise ModelTrainingError("Production-model training requires both label classes")
    model = _xgboost_classifier(**model_params)
    try:
        model.fit(full_labeled_df[feature_columns], full_labeled_df["label"])
    except Exception as exc:
        raise ModelTrainingError("Production-model fitting failed") from exc
    logger.info("Trained production model on %s labeled rows", len(full_labeled_df))
    return model


def feature_importance_frame(model: "XGBClassifier", feature_columns: list[str]) -> pd.DataFrame:
    """Return global gain, weight, and cover importances in schema order."""
    booster = model.get_booster()
    result = pd.DataFrame({"feature": feature_columns})
    for importance_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=importance_type)
        result[importance_type] = result["feature"].map(scores).fillna(0.0).astype("float64")
    return result
