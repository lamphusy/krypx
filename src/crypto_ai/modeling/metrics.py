"""Robust binary-classification metrics."""

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    """Calculate the complete Phase 1 binary-classification metric set."""
    actual = np.asarray(actual, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.int8)
    if not (len(actual) == len(probability) == len(predicted)) or len(actual) == 0:
        raise ValueError("Classification arrays must have the same positive length")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")

    labels = np.unique(actual)
    roc_auc: float | None = None
    pr_auc: float | None = None
    if len(labels) == 2:
        roc_auc = float(roc_auc_score(actual, probability))
        precision_curve, recall_curve, _ = precision_recall_curve(actual, probability)
        pr_auc = float(auc(recall_curve, precision_curve))
    else:
        warnings.warn("ROC-AUC and PR-AUC are undefined for a single-class sample", stacklevel=2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = float(balanced_accuracy_score(actual, predicted))
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": balanced,
        "precision_class_1": float(
            precision_score(actual, predicted, pos_label=1, zero_division=0)
        ),
        "recall_class_1": float(recall_score(actual, predicted, pos_label=1, zero_division=0)),
        "f1_class_1": float(f1_score(actual, predicted, pos_label=1, zero_division=0)),
        "precision_class_0": float(
            precision_score(actual, predicted, pos_label=0, zero_division=0)
        ),
        "recall_class_0": float(recall_score(actual, predicted, pos_label=0, zero_division=0)),
        "f1_class_0": float(f1_score(actual, predicted, pos_label=0, zero_division=0)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(actual, probability)),
        "confusion_matrix": matrix.tolist(),
        "positive_label_rate": float(actual.mean()),
        "predicted_positive_rate": float(predicted.mean()),
    }
