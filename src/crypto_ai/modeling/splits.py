"""Chronological holdout isolation and purged walk-forward splits."""

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai.config import settings
from crypto_ai.exceptions import DatasetSplitError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartitionRangeMetadata:
    """Positional and chronological boundaries for one dataset partition."""

    row_count: int
    start_position: int
    end_position: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp


@dataclass(frozen=True)
class DatasetSplitMetadata:
    """Auditable development, purge, and holdout partition metadata."""

    total_rows: int
    holdout_ratio: float
    label_lookahead_rows: int
    development: PartitionRangeMetadata
    boundary_purge: PartitionRangeMetadata
    holdout: PartitionRangeMetadata
    maximum_development_exit_timestamp: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardFoldMetadata:
    """Auditable boundaries and label provenance for one validation fold."""

    fold_number: int
    train: PartitionRangeMetadata
    gap: PartitionRangeMetadata
    validation: PartitionRangeMetadata
    maximum_training_exit_timestamp: pd.Timestamp


@dataclass(frozen=True)
class SplitPlan:
    """Isolated partitions, positional folds, and metadata for later model evaluation."""

    development: pd.DataFrame
    boundary_purge: pd.DataFrame
    holdout: pd.DataFrame
    partition_metadata: DatasetSplitMetadata
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    fold_metadata: tuple[WalkForwardFoldMetadata, ...]
    test_size_rows: int
    gap_rows: int


def _validate_labeled_frame(df: pd.DataFrame, operation: str) -> None:
    if df.empty:
        raise DatasetSplitError(f"Cannot {operation} an empty labeled dataset")
    if not df.columns.is_unique:
        raise DatasetSplitError("Labeled dataset column names must be unique")

    required_columns = ("timestamp", "exit_timestamp")
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise DatasetSplitError(
            f"Cannot {operation}; missing label-provenance columns: {missing_columns}"
        )

    for column in required_columns:
        values = df[column]
        if not isinstance(values.dtype, pd.DatetimeTZDtype) or str(values.dt.tz) != "UTC":
            raise DatasetSplitError(f"{column} must be a timezone-aware UTC datetime column")
        if values.isna().any():
            raise DatasetSplitError(f"{column} must not contain missing values")

    timestamps = df["timestamp"]
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise DatasetSplitError("Decision timestamps must be unique and chronological")
    if (df["exit_timestamp"] <= timestamps).any():
        raise DatasetSplitError("Every exit_timestamp must occur after its decision timestamp")


def _validate_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DatasetSplitError(f"{name} must be a positive integer")


def _partition_range(
    frame: pd.DataFrame,
    start_position: int,
    end_position: int,
) -> PartitionRangeMetadata:
    return PartitionRangeMetadata(
        row_count=len(frame),
        start_position=start_position,
        end_position=end_position,
        start_timestamp=frame["timestamp"].iloc[0],
        end_timestamp=frame["timestamp"].iloc[-1],
    )


def split_development_holdout(
    df: pd.DataFrame,
    holdout_ratio: float,
    label_lookahead_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return development, boundary-purge, and untouched holdout rows."""
    _validate_labeled_frame(df, "split")
    if not math.isfinite(holdout_ratio) or holdout_ratio <= 0.0 or holdout_ratio >= 1.0:
        raise DatasetSplitError("holdout_ratio must be finite and strictly between 0 and 1")
    _validate_positive_integer("label_lookahead_rows", label_lookahead_rows)
    if label_lookahead_rows < settings.LABEL_LOOKAHEAD_ROWS:
        raise DatasetSplitError(
            "label_lookahead_rows must be at least PREDICTION_HORIZON + 1 "
            f"({settings.LABEL_LOOKAHEAD_ROWS})"
        )

    holdout_size = max(1, math.ceil(len(df) * holdout_ratio))
    holdout_start = len(df) - holdout_size
    purge_start = holdout_start - label_lookahead_rows
    if purge_start <= 0:
        raise DatasetSplitError("Insufficient rows for development, purge, and holdout")

    development = df.iloc[:purge_start].copy()
    boundary_purge = df.iloc[purge_start:holdout_start].copy()
    holdout = df.iloc[holdout_start:].copy()

    if not (
        development["timestamp"].max()
        < boundary_purge["timestamp"].min()
        <= boundary_purge["timestamp"].max()
        < holdout["timestamp"].min()
    ):
        raise DatasetSplitError("Development, purge, and holdout timestamps overlap")
    if len(boundary_purge) != label_lookahead_rows:
        raise DatasetSplitError("Boundary purge does not cover the complete label lookahead")
    maximum_development_exit = development["exit_timestamp"].max()
    first_holdout_timestamp = holdout["timestamp"].min()
    if maximum_development_exit >= first_holdout_timestamp:
        raise DatasetSplitError(
            "Development label leakage detected: maximum exit_timestamp "
            f"{maximum_development_exit} is not before holdout start {first_holdout_timestamp}"
        )

    return development, boundary_purge, holdout


def build_dataset_split_metadata(
    full_df: pd.DataFrame,
    development: pd.DataFrame,
    boundary_purge: pd.DataFrame,
    holdout: pd.DataFrame,
    holdout_ratio: float,
    label_lookahead_rows: int,
) -> DatasetSplitMetadata:
    """Build manifest-ready metadata for an already validated holdout partition."""
    development_end = len(development) - 1
    purge_start = len(development)
    purge_end = purge_start + len(boundary_purge) - 1
    holdout_start = purge_end + 1
    return DatasetSplitMetadata(
        total_rows=len(full_df),
        holdout_ratio=holdout_ratio,
        label_lookahead_rows=label_lookahead_rows,
        development=_partition_range(development, 0, development_end),
        boundary_purge=_partition_range(boundary_purge, purge_start, purge_end),
        holdout=_partition_range(holdout, holdout_start, len(full_df) - 1),
        maximum_development_exit_timestamp=development["exit_timestamp"].max(),
    )


def walk_forward_splits(
    development_df: pd.DataFrame,
    n_splits: int,
    test_size_rows: int,
    gap_rows: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create expanding-window chronological splits with a provenance-checked purge gap."""
    _validate_labeled_frame(development_df, "create walk-forward folds from")
    _validate_positive_integer("n_splits", n_splits)
    _validate_positive_integer("test_size_rows", test_size_rows)
    _validate_positive_integer("gap_rows", gap_rows)
    if gap_rows < settings.LABEL_LOOKAHEAD_ROWS:
        raise DatasetSplitError(
            f"gap_rows must be at least PREDICTION_HORIZON + 1 "
            f"({settings.LABEL_LOOKAHEAD_ROWS})"
        )

    total_validation_rows = n_splits * test_size_rows
    first_validation_start = len(development_df) - total_validation_rows
    first_train_end = first_validation_start - gap_rows
    if first_validation_start <= 0 or first_train_end <= 0:
        raise DatasetSplitError(
            "Insufficient development rows for the requested folds, validation size, and gap"
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    previous_validation_end: int | None = None
    for fold_position in range(n_splits):
        validation_start = first_validation_start + fold_position * test_size_rows
        validation_end = validation_start + test_size_rows
        train_end = validation_start - gap_rows
        train_indices = np.arange(0, train_end, dtype=np.int64)
        validation_indices = np.arange(validation_start, validation_end, dtype=np.int64)

        if train_indices.size == 0 or validation_indices.size != test_size_rows:
            raise DatasetSplitError("A walk-forward fold has an invalid train or validation size")
        if int(train_indices[-1]) + gap_rows >= int(validation_indices[0]):
            raise DatasetSplitError("Walk-forward purge gap invariant was not satisfied")
        if previous_validation_end is not None and previous_validation_end != int(
            validation_indices[0]
        ):
            raise DatasetSplitError("Walk-forward validation blocks are not contiguous")

        maximum_training_exit = development_df.iloc[train_indices]["exit_timestamp"].max()
        first_validation_timestamp = development_df.iloc[validation_indices]["timestamp"].min()
        if maximum_training_exit >= first_validation_timestamp:
            raise DatasetSplitError(
                "Walk-forward label leakage detected in fold "
                f"{fold_position + 1}: maximum training exit_timestamp "
                f"{maximum_training_exit} is not before validation start "
                f"{first_validation_timestamp}"
            )

        splits.append((train_indices, validation_indices))
        previous_validation_end = int(validation_indices[-1]) + 1

    return splits


def build_walk_forward_metadata(
    development_df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[WalkForwardFoldMetadata, ...]:
    """Build manifest-ready boundaries for validated positional folds."""
    metadata: list[WalkForwardFoldMetadata] = []
    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        train_start = int(train_indices[0])
        train_end = int(train_indices[-1])
        validation_start = int(validation_indices[0])
        validation_end = int(validation_indices[-1])
        gap_start = train_end + 1
        gap_end = validation_start - 1
        train = development_df.iloc[train_indices]
        gap = development_df.iloc[gap_start : gap_end + 1]
        validation = development_df.iloc[validation_indices]
        metadata.append(
            WalkForwardFoldMetadata(
                fold_number=fold_number,
                train=_partition_range(train, train_start, train_end),
                gap=_partition_range(gap, gap_start, gap_end),
                validation=_partition_range(
                    validation,
                    validation_start,
                    validation_end,
                ),
                maximum_training_exit_timestamp=train["exit_timestamp"].max(),
            )
        )
    return tuple(metadata)


def create_split_plan(
    labeled_df: pd.DataFrame,
    *,
    holdout_ratio: float = settings.FINAL_HOLDOUT_RATIO,
    label_lookahead_rows: int = settings.LABEL_LOOKAHEAD_ROWS,
    n_splits: int = settings.N_WALK_FORWARD_SPLITS,
    test_ratio: float = settings.WALK_FORWARD_TEST_RATIO,
    gap_rows: int = settings.PURGE_GAP_ROWS,
) -> SplitPlan:
    """Create isolated partitions, purged folds, and manifest-ready metadata."""
    if not math.isfinite(test_ratio) or test_ratio <= 0.0 or test_ratio >= 1.0:
        raise DatasetSplitError("test_ratio must be finite and strictly between 0 and 1")
    development, boundary_purge, holdout = split_development_holdout(
        labeled_df,
        holdout_ratio=holdout_ratio,
        label_lookahead_rows=label_lookahead_rows,
    )
    test_size_rows = max(1, int(len(development) * test_ratio))
    fold_list = walk_forward_splits(
        development,
        n_splits=n_splits,
        test_size_rows=test_size_rows,
        gap_rows=gap_rows,
    )
    plan = SplitPlan(
        development=development,
        boundary_purge=boundary_purge,
        holdout=holdout,
        partition_metadata=build_dataset_split_metadata(
            labeled_df,
            development,
            boundary_purge,
            holdout,
            holdout_ratio,
            label_lookahead_rows,
        ),
        folds=tuple(fold_list),
        fold_metadata=build_walk_forward_metadata(development, fold_list),
        test_size_rows=test_size_rows,
        gap_rows=gap_rows,
    )
    logger.info(
        "Created split plan with %s development, %s boundary-purge, and %s holdout rows; "
        "%s folds of %s validation rows with a %s-row gap",
        len(development),
        len(boundary_purge),
        len(holdout),
        len(fold_list),
        test_size_rows,
        gap_rows,
    )
    return plan


def _json_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    return value


def save_split_metadata(plan: SplitPlan, destination: Path) -> None:
    """Atomically save partition and fold metadata without holdout outcomes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "partitions": _json_value(plan.partition_metadata),
        "folds": _json_value(plan.fold_metadata),
        "test_size_rows": plan.test_size_rows,
        "gap_rows": plan.gap_rows,
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
