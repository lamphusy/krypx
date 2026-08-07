"""Executable next-open forward-return labels."""

import logging
import math

import numpy as np
import pandas as pd

from crypto_ai.config import settings
from crypto_ai.exceptions import LabelGenerationError

logger = logging.getLogger(__name__)


def add_labels(
    feature_df: pd.DataFrame,
    horizon: int,
    minimum_required_return: float,
) -> pd.DataFrame:
    """Add executable forward-return provenance and binary labels.

    A decision at row ``t`` enters at the next candle open (``t + 1``) and exits
    at ``t + horizon + 1``. Rows without both executable prices are removed only
    from the labeled result; the input inference feature frame is not modified.
    """
    if not feature_df.columns.is_unique:
        raise LabelGenerationError("Feature column names must be unique")
    missing_columns = [column for column in settings.RAW_COLUMNS if column not in feature_df]
    if missing_columns:
        raise LabelGenerationError(f"Missing required market columns: {missing_columns}")
    if feature_df.empty:
        raise LabelGenerationError("Cannot add labels to an empty feature dataset")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise LabelGenerationError("horizon must be a positive integer")
    if not math.isfinite(minimum_required_return) or minimum_required_return < 0.0:
        raise LabelGenerationError("minimum_required_return must be finite and non-negative")

    timestamps = feature_df["timestamp"]
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype) or str(timestamps.dt.tz) != "UTC":
        raise LabelGenerationError("timestamp must be a timezone-aware UTC datetime column")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise LabelGenerationError("Feature timestamps must be unique and chronological")
    if not np.isfinite(feature_df["open"].to_numpy(dtype="float64", copy=False)).all():
        raise LabelGenerationError("open contains missing or infinite values")
    if (feature_df["open"] <= 0.0).any():
        raise LabelGenerationError("open prices must be positive")

    result = feature_df.copy()
    result["entry_timestamp"] = result["timestamp"].shift(-1)
    result["exit_timestamp"] = result["timestamp"].shift(-(horizon + 1))
    result["entry_open"] = result["open"].shift(-1).astype("float64")
    result["exit_open"] = result["open"].shift(-(horizon + 1)).astype("float64")
    result["gross_forward_return"] = result["exit_open"] / result["entry_open"] - 1.0

    realizable = (
        result[["entry_timestamp", "exit_timestamp", "entry_open", "exit_open"]].notna().all(axis=1)
    )
    result["label"] = (result["gross_forward_return"] > minimum_required_return).astype("int8")
    unrealizable_rows = int((~realizable).sum())
    result = result.loc[realizable].copy()
    if result.empty:
        raise LabelGenerationError(
            f"Insufficient rows for horizon {horizon}; at least {horizon + 2} are required"
        )
    if not np.isfinite(result["gross_forward_return"].to_numpy(dtype="float64")).all():
        raise LabelGenerationError("Executable forward returns contain invalid values")

    logger.info(
        "Added next-open labels to %s decision rows; removed %s unrealizable tail rows",
        len(result),
        unrealizable_rows,
    )
    return result
