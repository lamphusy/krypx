"""Validation helpers for chronological, closed OHLCV market data."""

import re
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from crypto_ai.config import settings
from crypto_ai.exceptions import MarketDataValidationError

_TIMEFRAME_PATTERN = re.compile(r"^(?P<count>[1-9]\d*)(?P<unit>[mhd])$")
_MILLISECONDS_PER_UNIT = {
    "m": 60_000,
    "h": 60 * 60_000,
    "d": 24 * 60 * 60_000,
}
_PRICE_COLUMNS = ("open", "high", "low", "close")
_NUMERIC_COLUMNS = (*_PRICE_COLUMNS, "volume")


def timeframe_to_milliseconds(timeframe: str) -> int:
    """Convert a fixed minute, hour, or day timeframe to milliseconds."""
    match = _TIMEFRAME_PATTERN.fullmatch(timeframe)
    if match is None:
        raise MarketDataValidationError(
            f"Unsupported fixed timeframe {timeframe!r}; expected forms such as '1m', '1h', or '1d'"
        )

    count = int(match.group("count"))
    return count * _MILLISECONDS_PER_UNIT[match.group("unit")]


def normalize_current_utc_time(
    current_utc_time: datetime | pd.Timestamp | None,
) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp, rejecting ambiguous naive values."""
    if current_utc_time is None:
        return pd.Timestamp.now(tz="UTC")

    timestamp = pd.Timestamp(current_utc_time)
    if timestamp.tzinfo is None:
        raise MarketDataValidationError("current_utc_time must be timezone-aware")
    return timestamp.tz_convert("UTC")


def empty_ohlcv_dataframe() -> pd.DataFrame:
    """Create an empty DataFrame with the canonical OHLCV dtypes."""
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )


def filter_closed_candles(
    df: pd.DataFrame,
    timeframe: str,
    current_utc_time: datetime | pd.Timestamp | None,
) -> pd.DataFrame:
    """Return only candles whose complete timeframe has elapsed."""
    if df.empty:
        return df.copy()
    if "timestamp" not in df.columns:
        raise MarketDataValidationError("Cannot filter candles without a timestamp column")

    now = normalize_current_utc_time(current_utc_time)
    duration = pd.to_timedelta(timeframe_to_milliseconds(timeframe), unit="ms")
    closed_mask = df["timestamp"] + duration <= now
    return df.loc[closed_mask].copy()


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    current_utc_time: datetime | pd.Timestamp | None = None,
) -> None:
    """Validate the complete Phase 1 raw-market-data contract."""
    missing_columns = [column for column in settings.RAW_COLUMNS if column not in df.columns]
    if missing_columns:
        raise MarketDataValidationError(f"Missing required OHLCV columns: {missing_columns}")
    if df.empty:
        raise MarketDataValidationError("OHLCV data must not be empty")

    timestamp_series = df["timestamp"]
    if not isinstance(timestamp_series.dtype, pd.DatetimeTZDtype):
        raise MarketDataValidationError("timestamp must use a timezone-aware datetime dtype")
    if str(timestamp_series.dt.tz) != "UTC":
        raise MarketDataValidationError("timestamp timezone must be UTC")
    if timestamp_series.duplicated().any():
        duplicate = timestamp_series.loc[timestamp_series.duplicated()].iloc[0]
        raise MarketDataValidationError(f"Duplicate candle timestamp detected: {duplicate}")
    if not timestamp_series.is_monotonic_increasing:
        raise MarketDataValidationError("Candle timestamps must be strictly increasing")

    for column in _NUMERIC_COLUMNS:
        if not is_numeric_dtype(df[column].dtype):
            raise MarketDataValidationError(f"{column} must be numeric")
        values = df[column].to_numpy(dtype="float64", copy=False)
        if not np.isfinite(values).all():
            raise MarketDataValidationError(f"{column} contains missing or infinite values")

    for column in _PRICE_COLUMNS:
        if df[column].dtype != np.dtype("float64"):
            raise MarketDataValidationError(f"{column} must use float64 dtype")
        if (df[column] <= 0.0).any():
            raise MarketDataValidationError(f"{column} must contain only positive values")

    if (df["volume"] < 0.0).any():
        raise MarketDataValidationError("volume must be non-negative")
    if (df["high"] < df["low"]).any():
        raise MarketDataValidationError("Every candle must satisfy high >= low")
    if ((df["open"] < df["low"]) | (df["open"] > df["high"])).any():
        raise MarketDataValidationError("Every candle must satisfy low <= open <= high")
    if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise MarketDataValidationError("Every candle must satisfy low <= close <= high")

    timeframe_ms = timeframe_to_milliseconds(timeframe)
    timeframe_ns = timeframe_ms * 1_000_000
    timestamp_ns = timestamp_series.astype("int64").to_numpy()
    if (timestamp_ns % timeframe_ns != 0).any():
        bad_timestamp = timestamp_series.iloc[int(np.flatnonzero(timestamp_ns % timeframe_ns)[0])]
        raise MarketDataValidationError(
            f"Candle timestamp {bad_timestamp} is not aligned to the {timeframe} grid"
        )

    if len(df) > 1:
        differences = timestamp_series.diff().iloc[1:]
        expected_difference = pd.to_timedelta(timeframe_ms, unit="ms")
        irregular = differences != expected_difference
        if irregular.any():
            irregular_position = int(np.flatnonzero(irregular.to_numpy())[0]) + 1
            previous_timestamp = timestamp_series.iloc[irregular_position - 1]
            actual_timestamp = timestamp_series.iloc[irregular_position]
            expected_timestamp = previous_timestamp + expected_difference
            raise MarketDataValidationError(
                "Missing or irregular candle interval: "
                f"expected {expected_timestamp}, received {actual_timestamp}"
            )

    now = normalize_current_utc_time(current_utc_time)
    final_close_time = timestamp_series.iloc[-1] + pd.to_timedelta(timeframe_ms, unit="ms")
    if final_close_time > now:
        raise MarketDataValidationError(
            f"Final {timeframe} candle is incomplete; expected close at {final_close_time}"
        )
