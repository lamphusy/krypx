"""Tests for the raw OHLCV validation contract."""

import pandas as pd
import pytest

from crypto_ai.data.validation import (
    filter_closed_candles,
    timeframe_to_milliseconds,
    validate_ohlcv,
)
from crypto_ai.exceptions import MarketDataValidationError


def _closed_time(data: pd.DataFrame) -> pd.Timestamp:
    return data["timestamp"].iloc[-1] + pd.Timedelta(hours=1)


def test_valid_ohlcv_passes_all_checks(synthetic_ohlcv: pd.DataFrame) -> None:
    """A complete deterministic fixture satisfies the full contract."""
    validate_ohlcv(synthetic_ohlcv, "1h", _closed_time(synthetic_ohlcv))


def test_timestamp_is_utc(synthetic_ohlcv: pd.DataFrame) -> None:
    """A timezone other than UTC is rejected even when it describes real instants."""
    data = synthetic_ohlcv.copy()
    data["timestamp"] = data["timestamp"].dt.tz_convert("Asia/Tokyo")

    with pytest.raises(MarketDataValidationError, match="timezone must be UTC"):
        validate_ohlcv(data, "1h")


def test_naive_timestamp_is_rejected(synthetic_ohlcv: pd.DataFrame) -> None:
    """Naive timestamps cannot silently inherit a timezone."""
    data = synthetic_ohlcv.copy()
    data["timestamp"] = data["timestamp"].dt.tz_localize(None)

    with pytest.raises(MarketDataValidationError, match="timezone-aware"):
        validate_ohlcv(data, "1h")


def test_timestamp_aligns_to_timeframe_grid(synthetic_ohlcv: pd.DataFrame) -> None:
    """Hourly candles must open exactly on a UTC hour boundary."""
    data = synthetic_ohlcv.iloc[:1].copy()
    data.loc[data.index[0], "timestamp"] += pd.Timedelta(minutes=30)

    with pytest.raises(MarketDataValidationError, match="not aligned"):
        validate_ohlcv(data, "1h")


def test_missing_candle_is_detected(synthetic_ohlcv: pd.DataFrame) -> None:
    """A gap in an otherwise regular sequence is reported."""
    data = synthetic_ohlcv.drop(index=3).reset_index(drop=True)

    with pytest.raises(MarketDataValidationError, match="Missing or irregular"):
        validate_ohlcv(data, "1h")


def test_duplicate_timestamp_is_rejected(synthetic_ohlcv: pd.DataFrame) -> None:
    """Raw validation never accepts two candles for the same open time."""
    data = pd.concat([synthetic_ohlcv.iloc[:2], synthetic_ohlcv.iloc[[1]]], ignore_index=True)

    with pytest.raises(MarketDataValidationError, match="Duplicate"):
        validate_ohlcv(data, "1h")


def test_invalid_ohlc_relationship_is_rejected(synthetic_ohlcv: pd.DataFrame) -> None:
    """Open and close prices must remain within the candle range."""
    data = synthetic_ohlcv.copy()
    data.loc[0, "close"] = data.loc[0, "high"] + 1.0

    with pytest.raises(MarketDataValidationError, match="low <= close <= high"):
        validate_ohlcv(data, "1h")


def test_negative_volume_is_rejected(synthetic_ohlcv: pd.DataFrame) -> None:
    """Trading volume cannot be negative."""
    data = synthetic_ohlcv.copy()
    data.loc[0, "volume"] = -1.0

    with pytest.raises(MarketDataValidationError, match="non-negative"):
        validate_ohlcv(data, "1h")


def test_non_float64_price_is_rejected(synthetic_ohlcv: pd.DataFrame) -> None:
    """Price columns use a stable float64 representation."""
    data = synthetic_ohlcv.copy()
    data["open"] = data["open"].astype("float32")

    with pytest.raises(MarketDataValidationError, match="open must use float64"):
        validate_ohlcv(data, "1h")


def test_incomplete_last_candle_is_rejected(synthetic_ohlcv: pd.DataFrame) -> None:
    """Validation identifies a final candle whose close time is still in the future."""
    data = synthetic_ohlcv.iloc[:2].copy()
    now = data["timestamp"].iloc[-1] + pd.Timedelta(minutes=30)

    with pytest.raises(MarketDataValidationError, match="incomplete"):
        validate_ohlcv(data, "1h", now)


def test_filter_closed_candles_uses_explicit_clock(synthetic_ohlcv: pd.DataFrame) -> None:
    """Filtering retains only rows whose full timeframe elapsed before the fixed clock."""
    data = synthetic_ohlcv.iloc[:3].copy()
    now = data["timestamp"].iloc[-1] + pd.Timedelta(minutes=30)

    filtered = filter_closed_candles(data, "1h", now)

    pd.testing.assert_frame_equal(filtered, data.iloc[:2])


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [("15m", 900_000), ("4h", 14_400_000), ("1d", 86_400_000)],
)
def test_timeframe_to_milliseconds_supports_fixed_units(timeframe: str, expected: int) -> None:
    """The supported fixed CCXT timeframe units convert exactly."""
    assert timeframe_to_milliseconds(timeframe) == expected


def test_unsupported_timeframe_is_rejected() -> None:
    """Calendar-dependent or malformed timeframes fail clearly."""
    with pytest.raises(MarketDataValidationError, match="Unsupported fixed timeframe"):
        timeframe_to_milliseconds("1M")
