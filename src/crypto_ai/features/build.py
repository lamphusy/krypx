"""Point-in-time technical feature construction."""

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from crypto_ai.config import settings
from crypto_ai.exceptions import FeatureEngineeringError

logger = logging.getLogger(__name__)

_BASE_FEATURE_COLUMNS = [
    "ema_short",
    "ema_long",
    "ema_ratio",
    "close_to_ema_short",
    "close_to_ema_long",
    "macd",
    "macd_signal",
    "macd_diff",
    "rsi",
    "stoch_rsi",
    "bb_width",
    "bb_pct",
    "atr",
    "atr_pct",
    "candle_range_pct",
    "body_return",
    "volume_change",
    "volume_ma_ratio",
]


def get_expected_feature_columns(
    return_periods: Sequence[int] | None = None,
) -> list[str]:
    """Return the exact ordered feature schema for configured lagged returns."""
    periods = list(settings.RETURN_PERIODS if return_periods is None else return_periods)
    if not periods or any(
        not isinstance(period, int) or isinstance(period, bool) or period <= 0 for period in periods
    ):
        raise FeatureEngineeringError("Return periods must be a non-empty sequence of integers")
    if len(set(periods)) != len(periods):
        raise FeatureEngineeringError("Return periods must be unique")
    return [*_BASE_FEATURE_COLUMNS, *(f"return_{period}" for period in periods)]


def _validate_feature_input(df: pd.DataFrame) -> None:
    if not df.columns.is_unique:
        raise FeatureEngineeringError("OHLCV column names must be unique")
    missing_columns = [column for column in settings.RAW_COLUMNS if column not in df.columns]
    if missing_columns:
        raise FeatureEngineeringError(f"Missing required OHLCV columns: {missing_columns}")
    if df.empty:
        raise FeatureEngineeringError("Cannot compute features from empty OHLCV data")
    present_label_columns = [column for column in settings.LABEL_COLUMNS if column in df.columns]
    if present_label_columns:
        raise FeatureEngineeringError(
            f"Feature input must not contain future-derived label columns: {present_label_columns}"
        )

    timestamps = df["timestamp"]
    if not isinstance(timestamps.dtype, pd.DatetimeTZDtype) or str(timestamps.dt.tz) != "UTC":
        raise FeatureEngineeringError("timestamp must be a timezone-aware UTC datetime column")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise FeatureEngineeringError("OHLCV timestamps must be unique and chronological")

    for column in settings.RAW_COLUMNS[1:]:
        if not is_numeric_dtype(df[column].dtype):
            raise FeatureEngineeringError(f"{column} must be numeric")
        if not np.isfinite(df[column].to_numpy(dtype="float64", copy=False)).all():
            raise FeatureEngineeringError(f"{column} contains missing or infinite values")

    if (df[["open", "high", "low", "close"]] <= 0.0).any().any():
        raise FeatureEngineeringError("OHLC prices must be positive")
    if (df["volume"] < 0.0).any():
        raise FeatureEngineeringError("volume must be non-negative")
    if (df["high"] < df["low"]).any():
        raise FeatureEngineeringError("Every candle must satisfy high >= low")
    if ((df["open"] < df["low"]) | (df["open"] > df["high"])).any():
        raise FeatureEngineeringError("Every candle must satisfy low <= open <= high")
    if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise FeatureEngineeringError("Every candle must satisfy low <= close <= high")


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing-only technical features and remove incomplete warm-up rows.

    Args:
        df: Valid chronological OHLCV data with timezone-aware UTC timestamps.

    Returns:
        Original market columns plus the exact ordered feature schema. Initial rows
        that lack sufficient trailing history are removed, while the latest complete
        row remains available for inference.

    Raises:
        FeatureEngineeringError: If input data or computed features are invalid, or
            if insufficient history exists for one complete feature row.
    """
    _validate_feature_input(df)
    minimum_input_rows = max(
        settings.EMA_LONG,
        settings.MACD_SLOW + settings.MACD_SIGNAL - 1,
        2 * settings.STOCH_RSI_PERIOD - 1,
        settings.BB_PERIOD,
        settings.ATR_PERIOD,
        settings.VOLUME_MA_PERIOD,
        max(settings.RETURN_PERIODS) + 1,
    )
    if len(df) < minimum_input_rows:
        raise FeatureEngineeringError(
            "Insufficient history to compute one complete feature row; "
            f"received {len(df)} rows, need at least {minimum_input_rows}"
        )

    result = df.copy()
    close = result["close"].astype("float64")
    high = result["high"].astype("float64")
    low = result["low"].astype("float64")
    open_price = result["open"].astype("float64")
    volume = result["volume"].astype("float64")

    result["ema_short"] = EMAIndicator(
        close=close,
        window=settings.EMA_SHORT,
        fillna=False,
    ).ema_indicator()
    result["ema_long"] = EMAIndicator(
        close=close,
        window=settings.EMA_LONG,
        fillna=False,
    ).ema_indicator()
    result["ema_ratio"] = result["ema_short"] / result["ema_long"] - 1.0
    result["close_to_ema_short"] = close / result["ema_short"] - 1.0
    result["close_to_ema_long"] = close / result["ema_long"] - 1.0

    macd = MACD(
        close=close,
        window_slow=settings.MACD_SLOW,
        window_fast=settings.MACD_FAST,
        window_sign=settings.MACD_SIGNAL,
        fillna=False,
    )
    result["macd"] = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_diff"] = macd.macd_diff()

    result["rsi"] = RSIIndicator(
        close=close,
        window=settings.RSI_PERIOD,
        fillna=False,
    ).rsi()
    result["stoch_rsi"] = StochRSIIndicator(
        close=close,
        window=settings.STOCH_RSI_PERIOD,
        fillna=False,
    ).stochrsi()

    bollinger = BollingerBands(
        close=close,
        window=settings.BB_PERIOD,
        window_dev=settings.BB_STD_DEV,
        fillna=False,
    )
    bb_middle = bollinger.bollinger_mavg()
    bb_upper = bollinger.bollinger_hband()
    bb_lower = bollinger.bollinger_lband()
    result["bb_width"] = (bb_upper - bb_lower) / bb_middle
    result["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower)

    result["atr"] = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=settings.ATR_PERIOD,
        fillna=False,
    ).average_true_range()
    result["atr_pct"] = result["atr"] / close
    result["candle_range_pct"] = (high - low) / close
    result["body_return"] = close / open_price - 1.0

    result["volume_change"] = volume / volume.shift(1) - 1.0
    volume_mean = volume.rolling(
        window=settings.VOLUME_MA_PERIOD,
        min_periods=settings.VOLUME_MA_PERIOD,
    ).mean()
    result["volume_ma_ratio"] = volume / volume_mean - 1.0

    for period in settings.RETURN_PERIODS:
        result[f"return_{period}"] = close / close.shift(period) - 1.0

    feature_columns = get_expected_feature_columns()
    derived_columns = [
        column for column in result.columns if column not in settings.NON_FEATURE_COLUMNS
    ]
    if derived_columns != feature_columns:
        raise FeatureEngineeringError(
            "Derived feature columns do not match the ordered expected schema"
        )

    result.loc[:, feature_columns] = result.loc[:, feature_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    complete_rows = result[feature_columns].notna().all(axis=1)
    if not complete_rows.any():
        raise FeatureEngineeringError(
            "Insufficient history to compute one complete feature row; provide more OHLCV data"
        )

    first_complete_position = int(np.flatnonzero(complete_rows.to_numpy())[0])
    if not complete_rows.iloc[first_complete_position:].all():
        first_bad_position = (
            int(np.flatnonzero((~complete_rows.iloc[first_complete_position:]).to_numpy())[0])
            + first_complete_position
        )
        bad_timestamp = result["timestamp"].iloc[first_bad_position]
        raise FeatureEngineeringError(
            f"Feature values became incomplete after warm-up at {bad_timestamp}"
        )

    warmup_rows = first_complete_position
    result = result.iloc[first_complete_position:].copy()
    if not all(is_numeric_dtype(result[column].dtype) for column in feature_columns):
        raise FeatureEngineeringError("Every derived feature must be numeric")
    if not np.isfinite(result[feature_columns].to_numpy(dtype="float64")).all():
        raise FeatureEngineeringError("Computed features contain missing or infinite values")

    logger.info(
        "Computed %s point-in-time features for %s rows; removed %s warm-up rows",
        len(feature_columns),
        len(result),
        warmup_rows,
    )
    return result
