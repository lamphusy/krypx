"""Tests for trailing-only technical feature construction."""

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from crypto_ai.config import settings
from crypto_ai.exceptions import FeatureEngineeringError
from crypto_ai.features.build import compute_features, get_expected_feature_columns


def test_compute_features_preserves_timestamp(synthetic_ohlcv: pd.DataFrame) -> None:
    """Every returned timestamp maps unchanged to its source market row."""
    result = compute_features(synthetic_ohlcv)

    pd.testing.assert_series_equal(
        result["timestamp"], synthetic_ohlcv.loc[result.index, "timestamp"]
    )


def test_compute_features_preserves_ohlcv_columns(synthetic_ohlcv: pd.DataFrame) -> None:
    """Feature construction never overwrites original market observations."""
    original = synthetic_ohlcv.copy(deep=True)
    result = compute_features(synthetic_ohlcv)

    pd.testing.assert_frame_equal(result[settings.RAW_COLUMNS], original.loc[result.index])
    pd.testing.assert_frame_equal(synthetic_ohlcv, original)


def test_expected_feature_columns_exist_in_exact_order(synthetic_ohlcv: pd.DataFrame) -> None:
    """The runtime schema exactly matches the frozen ordered indicator contract."""
    result = compute_features(synthetic_ohlcv)
    feature_columns = [column for column in result if column not in settings.NON_FEATURE_COLUMNS]

    assert feature_columns == get_expected_feature_columns()
    assert feature_columns[-6:] == [
        "return_1",
        "return_2",
        "return_3",
        "return_6",
        "return_12",
        "return_24",
    ]


def test_features_are_numeric(synthetic_ohlcv: pd.DataFrame) -> None:
    """Every model feature uses a numeric dtype."""
    result = compute_features(synthetic_ohlcv)

    assert all(is_numeric_dtype(result[column]) for column in get_expected_feature_columns())


def test_feature_rows_are_chronological(synthetic_ohlcv: pd.DataFrame) -> None:
    """Warm-up removal preserves chronological ordering and unique timestamps."""
    result = compute_features(synthetic_ohlcv)

    assert result["timestamp"].is_monotonic_increasing
    assert not result["timestamp"].duplicated().any()


def test_feature_output_has_no_missing_or_infinite_values(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """The inference-ready feature matrix is complete and finite."""
    result = compute_features(synthetic_ohlcv)
    matrix = result[get_expected_feature_columns()].to_numpy(dtype="float64")

    assert not np.isnan(matrix).any()
    assert np.isfinite(matrix).all()


def test_latest_closed_row_is_available_for_inference(synthetic_ohlcv: pd.DataFrame) -> None:
    """Feature warm-up never discards the newest complete market observation."""
    result = compute_features(synthetic_ohlcv)

    assert result["timestamp"].iloc[-1] == synthetic_ohlcv["timestamp"].iloc[-1]


def test_explicit_feature_formulas_match_market_values(synthetic_ohlcv: pd.DataFrame) -> None:
    """Shift-based returns and candle ratios follow their documented formulas."""
    result = compute_features(synthetic_ohlcv)
    row = result.iloc[-1]
    source_position = synthetic_ohlcv.index.get_loc(result.index[-1])

    assert row["body_return"] == pytest.approx(row["close"] / row["open"] - 1.0)
    assert row["candle_range_pct"] == pytest.approx((row["high"] - row["low"]) / row["close"])
    assert row["return_24"] == pytest.approx(
        row["close"] / synthetic_ohlcv["close"].iloc[source_position - 24] - 1.0
    )


def test_insufficient_history_raises_clear_error(synthetic_ohlcv: pd.DataFrame) -> None:
    """Short inference history fails instead of returning an unusable latest row."""
    with pytest.raises(FeatureEngineeringError, match="Insufficient history"):
        compute_features(synthetic_ohlcv.iloc[:20])


def test_future_data_changes_do_not_modify_past_features(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """Perturbing all future market values leaves every past feature bit-for-bit stable."""
    cutoff_position = 50
    cutoff_timestamp = synthetic_ohlcv["timestamp"].iloc[cutoff_position]
    changed = synthetic_ohlcv.copy()
    future = changed.index > cutoff_position
    changed.loc[future, ["open", "high", "low", "close"]] *= 3.0
    changed.loc[future, "volume"] *= 7.0

    original_features = compute_features(synthetic_ohlcv)
    changed_features = compute_features(changed)
    feature_columns = get_expected_feature_columns()
    original_past = original_features.loc[
        original_features["timestamp"] <= cutoff_timestamp, feature_columns
    ]
    changed_past = changed_features.loc[
        changed_features["timestamp"] <= cutoff_timestamp, feature_columns
    ]

    pd.testing.assert_frame_equal(original_past, changed_past, check_exact=True)


def test_unexpected_helper_column_is_not_silently_used(synthetic_ohlcv: pd.DataFrame) -> None:
    """An accidental derived column cannot enter the feature schema unnoticed."""
    data = synthetic_ohlcv.copy()
    data["future_helper"] = data["close"].shift(-1)

    with pytest.raises(FeatureEngineeringError, match="ordered expected schema"):
        compute_features(data)


def test_future_label_columns_are_rejected(synthetic_ohlcv: pd.DataFrame) -> None:
    """Feature construction cannot accidentally consume a future-derived target column."""
    data = synthetic_ohlcv.copy()
    data["exit_open"] = data["open"].shift(-5)

    with pytest.raises(FeatureEngineeringError, match="future-derived label"):
        compute_features(data)
