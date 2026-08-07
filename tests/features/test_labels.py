"""Tests for executable next-open labels and provenance."""

import numpy as np
import pandas as pd
import pytest

from crypto_ai.exceptions import LabelGenerationError
from crypto_ai.features.labels import add_labels


@pytest.fixture
def manual_feature_data() -> pd.DataFrame:
    """Return a small market frame with manually auditable opens and one feature."""
    opens = np.array([99.0, 100.0, 101.0, 103.0, 102.0, 105.0], dtype="float64")
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-02-01", periods=len(opens), freq="h", tz="UTC"),
            "open": opens,
            "high": opens + 2.0,
            "low": opens - 2.0,
            "close": opens + 1.0,
            "volume": np.arange(1_000.0, 1_000.0 + len(opens), dtype="float64"),
            "example_feature": np.linspace(0.0, 1.0, len(opens), dtype="float64"),
        }
    )


def test_entry_uses_next_open(manual_feature_data: pd.DataFrame) -> None:
    """A decision at row zero enters at row one's open."""
    result = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.0)

    assert result["entry_open"].iloc[0] == 100.0


def test_exit_uses_horizon_plus_one_open(manual_feature_data: pd.DataFrame) -> None:
    """A two-period holding horizon exits three row positions after the decision."""
    result = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.0)

    assert result["exit_open"].iloc[0] == 103.0


def test_label_records_entry_and_exit_timestamp_provenance(
    manual_feature_data: pd.DataFrame,
) -> None:
    """Both execution timestamps are retained for later leakage audits."""
    result = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.0)

    assert result["entry_timestamp"].iloc[0] == manual_feature_data["timestamp"].iloc[1]
    assert result["exit_timestamp"].iloc[0] == manual_feature_data["timestamp"].iloc[3]


def test_label_uses_executable_forward_return(manual_feature_data: pd.DataFrame) -> None:
    """The target return is based on entry and exit opens, not decision closes."""
    result = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.0)

    assert result["gross_forward_return"].iloc[0] == pytest.approx(0.03)


def test_label_uses_minimum_required_return(manual_feature_data: pd.DataFrame) -> None:
    """The binary label is positive only when gross return strictly clears its threshold."""
    below_return = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.031)
    above_return = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.029)

    assert below_return["label"].iloc[0] == 0
    assert above_return["label"].iloc[0] == 1
    assert above_return["label"].dtype == np.dtype("int8")


def test_unrealizable_tail_rows_are_removed(manual_feature_data: pd.DataFrame) -> None:
    """Exactly horizon plus one decision rows lack a known exit open."""
    result = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.0)

    assert len(result) == len(manual_feature_data) - 3
    assert result.index.tolist() == [0, 1, 2]


def test_features_are_unchanged_when_labels_are_added(manual_feature_data: pd.DataFrame) -> None:
    """Label generation appends provenance without altering any market or feature value."""
    original = manual_feature_data.copy(deep=True)
    result = add_labels(manual_feature_data, horizon=2, minimum_required_return=0.0)

    pd.testing.assert_frame_equal(result[original.columns], original.loc[result.index])
    pd.testing.assert_frame_equal(manual_feature_data, original)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_invalid_horizon_is_rejected(
    manual_feature_data: pd.DataFrame,
    horizon: object,
) -> None:
    """The positional execution horizon must be a positive integer."""
    with pytest.raises(LabelGenerationError, match="positive integer"):
        add_labels(manual_feature_data, horizon=horizon, minimum_required_return=0.0)  # type: ignore[arg-type]


def test_insufficient_rows_for_exit_are_rejected(manual_feature_data: pd.DataFrame) -> None:
    """A frame without one realizable decision row fails clearly."""
    with pytest.raises(LabelGenerationError, match="Insufficient rows"):
        add_labels(manual_feature_data.iloc[:3], horizon=2, minimum_required_return=0.0)
