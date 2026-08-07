"""Tests for processed feature and label dataset preparation."""

from pathlib import Path

import pandas as pd
import pytest

from crypto_ai.data.storage import load_or_update_ohlcv, sha256_file
from crypto_ai.exceptions import FeatureEngineeringError
from crypto_ai.features.dataset import load_labeled_dataset, prepare_datasets
from crypto_ai.modeling.splits import create_split_plan


def _create_raw_snapshot(
    tmp_path: Path,
    data: pd.DataFrame,
) -> tuple[Path, Path, pd.Timestamp]:
    raw_dir = tmp_path / "raw"
    snapshots_dir = tmp_path / "snapshots"
    now = data["timestamp"].iloc[-1] + pd.Timedelta(hours=1)

    def fetcher(**kwargs: object) -> pd.DataFrame:
        return data.copy()

    load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        30,
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        fetcher=fetcher,
    )
    return raw_dir, snapshots_dir, now


def test_prepare_datasets_uses_exact_immutable_raw_snapshot(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """Prepared data records and verifies the immutable raw bytes it consumed."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)

    result = prepare_datasets(
        "BTC/USDT",
        "1h",
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )

    assert result.source_snapshot_path.exists()
    assert result.source_snapshot_path.stem == result.source_sha256
    assert sha256_file(result.source_snapshot_path) == result.source_sha256
    assert result.feature_path.exists()
    assert result.labeled_path.exists()
    stored_features = pd.read_csv(result.feature_path)
    stored_labeled = pd.read_csv(result.labeled_path)
    assert len(stored_features) == len(result.features)
    assert len(stored_labeled) == len(result.labeled)
    assert stored_features.columns.tolist() == result.features.columns.tolist()
    assert stored_labeled.columns.tolist() == result.labeled.columns.tolist()
    loaded_labeled = load_labeled_dataset(result.labeled_path)
    assert len(loaded_labeled) == len(result.labeled)
    assert str(loaded_labeled["timestamp"].dt.tz) == "UTC"
    assert str(loaded_labeled["exit_timestamp"].dt.tz) == "UTC"
    assert loaded_labeled["label"].dtype == "int8"


def test_prepared_feature_matrix_is_complete_and_retains_inference_tail(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """Training is complete while the separate inference frame retains its latest row."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)
    result = prepare_datasets(
        "BTC/USDT",
        "1h",
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )

    assert not result.features[list(result.feature_columns)].isna().any().any()
    assert not result.labeled[list(result.feature_columns)].isna().any().any()
    assert result.features["timestamp"].iloc[-1] == synthetic_ohlcv["timestamp"].iloc[-1]
    assert result.unlabeled_rows_removed == 5
    assert len(result.features) - len(result.labeled) == 5


def test_prepare_fails_when_matching_snapshot_is_missing(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """The mutable latest file alone is never accepted as a research input."""
    raw_dir, _, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)

    with pytest.raises(FeatureEngineeringError, match="snapshot.*missing"):
        prepare_datasets(
            "BTC/USDT",
            "1h",
            current_utc_time=now,
            raw_dir=raw_dir,
            snapshots_dir=tmp_path / "empty-snapshots",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_persisted_labeled_dataset_builds_default_isolated_split_plan(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """Milestone 2 output flows into all default Milestone 3 boundaries and folds."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)
    prepared = prepare_datasets(
        "BTC/USDT",
        "1h",
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    labeled = load_labeled_dataset(prepared.labeled_path)

    plan = create_split_plan(labeled)

    assert len(plan.folds) == 5
    assert len(plan.boundary_purge) == 5
    assert plan.development["exit_timestamp"].max() < plan.holdout["timestamp"].min()
    for train_indices, validation_indices in plan.folds:
        assert (
            plan.development.iloc[train_indices]["exit_timestamp"].max()
            < plan.development.iloc[validation_indices]["timestamp"].min()
        )


def test_load_labeled_dataset_rejects_wrong_schema(tmp_path: Path) -> None:
    """Stored data cannot silently omit provenance or feature columns."""
    path = tmp_path / "invalid_labeled.csv"
    path.write_text("timestamp,label\n2026-01-01T00:00:00Z,1\n", encoding="utf-8")

    with pytest.raises(FeatureEngineeringError, match="ordered expected schema"):
        load_labeled_dataset(path)
