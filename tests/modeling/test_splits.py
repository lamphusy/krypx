"""Tests for holdout isolation and purged expanding walk-forward splits."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crypto_ai.config import settings
from crypto_ai.exceptions import ArtifactError, DatasetSplitError
from crypto_ai.modeling.splits import (
    create_split_plan,
    save_split_metadata,
    split_development_holdout,
    walk_forward_splits,
)


@pytest.fixture
def labeled_split_data() -> pd.DataFrame:
    """Return decisions with global-like indexes and auditable five-row exits."""
    row_count = 100
    timestamps = pd.date_range("2025-01-01", periods=row_count + 5, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps[:row_count],
            "exit_timestamp": timestamps[5:],
            "label": np.arange(row_count, dtype=np.int64) % 2,
        },
        index=np.arange(1_000, 1_000 + row_count),
    )


def test_development_precedes_holdout(labeled_split_data: pd.DataFrame) -> None:
    """Development, boundary purge, and holdout are strictly chronological."""
    development, boundary_purge, holdout = split_development_holdout(
        labeled_split_data,
        holdout_ratio=0.20,
        label_lookahead_rows=5,
    )

    assert development["timestamp"].max() < boundary_purge["timestamp"].min()
    assert boundary_purge["timestamp"].max() < holdout["timestamp"].min()


def test_holdout_ratio_is_correct_with_rounding(labeled_split_data: pd.DataFrame) -> None:
    """Holdout size is ceiling-rounded before boundary-purge rows are removed."""
    extra = labeled_split_data.iloc[[-1]].copy()
    extra.index = [1_100]
    extra["timestamp"] += pd.Timedelta(hours=1)
    extra["exit_timestamp"] += pd.Timedelta(hours=1)
    data = pd.concat([labeled_split_data, extra])

    development, boundary_purge, holdout = split_development_holdout(data, 0.20, 5)

    assert len(holdout) == math.ceil(len(data) * 0.20) == 21
    assert len(boundary_purge) == 5
    assert len(development) == 75


def test_partitioning_preserves_input_indexes(labeled_split_data: pd.DataFrame) -> None:
    """Positional slicing does not rewrite caller-owned index labels."""
    development, boundary_purge, holdout = split_development_holdout(labeled_split_data, 0.20, 5)

    assert development.index.tolist() == list(range(1_000, 1_075))
    assert boundary_purge.index.tolist() == list(range(1_075, 1_080))
    assert holdout.index.tolist() == list(range(1_080, 1_100))


def test_holdout_boundary_purge_covers_label_lookahead(
    labeled_split_data: pd.DataFrame,
) -> None:
    """Exactly horizon plus one rows are quarantined before the holdout."""
    _, boundary_purge, _ = split_development_holdout(labeled_split_data, 0.20, 5)

    assert settings.LABEL_LOOKAHEAD_ROWS == settings.PREDICTION_HORIZON + 1
    assert len(boundary_purge) == settings.LABEL_LOOKAHEAD_ROWS


def test_development_label_exits_precede_holdout_start(
    labeled_split_data: pd.DataFrame,
) -> None:
    """Actual development exit provenance ends strictly before holdout decisions."""
    development, _, holdout = split_development_holdout(labeled_split_data, 0.20, 5)

    assert development["exit_timestamp"].max() < holdout["timestamp"].min()


def test_development_label_leakage_is_rejected(labeled_split_data: pd.DataFrame) -> None:
    """A corrupted development exit crossing into holdout fails even if positions look safe."""
    data = labeled_split_data.copy()
    data.loc[1_074, "exit_timestamp"] = data.loc[1_080, "timestamp"]

    with pytest.raises(DatasetSplitError, match="Development label leakage"):
        split_development_holdout(data, 0.20, 5)


def test_walk_forward_training_precedes_validation(labeled_split_data: pd.DataFrame) -> None:
    """Every fold trains only on positions earlier than its validation block."""
    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)
    splits = walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=5)

    for train_indices, validation_indices in splits:
        assert train_indices.max() < validation_indices.min()


def test_walk_forward_training_expands(labeled_split_data: pd.DataFrame) -> None:
    """Each successive fold includes more chronological training observations."""
    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)
    splits = walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=5)

    assert [len(train) for train, _ in splits] == [40, 50, 60]
    assert all(np.array_equal(splits[0][0], split[0][:40]) for split in splits[1:])


def test_walk_forward_validation_blocks_are_contiguous(
    labeled_split_data: pd.DataFrame,
) -> None:
    """Fold validation blocks form one continuous non-overlapping OOF period."""
    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)
    splits = walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=5)

    assert int(splits[0][1][-1]) + 1 == int(splits[1][1][0])
    assert int(splits[1][1][-1]) + 1 == int(splits[2][1][0])
    combined = np.concatenate([validation for _, validation in splits])
    assert len(np.unique(combined)) == len(combined)


def test_purge_gap_is_applied(labeled_split_data: pd.DataFrame) -> None:
    """The complete label lookahead separates every training and validation block."""
    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)
    splits = walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=5)

    for train_indices, validation_indices in splits:
        assert int(train_indices.max()) + 5 < int(validation_indices.min())


def test_training_labels_do_not_touch_validation_prices(
    labeled_split_data: pd.DataFrame,
) -> None:
    """Stored training exit timestamps precede the first validation timestamp in every fold."""
    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)
    splits = walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=5)

    for train_indices, validation_indices in splits:
        training = development.iloc[train_indices]
        validation = development.iloc[validation_indices]
        assert training["exit_timestamp"].max() < validation["timestamp"].min()


def test_corrupt_training_exit_provenance_is_rejected(
    labeled_split_data: pd.DataFrame,
) -> None:
    """Direct provenance checks catch leakage that a positional gap alone would miss."""
    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)
    development = development.copy()
    development.loc[1_039, "exit_timestamp"] = development.loc[1_045, "timestamp"]

    with pytest.raises(DatasetSplitError, match="Walk-forward label leakage"):
        walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=5)


def test_holdout_and_boundary_purge_are_not_present_in_any_fold(
    labeled_split_data: pd.DataFrame,
) -> None:
    """Mapping positional fold indices back to labels never reaches quarantined indexes."""
    development, boundary_purge, holdout = split_development_holdout(labeled_split_data, 0.20, 5)
    excluded_indexes = set(boundary_purge.index) | set(holdout.index)
    splits = walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=5)

    for train_indices, validation_indices in splits:
        used_indexes = set(
            development.iloc[np.concatenate([train_indices, validation_indices])].index
        )
        assert used_indexes.isdisjoint(excluded_indexes)


def test_split_plan_contains_manifest_ready_metadata(
    tmp_path: Path,
    labeled_split_data: pd.DataFrame,
) -> None:
    """Partition and fold boundaries serialize without exposing holdout outcomes."""
    plan = create_split_plan(
        labeled_split_data,
        holdout_ratio=0.20,
        label_lookahead_rows=5,
        n_splits=3,
        test_ratio=0.10,
        gap_rows=5,
    )
    destination = tmp_path / "split_metadata.json"
    save_split_metadata(plan, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert plan.partition_metadata.development.row_count == 75
    assert plan.partition_metadata.boundary_purge.row_count == 5
    assert plan.partition_metadata.holdout.row_count == 20
    assert len(plan.fold_metadata) == 3
    assert payload["partitions"]["holdout"]["start_position"] == 80
    assert payload["folds"][0]["gap"]["row_count"] == 5
    assert "label" not in payload["partitions"]["holdout"]


def test_split_metadata_write_failure_is_an_artifact_error(
    tmp_path: Path,
    labeled_split_data: pd.DataFrame,
) -> None:
    plan = create_split_plan(
        labeled_split_data,
        holdout_ratio=0.20,
        label_lookahead_rows=5,
        n_splits=3,
        test_ratio=0.10,
        gap_rows=5,
    )
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ArtifactError, match="artifact directory"):
        save_split_metadata(plan, blocked_parent / "split_metadata.json")


def test_gap_shorter_than_label_lookahead_is_rejected(
    labeled_split_data: pd.DataFrame,
) -> None:
    """Callers cannot configure a purge shorter than the executable label lookahead."""
    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)

    with pytest.raises(DatasetSplitError, match="PREDICTION_HORIZON"):
        walk_forward_splits(development, n_splits=3, test_size_rows=10, gap_rows=4)


def test_insufficient_rows_raises_error(labeled_split_data: pd.DataFrame) -> None:
    """Impossible partition and fold requests fail with a clear project exception."""
    with pytest.raises(DatasetSplitError, match="Insufficient rows"):
        split_development_holdout(labeled_split_data.iloc[:7], 0.20, 5)

    development, _, _ = split_development_holdout(labeled_split_data, 0.20, 5)
    with pytest.raises(DatasetSplitError, match="Insufficient development rows"):
        walk_forward_splits(development, n_splits=10, test_size_rows=7, gap_rows=5)
