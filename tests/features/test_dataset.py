"""Tests for processed feature and label dataset preparation."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from crypto_ai.data.storage import load_or_update_ohlcv, sha256_file
from crypto_ai.exceptions import FeatureEngineeringError
from crypto_ai.features.dataset import (
    _manifest_absolute_path,
    get_prepared_dataset_manifest_path,
    load_labeled_dataset,
    load_prepared_dataset_bundle,
    prepare_datasets,
)
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
    assert (
        result.manifest_path
        == get_prepared_dataset_manifest_path("BTC/USDT", "1h", tmp_path / "processed").resolve()
    )
    assert result.manifest_path.exists()
    assert sha256_file(result.feature_path) == result.feature_sha256
    assert sha256_file(result.labeled_path) == result.labeled_sha256
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

    bundle = load_prepared_dataset_bundle(
        "BTC/USDT",
        "1h",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    required_manifest_fields = {
        "created_at_utc",
        "symbol",
        "timeframe",
        "source_snapshot_path",
        "source_snapshot_sha256",
        "feature_file_path",
        "feature_file_sha256",
        "labeled_file_path",
        "labeled_file_sha256",
        "feature_columns",
        "feature_schema_hash",
        "feature_configuration",
        "prediction_horizon",
        "label_lookahead_rows",
        "minimum_required_return",
        "minimum_net_edge_bps",
        "label_definition",
        "fee_assumptions",
        "slippage_assumptions",
        "spread_assumptions",
        "warmup_rows_removed",
        "unlabeled_rows_removed",
        "feature_row_count",
        "labeled_row_count",
        "feature_start_timestamp",
        "feature_end_timestamp",
        "labeled_start_timestamp",
        "labeled_end_timestamp",
    }
    assert required_manifest_fields <= bundle.manifest.keys()
    assert bundle.source_snapshot_sha256 == result.source_sha256
    assert bundle.feature_sha256 == result.feature_sha256
    assert bundle.labeled_sha256 == result.labeled_sha256
    assert bundle.manifest_bytes == result.manifest_path.read_bytes()
    assert bundle.manifest_sha256 == hashlib.sha256(bundle.manifest_bytes).hexdigest()
    assert json.loads(bundle.manifest_bytes) == bundle.manifest
    assert bundle.manifest["feature_row_count"] == len(bundle.features)
    assert bundle.manifest["labeled_row_count"] == len(bundle.labeled)


def test_bundle_captures_manifest_identity_from_the_same_validated_read(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path replacement after reading cannot change the returned manifest identity."""
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
    manifest_path = prepared.manifest_path
    assert manifest_path is not None
    original_read_bytes = Path.read_bytes
    validated_bytes = original_read_bytes(manifest_path)
    replacement_bytes = b'{"replacement": "must not be observed"}\n'
    replaced = False
    manifest_reads = 0

    def replace_path_after_read(path: Path) -> bytes:
        nonlocal manifest_reads, replaced
        content = original_read_bytes(path)
        if path == manifest_path:
            manifest_reads += 1
            if not replaced:
                path.write_bytes(replacement_bytes)
                replaced = True
        return content

    monkeypatch.setattr(Path, "read_bytes", replace_path_after_read)
    bundle = load_prepared_dataset_bundle(
        "BTC/USDT",
        "1h",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )

    assert replaced
    assert manifest_reads == 1
    assert original_read_bytes(manifest_path) == replacement_bytes
    assert bundle.manifest_bytes == validated_bytes
    assert bundle.manifest_sha256 == hashlib.sha256(validated_bytes).hexdigest()
    assert json.loads(bundle.manifest_bytes) == bundle.manifest
    assert bundle.manifest["symbol"] == "BTC/USDT"


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


def test_prepare_normalizes_latest_raw_hash_failure(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable latest raw bytes do not escape as a raw filesystem error."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)

    def fail_hash(path: Path) -> str:
        raise OSError("simulated latest hash failure")

    monkeypatch.setattr("crypto_ai.features.dataset.sha256_file", fail_hash)

    with pytest.raises(FeatureEngineeringError, match="Unable to hash latest raw data"):
        prepare_datasets(
            "BTC/USDT",
            "1h",
            current_utc_time=now,
            raw_dir=raw_dir,
            snapshots_dir=snapshots_dir,
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_prepare_normalizes_snapshot_hash_failure(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreadable immutable snapshot bytes yield a feature-domain failure."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)
    latest_path = raw_dir / "btc_usdt_1h.csv"
    source_digest = sha256_file(latest_path)
    snapshot_path = snapshots_dir / "btc_usdt_1h" / f"{source_digest}.csv"

    def selective_hash(path: Path) -> str:
        if Path(path) == snapshot_path:
            raise OSError("simulated snapshot hash failure")
        return sha256_file(Path(path))

    monkeypatch.setattr("crypto_ai.features.dataset.sha256_file", selective_hash)

    with pytest.raises(FeatureEngineeringError, match="Unable to hash immutable raw snapshot"):
        prepare_datasets(
            "BTC/USDT",
            "1h",
            current_utc_time=now,
            raw_dir=raw_dir,
            snapshots_dir=snapshots_dir,
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_prepare_normalizes_temporary_file_creation_failure(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared-bundle temporary-file errors are converted at the public boundary."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)

    def fail_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError("simulated prepared temporary-file failure")

    monkeypatch.setattr("crypto_ai.features.dataset.tempfile.mkstemp", fail_mkstemp)

    with pytest.raises(FeatureEngineeringError, match="simulated prepared temporary-file failure"):
        prepare_datasets(
            "BTC/USDT",
            "1h",
            current_utc_time=now,
            raw_dir=raw_dir,
            snapshots_dir=snapshots_dir,
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_bundle_loader_normalizes_invalid_utf8_manifest(
    tmp_path: Path,
) -> None:
    """Invalid UTF-8 manifest bytes are a feature-domain error, not a decode traceback."""
    manifest_path = get_prepared_dataset_manifest_path("BTC/USDT", "1h", tmp_path / "processed")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(FeatureEngineeringError, match="Unable to load prepared dataset manifest"):
        load_prepared_dataset_bundle(
            "BTC/USDT",
            "1h",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_manifest_path_resolution_failure_is_a_feature_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_resolve(self: Path, strict: bool = False) -> Path:
        raise RuntimeError("synthetic symlink loop")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    with pytest.raises(FeatureEngineeringError, match="synthetic symlink loop"):
        _manifest_absolute_path(
            {"source_snapshot_path": str(tmp_path / "snapshot.csv")},
            "source_snapshot_path",
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


def test_verified_bundle_remains_bound_to_snapshot_a_after_latest_advances_to_b(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """A later raw update cannot silently change a prepared bundle's provenance."""
    snapshot_a_data = synthetic_ohlcv.iloc[:60].copy()
    raw_dir, snapshots_dir, now_a = _create_raw_snapshot(tmp_path, snapshot_a_data)
    prepared_a = prepare_datasets(
        "BTC/USDT",
        "1h",
        current_utc_time=now_a,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )

    snapshot_b_tail = synthetic_ohlcv.iloc[60:].copy()
    now_b = synthetic_ohlcv["timestamp"].iloc[-1] + pd.Timedelta(hours=1)

    def fetch_b(**kwargs: object) -> pd.DataFrame:
        return snapshot_b_tail.copy()

    updated = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        30,
        current_utc_time=now_b,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        fetcher=fetch_b,
    )
    assert updated.sha256 != prepared_a.source_sha256

    bundle = load_prepared_dataset_bundle(
        "BTC/USDT",
        "1h",
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )

    assert bundle.source_snapshot_sha256 == prepared_a.source_sha256
    assert bundle.source_snapshot_path == prepared_a.source_snapshot_path
    assert sha256_file(bundle.source_snapshot_path) == prepared_a.source_sha256
    assert bundle.manifest["source_snapshot_sha256"] != updated.sha256
    pd.testing.assert_frame_equal(bundle.features, prepared_a.features)
    pd.testing.assert_frame_equal(bundle.labeled, prepared_a.labeled)


@pytest.mark.parametrize(
    ("path_attribute", "expected_message"),
    [
        ("feature_path", "feature file hash mismatch"),
        ("labeled_path", "labeled file hash mismatch"),
        ("source_snapshot_path", "source snapshot hash mismatch"),
    ],
)
def test_verified_bundle_rejects_corrupt_member(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    path_attribute: str,
    expected_message: str,
) -> None:
    """Every persisted member is verified before a prepared bundle is returned."""
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
    member_path = getattr(prepared, path_attribute)
    with member_path.open("ab") as file_handle:
        file_handle.write(b"corrupt")

    with pytest.raises(FeatureEngineeringError, match=expected_message):
        load_prepared_dataset_bundle(
            "BTC/USDT",
            "1h",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_verified_bundle_rejects_feature_configuration_drift(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bundle is not reused under settings different from those that created it."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)
    prepare_datasets(
        "BTC/USDT",
        "1h",
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    monkeypatch.setattr("crypto_ai.config.settings.EMA_SHORT", 10)

    with pytest.raises(FeatureEngineeringError, match="feature configuration"):
        load_prepared_dataset_bundle(
            "BTC/USDT",
            "1h",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_verified_bundle_rejects_label_and_cost_configuration_drift(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target threshold cannot be reused under different execution costs."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)
    prepare_datasets(
        "BTC/USDT",
        "1h",
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )
    monkeypatch.setattr("crypto_ai.config.settings.TAKER_FEE_RATE", 0.002)

    with pytest.raises(FeatureEngineeringError, match="label or execution-cost"):
        load_prepared_dataset_bundle(
            "BTC/USDT",
            "1h",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_verified_bundle_rejects_schema_manifest_mismatch(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """The ordered feature schema and its digest must both match current settings."""
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
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    manifest["feature_columns"] = list(reversed(manifest["feature_columns"]))
    prepared.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FeatureEngineeringError, match="feature schema"):
        load_prepared_dataset_bundle(
            "BTC/USDT",
            "1h",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_manifest_write_failure_leaves_no_loadable_incomplete_bundle(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The last-written manifest is required before finalized CSVs form a bundle."""
    raw_dir, snapshots_dir, now = _create_raw_snapshot(tmp_path, synthetic_ohlcv)

    def fail_manifest_write(payload: object, destination: Path) -> None:
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(
        "crypto_ai.features.dataset._atomic_write_json",
        fail_manifest_write,
    )
    with pytest.raises(FeatureEngineeringError, match="simulated manifest write failure"):
        prepare_datasets(
            "BTC/USDT",
            "1h",
            current_utc_time=now,
            raw_dir=raw_dir,
            snapshots_dir=snapshots_dir,
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )

    with pytest.raises(FeatureEngineeringError, match="manifest is missing"):
        load_prepared_dataset_bundle(
            "BTC/USDT",
            "1h",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )


def test_failed_reprepare_cannot_mix_new_csvs_with_old_completion_marker(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale manifest rejects partially published files from a newer snapshot."""
    snapshot_a_data = synthetic_ohlcv.iloc[:60].copy()
    raw_dir, snapshots_dir, now_a = _create_raw_snapshot(tmp_path, snapshot_a_data)
    prepare_datasets(
        "BTC/USDT",
        "1h",
        current_utc_time=now_a,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        interim_dir=tmp_path / "interim",
        processed_dir=tmp_path / "processed",
    )

    now_b = synthetic_ohlcv["timestamp"].iloc[-1] + pd.Timedelta(hours=1)

    def fetch_b(**kwargs: object) -> pd.DataFrame:
        return synthetic_ohlcv.iloc[60:].copy()

    load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        30,
        current_utc_time=now_b,
        raw_dir=raw_dir,
        snapshots_dir=snapshots_dir,
        fetcher=fetch_b,
    )

    def fail_manifest_write(payload: object, destination: Path) -> None:
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(
        "crypto_ai.features.dataset._atomic_write_json",
        fail_manifest_write,
    )
    with pytest.raises(FeatureEngineeringError, match="simulated manifest write failure"):
        prepare_datasets(
            "BTC/USDT",
            "1h",
            current_utc_time=now_b,
            raw_dir=raw_dir,
            snapshots_dir=snapshots_dir,
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )

    with pytest.raises(FeatureEngineeringError, match="hash mismatch"):
        load_prepared_dataset_bundle(
            "BTC/USDT",
            "1h",
            interim_dir=tmp_path / "interim",
            processed_dir=tmp_path / "processed",
        )
