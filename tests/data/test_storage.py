"""Tests for incremental, atomic, content-addressed market-data storage."""

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import crypto_ai.data.storage as storage
from crypto_ai.data.storage import (
    get_raw_data_path,
    load_ohlcv_csv,
    load_or_update_ohlcv,
    sha256_file,
    symbol_to_slug,
)
from crypto_ai.exceptions import MarketDataValidationError


def test_invalid_utf8_market_data_is_a_domain_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.csv"
    path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(MarketDataValidationError, match="Unable to load OHLCV file"):
        load_ohlcv_csv(path, "1h")


def _fixed_fetcher(data: pd.DataFrame, calls: list[dict[str, Any]] | None = None) -> Any:
    def fetcher(**kwargs: Any) -> pd.DataFrame:
        if calls is not None:
            calls.append(kwargs)
        return data.copy()

    return fetcher


def test_symbol_slug_is_filesystem_safe() -> None:
    """Exchange punctuation cannot escape or fragment the target filename."""
    assert symbol_to_slug("BTC/USDT:USDT-PERP") == "btc_usdt_usdt_perp"
    assert "/" not in symbol_to_slug("ETH/USDT")


def test_empty_symbol_slug_is_rejected() -> None:
    """A symbol containing no safe characters cannot become a data path."""
    with pytest.raises(MarketDataValidationError, match="safe filename"):
        symbol_to_slug("///::---")


def test_raw_data_path_uses_symbol_and_timeframe(tmp_path: Path) -> None:
    """The mutable convenience path is deterministic and scoped to raw storage."""
    assert get_raw_data_path("BTC/USDT", "1h", tmp_path) == tmp_path / "btc_usdt_1h.csv"


def test_initial_update_writes_latest_and_snapshot(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """A first download creates both mutable latest data and immutable bytes."""
    data = synthetic_ohlcv.iloc[:3].copy()
    result = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=data["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        raw_dir=tmp_path / "raw",
        snapshots_dir=tmp_path / "snapshots",
        fetcher=_fixed_fetcher(data),
    )

    assert result.latest_path.exists()
    assert result.snapshot_path.exists()
    assert result.latest_path.read_bytes() == result.snapshot_path.read_bytes()
    pd.testing.assert_frame_equal(result.data, data.reset_index(drop=True))


def test_incremental_fetch_starts_after_last_timestamp(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """An update cursor advances exactly one timeframe after the latest stored candle."""
    raw_dir = tmp_path / "raw"
    snapshot_dir = tmp_path / "snapshots"
    initial = synthetic_ohlcv.iloc[:3].copy()
    load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=initial["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        raw_dir=raw_dir,
        snapshots_dir=snapshot_dir,
        fetcher=_fixed_fetcher(initial),
    )
    calls: list[dict[str, Any]] = []
    new_row = synthetic_ohlcv.iloc[[3]].copy()

    result = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=new_row["timestamp"].iloc[0] + pd.Timedelta(hours=1),
        raw_dir=raw_dir,
        snapshots_dir=snapshot_dir,
        fetcher=_fixed_fetcher(new_row, calls),
    )

    expected_since = int(new_row["timestamp"].iloc[0].timestamp() * 1000)
    assert calls[0]["since_ms"] == expected_since
    assert len(result.data) == 4


def test_duplicate_timestamps_are_removed_during_incremental_merge(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """An overlapping response cannot produce duplicate persisted timestamps."""
    initial = synthetic_ohlcv.iloc[:3].copy()
    now = synthetic_ohlcv["timestamp"].iloc[3] + pd.Timedelta(hours=1)
    raw_dir = tmp_path / "raw"
    snapshot_dir = tmp_path / "snapshots"
    load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshot_dir,
        fetcher=_fixed_fetcher(initial),
    )

    overlapping = synthetic_ohlcv.iloc[[2, 3]].copy()
    result = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=now,
        raw_dir=raw_dir,
        snapshots_dir=snapshot_dir,
        fetcher=_fixed_fetcher(overlapping),
    )

    assert len(result.data) == 4
    assert not result.data["timestamp"].duplicated().any()


def test_atomic_write_preserves_previous_file_on_failure(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed temporary write leaves the prior latest file byte-for-byte valid."""
    initial = synthetic_ohlcv.iloc[:3].copy()
    raw_dir = tmp_path / "raw"
    snapshot_dir = tmp_path / "snapshots"
    first = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=synthetic_ohlcv["timestamp"].iloc[3],
        raw_dir=raw_dir,
        snapshots_dir=snapshot_dir,
        fetcher=_fixed_fetcher(initial),
    )
    previous_latest = first.latest_path.read_bytes()
    previous_snapshot = first.snapshot_path.read_bytes()

    def fail_write(data: pd.DataFrame, path: Path) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(storage, "_write_csv", fail_write)
    with pytest.raises(MarketDataValidationError, match="simulated disk failure"):
        load_or_update_ohlcv(
            "BTC/USDT",
            "1h",
            1,
            current_utc_time=synthetic_ohlcv["timestamp"].iloc[4],
            raw_dir=raw_dir,
            snapshots_dir=snapshot_dir,
            fetcher=_fixed_fetcher(synthetic_ohlcv.iloc[[3]].copy()),
        )

    assert first.latest_path.read_bytes() == previous_latest
    assert first.snapshot_path.read_bytes() == previous_snapshot


def test_incremental_update_preserves_immutable_snapshot(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """A later update creates a new snapshot and never rewrites the earlier one."""
    raw_dir = tmp_path / "raw"
    snapshot_dir = tmp_path / "snapshots"
    first = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=synthetic_ohlcv["timestamp"].iloc[3],
        raw_dir=raw_dir,
        snapshots_dir=snapshot_dir,
        fetcher=_fixed_fetcher(synthetic_ohlcv.iloc[:3].copy()),
    )
    original_snapshot_bytes = first.snapshot_path.read_bytes()

    second = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=synthetic_ohlcv["timestamp"].iloc[4],
        raw_dir=raw_dir,
        snapshots_dir=snapshot_dir,
        fetcher=_fixed_fetcher(synthetic_ohlcv.iloc[[3]].copy()),
    )

    assert first.snapshot_path != second.snapshot_path
    assert first.snapshot_path.read_bytes() == original_snapshot_bytes
    assert second.snapshot_path.exists()


def test_market_data_result_identifies_exact_snapshot(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """The result hash identifies the exact persisted CSV bytes and path."""
    data = synthetic_ohlcv.iloc[:3].copy()
    result = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        1,
        current_utc_time=data["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        raw_dir=tmp_path / "raw",
        snapshots_dir=tmp_path / "snapshots",
        fetcher=_fixed_fetcher(data),
    )

    exact_digest = hashlib.sha256(result.snapshot_path.read_bytes()).hexdigest()
    assert result.sha256 == exact_digest
    assert result.snapshot_path.stem == exact_digest
    assert sha256_file(result.latest_path) == exact_digest


def test_existing_invalid_file_is_not_silently_replaced(tmp_path: Path) -> None:
    """Incremental loading fails closed when the mutable input is corrupt."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    latest = raw_dir / "btc_usdt_1h.csv"
    latest.write_text("timestamp,open\nnot-a-date,1\n", encoding="utf-8")

    with pytest.raises(MarketDataValidationError, match="missing required columns"):
        load_or_update_ohlcv(
            "BTC/USDT",
            "1h",
            1,
            current_utc_time=pd.Timestamp("2026-01-01", tz="UTC"),
            raw_dir=raw_dir,
            snapshots_dir=tmp_path / "snapshots",
            fetcher=_fixed_fetcher(pd.DataFrame()),
        )
