"""Tests for the Phase 1 command-line interface."""

from pathlib import Path

import pandas as pd
import pytest

from crypto_ai.cli import main
from crypto_ai.data.storage import MarketDataResult
from crypto_ai.exceptions import MarketDataNetworkError
from crypto_ai.features.dataset import PreparedDatasetResult


def test_fetch_command_reports_snapshot_identity(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful fetch prints its row range, immutable path, and exact hash."""
    data = synthetic_ohlcv.iloc[:2].copy()
    expected_arguments: dict[str, object] = {}

    def update(**kwargs: object) -> MarketDataResult:
        expected_arguments.update(kwargs)
        return MarketDataResult(
            data=data,
            latest_path=tmp_path / "raw" / "eth_usdt_4h.csv",
            snapshot_path=tmp_path / "snapshots" / "abc123.csv",
            sha256="abc123",
        )

    monkeypatch.setattr("crypto_ai.cli.load_or_update_ohlcv", update)

    assert (
        main(["fetch", "--symbol", "ETH/USDT", "--timeframe", "4h", "--lookback-days", "30"]) == 0
    )
    output = capsys.readouterr().out
    assert expected_arguments == {
        "symbol": "ETH/USDT",
        "timeframe": "4h",
        "lookback_days": 30,
    }
    assert "Stored 2 closed candles" in output
    assert "Immutable snapshot:" in output
    assert "SHA-256: abc123" in output


def test_fetch_command_returns_failure_for_project_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected market-data failures produce a nonzero process status."""

    def fail(**kwargs: object) -> MarketDataResult:
        raise MarketDataNetworkError("network unavailable")

    monkeypatch.setattr("crypto_ai.cli.load_or_update_ohlcv", fail)

    assert main(["fetch"]) == 1


def test_prepare_command_reports_rows_schema_and_raw_provenance(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Preparation reports structure and provenance without revealing outcome summaries."""
    features = synthetic_ohlcv.iloc[:10].copy()
    labeled = features.iloc[:5].copy()
    arguments: dict[str, object] = {}

    def prepare(**kwargs: object) -> PreparedDatasetResult:
        arguments.update(kwargs)
        return PreparedDatasetResult(
            features=features,
            labeled=labeled,
            feature_path=tmp_path / "interim" / "features.csv",
            labeled_path=tmp_path / "processed" / "labeled.csv",
            source_snapshot_path=tmp_path / "snapshots" / "abc123.csv",
            source_sha256="abc123",
            feature_columns=("ema_short", "return_1"),
            warmup_rows_removed=33,
            unlabeled_rows_removed=5,
            minimum_required_return=0.003,
        )

    monkeypatch.setattr("crypto_ai.cli.prepare_datasets", prepare)

    assert main(["prepare", "--symbol", "ETH/USDT", "--timeframe", "4h"]) == 0
    output = capsys.readouterr().out
    assert arguments == {"symbol": "ETH/USDT", "timeframe": "4h"}
    assert "Prepared 10 inference rows" in output
    assert "Labeled decision rows: 5" in output
    assert "Feature count: 2" in output
    assert "Warm-up rows removed: 33" in output
    assert "Unrealizable tail rows removed: 5" in output
    assert "Raw SHA-256: abc123" in output
    assert "label distribution" not in output.lower()
