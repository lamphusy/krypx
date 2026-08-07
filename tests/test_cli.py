"""Tests for the Phase 1 command-line interface."""

from pathlib import Path

import pandas as pd
import pytest

from crypto_ai.cli import main
from crypto_ai.data.storage import MarketDataResult
from crypto_ai.exceptions import MarketDataNetworkError


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
