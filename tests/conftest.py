"""Shared deterministic pytest fixtures for the Phase 1 pipeline."""

import socket

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def prevent_real_network_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that accidentally attempts to open a network connection."""

    def reject_connection(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tests must not make real network requests")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Return deterministic, valid UTC hourly OHLCV with varied prices and volume."""
    row_count = 72
    index = np.arange(row_count, dtype="float64")
    random = np.random.default_rng(42)
    close = 100.0 + 0.01 * index + 2.0 * np.sin(index / 6.0) + random.normal(0, 0.05, row_count)
    open_price = np.concatenate(([close[0] - 0.1], close[:-1]))
    high = np.maximum(open_price, close) + 0.5
    low = np.minimum(open_price, close) - 0.5
    volume = 1_000.0 + 10.0 * index + 50.0 * np.cos(index / 5.0)

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=row_count, freq="h", tz="UTC"),
            "open": open_price.astype("float64"),
            "high": high.astype("float64"),
            "low": low.astype("float64"),
            "close": close.astype("float64"),
            "volume": volume.astype("float64"),
        }
    )
