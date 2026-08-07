"""Tests for paginated, retrying CCXT OHLCV retrieval."""

from collections.abc import Sequence
from typing import Any

import ccxt
import pandas as pd
import pytest

from crypto_ai.data.fetch import create_exchange, fetch_ohlcv, rows_to_ohlcv_dataframe
from crypto_ai.exceptions import (
    MarketDataExchangeError,
    MarketDataNetworkError,
    MarketDataValidationError,
)


class FakeExchange:
    """Return scripted pages or exceptions without any network access."""

    def __init__(self, responses: Sequence[list[list[Any]] | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def fetch_ohlcv(self, **kwargs: Any) -> list[list[Any]]:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _exchange_rows(data: pd.DataFrame) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for row in data.itertuples(index=False):
        rows.append(
            [
                int(row.timestamp.timestamp() * 1000),
                row.open,
                row.high,
                row.low,
                row.close,
                row.volume,
            ]
        )
    return rows


def test_create_exchange_enables_ccxt_rate_limiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public exchange client is configured to let CCXT enforce request spacing."""
    configurations: list[dict[str, Any]] = []

    class StubExchange:
        def __init__(self, configuration: dict[str, Any]) -> None:
            configurations.append(configuration)
            self.has = {"fetchOHLCV": True}

    monkeypatch.setattr(ccxt, "stubexchange", StubExchange, raising=False)

    exchange = create_exchange("stubexchange")

    assert isinstance(exchange, StubExchange)
    assert configurations == [{"enableRateLimit": True}]


def test_create_exchange_rejects_unknown_exchange() -> None:
    """An unknown configured exchange is converted to a project-specific error."""
    with pytest.raises(MarketDataExchangeError, match="Unsupported CCXT exchange"):
        create_exchange("not_a_real_ccxt_exchange")


def test_fetch_paginates_until_end(synthetic_ohlcv: pd.DataFrame) -> None:
    """The next page begins one timeframe after the last returned timestamp."""
    data = synthetic_ohlcv.iloc[:4]
    rows = _exchange_rows(data)
    exchange = FakeExchange([rows[:2], rows[2:]])
    since_ms = rows[0][0]
    until_ms = rows[-1][0] + 3_600_000

    result = fetch_ohlcv(
        "BTC/USDT",
        "1h",
        since_ms,
        until_ms,
        limit=2,
        current_utc_time=data["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        exchange=exchange,  # type: ignore[arg-type]
    )

    pd.testing.assert_frame_equal(result, data.reset_index(drop=True))
    assert [call["since"] for call in exchange.calls] == [since_ms, rows[2][0]]


def test_fetch_retries_network_errors(synthetic_ohlcv: pd.DataFrame) -> None:
    """Transient failures use exponential backoff and then return the page."""
    row = _exchange_rows(synthetic_ohlcv.iloc[:1])[0]
    exchange = FakeExchange(
        [ccxt.NetworkError("temporary one"), ccxt.NetworkError("temporary two"), [row]]
    )
    sleeps: list[float] = []

    result = fetch_ohlcv(
        "BTC/USDT",
        "1h",
        row[0],
        row[0] + 3_600_000,
        current_utc_time=synthetic_ohlcv["timestamp"].iloc[0] + pd.Timedelta(hours=1),
        exchange=exchange,  # type: ignore[arg-type]
        sleep=sleeps.append,
    )

    assert len(result) == 1
    assert sleeps == [2.0, 4.0]
    assert len(exchange.calls) == 3


def test_fetch_raises_after_retry_exhaustion(synthetic_ohlcv: pd.DataFrame) -> None:
    """The fetch fails instead of returning partial data after all retries."""
    start = int(synthetic_ohlcv["timestamp"].iloc[0].timestamp() * 1000)
    exchange = FakeExchange([ccxt.NetworkError("offline")] * 4)
    sleeps: list[float] = []

    with pytest.raises(MarketDataNetworkError, match="retries exhausted"):
        fetch_ohlcv(
            "BTC/USDT",
            "1h",
            start,
            start + 3_600_000,
            current_utc_time=synthetic_ohlcv["timestamp"].iloc[0] + pd.Timedelta(hours=1),
            exchange=exchange,  # type: ignore[arg-type]
            sleep=sleeps.append,
        )

    assert sleeps == [2.0, 4.0, 8.0]
    assert len(exchange.calls) == 4


def test_fetch_does_not_retry_permanent_exchange_error(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    """A permanent exchange rejection is surfaced on its first occurrence."""
    start = int(synthetic_ohlcv["timestamp"].iloc[0].timestamp() * 1000)
    exchange = FakeExchange([ccxt.ExchangeError("bad symbol")])
    sleeps: list[float] = []

    with pytest.raises(MarketDataExchangeError, match="Exchange rejected"):
        fetch_ohlcv(
            "BAD/PAIR",
            "1h",
            start,
            start + 3_600_000,
            current_utc_time=synthetic_ohlcv["timestamp"].iloc[0] + pd.Timedelta(hours=1),
            exchange=exchange,  # type: ignore[arg-type]
            sleep=sleeps.append,
        )

    assert sleeps == []
    assert len(exchange.calls) == 1


def test_incomplete_last_candle_is_removed(synthetic_ohlcv: pd.DataFrame) -> None:
    """The open exchange candle is filtered against a single explicit UTC clock."""
    data = synthetic_ohlcv.iloc[:2]
    rows = _exchange_rows(data)
    exchange = FakeExchange([rows])
    now = data["timestamp"].iloc[-1] + pd.Timedelta(minutes=30)

    result = fetch_ohlcv(
        "BTC/USDT",
        "1h",
        rows[0][0],
        rows[-1][0] + 3_600_000,
        current_utc_time=now,
        exchange=exchange,  # type: ignore[arg-type]
    )

    pd.testing.assert_frame_equal(result, data.iloc[:1].reset_index(drop=True))


def test_duplicate_timestamps_are_removed(synthetic_ohlcv: pd.DataFrame) -> None:
    """Overlapping exchange rows are collapsed before validation."""
    data = synthetic_ohlcv.iloc[:2]
    rows = _exchange_rows(data)
    exchange = FakeExchange([[rows[0], rows[1], rows[1]]])

    result = fetch_ohlcv(
        "BTC/USDT",
        "1h",
        rows[0][0],
        rows[-1][0] + 3_600_000,
        current_utc_time=data["timestamp"].iloc[-1] + pd.Timedelta(hours=1),
        exchange=exchange,  # type: ignore[arg-type]
    )

    pd.testing.assert_frame_equal(result, data.reset_index(drop=True))


def test_empty_requested_range_returns_without_exchange_call() -> None:
    """An already-current local dataset does not issue an invalid zero-width request."""
    exchange = FakeExchange([])

    result = fetch_ohlcv(
        "BTC/USDT",
        "1h",
        1_000,
        1_000,
        current_utc_time=pd.Timestamp("2026-01-01", tz="UTC"),
        exchange=exchange,  # type: ignore[arg-type]
    )

    assert result.empty
    assert exchange.calls == []


def test_short_exchange_row_is_rejected() -> None:
    """Malformed exchange records produce a project validation error."""
    with pytest.raises(MarketDataValidationError, match="fewer than 6"):
        rows_to_ohlcv_dataframe([[1, 2, 3]])
