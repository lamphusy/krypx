"""Paginated public OHLCV fetching through CCXT."""

import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import ccxt
import pandas as pd

from crypto_ai.config import settings
from crypto_ai.data.validation import (
    empty_ohlcv_dataframe,
    filter_closed_candles,
    normalize_current_utc_time,
    timeframe_to_milliseconds,
    validate_ohlcv,
)
from crypto_ai.exceptions import (
    MarketDataExchangeError,
    MarketDataNetworkError,
    MarketDataValidationError,
)

logger = logging.getLogger(__name__)

SleepFunction = Callable[[float], None]


def create_exchange(exchange_id: str = settings.EXCHANGE_ID) -> ccxt.Exchange:
    """Create a rate-limited CCXT exchange for public market-data requests."""
    try:
        exchange_class = getattr(ccxt, exchange_id)
    except AttributeError as exc:
        raise MarketDataExchangeError(f"Unsupported CCXT exchange: {exchange_id}") from exc

    exchange = exchange_class({"enableRateLimit": True})
    if not exchange.has.get("fetchOHLCV", False):
        raise MarketDataExchangeError(f"Exchange {exchange_id} does not support OHLCV fetching")
    return exchange


def rows_to_ohlcv_dataframe(rows: Sequence[Sequence[Any]]) -> pd.DataFrame:
    """Convert raw CCXT OHLCV rows to the canonical typed DataFrame."""
    if not rows:
        return empty_ohlcv_dataframe()
    if any(len(row) < len(settings.RAW_COLUMNS) for row in rows):
        raise MarketDataValidationError("Exchange returned an OHLCV row with fewer than 6 values")

    try:
        data = [list(row[: len(settings.RAW_COLUMNS)]) for row in rows]
        result = pd.DataFrame(data, columns=settings.RAW_COLUMNS)
        result["timestamp"] = pd.to_datetime(result["timestamp"], unit="ms", utc=True)
        for column in settings.RAW_COLUMNS[1:]:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketDataValidationError(f"Exchange returned invalid OHLCV values: {exc}") from exc

    return result


def _fetch_page_with_retries(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    limit: int,
    max_retries: int,
    retry_base_seconds: float,
    sleep: SleepFunction,
) -> list[list[Any]]:
    network_failures = 0
    while True:
        try:
            return exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=limit,
            )
        except ccxt.NetworkError as exc:
            if network_failures >= max_retries:
                raise MarketDataNetworkError(
                    f"Network retries exhausted fetching {symbol} {timeframe} from {since_ms}"
                ) from exc

            delay = retry_base_seconds * (2**network_failures)
            network_failures += 1
            logger.warning(
                "Transient network error fetching %s %s from %s; retry %s/%s in %.1fs",
                symbol,
                timeframe,
                since_ms,
                network_failures,
                max_retries,
                delay,
            )
            sleep(delay)
        except ccxt.BaseError as exc:
            raise MarketDataExchangeError(
                f"Exchange rejected OHLCV request for {symbol} {timeframe} from {since_ms}: {exc}"
            ) from exc


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    limit: int = settings.CANDLES_PER_REQUEST,
    *,
    current_utc_time: datetime | pd.Timestamp | None = None,
    exchange: ccxt.Exchange | None = None,
    max_retries: int = settings.MAX_FETCH_RETRIES,
    retry_base_seconds: float = settings.RETRY_BASE_SECONDS,
    sleep: SleepFunction = time.sleep,
) -> pd.DataFrame:
    """Fetch closed OHLCV candles from the configured exchange.

    Args:
        symbol: CCXT trading pair, such as ``BTC/USDT``.
        timeframe: Fixed candle timeframe, such as ``1h``.
        since_ms: Inclusive start timestamp in milliseconds.
        until_ms: Optional exclusive end timestamp in milliseconds.
        limit: Maximum records requested per API call.
        current_utc_time: Explicit clock used to exclude incomplete candles.
        exchange: Optional injected CCXT exchange used by deterministic tests.
        max_retries: Number of retries after transient network failures.
        retry_base_seconds: Initial exponential-backoff delay.
        sleep: Injectable sleep function used for retry backoff.

    Returns:
        Chronologically sorted, unique, closed OHLCV candles with UTC timestamps.

    Raises:
        MarketDataNetworkError: If network retries are exhausted.
        MarketDataExchangeError: If the exchange rejects the request.
        MarketDataValidationError: If inputs or returned data are invalid.
    """
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    now = normalize_current_utc_time(current_utc_time)
    if since_ms < 0:
        raise MarketDataValidationError("since_ms must be non-negative")
    if until_ms is not None and until_ms < since_ms:
        raise MarketDataValidationError("until_ms must be greater than or equal to since_ms")
    if limit <= 0:
        raise MarketDataValidationError("limit must be positive")
    if max_retries < 0:
        raise MarketDataValidationError("max_retries must be non-negative")
    if retry_base_seconds < 0:
        raise MarketDataValidationError("retry_base_seconds must be non-negative")
    if until_ms == since_ms:
        return empty_ohlcv_dataframe()

    market_exchange = exchange or create_exchange()
    cursor_ms = since_ms
    previous_cursor_ms: int | None = None
    fetched_rows: list[Sequence[Any]] = []

    logger.info(
        "Fetching %s %s candles from %s to %s",
        symbol,
        timeframe,
        since_ms,
        until_ms if until_ms is not None else "latest",
    )

    while until_ms is None or cursor_ms < until_ms:
        page = _fetch_page_with_retries(
            exchange=market_exchange,
            symbol=symbol,
            timeframe=timeframe,
            since_ms=cursor_ms,
            limit=limit,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
            sleep=sleep,
        )
        if not page:
            break

        page_frame = rows_to_ohlcv_dataframe(page)
        page_timestamp_ms = page_frame["timestamp"].astype("int64") // 1_000_000
        eligible = page_timestamp_ms >= since_ms
        if until_ms is not None:
            eligible &= page_timestamp_ms < until_ms
        fetched_rows.extend(page_frame.loc[eligible, settings.RAW_COLUMNS].itertuples(index=False))

        last_returned_ms = int(page_timestamp_ms.max())
        next_cursor_ms = last_returned_ms + timeframe_ms
        if next_cursor_ms <= cursor_ms or next_cursor_ms == previous_cursor_ms:
            logger.warning(
                "Stopping %s %s pagination because the exchange returned no newer timestamps",
                symbol,
                timeframe,
            )
            break

        previous_cursor_ms = cursor_ms
        cursor_ms = next_cursor_ms

        if until_ms is not None and cursor_ms >= until_ms:
            break

    if not fetched_rows:
        return empty_ohlcv_dataframe()

    result = pd.DataFrame(fetched_rows, columns=settings.RAW_COLUMNS)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    for column in settings.RAW_COLUMNS[1:]:
        result[column] = result[column].astype("float64")
    result = result.drop_duplicates(subset="timestamp", keep="last")
    result = result.sort_values("timestamp", kind="stable").reset_index(drop=True)
    result = filter_closed_candles(result, timeframe=timeframe, current_utc_time=now)
    if result.empty:
        return empty_ohlcv_dataframe()

    result = result.reset_index(drop=True)
    validate_ohlcv(result, timeframe=timeframe, current_utc_time=now)
    logger.info(
        "Fetched %s closed %s %s candles from %s through %s",
        len(result),
        symbol,
        timeframe,
        result["timestamp"].iloc[0],
        result["timestamp"].iloc[-1],
    )
    return result
