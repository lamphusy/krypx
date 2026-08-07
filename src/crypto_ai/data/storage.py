"""Incremental, atomic, and content-addressed OHLCV storage."""

import hashlib
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from crypto_ai.config import settings
from crypto_ai.data.fetch import fetch_ohlcv
from crypto_ai.data.validation import (
    filter_closed_candles,
    normalize_current_utc_time,
    timeframe_to_milliseconds,
    validate_ohlcv,
)
from crypto_ai.exceptions import MarketDataValidationError

logger = logging.getLogger(__name__)

FetchFunction = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class MarketDataResult:
    """Validated market data and its persisted snapshot identity."""

    data: pd.DataFrame
    latest_path: Path
    snapshot_path: Path
    sha256: str


def symbol_to_slug(symbol: str) -> str:
    """Convert an exchange symbol to a safe lowercase filename component."""
    slug = symbol.lower().replace("/", "_").replace(":", "_").replace("-", "_")
    slug = re.sub(r"[^a-z0-9_]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise MarketDataValidationError(f"Symbol {symbol!r} cannot produce a safe filename")
    return slug


def get_raw_data_path(
    symbol: str,
    timeframe: str,
    raw_dir: Path | None = None,
) -> Path:
    """Return the mutable latest-data convenience path."""
    timeframe_to_milliseconds(timeframe)
    destination = raw_dir if raw_dir is not None else settings.DATA_RAW_DIR
    return destination / f"{symbol_to_slug(symbol)}_{timeframe}.csv"


def sha256_file(path: Path) -> str:
    """Calculate the lowercase SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ohlcv_csv(
    path: Path,
    timeframe: str,
    current_utc_time: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load and validate a canonical OHLCV CSV file."""
    try:
        result = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise MarketDataValidationError(f"Unable to load OHLCV file {path}: {exc}") from exc

    missing_columns = [column for column in settings.RAW_COLUMNS if column not in result.columns]
    if missing_columns:
        raise MarketDataValidationError(
            f"OHLCV file {path} is missing required columns: {missing_columns}"
        )

    try:
        result = result.loc[:, settings.RAW_COLUMNS].copy()
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
        for column in settings.RAW_COLUMNS[1:]:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketDataValidationError(
            f"OHLCV file {path} contains invalid values: {exc}"
        ) from exc

    validate_ohlcv(result, timeframe=timeframe, current_utc_time=current_utc_time)
    return result


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        index=False,
        columns=settings.RAW_COLUMNS,
        date_format="%Y-%m-%dT%H:%M:%S.%fZ",
        float_format="%.17g",
        lineterminator="\n",
    )


def _persist_snapshot(source_path: Path, snapshot_path: Path, expected_sha256: str) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        existing_sha256 = sha256_file(snapshot_path)
        if existing_sha256 != expected_sha256:
            raise MarketDataValidationError(
                f"Existing snapshot hash mismatch at {snapshot_path}: {existing_sha256}"
            )
        return

    created = False
    try:
        with snapshot_path.open("xb") as destination, source_path.open("rb") as source:
            created = True
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
    except FileExistsError:
        existing_sha256 = sha256_file(snapshot_path)
        if existing_sha256 != expected_sha256:
            raise MarketDataValidationError(
                f"Concurrent snapshot hash mismatch at {snapshot_path}: {existing_sha256}"
            ) from None
    except Exception:
        if created:
            snapshot_path.unlink(missing_ok=True)
        raise


def _persist_market_data(
    df: pd.DataFrame,
    latest_path: Path,
    snapshot_root: Path,
    symbol: str,
    timeframe: str,
) -> tuple[Path, str]:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=latest_path.parent,
        prefix=f"{latest_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        _write_csv(df, temporary_path)
        digest = sha256_file(temporary_path)
        snapshot_path = snapshot_root / f"{symbol_to_slug(symbol)}_{timeframe}" / f"{digest}.csv"
        _persist_snapshot(temporary_path, snapshot_path, digest)
        os.replace(temporary_path, latest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return snapshot_path, digest


def load_or_update_ohlcv(
    symbol: str,
    timeframe: str,
    lookback_days: int,
    *,
    current_utc_time: datetime | pd.Timestamp | None = None,
    raw_dir: Path | None = None,
    snapshots_dir: Path | None = None,
    fetcher: FetchFunction | None = None,
) -> MarketDataResult:
    """Update OHLCV and return data with its immutable snapshot identity."""
    if lookback_days <= 0:
        raise MarketDataValidationError("lookback_days must be positive")

    now = normalize_current_utc_time(current_utc_time)
    timeframe_ms = timeframe_to_milliseconds(timeframe)
    destination = raw_dir if raw_dir is not None else settings.DATA_RAW_DIR
    snapshot_root = snapshots_dir if snapshots_dir is not None else destination / "snapshots"
    latest_path = get_raw_data_path(symbol, timeframe, destination)
    market_fetcher = fetcher if fetcher is not None else fetch_ohlcv

    existing: pd.DataFrame | None = None
    if latest_path.exists():
        existing = load_ohlcv_csv(
            latest_path,
            timeframe=timeframe,
            current_utc_time=now,
        )
        latest_timestamp = existing["timestamp"].iloc[-1]
        since_ms = int(latest_timestamp.timestamp() * 1000) + timeframe_ms
        logger.info(
            "Loaded %s existing %s %s candles through %s",
            len(existing),
            symbol,
            timeframe,
            latest_timestamp,
        )
    else:
        now_ms = int(now.timestamp() * 1000)
        raw_since_ms = now_ms - lookback_days * 24 * 60 * 60 * 1000
        since_ms = raw_since_ms - raw_since_ms % timeframe_ms

    fetched = market_fetcher(
        symbol=symbol,
        timeframe=timeframe,
        since_ms=since_ms,
        until_ms=int(now.timestamp() * 1000),
        current_utc_time=now,
    )

    frames = [frame for frame in (existing, fetched) if frame is not None and not frame.empty]
    if not frames:
        raise MarketDataValidationError(f"No closed OHLCV data available for {symbol} {timeframe}")

    combined = pd.concat(frames, ignore_index=True)
    before_deduplication = len(combined)
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    duplicates_removed = before_deduplication - len(combined)
    combined = combined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    combined = filter_closed_candles(
        combined,
        timeframe=timeframe,
        current_utc_time=now,
    ).reset_index(drop=True)
    validate_ohlcv(combined, timeframe=timeframe, current_utc_time=now)

    snapshot_path, digest = _persist_market_data(
        combined,
        latest_path=latest_path,
        snapshot_root=snapshot_root,
        symbol=symbol,
        timeframe=timeframe,
    )

    logger.info(
        "Stored %s %s %s candles at %s; removed %s duplicates; snapshot %s",
        len(combined),
        symbol,
        timeframe,
        latest_path,
        duplicates_removed,
        digest,
    )
    return MarketDataResult(
        data=combined,
        latest_path=latest_path,
        snapshot_path=snapshot_path,
        sha256=digest,
    )
