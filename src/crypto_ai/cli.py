"""Command-line entry point for KrypX."""

import argparse
import logging
from collections.abc import Sequence

from crypto_ai import __version__
from crypto_ai.config import settings
from crypto_ai.data.storage import load_or_update_ohlcv
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.logging_config import configure_logging

logger = logging.getLogger(__name__)


def _run_fetch(args: argparse.Namespace) -> int:
    """Fetch, validate, and persist the requested market data."""
    configure_logging()
    try:
        result = load_or_update_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
        )
    except CryptoAIError as exc:
        logger.error("Market-data update failed: %s", exc)
        return 1

    first_timestamp = result.data["timestamp"].iloc[0]
    last_timestamp = result.data["timestamp"].iloc[-1]
    print(
        f"Stored {len(result.data)} closed candles from {first_timestamp} through {last_timestamp}"
    )
    print(f"Latest data: {result.latest_path}")
    print(f"Immutable snapshot: {result.snapshot_path}")
    print(f"SHA-256: {result.sha256}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 command-line parser."""
    parser = argparse.ArgumentParser(
        prog="krypx",
        description="KrypX Phase 1 cryptocurrency research pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="fetch, validate, and store closed OHLCV candles",
    )
    fetch_parser.add_argument("--symbol", default=settings.SYMBOL)
    fetch_parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    fetch_parser.add_argument("--lookback-days", type=int, default=settings.LOOKBACK_DAYS)
    fetch_parser.set_defaults(handler=_run_fetch)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
