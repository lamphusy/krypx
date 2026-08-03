"""Logging configuration shared by CLI and pipeline modules."""

import logging
import time
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


class UTCFormatter(logging.Formatter):
    """Format log timestamps in UTC."""

    converter = time.gmtime


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure deterministic UTC console logging and an optional file handler."""
    if level < 0:
        raise ValueError("Logging level must be non-negative")

    formatter = UTCFormatter(DEFAULT_LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%SZ")
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handlers, force=True)
