"""Command-line entry point for KrypX."""

import argparse
from collections.abc import Sequence

from crypto_ai import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 command-line parser."""
    parser = argparse.ArgumentParser(
        prog="krypx",
        description="KrypX Phase 1 research pipeline (bootstrap only)",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
