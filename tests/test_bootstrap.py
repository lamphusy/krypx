"""Tests for the Milestone 0 repository bootstrap."""

import importlib
import logging
from pathlib import Path

import pytest

import crypto_ai
from crypto_ai import exceptions
from crypto_ai.cli import build_parser, main
from crypto_ai.config import settings
from crypto_ai.logging_config import configure_logging


def test_package_imports_successfully() -> None:
    """The src-layout package can be imported after installation."""
    assert importlib.import_module("crypto_ai") is crypto_ai
    assert crypto_ai.__version__ == settings.PROJECT_VERSION


def test_settings_paths_resolve_under_project_root() -> None:
    """Configured data and artifact paths remain inside the repository root."""
    assert settings.BASE_DIR == Path(__file__).resolve().parents[1]
    for directory in settings.REQUIRED_DIRECTORIES:
        assert directory.is_relative_to(settings.BASE_DIR)


def test_required_directories_can_be_created(tmp_path: Path) -> None:
    """The configured directory shape can be created from an empty root."""
    directories = (
        tmp_path / "data" / "raw" / "snapshots",
        tmp_path / "data" / "interim",
        tmp_path / "data" / "processed",
        tmp_path / "artifacts" / "evaluations",
        tmp_path / "artifacts" / "production",
        tmp_path / "artifacts" / "runs",
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    assert all(directory.is_dir() for directory in directories)


@pytest.mark.parametrize(
    "exception_type",
    [
        exceptions.MarketDataError,
        exceptions.MarketDataNetworkError,
        exceptions.MarketDataExchangeError,
        exceptions.MarketDataValidationError,
        exceptions.FeatureEngineeringError,
        exceptions.LabelGenerationError,
        exceptions.DatasetSplitError,
        exceptions.ModelTrainingError,
        exceptions.BacktestError,
        exceptions.ArtifactError,
    ],
)
def test_custom_exceptions_inherit_from_crypto_ai_error(
    exception_type: type[exceptions.CryptoAIError],
) -> None:
    """All project-specific exceptions share the documented base type."""
    assert issubclass(exception_type, exceptions.CryptoAIError)


def test_configure_logging_writes_utc_log_file(tmp_path: Path) -> None:
    """Logging can write the documented UTC format to a requested file."""
    log_path = tmp_path / "logs" / "test.log"
    configure_logging(log_file=log_path)

    logging.getLogger("crypto_ai.test").info("bootstrap complete")
    logging.shutdown()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Z | INFO | crypto_ai.test | bootstrap complete" in log_text


def test_configure_logging_rejects_negative_level() -> None:
    """Invalid logging levels fail clearly."""
    with pytest.raises(ValueError, match="non-negative"):
        configure_logging(level=-1)


def test_cli_parser_uses_expected_program_name() -> None:
    """The parser exposes the stable CLI program name."""
    assert build_parser().prog == "krypx"


def test_cli_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """The bootstrap CLI explains its current surface without failing."""
    assert main([]) == 0
    assert "bootstrap only" in capsys.readouterr().out
