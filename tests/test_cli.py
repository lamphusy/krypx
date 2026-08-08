"""Tests for the Phase 1 command-line interface."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from crypto_ai.cli import main
from crypto_ai.data.storage import MarketDataResult
from crypto_ai.exceptions import ArtifactError, MarketDataNetworkError
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
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected market-data failures produce a clean nonzero process status."""

    def fail(**kwargs: object) -> MarketDataResult:
        raise MarketDataNetworkError("network unavailable")

    monkeypatch.setattr("crypto_ai.cli.load_or_update_ohlcv", fail)

    assert main(["fetch"]) == 1
    assert "Traceback" not in capsys.readouterr().err


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


def test_validate_command_reports_artifacts_without_evaluating_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_directory = tmp_path / "runs" / "run-1"
    evaluation_directory = tmp_path / "evaluations" / "run-1"
    monkeypatch.setattr(
        "crypto_ai.cli.run_development_validation",
        lambda symbol, timeframe: SimpleNamespace(
            run_id="run-1",
            run_directory=run_directory,
            evaluation_directory=evaluation_directory,
            xgboost_metrics={
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "pr_auc": 0.5,
                "log_loss": 0.7,
                "brier_score": 0.25,
            },
            logistic_metrics={
                "accuracy": 0.4,
                "balanced_accuracy": 0.4,
                "pr_auc": 0.4,
                "log_loss": 0.8,
                "brier_score": 0.3,
            },
            development_summary={
                "development_start": "2025-01-01",
                "development_end": "2025-06-30",
                "development_rows": 100,
                "development_positive_label_rate": 0.45,
                "boundary_purge_start": "2025-07-01",
                "boundary_purge_end": "2025-07-05",
                "boundary_purge_rows": 5,
                "holdout_start": "2025-07-06",
                "holdout_end": "2025-08-01",
                "holdout_rows": 20,
            },
        ),
    )
    assert main(["validate"]) == 0
    output = capsys.readouterr().out
    assert "Development run: run-1" in output
    assert "Final holdout has not been evaluated" in output


def test_evaluate_holdout_prints_irreversible_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "crypto_ai.cli.evaluate_final_holdout",
        lambda run_id: {
            "metrics": {
                "total_return": 0.1,
                "sharpe_ratio": 1.0,
                "maximum_drawdown": -0.05,
                "market_exposure": 0.3,
                "num_trades": 4,
            },
            "baselines": {},
        },
    )
    assert main(["evaluate-holdout", "--run-id", "run-1"]) == 0
    output = capsys.readouterr().out
    assert "You are evaluating the final holdout" in output
    assert "Do not use these results for iterative model tuning" in output


def test_train_production_reports_non_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    version = tmp_path / "production" / "versions" / "v1"
    arguments: dict[str, object] = {}

    def train(
        evaluation_run_id: str,
        *,
        symbol: str,
        timeframe: str,
    ) -> Path:
        arguments.update(
            {
                "evaluation_run_id": evaluation_run_id,
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )
        return version

    monkeypatch.setattr(
        "crypto_ai.cli.train_versioned_production_model",
        train,
    )
    assert (
        main(
            [
                "train-production",
                "--evaluation-run-id",
                "evaluation-1",
                "--symbol",
                "ETH/USDT",
                "--timeframe",
                "4h",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert arguments == {
        "evaluation_run_id": "evaluation-1",
        "symbol": "ETH/USDT",
        "timeframe": "4h",
    }
    assert str(version) in output
    assert "Authorized by completed evaluation: evaluation-1" in output
    assert "not automatically activated" in output


def test_train_production_requires_evaluation_run_id() -> None:
    """Production training cannot be requested without an accepted evaluation."""
    with pytest.raises(SystemExit) as exc_info:
        main(["train-production"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("arguments", "target", "message"),
    [
        (["prepare"], "crypto_ai.cli.prepare_datasets", "prepared manifest is corrupt"),
        (
            ["validate"],
            "crypto_ai.cli.run_development_validation",
            "development manifest contains invalid JSON",
        ),
        (
            ["evaluate-holdout", "--run-id", "run-1"],
            "crypto_ai.cli.evaluate_final_holdout",
            "evaluation model could not be loaded",
        ),
        (
            ["train-production", "--evaluation-run-id", "run-1"],
            "crypto_ai.cli.train_versioned_production_model",
            "evaluation artifacts are incomplete",
        ),
    ],
)
def test_expected_operational_errors_return_nonzero_without_traceback(
    arguments: list[str],
    target: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected artifact failures are logged cleanly by every artifact command."""

    def fail(*args: object, **kwargs: object) -> object:
        raise ArtifactError(message)

    monkeypatch.setattr(target, fail)

    assert main(arguments) == 1
    error_output = capsys.readouterr().err
    assert message in error_output
    assert "Traceback" not in error_output
