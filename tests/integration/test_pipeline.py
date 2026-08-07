"""Synthetic end-to-end Phase 1 integration test."""

from pathlib import Path

import pandas as pd
import pytest

from crypto_ai.config import settings
from crypto_ai.data.storage import load_or_update_ohlcv
from crypto_ai.exceptions import ArtifactError
from crypto_ai.features.dataset import prepare_datasets
from crypto_ai.workflow import (
    evaluate_final_holdout,
    run_development_validation,
    train_versioned_production_model,
)


def test_phase1_pipeline_runs_end_to_end_on_synthetic_data(
    tmp_path: Path,
    synthetic_ohlcv: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(settings, "DATA_RAW_DIR", data_root / "raw")
    monkeypatch.setattr(settings, "DATA_RAW_SNAPSHOTS_DIR", data_root / "raw" / "snapshots")
    monkeypatch.setattr(settings, "DATA_INTERIM_DIR", data_root / "interim")
    monkeypatch.setattr(settings, "DATA_PROCESSED_DIR", data_root / "processed")
    monkeypatch.setattr(settings, "RUNS_DIR", artifact_root / "runs")
    monkeypatch.setattr(settings, "EVALUATIONS_DIR", artifact_root / "evaluations")
    monkeypatch.setattr(settings, "PRODUCTION_DIR", artifact_root / "production")
    monkeypatch.setattr(settings, "RANDOM_BASELINE_SIMULATIONS", 5)
    monkeypatch.setattr(
        settings,
        "XGBOOST_PARAMS",
        {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": 1,
            "eval_metric": "logloss",
        },
    )
    now = synthetic_ohlcv["timestamp"].iloc[-1] + pd.Timedelta(hours=1)

    def fetcher(**kwargs: object) -> pd.DataFrame:
        return synthetic_ohlcv.copy()

    load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        30,
        current_utc_time=now,
        fetcher=fetcher,
    )
    prepared = prepare_datasets("BTC/USDT", "1h", current_utc_time=now)
    assert prepared.labeled_path.exists()

    development = run_development_validation()
    for filename in (
        "config.json",
        "manifest.json",
        "feature_columns.json",
        "fold_metrics.json",
        "oof_predictions.csv",
        "classification_report.json",
        "feature_importance.csv",
        "development_strategy_metrics.json",
        "development_report.md",
        "logs.txt",
    ):
        assert (development.run_directory / filename).exists()
    for filename in (
        "input_data_snapshot.csv",
        "evaluation_model.json",
        "model_metadata.json",
        "feature_columns.json",
    ):
        assert (development.evaluation_directory / filename).exists()

    holdout = evaluate_final_holdout(development.run_id)
    assert "total_return" in holdout["metrics"]
    for filename in (
        "holdout_predictions.csv",
        "trade_ledger.csv",
        "equity_curve.csv",
        "metrics.json",
        "baseline_metrics.json",
        "cost_sensitivity.json",
        "evaluation_manifest.json",
    ):
        assert (development.evaluation_directory / filename).exists()
    with pytest.raises(ArtifactError, match="already exists"):
        evaluate_final_holdout(development.run_id)

    production = train_versioned_production_model()
    assert (production / "model.json").exists()
    assert (production / "feature_columns.json").exists()
    assert (production / "manifest.json").exists()
