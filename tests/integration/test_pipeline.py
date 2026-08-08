"""Synthetic end-to-end Phase 1 integration test."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import crypto_ai.workflow as workflow
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
    assert prepared.manifest_path is not None and prepared.manifest_path.exists()
    prepared_manifest_a = prepared.manifest_path.read_bytes()
    prepared_manifest_b_payload = json.loads(prepared_manifest_a.decode("utf-8"))
    prepared_manifest_b_payload["created_at_utc"] = "2099-01-01T00:00:00Z"
    prepared_manifest_b = (
        json.dumps(prepared_manifest_b_payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    # Advance only the mutable latest raw file after preparation. Validation and
    # production must remain bound to the already completed snapshot-A bundle.
    last = synthetic_ohlcv.iloc[-1]
    next_open = float(last["close"])
    next_close = next_open + 0.2
    snapshot_b_tail = pd.DataFrame(
        {
            "timestamp": [last["timestamp"] + pd.Timedelta(hours=1)],
            "open": [next_open],
            "high": [next_close + 0.5],
            "low": [next_open - 0.5],
            "close": [next_close],
            "volume": [float(last["volume"]) + 10.0],
        }
    )
    updated = load_or_update_ohlcv(
        "BTC/USDT",
        "1h",
        30,
        current_utc_time=snapshot_b_tail["timestamp"].iloc[0] + pd.Timedelta(hours=1),
        fetcher=lambda **kwargs: snapshot_b_tail.copy(),
    )
    assert updated.sha256 != prepared.source_sha256

    original_evaluation_fit = workflow.train_evaluation_model

    def replace_manifest_during_evaluation_fit(*args: object, **kwargs: object) -> object:
        prepared.manifest_path.write_bytes(prepared_manifest_b)
        return original_evaluation_fit(*args, **kwargs)

    monkeypatch.setattr(workflow, "train_evaluation_model", replace_manifest_during_evaluation_fit)
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
        "prepared_dataset_manifest.json",
    ):
        assert (development.run_directory / filename).exists()
    for filename in (
        "input_data_snapshot.csv",
        "evaluation_model.json",
        "model_metadata.json",
        "feature_columns.json",
    ):
        assert (development.evaluation_directory / filename).exists()
    run_manifest = json.loads(
        (development.run_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["data_hash"] == prepared.source_sha256
    assert run_manifest["data_hash"] != updated.sha256
    assert run_manifest["feature_file_sha256"] == prepared.feature_sha256
    frozen_prepared_manifest = development.run_directory / "prepared_dataset_manifest.json"
    assert prepared.manifest_path.read_bytes() == prepared_manifest_b
    assert frozen_prepared_manifest.read_bytes() == prepared_manifest_a
    assert (
        run_manifest["prepared_dataset_manifest_sha256"]
        == hashlib.sha256(prepared_manifest_a).hexdigest()
    )
    prepared.manifest_path.write_bytes(prepared_manifest_a)
    assert any(warning.startswith("Cash:") for warning in run_manifest["warnings"])
    report = (development.run_directory / "development_report.md").read_text(encoding="utf-8")
    for heading in (
        "Per-fold classification results",
        "Development OOF trading comparison",
        "Random-exposure baseline",
        "Global XGBoost feature importance",
        "Fold stability",
        "Frozen execution and cost assumptions",
        "Final holdout status",
    ):
        assert heading in report
    for strategy in ("XGBoost OOF", "Cash", "Buy & Hold", "EMA", "Momentum"):
        assert f"| {strategy} |" in report
    xgboost_fold_section = report.split("### XGBoost", maxsplit=1)[1].split(
        "### Logistic regression", maxsplit=1
    )[0]
    logistic_fold_section = report.split("### Logistic regression", maxsplit=1)[1].split(
        "## Aggregate model comparison", maxsplit=1
    )[0]
    assert "| 1 |" in xgboost_fold_section and "Confusion matrix" in xgboost_fold_section
    assert "| 1 |" in logistic_fold_section and "Confusion matrix" in logistic_fold_section
    assert "| XGBoost |" in report
    assert "| Logistic regression |" in report
    assert "- Simulations: 5" in report
    assert "Total-return median / p05 / p95" in report
    assert "Sharpe median / p05 / p95" in report
    assert "Maximum-drawdown median / p05 / p95" in report
    assert "Fraction with return at least the model" in report
    importance = pd.read_csv(development.run_directory / "feature_importance.csv").sort_values(
        ["gain", "weight"], ascending=False
    )
    assert not importance.empty
    assert f"| {importance.iloc[0]['feature']} |" in report
    for label, value in (
        ("Entry/exit fee rate", run_manifest["base_cost"]["fee_rate"]),
        ("Slippage", run_manifest["base_cost"]["slippage_bps_per_side"]),
        ("Half-spread", run_manifest["base_cost"]["half_spread_bps_per_side"]),
    ):
        assert f"- {label}: {value}" in report
    assert "## Warnings" in report
    assert "## Limitations" in report
    assert "The final holdout was not evaluated" in report
    assert "Holdout total return" not in report
    assert "Holdout ROC-AUC" not in report
    assert "Holdout PR-AUC" not in report

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
    with pytest.raises(ArtifactError, match="existing evaluation artifacts"):
        evaluate_final_holdout(development.run_id)

    original_production_fit = workflow.train_production_model

    def replace_manifest_during_production_fit(*args: object, **kwargs: object) -> object:
        model = original_production_fit(*args, **kwargs)
        prepared.manifest_path.write_bytes(prepared_manifest_b)
        return model

    monkeypatch.setattr(workflow, "train_production_model", replace_manifest_during_production_fit)
    with pytest.raises(ArtifactError, match="hash mismatch"):
        train_versioned_production_model(development.run_id)
    versions_root = settings.PRODUCTION_DIR / "versions"
    assert not versions_root.exists() or not list(versions_root.iterdir())

    prepared.manifest_path.write_bytes(prepared_manifest_a)
    monkeypatch.setattr(workflow, "train_production_model", original_production_fit)
    production = train_versioned_production_model(development.run_id)
    assert (production / "model.json").exists()
    assert (production / "feature_columns.json").exists()
    assert (production / "prepared_dataset_manifest.json").read_bytes() == prepared_manifest_a
    assert (production / "manifest.json").exists()
    production_manifest = json.loads((production / "manifest.json").read_text(encoding="utf-8"))
    assert production_manifest["authorized_evaluation_run_id"] == development.run_id
    assert production_manifest["data_hash"] == prepared.source_sha256
    assert not (settings.PRODUCTION_DIR / "active_model.json").exists()
