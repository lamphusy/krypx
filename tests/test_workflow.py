"""Irreversible holdout and production-authorization workflow safety tests."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import crypto_ai.workflow as workflow
from crypto_ai.config import settings
from crypto_ai.data.storage import sha256_file
from crypto_ai.exceptions import ArtifactError
from crypto_ai.modeling.train import feature_schema_hash


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_raw_snapshot(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1_000.0,
        }
    )
    path.write_text(frame.to_csv(index=False, lineterminator="\n"), encoding="utf-8")


def _frozen_evaluation_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completed: bool = False,
) -> dict[str, Any]:
    """Create deterministic fake artifacts that pass every pre-claim hash check."""
    run_id = "synthetic-evaluation-run"
    runs_root = tmp_path / "runs"
    evaluations_root = tmp_path / "evaluations"
    production_root = tmp_path / "production"
    monkeypatch.setattr(settings, "RUNS_DIR", runs_root)
    monkeypatch.setattr(settings, "EVALUATIONS_DIR", evaluations_root)
    monkeypatch.setattr(settings, "PRODUCTION_DIR", production_root)
    run_directory = runs_root / run_id
    evaluation_directory = evaluations_root / run_id
    run_directory.mkdir(parents=True)
    evaluation_directory.mkdir(parents=True)

    feature_columns = ["feature_alpha", "feature_beta"]
    schema_hash = feature_schema_hash(feature_columns)
    config: dict[str, Any] = {
        "feature_configuration": workflow._feature_configuration(),
        "prediction_horizon": settings.PREDICTION_HORIZON,
        "label_lookahead_rows": settings.LABEL_LOOKAHEAD_ROWS,
        "label_definition": "synthetic fixed label contract",
        "feature_columns": feature_columns,
        "feature_schema_hash": schema_hash,
        "model_parameters": settings.XGBOOST_PARAMS,
        "minimum_required_return": 0.002,
        "signal_threshold": settings.SIGNAL_THRESHOLD,
        "walk_forward_configuration": {
            "n_splits": settings.N_WALK_FORWARD_SPLITS,
            "test_ratio": settings.WALK_FORWARD_TEST_RATIO,
            "test_size_rows": 4,
            "gap_rows": settings.LABEL_LOOKAHEAD_ROWS,
        },
        "final_holdout_ratio": settings.FINAL_HOLDOUT_RATIO,
        "initial_capital": settings.INITIAL_CAPITAL,
        "base_cost": {
            "fee_rate": settings.TAKER_FEE_RATE,
            "slippage_bps_per_side": settings.SLIPPAGE_BPS_PER_SIDE,
            "half_spread_bps_per_side": settings.HALF_SPREAD_BPS_PER_SIDE,
        },
        "cost_scenarios": settings.COST_SCENARIOS,
        "random_baseline_simulations": settings.RANDOM_BASELINE_SIMULATIONS,
        "random_seed": settings.RANDOM_SEED,
    }
    config_path = run_directory / "config.json"
    prepared_manifest_path = run_directory / "prepared_dataset_manifest.json"
    model_path = evaluation_directory / "evaluation_model.json"
    metadata_path = evaluation_directory / "model_metadata.json"
    schema_path = evaluation_directory / "feature_columns.json"
    snapshot_path = evaluation_directory / "input_data_snapshot.csv"
    _write_json(config_path, config)
    model_path.write_bytes(b"synthetic immutable model bytes\n")
    _write_raw_snapshot(snapshot_path)
    data_hash = sha256_file(snapshot_path)
    feature_file_sha256 = "a" * 64
    labeled_file_sha256 = "b" * 64
    _write_json(
        prepared_manifest_path,
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "source_snapshot_sha256": data_hash,
            "feature_file_sha256": feature_file_sha256,
            "labeled_file_sha256": labeled_file_sha256,
            "feature_columns": feature_columns,
            "feature_schema_hash": schema_hash,
            "feature_configuration": config["feature_configuration"],
            "prediction_horizon": config["prediction_horizon"],
            "label_lookahead_rows": config["label_lookahead_rows"],
            "minimum_required_return": config["minimum_required_return"],
            "label_definition": config["label_definition"],
            "fee_assumptions": {"taker_fee_rate": config["base_cost"]["fee_rate"]},
            "slippage_assumptions": {
                "slippage_bps_per_side": config["base_cost"]["slippage_bps_per_side"]
            },
            "spread_assumptions": {
                "half_spread_bps_per_side": config["base_cost"]["half_spread_bps_per_side"]
            },
        },
    )
    _write_json(
        metadata_path,
        {
            "model_version": run_id,
            "model_type": "XGBClassifier",
            "feature_schema_hash": schema_hash,
            "feature_columns": feature_columns,
            "model_parameters": settings.XGBOOST_PARAMS,
            "prediction_horizon": settings.PREDICTION_HORIZON,
            "label_threshold": config["minimum_required_return"],
            "signal_threshold": settings.SIGNAL_THRESHOLD,
            "data_hash": data_hash,
            "training_start": "2026-01-01T00:00:00+00:00",
            "training_end": "2026-01-02T23:00:00+00:00",
            "training_exit_end": "2026-01-03T04:00:00+00:00",
            "training_row_count": 48,
            "holdout_start": "2026-01-03T05:00:00+00:00",
        },
    )
    _write_json(schema_path, {"feature_columns": feature_columns})
    frozen_paths = {
        "config.json": config_path,
        "evaluation_model.json": model_path,
        "model_metadata.json": metadata_path,
        "feature_columns.json": schema_path,
        "input_data_snapshot.csv": snapshot_path,
        "prepared_dataset_manifest.json": prepared_manifest_path,
    }
    frozen_hashes = {name: sha256_file(path) for name, path in frozen_paths.items()}
    manifest: dict[str, Any] = {
        **config,
        "run_id": run_id,
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "data_hash": data_hash,
        "feature_file_sha256": feature_file_sha256,
        "labeled_file_sha256": labeled_file_sha256,
        "development_boundary": {
            "row_count": 48,
            "start_timestamp": "2026-01-01T00:00:00+00:00",
            "end_timestamp": "2026-01-02T23:00:00+00:00",
        },
        "holdout_boundary": {
            "row_count": 10,
            "start_timestamp": "2026-01-03T05:00:00+00:00",
            "end_timestamp": "2026-01-03T14:00:00+00:00",
        },
        "prepared_dataset_manifest_sha256": frozen_hashes["prepared_dataset_manifest.json"],
        "frozen_artifact_hashes": frozen_hashes,
    }
    _write_json(run_directory / "manifest.json", manifest)

    paths: dict[str, Any] = {
        "run_id": run_id,
        "run_directory": run_directory,
        "evaluation_directory": evaluation_directory,
        "claim": run_directory / "holdout_evaluation_claim.json",
        "manifest": run_directory / "manifest.json",
        "config": config_path,
        "prepared_manifest": prepared_manifest_path,
        "model": model_path,
        "metadata": metadata_path,
        "schema": schema_path,
        "snapshot": snapshot_path,
    }
    if not completed:
        return paths

    claim_path = paths["claim"]
    _write_json(claim_path, {"status": "completed", "claimed_at_utc": "2026-01-01T00:00:00Z"})
    evaluation_artifacts = {
        "input_data_snapshot.csv": snapshot_path,
        "evaluation_model.json": model_path,
        "model_metadata.json": metadata_path,
        "feature_columns.json": schema_path,
        "holdout_predictions.csv": evaluation_directory / "holdout_predictions.csv",
        "trade_ledger.csv": evaluation_directory / "trade_ledger.csv",
        "equity_curve.csv": evaluation_directory / "equity_curve.csv",
        "metrics.json": evaluation_directory / "metrics.json",
        "baseline_metrics.json": evaluation_directory / "baseline_metrics.json",
        "cost_sensitivity.json": evaluation_directory / "cost_sensitivity.json",
    }
    evaluation_artifacts["holdout_predictions.csv"].write_text(
        "timestamp,probability_score\n2026-01-03T05:00:00Z,0.5\n", encoding="utf-8"
    )
    evaluation_artifacts["trade_ledger.csv"].write_text("trade_id,pnl\n", encoding="utf-8")
    evaluation_artifacts["equity_curve.csv"].write_text(
        "timestamp,equity\n2026-01-03T05:00:00Z,10000\n", encoding="utf-8"
    )
    for name in ("metrics.json", "baseline_metrics.json", "cost_sensitivity.json"):
        _write_json(evaluation_artifacts[name], {"synthetic": True})
    evaluation_hashes = {name: sha256_file(path) for name, path in evaluation_artifacts.items()}
    evaluation_manifest_path = evaluation_directory / "evaluation_manifest.json"
    _write_json(
        evaluation_manifest_path,
        {
            **manifest,
            "holdout_evaluation_claim_status": "claimed_at_manifest_write",
            "evaluation_artifacts_status": "complete",
            "evaluation_artifact_hashes": evaluation_hashes,
            "strategy_metrics": {"synthetic": True},
            "baseline_metrics": {"synthetic": True},
            "cost_sensitivity": {"synthetic": True},
            "evaluated_at_utc": "2026-01-07T00:00:00Z",
        },
    )
    paths.update(evaluation_artifacts)
    paths["evaluation_manifest"] = evaluation_manifest_path
    return paths


def _forbid_production_data_or_fit(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"bundle": 0, "fit": 0}

    def reject_bundle(*args: object, **kwargs: object) -> None:
        calls["bundle"] += 1
        raise AssertionError("Production data must not load before authorization succeeds")

    def reject_fit(*args: object, **kwargs: object) -> None:
        calls["fit"] += 1
        raise AssertionError("Production model fitting must not start before authorization")

    monkeypatch.setattr(workflow, "load_prepared_dataset_bundle", reject_bundle)
    monkeypatch.setattr(workflow, "train_production_model", reject_fit)
    return calls


def _configure_synthetic_production(
    paths: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: str = "synthetic-production-version",
) -> tuple[SimpleNamespace, Path]:
    """Provide compatible synthetic production inputs without real training."""
    prepared_manifest_path = paths["prepared_manifest"]
    manifest_bytes = prepared_manifest_path.read_bytes()
    prepared = json.loads(manifest_bytes)
    feature_columns = tuple(prepared["feature_columns"])
    labeled = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            feature_columns[0]: [0.1, 0.2, 0.3],
            feature_columns[1]: [0.4, 0.5, 0.6],
            "label": [0, 1, 0],
        }
    )
    bundle = SimpleNamespace(
        manifest=prepared,
        manifest_path=prepared_manifest_path,
        manifest_bytes=manifest_bytes,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        source_snapshot_path=paths["snapshot"],
        source_snapshot_sha256=sha256_file(paths["snapshot"]),
        feature_path=paths["run_directory"] / "synthetic_features.csv",
        feature_sha256="a" * 64,
        labeled_path=paths["run_directory"] / "synthetic_labeled.csv",
        labeled_sha256="b" * 64,
        feature_columns=feature_columns,
        labeled=labeled,
    )

    class SyntheticModel:
        def save_model(self, destination: Path) -> None:
            Path(destination).write_bytes(b"synthetic production model\n")

    monkeypatch.setattr(workflow, "load_prepared_dataset_bundle", lambda *args: bundle)
    monkeypatch.setattr(workflow, "train_production_model", lambda *args: SyntheticModel())
    monkeypatch.setattr(workflow, "generate_run_id", lambda *args: version)
    return bundle, settings.PRODUCTION_DIR / "versions" / version


@pytest.mark.parametrize("identifier_kind", ["traversal", "absolute"])
def test_holdout_run_id_cannot_escape_artifact_roots(
    identifier_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(settings, "EVALUATIONS_DIR", tmp_path / "evaluations")
    outside = tmp_path / "outside"
    run_id = "../outside" if identifier_kind == "traversal" else str(outside)

    with pytest.raises(ArtifactError, match="one path component"):
        workflow.evaluate_final_holdout(run_id)

    assert not (outside / "holdout_evaluation_claim.json").exists()


@pytest.mark.parametrize("identifier_kind", ["traversal", "absolute"])
def test_production_evaluation_id_cannot_escape_artifact_roots(
    identifier_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(settings, "EVALUATIONS_DIR", tmp_path / "evaluations")
    outside = tmp_path / "outside"
    run_id = "../outside" if identifier_kind == "traversal" else str(outside)
    calls = _forbid_production_data_or_fit(monkeypatch)

    with pytest.raises(ArtifactError, match="one path component"):
        workflow.train_versioned_production_model(run_id)

    assert calls == {"bundle": 0, "fit": 0}


def test_holdout_missing_frozen_artifact_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    snapshot_before = paths["snapshot"].read_bytes()
    paths["model"].unlink()

    with pytest.raises(ArtifactError, match="Required artifacts are missing"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()
    assert paths["snapshot"].read_bytes() == snapshot_before


def test_holdout_corrupt_frozen_artifact_fails_before_claim_without_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    corrupted_bytes = paths["model"].read_bytes() + b"tampered\n"
    paths["model"].write_bytes(corrupted_bytes)

    with pytest.raises(ArtifactError, match="hash mismatch"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()
    assert paths["model"].read_bytes() == corrupted_bytes


@pytest.mark.parametrize(
    "filename",
    [
        "holdout_predictions.csv",
        "trade_ledger.csv",
        "equity_curve.csv",
        "metrics.json",
        "baseline_metrics.json",
        "cost_sensitivity.json",
        "evaluation_manifest.json",
    ],
)
def test_each_preexisting_holdout_output_fails_before_claim_without_overwrite(
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    output = paths["evaluation_directory"] / filename
    sentinel = b"existing irreversible result bytes\n"
    output.write_bytes(sentinel)

    with pytest.raises(ArtifactError, match="existing evaluation artifacts"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()
    assert output.read_bytes() == sentinel


def test_invalid_frozen_configuration_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    config["initial_capital"] = 0.0
    manifest["initial_capital"] = 0.0
    _write_json(paths["config"], config)
    manifest["frozen_artifact_hashes"]["config.json"] = sha256_file(paths["config"])
    _write_json(paths["manifest"], manifest)

    with pytest.raises(ArtifactError, match="initial capital"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()


def test_negative_frozen_random_seed_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    config["random_seed"] = -1
    manifest["random_seed"] = -1
    _write_json(paths["config"], config)
    manifest["frozen_artifact_hashes"]["config.json"] = sha256_file(paths["config"])
    _write_json(paths["manifest"], manifest)

    with pytest.raises(ArtifactError, match="non-negative integer"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()


def test_unhashable_frozen_feature_schema_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    _write_json(paths["schema"], {"feature_columns": [["nested-invalid-column"]]})
    manifest["frozen_artifact_hashes"]["feature_columns.json"] = sha256_file(paths["schema"])
    _write_json(paths["manifest"], manifest)

    with pytest.raises(ArtifactError, match="data hash or feature schema"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()


@pytest.mark.parametrize(
    ("walk_forward_updates", "message"),
    [
        ({"test_size_rows": 5}, "test size does not match"),
        ({"n_splits": 12}, "leaves no initial training rows"),
    ],
)
def test_invalid_walk_forward_geometry_fails_before_claim(
    walk_forward_updates: dict[str, int],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    config["walk_forward_configuration"].update(walk_forward_updates)
    manifest["walk_forward_configuration"] = config["walk_forward_configuration"]
    _write_json(paths["config"], config)
    manifest["frozen_artifact_hashes"]["config.json"] = sha256_file(paths["config"])
    _write_json(paths["manifest"], manifest)

    with pytest.raises(ArtifactError, match=message):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()


def test_impossible_frozen_time_geometry_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metadata["training_exit_end"] = "2026-01-03T03:00:00+00:00"
    _write_json(paths["metadata"], metadata)
    manifest["frozen_artifact_hashes"]["model_metadata.json"] = sha256_file(paths["metadata"])
    _write_json(paths["manifest"], manifest)

    with pytest.raises(ArtifactError, match="label lookahead"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()


def test_frozen_training_row_count_span_mismatch_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metadata.update(
        {
            "training_end": "2026-01-03T00:00:00+00:00",
            "training_exit_end": "2026-01-03T05:00:00+00:00",
            "holdout_start": "2026-01-03T06:00:00+00:00",
        }
    )
    manifest["development_boundary"]["end_timestamp"] = metadata["training_end"]
    manifest["holdout_boundary"]["start_timestamp"] = metadata["holdout_start"]
    _write_json(paths["metadata"], metadata)
    manifest["frozen_artifact_hashes"]["model_metadata.json"] = sha256_file(paths["metadata"])
    _write_json(paths["manifest"], manifest)

    with pytest.raises(ArtifactError, match="row count does not match its timestamp span"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()


def test_frozen_boundary_overflow_is_normalized_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    metadata.update(
        {
            "training_start": "2262-04-09T23:00:00+00:00",
            "training_end": "2262-04-11T22:00:00+00:00",
            "training_exit_end": "2262-04-11T23:00:00+00:00",
            "holdout_start": "2262-04-11T23:30:00+00:00",
        }
    )
    manifest["development_boundary"].update(
        {
            "start_timestamp": metadata["training_start"],
            "end_timestamp": metadata["training_end"],
        }
    )
    manifest["holdout_boundary"]["start_timestamp"] = metadata["holdout_start"]
    _write_json(paths["metadata"], metadata)
    manifest["frozen_artifact_hashes"]["model_metadata.json"] = sha256_file(paths["metadata"])
    _write_json(paths["manifest"], manifest)

    with pytest.raises(ArtifactError, match="boundary geometry is numerically invalid"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert not paths["claim"].exists()


def test_dangling_output_symlink_fails_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)
    output = paths["evaluation_directory"] / "holdout_predictions.csv"
    output.symlink_to(tmp_path / "missing-target.csv")

    with pytest.raises(ArtifactError, match="existing evaluation artifacts"):
        workflow.evaluate_final_holdout(paths["run_id"])

    assert output.is_symlink()
    assert not paths["claim"].exists()


def test_post_claim_failure_is_recorded_and_retry_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)

    def fail_after_claim(*args: object, **kwargs: object) -> None:
        raise OSError("synthetic post-claim read failure")

    monkeypatch.setattr(workflow.pd, "read_csv", fail_after_claim)
    with pytest.raises(ArtifactError, match="failed after claim creation"):
        workflow.evaluate_final_holdout(paths["run_id"])

    claim = json.loads(paths["claim"].read_text(encoding="utf-8"))
    assert claim["status"] == "failed"
    assert "synthetic post-claim read failure" in claim["error"]
    with pytest.raises(ArtifactError, match="already exists"):
        workflow.evaluate_final_holdout(paths["run_id"])
    assert json.loads(paths["claim"].read_text(encoding="utf-8"))["status"] == "failed"


def test_keyboard_interrupt_preserves_claimed_state_and_blocks_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch)

    def interrupt_after_claim(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(workflow.pd, "read_csv", interrupt_after_claim)
    with pytest.raises(KeyboardInterrupt):
        workflow.evaluate_final_holdout(paths["run_id"])

    claim = json.loads(paths["claim"].read_text(encoding="utf-8"))
    assert claim["status"] == "claimed"
    with pytest.raises(ArtifactError, match="already exists"):
        workflow.evaluate_final_holdout(paths["run_id"])
    assert json.loads(paths["claim"].read_text(encoding="utf-8"))["status"] == "claimed"


def test_production_rejects_missing_evaluation_artifact_before_data_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    paths["metrics.json"].unlink()
    calls = _forbid_production_data_or_fit(monkeypatch)

    with pytest.raises(ArtifactError, match="Required artifacts are missing"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert calls == {"bundle": 0, "fit": 0}


def test_production_rejects_noncompleted_claim_before_data_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _write_json(paths["claim"], {"status": "failed", "error": "synthetic failure"})
    calls = _forbid_production_data_or_fit(monkeypatch)

    with pytest.raises(ArtifactError, match="is not completed"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert calls == {"bundle": 0, "fit": 0}


def test_production_rejects_corrupt_evaluation_artifact_before_data_or_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _write_json(paths["metrics.json"], {"synthetic": True, "tampered": True})
    calls = _forbid_production_data_or_fit(monkeypatch)

    with pytest.raises(ArtifactError, match="hash mismatch"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert calls == {"bundle": 0, "fit": 0}


@pytest.mark.parametrize("mismatch", ["feature_order", "base_cost"])
def test_production_rejects_incompatible_prepared_bundle_before_fit(
    mismatch: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    prepared = json.loads(paths["prepared_manifest"].read_text(encoding="utf-8"))
    feature_columns = tuple(prepared["feature_columns"])
    if mismatch == "feature_order":
        feature_columns = tuple(reversed(feature_columns))
    else:
        prepared["fee_assumptions"]["taker_fee_rate"] += 0.0001
    bundle = SimpleNamespace(manifest=prepared, feature_columns=feature_columns)
    calls = {"fit": 0}

    def reject_fit(*args: object, **kwargs: object) -> None:
        calls["fit"] += 1
        raise AssertionError("Production fitting must not start with incompatible inputs")

    monkeypatch.setattr(workflow, "load_prepared_dataset_bundle", lambda *args: bundle)
    monkeypatch.setattr(workflow, "train_production_model", reject_fit)

    with pytest.raises(ArtifactError, match="incompatible with the accepted evaluation"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert calls == {"fit": 0}


@pytest.mark.parametrize(
    "failure_step",
    [
        "model_save",
        "feature_schema_write",
        "prepared_manifest_copy",
        "prepared_manifest_reverification",
        "production_manifest_write",
        "final_publication",
    ],
)
def test_production_publication_failure_never_exposes_partial_version(
    failure_step: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    versions_root = final_directory.parent
    existing_version = versions_root / "existing-version"
    existing_version.mkdir(parents=True)
    existing_sentinel = existing_version / "manifest.json"
    existing_sentinel.write_bytes(b"existing immutable version\n")
    active_model = settings.PRODUCTION_DIR / "active_model.json"
    active_model.parent.mkdir(parents=True, exist_ok=True)
    active_model.write_bytes(b"existing active model pointer\n")

    original_atomic_write_json = workflow.atomic_write_json
    original_verified_hash = workflow._verified_hash

    if failure_step == "model_save":
        monkeypatch.setattr(
            workflow,
            "save_xgboost_model",
            lambda *args: (_ for _ in ()).throw(OSError("synthetic model-save failure")),
        )
    elif failure_step == "feature_schema_write":

        def fail_feature_schema(
            path: Path,
            payload: dict[str, Any],
            *,
            exclusive: bool = False,
        ) -> None:
            if path.name == "feature_columns.json" and path.parent.name.startswith(".staging-"):
                raise OSError("synthetic feature-schema failure")
            original_atomic_write_json(path, payload, exclusive=exclusive)

        monkeypatch.setattr(workflow, "atomic_write_json", fail_feature_schema)
    elif failure_step == "prepared_manifest_copy":
        monkeypatch.setattr(
            workflow,
            "_write_verified_bytes",
            lambda *args: (_ for _ in ()).throw(OSError("synthetic prepared-copy failure")),
        )
    elif failure_step == "prepared_manifest_reverification":

        def fail_reverification(path: Path, expected: str, description: str) -> None:
            if description == "prepared dataset manifest before production manifest commit":
                raise ArtifactError("synthetic prepared-manifest reverification failure")
            original_verified_hash(path, expected, description)

        monkeypatch.setattr(workflow, "_verified_hash", fail_reverification)
    elif failure_step == "production_manifest_write":

        def fail_production_manifest(
            path: Path,
            payload: dict[str, Any],
            *,
            exclusive: bool = False,
        ) -> None:
            if path.name == "manifest.json" and path.parent.name.startswith(".staging-"):
                raise OSError("synthetic production-manifest failure")
            original_atomic_write_json(path, payload, exclusive=exclusive)

        monkeypatch.setattr(workflow, "atomic_write_json", fail_production_manifest)
    else:
        monkeypatch.setattr(
            workflow,
            "_publish_production_version",
            lambda *args: (_ for _ in ()).throw(OSError("synthetic publication failure")),
        )

    with pytest.raises(ArtifactError, match="synthetic"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert not final_directory.exists()
    assert existing_sentinel.read_bytes() == b"existing immutable version\n"
    assert active_model.read_bytes() == b"existing active model pointer\n"
    assert {path.name for path in versions_root.iterdir()} == {"existing-version"}


def test_successful_production_publication_is_complete_and_hash_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    bundle, expected_directory = _configure_synthetic_production(paths, monkeypatch)

    published = workflow.train_versioned_production_model(paths["run_id"])

    assert published == expected_directory
    assert {path.name for path in published.iterdir()} == {
        "model.json",
        "feature_columns.json",
        "prepared_dataset_manifest.json",
        "manifest.json",
    }
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["production_artifact_hashes"]) == {
        "model.json",
        "feature_columns.json",
        "prepared_dataset_manifest.json",
    }
    for filename, expected_hash in manifest["production_artifact_hashes"].items():
        assert sha256_file(published / filename) == expected_hash
    assert (published / "prepared_dataset_manifest.json").read_bytes() == bundle.manifest_bytes
    assert manifest["prepared_dataset_manifest_copy"] == str(
        published / "prepared_dataset_manifest.json"
    )
    assert not (settings.PRODUCTION_DIR / "active_model.json").exists()
    assert not any(path.name.startswith(".staging-") for path in published.parent.iterdir())


def test_existing_production_version_id_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    final_directory.mkdir(parents=True)
    sentinel = final_directory / "manifest.json"
    sentinel.write_bytes(b"existing production evidence\n")

    with pytest.raises(ArtifactError, match="already exists"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert sentinel.read_bytes() == b"existing production evidence\n"
    assert not any(path.name.startswith(".staging-") for path in final_directory.parent.iterdir())


def test_existing_empty_production_version_directory_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    final_directory.mkdir(parents=True)
    original_inode = final_directory.stat().st_ino
    active_model = settings.PRODUCTION_DIR / "active_model.json"
    active_model.write_bytes(b"existing active model pointer\n")

    with pytest.raises(ArtifactError, match="already exists"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert final_directory.is_dir()
    assert final_directory.stat().st_ino == original_inode
    assert list(final_directory.iterdir()) == []
    assert active_model.read_bytes() == b"existing active model pointer\n"
    assert not any(
        path.name.startswith((".staging-", ".publish-"))
        for path in final_directory.parent.iterdir()
    )


def test_destination_appearing_at_atomic_publish_is_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    versions_root = final_directory.parent
    existing_version = versions_root / "existing-version"
    existing_version.mkdir(parents=True)
    existing_sentinel = existing_version / "manifest.json"
    existing_sentinel.write_bytes(b"existing immutable version\n")
    active_model = settings.PRODUCTION_DIR / "active_model.json"
    active_model.write_bytes(b"existing active model pointer\n")
    original_atomic_rename = workflow._atomic_rename_directory_no_replace
    injected_inode: list[int] = []

    def create_empty_destination_then_publish(source: Path, destination: Path) -> None:
        destination.mkdir()
        injected_inode.append(destination.stat().st_ino)
        original_atomic_rename(source, destination)

    monkeypatch.setattr(
        workflow,
        "_atomic_rename_directory_no_replace",
        create_empty_destination_then_publish,
    )

    with pytest.raises(ArtifactError, match="already exists"):
        workflow.train_versioned_production_model(paths["run_id"])

    assert injected_inode
    assert final_directory.is_dir()
    assert final_directory.stat().st_ino == injected_inode[0]
    assert list(final_directory.iterdir()) == []
    assert existing_sentinel.read_bytes() == b"existing immutable version\n"
    assert active_model.read_bytes() == b"existing active model pointer\n"
    assert not any(path.name.startswith(".staging-") for path in versions_root.iterdir())
    assert not any(path.name.startswith(".publish-") for path in versions_root.iterdir())


def test_competing_production_publisher_is_rejected_by_version_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    versions_root = final_directory.parent
    existing_version = versions_root / "existing-version"
    existing_version.mkdir(parents=True)
    existing_sentinel = existing_version / "manifest.json"
    existing_sentinel.write_bytes(b"existing immutable version\n")
    active_model = settings.PRODUCTION_DIR / "active_model.json"
    active_model.write_bytes(b"existing active model pointer\n")
    first_publisher_has_lock = threading.Event()
    release_first_publisher = threading.Event()
    original_save_xgboost_model = workflow.save_xgboost_model

    def block_first_publisher(model: object, path: Path) -> None:
        first_publisher_has_lock.set()
        if not release_first_publisher.wait(timeout=5):
            raise AssertionError("Timed out waiting to release the first publisher")
        original_save_xgboost_model(model, path)

    monkeypatch.setattr(workflow, "save_xgboost_model", block_first_publisher)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_result = executor.submit(
            workflow.train_versioned_production_model,
            paths["run_id"],
        )
        assert first_publisher_has_lock.wait(timeout=5)
        with pytest.raises(ArtifactError, match="publication is already locked"):
            workflow.train_versioned_production_model(paths["run_id"])
        assert (versions_root / f".publish-{final_directory.name}.lock").exists()
        release_first_publisher.set()
        assert first_result.result(timeout=5) == final_directory

    assert existing_sentinel.read_bytes() == b"existing immutable version\n"
    assert active_model.read_bytes() == b"existing active model pointer\n"
    assert {path.name for path in versions_root.iterdir()} == {
        "existing-version",
        final_directory.name,
    }


def test_successful_publication_fsyncs_parent_after_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    original_fsync_directory = workflow._fsync_directory
    observations: list[tuple[str, bool, bool]] = []

    def observe_fsync(path: Path, description: str) -> None:
        lock_path = final_directory.parent / f".publish-{final_directory.name}.lock"
        observations.append((description, final_directory.exists(), lock_path.exists()))
        original_fsync_directory(path, description)

    monkeypatch.setattr(workflow, "_fsync_directory", observe_fsync)

    published = workflow.train_versioned_production_model(paths["run_id"])

    assert published == final_directory
    assert observations == [
        ("production staging directory", False, True),
        ("production versions parent directory", True, True),
    ]
    assert not (final_directory.parent / f".publish-{final_directory.name}.lock").exists()


def test_parent_fsync_failure_after_rename_preserves_published_and_existing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    versions_root = final_directory.parent
    existing_version = versions_root / "existing-version"
    existing_version.mkdir(parents=True)
    existing_sentinel = existing_version / "manifest.json"
    existing_sentinel.write_bytes(b"existing immutable version\n")
    active_model = settings.PRODUCTION_DIR / "active_model.json"
    active_model.write_bytes(b"existing active model pointer\n")
    original_fsync_directory = workflow._fsync_directory

    def fail_parent_fsync(path: Path, description: str) -> None:
        if description == "production versions parent directory":
            assert path == versions_root
            assert final_directory.is_dir()
            assert (final_directory / "manifest.json").is_file()
            raise ArtifactError("synthetic parent fsync failure")
        original_fsync_directory(path, description)

    monkeypatch.setattr(workflow, "_fsync_directory", fail_parent_fsync)

    with pytest.raises(
        ArtifactError,
        match="published completely, but parent-directory durability confirmation failed",
    ):
        workflow.train_versioned_production_model(paths["run_id"])

    assert {path.name for path in final_directory.iterdir()} == {
        "model.json",
        "feature_columns.json",
        "prepared_dataset_manifest.json",
        "manifest.json",
    }
    manifest = json.loads((final_directory / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected_hash in manifest["production_artifact_hashes"].items():
        assert sha256_file(final_directory / filename) == expected_hash
    assert existing_sentinel.read_bytes() == b"existing immutable version\n"
    assert active_model.read_bytes() == b"existing active model pointer\n"
    assert not any(path.name.startswith(".staging-") for path in versions_root.iterdir())
    assert not any(path.name.startswith(".publish-") for path in versions_root.iterdir())


def test_failed_staging_cleanup_is_logged_and_remains_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    paths = _frozen_evaluation_run(tmp_path, monkeypatch, completed=True)
    _, final_directory = _configure_synthetic_production(paths, monkeypatch)
    versions_root = final_directory.parent
    original_atomic_write_json = workflow.atomic_write_json

    def fail_feature_schema(
        path: Path,
        payload: dict[str, Any],
        *,
        exclusive: bool = False,
    ) -> None:
        if path.name == "feature_columns.json":
            raise OSError("synthetic staging failure")
        original_atomic_write_json(path, payload, exclusive=exclusive)

    monkeypatch.setattr(workflow, "atomic_write_json", fail_feature_schema)
    monkeypatch.setattr(
        workflow.shutil,
        "rmtree",
        lambda *args: (_ for _ in ()).throw(OSError("synthetic cleanup failure")),
    )

    with caplog.at_level("ERROR"), pytest.raises(ArtifactError, match="synthetic staging failure"):
        workflow.train_versioned_production_model(paths["run_id"])

    hidden_staging = [path for path in versions_root.iterdir() if path.name.startswith(".staging-")]
    assert len(hidden_staging) == 1
    assert not (hidden_staging[0] / "manifest.json").exists()
    assert not final_directory.exists()
    assert "Unable to clean failed production staging directory" in caplog.text
    assert "synthetic cleanup failure" in caplog.text
    assert not any(path.name.startswith(".publish-") for path in versions_root.iterdir())
