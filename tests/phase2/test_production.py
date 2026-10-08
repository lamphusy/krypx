"""Synthetic-only production registry tests; no real promotion or activation.

The passing parent is a genuine Milestone 9 evaluation of deterministic generated
prices. Its gate calculation, source verification, model replay, claim and durable
completion are never mocked. A reduced random-baseline fixture is explicitly
retained as synthetic evidence, not a real production approval.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from crypto_ai.costs import minimum_gross_return_for_net_edge
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.features.build import compute_features
from crypto_ai.phase2 import backtests, evaluation, holdout, production, production_store
from crypto_ai.phase2.dataset import COMBINED_COLUMNS, SENTIMENT_COLUMNS, TECHNICAL_COLUMNS
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp

from .test_evaluation import (
    CUTOFF,
    FIRST_HOLDOUT,
    HOUR,
    LAST_MARKET,
    MARKET_START,
    _reissue_readiness,
    _snapshot,
    _synthetic_request,
)


@dataclass(frozen=True)
class ProductionCase:
    source: evaluation.SyntheticEvaluationRequest
    evaluation_artifact: Any
    registry_path: Path


def _passing_request(
    root: Path, *, seed: int, drift: float
) -> evaluation.SyntheticEvaluationRequest:
    """Build an independently replayable D/B case using only made-up data.

    Constant no-news features remain truthful. Frozen XGBoost feature subsampling
    differs between D's 37 columns and B's 24 columns. A bounded test-only
    selector below finds a generated path with a genuine final-gate PASS on the
    test host. This is fixture design, not evidence of real news trading value.
    """
    request = _synthetic_request(root)
    rng = np.random.default_rng(seed)
    previous = 100.0
    records: list[dict[str, Any]] = []
    lines = ["market_ordinal,timestamp,open,high,low,close,volume"]
    for ordinal in range(LAST_MARKET + 1):
        body_return = float(
            drift + 0.0025 * np.sin(ordinal * 2.0 * np.pi / 24.0) + rng.normal(0.0, 0.0012)
        )
        close = previous * (1.0 + body_return)
        at = MARKET_START + ordinal * HOUR
        row = {
            "timestamp": pd.Timestamp(at),
            "open": previous,
            "high": max(previous, close) * 1.0015,
            "low": min(previous, close) * 0.9985,
            "close": close,
            "volume": float(1000 + rng.integers(0, 50)),
        }
        records.append(row)
        lines.append(
            f"{ordinal},{format_utc_timestamp(at)},"
            + ",".join(
                repr(float(row[name])) for name in ("open", "high", "low", "close", "volume")
            )
        )
        previous = close
    market_raw = ("\n".join(lines) + "\n").encode()
    development_raw = ("\n".join(lines[: CUTOFF + holdout.HORIZON + 3]) + "\n").encode()
    market_ref = _snapshot(request.market_snapshot.path.parent, "market.csv", market_raw)
    development_ref = _snapshot(
        request.development_run_dir, "development_market.csv", development_raw
    )
    rows_raw, augmented, control = evaluation.build_synthetic_development_fit(
        development_raw, CUTOFF, ("D", "B")
    )
    rows_ref = _snapshot(request.development_run_dir, "development_rows.json", rows_raw)
    original_fit = request.boundary.frozen_fit
    fit_manifest = json.loads(original_fit.fit_manifest_bytes)
    fit_manifest.update(
        selected_augmented_cell="D",
        matched_control_cell="B",
        shared_labeled_rows_sha256=rows_ref.sha256,
        augmented_model_sha256=sha256_bytes(augmented),
        control_model_sha256=sha256_bytes(control),
    )
    fit_bytes = canonicalize(fit_manifest)
    fit = replace(
        original_fit,
        selected_augmented_cell="D",
        matched_control_cell="B",
        shared_labeled_rows_sha256=rows_ref.sha256,
        augmented_model_sha256=sha256_bytes(augmented),
        control_model_sha256=sha256_bytes(control),
        fit_manifest_sha256=sha256_bytes(fit_bytes),
        fit_manifest_bytes=fit_bytes,
        augmented_model_bytes=augmented,
        control_model_bytes=control,
    )
    development_manifest = json.loads(
        (request.development_run_dir / "development_manifest.json").read_bytes()
    )
    development_manifest.update(
        selected_augmented_cell="D",
        matched_control_cell="B",
        model_family="XGBClassifier",
        development_market_sha256=development_ref.sha256,
        development_rows_sha256=rows_ref.sha256,
        shared_labeled_rows_sha256=rows_ref.sha256,
        fit_manifest_sha256=sha256_bytes(fit_bytes),
        augmented_model_sha256=sha256_bytes(augmented),
        control_model_sha256=sha256_bytes(control),
    )
    for name, raw in {
        "development_manifest.json": canonicalize(development_manifest),
        "augmented_model.json": augmented,
        "control_model.json": control,
        "fit_manifest.json": fit_bytes,
        "development_dataset_manifest.json": evaluation._development_dataset_manifest_bytes(
            development_raw, rows_raw, CUTOFF
        ),
    }.items():
        (request.development_run_dir / name).write_bytes(raw)

    technical = compute_features(pd.DataFrame(records))
    no_news = {name: 0.0 for name in SENTIMENT_COLUMNS}
    no_news.update(hours_since_latest_article=24.0, news_missing_24h=1.0)
    feature_lines = ["market_ordinal,decision_at," + ",".join(COMBINED_COLUMNS)]
    for ordinal in range(FIRST_HOLDOUT, LAST_MARKET - holdout.HORIZON):
        values = [float(technical.loc[ordinal, name]) for name in TECHNICAL_COLUMNS]
        values += [no_news[name] for name in SENTIMENT_COLUMNS]
        feature_lines.append(
            f"{ordinal},{format_utc_timestamp(MARKET_START + (ordinal + 1) * HOUR)},"
            + ",".join(repr(value) for value in values)
        )
    request = replace(
        request,
        market_snapshot=market_ref,
        development_market=development_ref,
        development_rows=rows_ref,
        boundary=holdout.BoundaryPurgeManager.validate(
            development_cutoff_ordinal=request.boundary.development_cutoff_ordinal,
            development_labels=request.boundary.development_labels,
            purge_ordinals=request.boundary.purge_ordinals,
            first_holdout_ordinal=request.boundary.first_holdout_ordinal,
            first_holdout_decision_at=request.boundary.first_holdout_decision_at,
            frozen_fit=fit,
        ),
    )
    return _reissue_readiness(request, feature_raw=("\n".join(feature_lines) + "\n").encode())


_PASS_CANDIDATES = (
    (69, 0.0002),
    (0, 0.00085),
    (69, 0.0005),
    (0, 0.0005),
) + tuple((seed, (0.0002, 0.0005, 0.00085)[(seed - 1) % 3]) for seed in range(1, 37))


def _screen_final_gates(request: evaluation.SyntheticEvaluationRequest) -> dict[str, Any]:
    """Run the frozen base-cost models and gate arithmetic before the full M9 claim."""
    market = evaluation._market_frame(request.market_snapshot.path.read_bytes(), LAST_MARKET)
    features = evaluation._feature_frame(
        request.feature_snapshot.path.read_bytes(), market, request.boundary
    )
    fit = request.boundary.frozen_fit
    first_open = int(features.market_ordinal.iloc[0]) + 1
    final_open = int(features.market_ordinal.iloc[-1]) + holdout.HORIZON + 1

    def base_payload(scores: pd.Series) -> dict[str, Any]:
        result = backtests._simulate(
            market,
            scores,
            None,
            backtests.SCENARIOS["base"],
            expected_start=first_open,
            expected_end=final_open,
        )
        return backtests._result_payload(result)

    augmented = base_payload(
        evaluation._predict(
            features,
            COMBINED_COLUMNS,
            evaluation._verified_model(fit.augmented_model_bytes, COMBINED_COLUMNS),
        )
    )
    control = base_payload(
        evaluation._predict(
            features,
            TECHNICAL_COLUMNS,
            evaluation._verified_model(fit.control_model_bytes, TECHNICAL_COLUMNS),
        )
    )
    cash = base_payload(pd.Series(0.0, index=features.market_ordinal.to_numpy(dtype=np.int64)))
    return evaluation.evaluate_final_gates(
        augmented_metrics=augmented["metrics"],
        control_metrics=control["metrics"],
        cash_metrics=cash["metrics"],
        augmented_ledger=augmented["trade_ledger"],
        control_ledger=control["trade_ledger"],
        elapsed_days=request.readiness.elapsed_days,
        planned_minimum_days=request.readiness.planned_minimum_days,
    )


@pytest.fixture(scope="module")
def passing_evaluation(tmp_path_factory: pytest.TempPathFactory) -> ProductionCase:
    root = tmp_path_factory.mktemp("production-genuine-synthetic-pass").resolve()
    registry = root / holdout.GENERATION_REGISTRY_NAME

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("production fixtures must never access the network")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", forbidden)
        patch.setattr(socket.socket, "connect_ex", forbidden)
        patch.setattr(socket, "create_connection", forbidden)
        patch.setattr(holdout, "_generation_registry_path", lambda: registry)
        diagnostics = []
        for index, (seed, drift) in enumerate(_PASS_CANDIDATES):
            candidate_root = root / f"candidate-{index:02d}"
            candidate_root.mkdir()
            try:
                candidate = _passing_request(candidate_root, seed=seed, drift=drift)
                gates = _screen_final_gates(candidate)
            except CryptoAIError as exc:
                diagnostics.append(f"{seed}/{drift:g}: {type(exc).__name__}: {exc}")
                continue
            operands = gates["operands"]
            diagnostics.append(
                f"{seed}/{drift:g}: {gates['research_verdict']} "
                f"D-B={operands['augmented_total_return'] - operands['control_total_return']:.3f} "
                f"concentration={operands['rolling_incremental_concentration']}"
            )
            if gates["research_verdict"] == "PASS":
                request = candidate
                break
        else:
            pytest.fail(
                "no synthetic production PASS after "
                f"{len(_PASS_CANDIDATES)} predeclared candidates: " + "; ".join(diagnostics),
                pytrace=False,
            )
        artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
        metrics = json.loads(artifact.files["metrics.json"])
        assert metrics["research_verdict"] == "PASS", metrics["final_gates"]
        assert all(metrics["final_gates"]["gates"].values())
        assert metrics["production_decision"] == "NO-GO"
        assert metrics["engineering_status"] == "REDUCED_SYNTHETIC_FIXTURE"
        return ProductionCase(request, artifact, registry)


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("production tests must remain completely offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def production_request(
    passing_evaluation: ProductionCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> production.SyntheticProductionRequest:
    source = passing_evaluation.source
    monkeypatch.setattr(
        holdout, "_generation_registry_path", lambda: passing_evaluation.registry_path
    )
    now = datetime.now(UTC) + timedelta(seconds=1)
    authorization = production.SyntheticProductionAuthorization(
        evaluation_run_id=source.run_id,
        evaluation_manifest_sha256=sha256_bytes(
            canonicalize(passing_evaluation.evaluation_artifact.manifest)
        ),
        model_version="synthetic-production-v1",
        authorized_at_utc=now,
        authorization_id="synthetic-human-approval",
    )
    return production.SyntheticProductionRequest(
        evaluation_root=source.evaluation_root,
        development_run_dir=source.development_run_dir,
        versions_root=tmp_path / "production" / "versions",
        evaluation_run_id=source.run_id,
        model_version=authorization.model_version,
        authorization=authorization,
        training_as_of_utc=now,
        created_at_utc=now,
    )


def _registry(request: production.SyntheticProductionRequest) -> production.ProductionRegistry:
    return production.ProductionRegistry(
        request.versions_root, request.evaluation_root, request.development_run_dir
    )


@contextmanager
def _changed_bytes(path: Path, raw: bytes | None):
    original = path.read_bytes()
    try:
        if raw is None:
            path.unlink()
        else:
            path.write_bytes(raw)
        yield
    finally:
        path.write_bytes(original)


def test_end_to_end_genuine_verified_pass_trains_separate_production_version(
    production_request: production.SyntheticProductionRequest,
    passing_evaluation: ProductionCase,
) -> None:
    request = production_request
    originals = {
        path: path.read_bytes() for path in request.development_run_dir.iterdir() if path.is_file()
    }
    request.versions_root.parent.mkdir(parents=True)
    active = request.versions_root.parent / "active_model.json"
    active.write_bytes(b'{"unrelated_existing_active_model":"do-not-change"}')
    active_before = active.read_bytes()
    artifact = production.OfflineProductionEngine().train(request)
    metadata = artifact.manifest["metadata"]
    assert metadata["model_version"] == request.model_version
    assert metadata["model_type"] == "XGBClassifier"
    assert metadata["evaluation_run_id_provenance"] == request.evaluation_run_id
    assert metadata["feature_columns"] == list(COMBINED_COLUMNS)
    assert len(metadata["feature_columns"]) == 37
    assert metadata["training_row_count"] > 4_000
    assert (
        artifact.files["model.json"]
        != passing_evaluation.evaluation_artifact.files["evaluation_models/augmented.json"]
    )
    assert json.loads(artifact.files["feature_columns.json"]) == list(COMBINED_COLUMNS)
    assert metadata["feature_schema_hash"] == sha256_bytes(canonicalize(list(COMBINED_COLUMNS)))
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    assert active.read_bytes() == active_before
    assert not (request.versions_root / "active_model.json").exists()
    assert metadata["production_decision"] == "NO-GO"
    assert metadata["activation_authorized"] is False
    assert metadata["source_baseline_mode"] == "REDUCED_SYNTHETIC_FIXTURE"
    rows = json.loads(artifact.files["training_rows.json"])
    prepared = json.loads(artifact.files["prepared_dataset_manifest.json"])
    model = json.loads(artifact.files["model.json"])
    assert prepared["labeled_dataset_sha256"] == sha256_bytes(artifact.files["training_rows.json"])
    assert prepared["training_row_count"] == len(rows["rows"]) == metadata["training_row_count"]
    assert len(rows["rows"]) == rows["development_row_count"] + rows["holdout_row_count"]
    assert rows["holdout_row_count"] == LAST_MARKET - holdout.HORIZON - FIRST_HOLDOUT
    ordinals = [row["market_ordinal"] for row in rows["rows"]]
    assert ordinals == sorted(set(ordinals))
    assert not set(range(CUTOFF + 1, FIRST_HOLDOUT)).intersection(ordinals)
    assert metadata["training_start"] == rows["rows"][0]["decision_at"]
    assert metadata["training_end"] == rows["rows"][-1]["decision_at"]
    market = evaluation._market_frame(
        passing_evaluation.evaluation_artifact.files["input_market_snapshot.csv"], None
    )
    threshold = minimum_gross_return_for_net_edge(0.001, 2.0, 1.0, 5.0)
    for row in rows["rows"]:
        ordinal = row["market_ordinal"]
        assert row["exit_ordinal"] == ordinal + 5
        assert row["label"] == int(
            market.iloc[ordinal + 5].open / market.iloc[ordinal + 1].open - 1.0 > threshold
        )
        assert len(row["features"]) == 37
    assert model["training_rows_sha256"] == sha256_bytes(artifact.files["training_rows.json"])
    assert model["purpose"] == "production_refit_never_historical_holdout_evidence"
    reloaded = _registry(request).get(request.model_version)
    assert reloaded.manifest == artifact.manifest
    assert reloaded.files == artifact.files


@pytest.mark.parametrize(
    "change",
    [
        {"approved": False},
        {"approved": 1},
        {"synthetic": False},
        {"scope": "real_production"},
        {"authorization_id": ""},
        {"evaluation_run_id": "another-run"},
        {"evaluation_manifest_sha256": "0" * 64},
        {"model_version": "another-version"},
        {"authorized_at_utc": datetime(2025, 1, 1)},
    ],
)
def test_unapproved_or_substituted_authority_fails_before_fitting(
    production_request: production.SyntheticProductionRequest,
    change: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("authorization failure reached production fitting")

    monkeypatch.setattr(production, "_model", forbidden_fit)
    changed = replace(
        production_request, authorization=replace(production_request.authorization, **change)
    )
    with pytest.raises(production.ProductionError):
        production.OfflineProductionEngine().train(changed)
    assert not (changed.versions_root / changed.model_version).exists()


@pytest.mark.parametrize("version", ["", ".", "..", "../escaped", "a/b", "a\\b", "/absolute"])
def test_unsafe_version_names_fail_closed(
    production_request: production.SyntheticProductionRequest, version: str
) -> None:
    request = replace(
        production_request,
        model_version=version,
        authorization=replace(production_request.authorization, model_version=version),
    )
    with pytest.raises(production.ProductionError):
        production.OfflineProductionEngine().train(request)


@pytest.mark.parametrize("kind", ["empty", "occupied", "symlink"])
def test_existing_version_is_never_replaced(
    production_request: production.SyntheticProductionRequest, kind: str, tmp_path: Path
) -> None:
    request = production_request
    request.versions_root.mkdir(parents=True)
    destination = request.versions_root / request.model_version
    if kind == "symlink":
        other = tmp_path / "unrelated"
        other.mkdir()
        (other / "sentinel").write_bytes(b"never replace")
        destination.symlink_to(other, target_is_directory=True)
    else:
        destination.mkdir()
        if kind == "occupied":
            (destination / "sentinel").write_bytes(b"never replace")
    with pytest.raises(production.ProductionError):
        production.OfflineProductionEngine().train(request)
    assert destination.exists()
    if kind != "empty":
        assert (destination / "sentinel").read_bytes() == b"never replace"


@pytest.mark.parametrize(
    "filename",
    [
        "evaluation_manifest.json",
        "metrics.json",
        "input_market_snapshot.csv",
        "evaluation_models/augmented.json",
        "holdout_predictions.csv",
    ],
)
def test_corrupted_evaluation_evidence_rejects_production(
    production_request: production.SyntheticProductionRequest, filename: str
) -> None:
    request = production_request
    path = request.evaluation_root / request.evaluation_run_id / filename
    with _changed_bytes(path, path.read_bytes() + b" "):
        with pytest.raises(production.ProductionError):
            production.OfflineProductionEngine().train(request)
    assert not (request.versions_root / request.model_version).exists()


@pytest.mark.parametrize(
    "filename",
    ["holdout_evaluation_completed.json", "development_manifest.json", "augmented_model.json"],
)
def test_missing_completed_claim_or_development_evidence_rejects_production(
    production_request: production.SyntheticProductionRequest, filename: str
) -> None:
    request = production_request
    with _changed_bytes(request.development_run_dir / filename, None):
        with pytest.raises(production.ProductionError):
            production.OfflineProductionEngine().train(request)
    assert not (request.versions_root / request.model_version).exists()


def test_project_error_contract() -> None:
    assert issubclass(production.ProductionError, CryptoAIError)


def test_missing_frozen_dependency_metadata_uses_project_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(production, "version", missing_version)
    with pytest.raises(production.ProductionIntegrityError) as caught:
        with production._errors():
            production._context()
    assert isinstance(caught.value.__cause__, PackageNotFoundError)


def test_genuine_failed_research_never_authorizes_even_synthetic_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "failed-evaluation"
    root.mkdir()
    monkeypatch.setattr(
        holdout, "_generation_registry_path", lambda: root / holdout.GENERATION_REGISTRY_NAME
    )
    source = _synthetic_request(root)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(source)
    assert json.loads(artifact.files["metrics.json"])["research_verdict"] == "FAIL"
    now = datetime.now(UTC) + timedelta(seconds=1)
    authority = production.SyntheticProductionAuthorization(
        evaluation_run_id=source.run_id,
        evaluation_manifest_sha256=sha256_bytes(canonicalize(artifact.manifest)),
        model_version="never-publish-failed-evidence",
        authorized_at_utc=now,
        authorization_id="human-cannot-override-failed-gates",
    )
    request = production.SyntheticProductionRequest(
        evaluation_root=source.evaluation_root,
        development_run_dir=source.development_run_dir,
        versions_root=tmp_path / "versions",
        evaluation_run_id=source.run_id,
        model_version=authority.model_version,
        authorization=authority,
        training_as_of_utc=now,
        created_at_utc=now,
    )

    def forbidden_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("failed research reached production model fitting")

    monkeypatch.setattr(production, "_model", forbidden_fit)
    with pytest.raises(production.ProductionError, match="PASS"):
        production.OfflineProductionEngine().train(request)
    assert not (request.versions_root / request.model_version).exists()


@pytest.mark.parametrize("field", ["training_as_of_utc", "created_at_utc"])
def test_naive_production_timestamps_rejected(
    production_request: production.SyntheticProductionRequest, field: str
) -> None:
    with pytest.raises(production.ProductionError):
        production.OfflineProductionEngine().train(
            replace(production_request, **{field: datetime(2026, 10, 5)})
        )


def test_missing_evaluation_run_rejected(
    production_request: production.SyntheticProductionRequest,
) -> None:
    request = replace(
        production_request,
        evaluation_run_id="missing-run",
        authorization=replace(production_request.authorization, evaluation_run_id="missing-run"),
    )
    with pytest.raises(production.ProductionError):
        production.OfflineProductionEngine().train(request)
    assert not (request.versions_root / request.model_version).exists()


def test_unlabeled_training_as_of_fails_closed(
    production_request: production.SyntheticProductionRequest,
) -> None:
    with pytest.raises(production.ProductionError):
        production.OfflineProductionEngine().train(
            replace(
                production_request,
                training_as_of_utc=MARKET_START + (FIRST_HOLDOUT + 2) * HOUR,
            )
        )


def _authorization_payload() -> dict:
    return {
        "schema_version": production.AUTHORIZATION_SCHEMA,
        "synthetic": True,
        "scope": "offline_synthetic_only",
        "approved": True,
        "authorization_id": "synthetic-human-approval",
        "evaluation_run_id": "synthetic-evaluation",
        "evaluation_manifest_sha256": "a" * 64,
        "model_version": "synthetic-v1",
        "authorized_at_utc": "2026-10-06T00:00:00Z",
    }


def test_authorization_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    raw = canonicalize(_authorization_payload())
    path.write_bytes(raw)
    result = production.load_synthetic_authorization(path)
    assert canonicalize(production._authorization(result)) == raw


@pytest.mark.parametrize(
    "change",
    [
        {"approved": 1},
        {"approved": False},
        {"synthetic": False},
        {"scope": "real_production"},
        {"extra": "not-authorized"},
        {"evaluation_manifest_sha256": "A" * 64},
        {"schema_version": "unknown"},
        {"model_version": "../escape"},
        {"authorized_at_utc": "2026-10-06T00:00:00"},
    ],
)
def test_authorization_file_rejects_invalid_contract(tmp_path: Path, change: dict) -> None:
    path = tmp_path / "authorization.json"
    path.write_bytes(canonicalize({**_authorization_payload(), **change}))
    with pytest.raises(production.ProductionError):
        production.load_synthetic_authorization(path)


@pytest.mark.parametrize("raw", [b"{} trailing", b'{"x":1,"x":2}', b"NaN", b"[]", b"null"])
def test_authorization_file_rejects_malformed_json(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "authorization.json"
    path.write_bytes(raw)
    with pytest.raises(production.ProductionError):
        production.load_synthetic_authorization(path)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "hardlink"])
def test_authorization_rejects_nonunique_or_nonregular_files(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "authorization.json"
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        original = tmp_path / "source.json"
        original.write_bytes(canonicalize(_authorization_payload()))
        if kind == "symlink":
            path.symlink_to(original)
        else:
            os.link(original, path)
    with pytest.raises(production.ProductionError):
        production.load_synthetic_authorization(path)


def test_oversized_authorization_rejected_before_body_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "authorization.json"
    path.write_bytes(b"x" * 16_385)
    monkeypatch.setattr(production.os, "read", lambda *a: pytest.fail("oversize body was read"))
    with pytest.raises(production.ProductionError):
        production.load_synthetic_authorization(path)


@pytest.mark.parametrize("cell", ["C", "D"])
def test_both_frozen_augmented_families_preserve_model_schema(
    passing_evaluation: ProductionCase, cell: str
) -> None:
    rows = json.loads(
        passing_evaluation.evaluation_artifact.files["evaluation_models/development_rows.json"]
    )
    raw = production._model(rows, cell)
    model = json.loads(raw)
    assert model["schema_version"] == production.MODEL_SCHEMA
    assert model["feature_columns"] == list(COMBINED_COLUMNS)
    assert model["training_rows_sha256"] == sha256_bytes(canonicalize(rows))
    assert model["purpose"] == "production_refit_never_historical_holdout_evidence"
    if cell == "C":
        assert model["model_family"] == "LogisticRegression"
        assert len(model["coefficients"]) == len(model["scaler_mean"]) == 37
        assert model["hyperparameters"] == production.experiments.LOGISTIC_PARAMS
    else:
        assert model["model_family"] == "XGBClassifier"
        assert model["hyperparameters"] == production.experiments.XGBOOST_PARAMS
    assert production._model(rows, cell) == raw


@pytest.fixture
def small_registry(tmp_path: Path) -> tuple[production_store.ProductionStore, dict[str, bytes]]:
    store = production_store.ProductionStore(tmp_path / "versions")
    files = {
        "model.json": b"{}",
        "feature_columns.json": b"[]",
        "prepared_dataset_manifest.json": b"{}",
    }
    return store, files


def test_version_reservation_uses_parent_descriptor_before_payload_writes(
    small_registry: tuple[production_store.ProductionStore, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, files = small_registry
    events: list[str] = []
    mkdir, opening, lookup, write = (
        production_store.os.mkdir,
        production_store.os.open,
        production_store.os.stat,
        production_store._write_fsynced_at,
    )

    def tracked_mkdir(path: str, *args: Any, **kwargs: Any) -> None:
        if path == "pinned-version":
            parent = kwargs["dir_fd"]
            assert os.fstat(parent).st_ino == store.root.stat().st_ino
            events.append("mkdir")
        mkdir(path, *args, **kwargs)

    def tracked_stat(path: str, *args: Any, **kwargs: Any) -> os.stat_result:
        if path == "pinned-version":
            assert kwargs["follow_symlinks"] is False
            assert "dir_fd" in kwargs
            events.append("lookup")
        return lookup(path, *args, **kwargs)

    def tracked_open(path: str, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "pinned-version":
            assert flags & os.O_DIRECTORY
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_ACCMODE == os.O_RDONLY
            assert "dir_fd" in kwargs
            events.append("open")
        return opening(path, flags, *args, **kwargs)

    def tracked_write(descriptor: int, name: str, raw: bytes) -> None:
        events.append("payload-write")
        write(descriptor, name, raw)

    monkeypatch.setattr(production_store.os, "mkdir", tracked_mkdir)
    monkeypatch.setattr(production_store.os, "stat", tracked_stat)
    monkeypatch.setattr(production_store.os, "open", tracked_open)
    monkeypatch.setattr(production_store, "_write_fsynced_at", tracked_write)
    store.publish("pinned-version", files, {"synthetic": True}, verify=lambda *_: None)
    reserved_at = events.index("mkdir")
    assert events[reserved_at : reserved_at + 4] == ["mkdir", "lookup", "open", "lookup"]
    assert events.index("payload-write") > reserved_at + 3


def test_directory_swap_between_parent_lookup_and_open_aborts_before_payload_write(
    small_registry: tuple[production_store.ProductionStore, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, files = small_registry
    opening = production_store.os.open
    writes: list[str] = []

    def replace_at_open(path: str, flags: int, *args: Any, **kwargs: Any) -> int:
        if path == "pinned-version":
            original = store.root / "pinned-version"
            original.rename(store.root / ".displaced-original")
            original.mkdir()
            (original / "unrelated").write_bytes(b"preserve")
        return opening(path, flags, *args, **kwargs)

    def forbidden_write(descriptor: int, name: str, raw: bytes) -> None:
        writes.append(name)
        raise AssertionError("swapped directory reached payload writes")

    monkeypatch.setattr(production_store.os, "open", replace_at_open)
    monkeypatch.setattr(production_store, "_write_fsynced_at", forbidden_write)
    with pytest.raises(production_store.ProductionIntegrityError, match="directory was replaced"):
        store.publish("pinned-version", files, {"synthetic": True}, verify=lambda *_: None)
    assert writes == []
    assert (store.root / "pinned-version" / "unrelated").read_bytes() == b"preserve"
    assert not (store.root / "pinned-version" / "manifest.json").exists()


def test_version_scoped_lock_serializes_cooperative_writers(
    small_registry: tuple[production_store.ProductionStore, dict[str, bytes]],
) -> None:
    store, files = small_registry
    lock = store.root / ".lock-pinned-version"
    with lock.open("wb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(production_store.ProductionCollisionError, match="locked"):
            store.publish("pinned-version", files, {"synthetic": True}, verify=lambda *_: None)
    assert not (store.root / "pinned-version").exists()
    store.publish("pinned-version", files, {"synthetic": True}, verify=lambda *_: None)
