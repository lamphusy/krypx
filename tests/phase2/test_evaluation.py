"""Synthetic-only adversarial tests for the Milestone 9 evaluation contract."""

from __future__ import annotations

import gzip
import json
import shutil
import socket
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.features.build import compute_features
from crypto_ai.phase2 import backtests, evaluation, evaluation_store, holdout
from crypto_ai.phase2.dataset import COMBINED_COLUMNS, SENTIMENT_COLUMNS, TECHNICAL_COLUMNS
from crypto_ai.phase2.experiments import generate_run_id
from crypto_ai.sentiment import aggregation as sentiment_aggregation
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import (
    ArticleRecord,
    ScorePayload,
    ScoreRecord,
    derive_article_id,
    derive_article_version_id,
    derive_content_hash,
    derive_duplicate_group_id,
    derive_score_id,
    format_utc_timestamp,
    validate_article_record,
    validate_score_record,
)
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GapAttempt,
    GroupAnchor,
    TerminalGapEvidence,
    _dedup_fingerprint,
    _expected_source_locator_at,
)

MARKET_START = datetime(2025, 12, 1, tzinfo=UTC)
OOF_START = datetime(2025, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
DAY = timedelta(days=1)
CUTOFF = 100
FIRST_HOLDOUT = 106
LAST_MARKET = FIRST_HOLDOUT + 180 * 24 - 1
PROTOCOL_SHA256 = "1" * 64
CODE_COMMIT = "3" * 40
LOCK_SHA256 = "4" * 64


def _snapshot(directory: Path, name: str, raw: bytes) -> evaluation.SnapshotRef:
    path = directory / name
    path.write_bytes(raw)
    return evaluation.SnapshotRef(path=path, sha256=sha256_bytes(raw))


def _synthetic_request(
    tmp_path: Path,
    *,
    market_open_token: str | None = None,
    market_timestamp_gap_at: int | None = None,
    scheduled_trades: int = 50,
    article_raw: bytes = b"",
    score_raw: bytes = b"",
) -> evaluation.SyntheticEvaluationRequest:
    """Construct a fully local future-looking fixture, never a real holdout."""
    development_run_dir = tmp_path / f"synthetic-development-{uuid4().hex}"
    development_run_dir.mkdir()
    evaluation_root = tmp_path / "evaluations"
    evaluation_root.mkdir()
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    holdout_started_at = MARKET_START + FIRST_HOLDOUT * HOUR
    inspected_at = holdout_started_at + 180 * DAY
    signal_ordinals = {FIRST_HOLDOUT + 24 * index for index in range(scheduled_trades)}

    market_lines = ["market_ordinal,timestamp,open,high,low,close,volume"]
    market_rows: list[dict[str, Any]] = []
    previous_close = 100.0
    for ordinal in range(LAST_MARKET + 1):
        candle_at = MARKET_START + ordinal * HOUR
        opened_at = candle_at
        if ordinal == market_timestamp_gap_at:
            opened_at += HOUR
        open_price = previous_close
        development_signal = ordinal in {36, 48, 60, 72, 84, 96}
        body_return = (
            0.08
            if ordinal in signal_ordinals or development_signal
            else 0.0003 + 0.0001 * (ordinal % 11 - 5)
        )
        close_price = open_price * (1.0 + body_return)
        high_price = max(open_price, close_price) * 1.0015
        low_price = min(open_price, close_price) * 0.9985
        volume = float(1000 + ordinal % 23)
        market_rows.append(
            {
                "timestamp": pd.Timestamp(candle_at),
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )
        market_lines.append(
            f"{ordinal},{format_utc_timestamp(opened_at)},"
            + ",".join(
                [
                    (
                        market_open_token
                        if market_open_token is not None and ordinal >= FIRST_HOLDOUT
                        else repr(open_price)
                    ),
                    repr(high_price),
                    repr(low_price),
                    repr(close_price),
                    repr(volume),
                ]
            )
        )
        previous_close = close_price
    market_raw = ("\n".join(market_lines) + "\n").encode("utf-8")
    market_ref = _snapshot(snapshots_dir, "market.csv", market_raw)
    development_market_raw = (
        "\n".join(market_lines[: CUTOFF + holdout.HORIZON + 3]) + "\n"
    ).encode("utf-8")
    development_market_ref = _snapshot(
        development_run_dir, "development_market.csv", development_market_raw
    )
    development_rows_raw, augmented_model_bytes, control_model_bytes = (
        evaluation.build_synthetic_development_fit(development_market_raw, CUTOFF)
    )
    development_rows_ref = _snapshot(
        development_run_dir, "development_rows.json", development_rows_raw
    )
    article_ref = _snapshot(snapshots_dir, "articles.jsonl", article_raw)
    score_ref = _snapshot(snapshots_dir, "scores.jsonl", score_raw)

    technical = compute_features(pd.DataFrame(market_rows))
    no_news = {name: 0.0 for name in SENTIMENT_COLUMNS}
    no_news["hours_since_latest_article"] = 24.0
    no_news["news_missing_24h"] = 1.0
    feature_lines = ["market_ordinal,decision_at," + ",".join(COMBINED_COLUMNS)]
    prediction_rows = []
    for ordinal in range(FIRST_HOLDOUT, LAST_MARKET - 4):
        decision_at = MARKET_START + (ordinal + 1) * HOUR
        values = [float(technical.loc[ordinal, name]) for name in TECHNICAL_COLUMNS]
        values += [no_news[name] for name in SENTIMENT_COLUMNS]
        prediction_rows.append(
            {"market_ordinal": ordinal, **dict(zip(COMBINED_COLUMNS, values, strict=True))}
        )
        feature_lines.append(
            f"{ordinal},{format_utc_timestamp(decision_at)},"
            + ",".join(repr(value) for value in values)
        )
    feature_raw = ("\n".join(feature_lines) + "\n").encode("utf-8")
    feature_ref = _snapshot(snapshots_dir, "features.csv", feature_raw)

    prediction_frame = pd.DataFrame(prediction_rows)
    augmented_probabilities = evaluation._predict(
        prediction_frame,
        COMBINED_COLUMNS,
        evaluation._verified_model(augmented_model_bytes, COMBINED_COLUMNS),
    )
    completed_exit_ordinals = evaluation._scheduled_exit_ordinals(
        prediction_frame, augmented_probabilities
    )
    if scheduled_trades < 50:
        completed_exit_ordinals = completed_exit_ordinals[:scheduled_trades]
    augmented_model_sha256 = sha256_bytes(augmented_model_bytes)
    control_model_sha256 = sha256_bytes(control_model_bytes)
    frozen_fit_at = MARKET_START + 105 * HOUR + timedelta(minutes=30)
    shared_rows_sha256 = development_rows_ref.sha256
    fit_manifest_bytes = canonicalize(
        {
            "schema_version": holdout.FIT_SCHEMA,
            "specification_id": holdout.SPECIFICATION_ID,
            "synthetic": True,
            "selected_augmented_cell": "C",
            "matched_control_cell": "A",
            "development_cutoff_ordinal": CUTOFF,
            "shared_labeled_rows_sha256": shared_rows_sha256,
            "augmented_model_sha256": augmented_model_sha256,
            "control_model_sha256": control_model_sha256,
            "augmented_fit_count": 1,
            "control_fit_count": 1,
            "frozen_at_utc": format_utc_timestamp(frozen_fit_at),
        }
    )
    frozen_fit = holdout.FrozenFitProof(
        selected_augmented_cell="C",
        matched_control_cell="A",
        development_cutoff_ordinal=CUTOFF,
        shared_labeled_rows_sha256=shared_rows_sha256,
        augmented_model_sha256=augmented_model_sha256,
        control_model_sha256=control_model_sha256,
        fit_manifest_sha256=sha256_bytes(fit_manifest_bytes),
        augmented_fit_count=1,
        control_fit_count=1,
        frozen_at=frozen_fit_at,
        fit_manifest_bytes=fit_manifest_bytes,
        augmented_model_bytes=augmented_model_bytes,
        control_model_bytes=control_model_bytes,
    )
    labels = (
        holdout.DevelopmentLabel(99, 104, MARKET_START + 104 * HOUR),
        holdout.DevelopmentLabel(100, 105, MARKET_START + 105 * HOUR),
    )
    boundary = holdout.BoundaryPurgeManager.validate(
        development_cutoff_ordinal=CUTOFF,
        development_labels=labels,
        purge_ordinals=tuple(range(CUTOFF + 1, FIRST_HOLDOUT)),
        first_holdout_ordinal=FIRST_HOLDOUT,
        first_holdout_decision_at=MARKET_START + (FIRST_HOLDOUT + 1) * HOUR,
        frozen_fit=frozen_fit,
    )
    oof_spans = tuple(
        holdout.OofSpan(
            first_test_decision_at=OOF_START + 36 * index * DAY + HOUR,
            last_test_decision_at=OOF_START + 36 * (index + 1) * DAY,
        )
        for index in range(5)
    )
    readiness_plan = holdout.ZeroOutcomeReadinessInspector.plan(
        oof_spans, 50, frozen_at=MARKET_START + 105 * HOUR
    )
    candles = tuple(
        holdout.ClosedCandle(
            ordinal=ordinal,
            opened_at=MARKET_START + ordinal * HOUR,
            closed_at=MARKET_START + (ordinal + 1) * HOUR,
        )
        for ordinal in range(73, LAST_MARKET + 1)
    )
    scheduled_exits = tuple(
        holdout.ScheduledExit(
            decision_ordinal=exit_ordinal - holdout.HORIZON - 1,
            ordinal=exit_ordinal,
            exit_at=MARKET_START + exit_ordinal * HOUR,
            policy_sha256=PROTOCOL_SHA256,
        )
        for exit_ordinal in completed_exit_ordinals
    )
    readiness = holdout.ZeroOutcomeReadinessInspector.inspect(
        plan=readiness_plan,
        market_first_ordinal=73,
        first_holdout_ordinal=FIRST_HOLDOUT,
        market_last_ordinal=LAST_MARKET,
        candles=candles,
        raw_snapshots=tuple(
            holdout.RawSnapshot(raw, sha256_bytes(raw))
            for raw in (
                market_raw,
                article_ref.path.read_bytes(),
                score_ref.path.read_bytes(),
                feature_raw,
            )
        ),
        provider_outage_state="CLEAR",
        provider_gap_exclusions_verified=True,
        holdout_started_at=holdout_started_at,
        inspected_at=inspected_at,
        scheduled_exits=scheduled_exits,
        frozen_policy_sha256=PROTOCOL_SHA256,
    )
    development_manifest = {
        "schema_version": "phase2-synthetic-development-provenance-v1",
        "synthetic": True,
        "run_id": development_run_dir.name,
        "development_cutoff_ordinal": CUTOFF,
        "purge_ordinals": list(range(CUTOFF + 1, FIRST_HOLDOUT)),
        "first_holdout_ordinal": FIRST_HOLDOUT,
        "selected_augmented_cell": "C",
        "matched_control_cell": "A",
        "shared_labeled_rows_sha256": shared_rows_sha256,
        "fit_manifest_sha256": sha256_bytes(fit_manifest_bytes),
        "augmented_model_sha256": augmented_model_sha256,
        "control_model_sha256": control_model_sha256,
        "augmented_fit_count": 1,
        "control_fit_count": 1,
        "fitted_max_decision_ordinal": CUTOFF,
        "augmented_feature_columns": list(COMBINED_COLUMNS),
        "control_feature_columns": list(TECHNICAL_COLUMNS),
        "signal_threshold": 0.5,
        "development_market_sha256": development_market_ref.sha256,
        "development_rows_sha256": development_rows_ref.sha256,
        "model_family": "LogisticRegression",
    }
    (development_run_dir / "development_manifest.json").write_bytes(
        canonicalize(development_manifest)
    )
    for name, raw in {
        "augmented_model.json": augmented_model_bytes,
        "control_model.json": control_model_bytes,
        "fit_manifest.json": fit_manifest_bytes,
        "development_dataset_manifest.json": evaluation._development_dataset_manifest_bytes(
            development_market_raw, development_rows_raw, CUTOFF
        ),
    }.items():
        (development_run_dir / name).write_bytes(raw)
    run_id = generate_run_id()
    inventory_sha256 = sha256_bytes(
        canonicalize(
            {
                "evaluation_run_id": run_id,
                "market_snapshot_sha256": market_ref.sha256,
                "article_snapshot_sha256": article_ref.sha256,
                "score_snapshot_sha256": score_ref.sha256,
                "feature_snapshot_sha256": feature_ref.sha256,
            }
        )
    )
    return evaluation.SyntheticEvaluationRequest(
        run_id=run_id,
        development_run_dir=development_run_dir,
        evaluation_root=evaluation_root,
        boundary=boundary,
        readiness=readiness,
        protocol_sha256=PROTOCOL_SHA256,
        input_inventory_sha256=inventory_sha256,
        code_commit=CODE_COMMIT,
        dependency_lock_sha256=LOCK_SHA256,
        market_snapshot=market_ref,
        article_snapshot=article_ref,
        score_snapshot=score_ref,
        feature_snapshot=feature_ref,
        development_market=development_market_ref,
        development_rows=development_rows_ref,
        random_simulations=1,
        reduced_fixture_mode=True,
    )


def _reissue_readiness(
    request: evaluation.SyntheticEvaluationRequest,
    *,
    feature_raw: bytes | None = None,
    scheduled_decisions: tuple[int, ...] | None = None,
    aggregation_evidence: evaluation.SnapshotRef | None = None,
    provider_outage_state: str = "CLEAR",
) -> evaluation.SyntheticEvaluationRequest:
    """Reissue an otherwise valid proof after a targeted adversarial change."""
    feature_ref = request.feature_snapshot
    if feature_raw is not None:
        feature_ref.path.write_bytes(feature_raw)
        feature_ref = replace(feature_ref, sha256=sha256_bytes(feature_raw))
    oof_spans = tuple(
        holdout.OofSpan(
            first_test_decision_at=OOF_START + 36 * index * DAY + HOUR,
            last_test_decision_at=OOF_START + 36 * (index + 1) * DAY,
        )
        for index in range(5)
    )
    plan = holdout.ZeroOutcomeReadinessInspector.plan(
        oof_spans, 50, frozen_at=MARKET_START + 105 * HOUR
    )
    market_first = FIRST_HOLDOUT - holdout.TECHNICAL_CONTEXT_ROWS + 1
    candles = tuple(
        holdout.ClosedCandle(
            ordinal=ordinal,
            opened_at=MARKET_START + ordinal * HOUR,
            closed_at=MARKET_START + (ordinal + 1) * HOUR,
        )
        for ordinal in range(market_first, LAST_MARKET + 1)
    )
    if scheduled_decisions is None:
        rows = evaluation._csv_rows(
            feature_ref.path.read_bytes(),
            ("market_ordinal", "decision_at") + COMBINED_COLUMNS,
        )
        prediction_frame = pd.DataFrame(
            [
                {
                    "market_ordinal": int(row[0]),
                    **dict(zip(COMBINED_COLUMNS, (float(value) for value in row[2:]), strict=True)),
                }
                for row in rows
            ]
        )
        probabilities = evaluation._predict(
            prediction_frame,
            COMBINED_COLUMNS,
            evaluation._verified_model(
                request.boundary.frozen_fit.augmented_model_bytes, COMBINED_COLUMNS
            ),
        )
        decisions = tuple(
            exit_ordinal - holdout.HORIZON - 1
            for exit_ordinal in evaluation._scheduled_exit_ordinals(prediction_frame, probabilities)
        )
    else:
        decisions = scheduled_decisions
    scheduled_exits = tuple(
        holdout.ScheduledExit(
            decision_ordinal=decision,
            ordinal=decision + holdout.HORIZON + 1,
            exit_at=MARKET_START + (decision + holdout.HORIZON + 1) * HOUR,
            policy_sha256=PROTOCOL_SHA256,
        )
        for decision in decisions
    )
    readiness = holdout.ZeroOutcomeReadinessInspector.inspect(
        plan=plan,
        market_first_ordinal=market_first,
        first_holdout_ordinal=FIRST_HOLDOUT,
        market_last_ordinal=LAST_MARKET,
        candles=candles,
        raw_snapshots=tuple(
            holdout.RawSnapshot(raw, sha256_bytes(raw))
            for raw in (
                request.market_snapshot.path.read_bytes(),
                request.article_snapshot.path.read_bytes(),
                request.score_snapshot.path.read_bytes(),
                feature_ref.path.read_bytes(),
            )
        ),
        provider_outage_state=provider_outage_state,
        provider_gap_exclusions_verified=True,
        holdout_started_at=request.readiness.holdout_started_at,
        inspected_at=request.readiness.inspected_at,
        scheduled_exits=scheduled_exits,
        frozen_policy_sha256=PROTOCOL_SHA256,
    )
    inventory = {
        "evaluation_run_id": request.run_id,
        "market_snapshot_sha256": request.market_snapshot.sha256,
        "article_snapshot_sha256": request.article_snapshot.sha256,
        "score_snapshot_sha256": request.score_snapshot.sha256,
        "feature_snapshot_sha256": feature_ref.sha256,
    }
    if aggregation_evidence is not None:
        inventory["aggregation_evidence_sha256"] = aggregation_evidence.sha256
        if request.verified_news_parent is not None:
            inventory["verified_news_parent_sha256"] = sha256_bytes(
                evaluation._news_parent_bytes(request.verified_news_parent)
            )
    inventory_sha256 = sha256_bytes(canonicalize(inventory))
    return replace(
        request,
        readiness=readiness,
        feature_snapshot=feature_ref,
        aggregation_evidence=aggregation_evidence,
        input_inventory_sha256=inventory_sha256,
    )


def _verified_news_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_gap: bool,
    gap_offset_hours: int = 36,
) -> evaluation.SyntheticEvaluationRequest:
    """Build causal news fixtures; mock only the costly immutable-parent replay."""
    request = _synthetic_request(
        tmp_path, scheduled_trades=51 if with_gap and gap_offset_hours == 0 else 50
    )
    seen = MARKET_START + (FIRST_HOLDOUT + 15) * HOUR
    seen_at = format_utc_timestamp(seen)
    url = "https://example.test/synthetic-bitcoin-title"
    title = "Synthetic Bitcoin title for an offline fixture"
    source = "example.test"
    article_id = derive_article_id("gdelt_gsg", "synthetic-title-1", url)
    content_hash = derive_content_hash(
        asset="BTC", content=None, language="en", source=source, title=title
    )
    article = validate_article_record(
        ArticleRecord(
            article_id=article_id,
            article_version_id=derive_article_version_id(
                article_id=article_id,
                first_seen_at=seen_at,
                language="en",
                content_hash=content_hash,
            ),
            provider="gdelt_gsg",
            provider_article_id="synthetic-title-1",
            provider_observation_id="a" * 64,
            source=source,
            canonical_url=url,
            title=title,
            content=None,
            language="en",
            published_at=seen_at,
            provider_first_seen_at=seen_at,
            first_seen_at=seen_at,
            ingested_at=seen_at,
            provider_updated_at=None,
            asset="BTC",
            content_hash=content_hash,
            raw_snapshot_sha256="b" * 64,
            point_in_time_eligible=True,
            exclusion_reason=None,
            duplicate_group_id=derive_duplicate_group_id(article_id),
        ).to_dict()
    )
    payload = ScorePayload(sentiment_score=0.6, relevance_score=0.8)
    model_id, model_version, prompt_version, config_hash = (
        "synthetic-mock",
        "v1",
        "v1",
        "c" * 64,
    )
    score = validate_score_record(
        ScoreRecord(
            score_id=derive_score_id(
                content_hash=content_hash,
                asset="BTC",
                sentiment_model_id=model_id,
                sentiment_model_version=model_version,
                prompt_version=prompt_version,
                scoring_config_hash=config_hash,
            ),
            article_version_id=article.article_version_id,
            content_hash=content_hash,
            asset="BTC",
            sentiment_model_id=model_id,
            sentiment_model_version=model_version,
            prompt_version=prompt_version,
            scoring_config_hash=config_hash,
            input_sha256="d" * 64,
            raw_response_sha256="e" * 64,
            score_payload_sha256=canonical_sha256(payload.to_dict()),
            scored_at=format_utc_timestamp(seen + HOUR),
            state="succeeded",
            payload=payload,
        ).to_dict()
    )
    article_ref = _snapshot(
        request.article_snapshot.path.parent,
        request.article_snapshot.path.name,
        canonicalize(article.to_dict()) + b"\n",
    )
    score_ref = _snapshot(
        request.score_snapshot.path.parent,
        request.score_snapshot.path.name,
        canonicalize(score.to_dict()) + b"\n",
    )
    request = replace(request, article_snapshot=article_ref, score_snapshot=score_ref)
    anchor = GroupAnchor(
        duplicate_group_id=article.duplicate_group_id,
        anchor_article_id=article.article_id,
        initial_first_seen_at=article.first_seen_at,
        dedup_fingerprint=_dedup_fingerprint(article.title, article.language),
        canonical_url=url,
        source=source,
    )
    original_lines = request.feature_snapshot.path.read_text(encoding="utf-8").splitlines()
    first_close = MARKET_START + (FIRST_HOLDOUT + 1) * HOUR
    last_close = MARKET_START + (LAST_MARKET - 4) * HOUR
    coverage_start = first_close - timedelta(hours=24, minutes=1)
    coverage_end = last_close + timedelta(minutes=1)
    gaps: list[TerminalGapEvidence] = []
    if with_gap:
        gap_start = first_close + timedelta(hours=gap_offset_hours)
        gaps.append(
            TerminalGapEvidence.create(
                interval_start=format_utc_timestamp(gap_start),
                interval_end_exclusive=format_utc_timestamp(gap_start + timedelta(minutes=1)),
                expected_source_locator=_expected_source_locator_at(
                    format_utc_timestamp(gap_start)
                ),
                attempts=(
                    GapAttempt(
                        attempt_number=1,
                        attempted_at=format_utc_timestamp(gap_start + timedelta(minutes=30)),
                        http_status=404,
                        error_kind=None,
                        retry_after_seconds=None,
                        retry_disposition="gap",
                    ),
                ),
                terminal_at=format_utc_timestamp(gap_start + timedelta(minutes=31)),
                protocol_config_sha256=PROTOCOL_SHA256,
            )
        )
    intervals = [
        evaluation._SyntheticCoverageInterval(
            format_utc_timestamp(coverage_start),
            format_utc_timestamp(coverage_start),
            "complete",
        )
    ]
    intervals.extend(
        evaluation._SyntheticCoverageInterval(
            item.interval_start, item.interval_end_exclusive, "provider_gap"
        )
        for item in gaps
    )
    intervals.append(
        evaluation._SyntheticCoverageInterval(
            format_utc_timestamp(coverage_end),
            format_utc_timestamp(coverage_end),
            "complete",
        )
    )
    updated_lines = [original_lines[0]]
    serialized_rows = []
    excluded = []
    for line in original_lines[1:]:
        fields = line.split(",")
        ordinal = int(fields[0])
        decision_at = fields[1]
        visible_scores = (
            (score,)
            if seen + HOUR <= datetime.fromisoformat(decision_at.replace("Z", "+00:00"))
            else ()
        )
        prepared = sentiment_aggregation._Prepared(
            (article,), (anchor,), visible_scores, tuple(intervals), ()
        )
        row = sentiment_aggregation._aggregate_rows(prepared, (decision_at,))[0]
        serialized_rows.append(row.to_dict())
        if row.features is None:
            excluded.append(ordinal)
            continue
        values = row.features.to_dict()
        updated_lines.append(
            ",".join(fields[: 2 + len(TECHNICAL_COLUMNS)])
            + ","
            + ",".join(repr(float(values[name])) for name in SENTIMENT_COLUMNS)
        )
    feature_raw = ("\n".join(updated_lines) + "\n").encode("utf-8")
    feature_sha = sha256_bytes(feature_raw)
    parent = {
        "schema_version": "phase2-synthetic-aggregation-parent-v1",
        "synthetic": True,
        "article_snapshot_sha256": article_ref.sha256,
        "score_snapshot_sha256": score_ref.sha256,
        "feature_snapshot_sha256": feature_sha,
        "aggregation_config_sha256": canonical_sha256(sentiment_aggregation._config()),
        "coverage_start_at": format_utc_timestamp(coverage_start),
        "coverage_end_at_exclusive": format_utc_timestamp(coverage_end),
        "groups": [anchor.to_dict()],
        "terminal_gap_evidence": [gap.to_dict() for gap in gaps],
        "prepared_dataset_manifest": {
            "schema_version": "phase2-synthetic-prepared-dataset-parent-v1",
            "synthetic": True,
            "article_snapshot_sha256": article_ref.sha256,
            "score_snapshot_sha256": score_ref.sha256,
            "sentiment_feature_sha256": canonical_sha256(serialized_rows),
            "combined_feature_sha256": feature_sha,
            "technical_columns": list(TECHNICAL_COLUMNS),
            "sentiment_columns": list(SENTIMENT_COLUMNS),
            "excluded_provider_gap_ordinals": excluded,
        },
    }
    evidence_ref = _snapshot(
        request.feature_snapshot.path.parent,
        "aggregation-evidence.json",
        canonicalize({**parent, "parent_manifest_sha256": canonical_sha256(parent)}),
    )
    cas_root = tmp_path / "mock-verified-news-cas"
    cas_root.mkdir()
    request = replace(
        request,
        verified_news_parent=evaluation.VerifiedNewsParent(cas_root, "a" * 64, "b" * 64),
    )
    replay = evaluation._aggregation_expectations

    def verified_parent_fixture(_parent: Any, **kwargs: Any) -> Any:
        raw = kwargs["evidence_raw"]
        value = json.loads(raw)
        return replay(
            raw,
            articles=kwargs["articles"],
            scores=kwargs["scores"],
            article_sha256=value["article_snapshot_sha256"],
            score_sha256=value["score_snapshot_sha256"],
            feature_sha256=value["feature_snapshot_sha256"],
            protocol_sha256=kwargs["protocol_sha256"],
            opened_at=kwargs["opened_at"],
            first_ordinal=kwargs["first_ordinal"],
            last_decision_ordinal=kwargs["last_decision_ordinal"],
            inspected_at=request.readiness.inspected_at,
        )

    monkeypatch.setattr(evaluation, "_verified_news_parent_expectations", verified_parent_fixture)
    return _reissue_readiness(
        request,
        feature_raw=feature_raw,
        aggregation_evidence=evidence_ref,
        provider_outage_state="VERIFIED_GAP" if with_gap else "CLEAR",
    )


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Every Milestone 9 test must fail if an execution path attempts transport."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Milestone 9 tests must remain wholly offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(
        holdout,
        "_generation_registry_path",
        lambda: tmp_path / holdout.GENERATION_REGISTRY_NAME,
    )


def _ledger(increments: list[tuple[int, float]]) -> list[dict[str, Any]]:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return [
        {"exit_timestamp": format_utc_timestamp(start + timedelta(days=day)), "pnl": pnl}
        for day, pnl in increments
    ]


def _spread_ledgers() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fifty exits across five non-overlapping 30-day calendar windows."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    stamps = [
        start + timedelta(days=31 * group, hours=hour) for group in range(5) for hour in range(10)
    ]
    augmented = [{"exit_timestamp": format_utc_timestamp(at), "pnl": 1.0} for at in stamps]
    control = [{"exit_timestamp": format_utc_timestamp(at), "pnl": 0.0} for at in stamps]
    return augmented, control


def _passing_operands() -> dict[str, Any]:
    augmented, control = _spread_ledgers()
    return {
        "augmented_metrics": {
            "total_return": 0.10,
            "sharpe_ratio": 0.7,
            "profit_factor": 1.2,
            "maximum_drawdown": -0.15,
            "num_trades": 50,
        },
        "control_metrics": {"total_return": 0.06, "maximum_drawdown": -0.14},
        "cash_metrics": {"total_return": 0.0},
        "augmented_ledger": augmented,
        "control_ledger": control,
        "elapsed_days": 180.0,
        "planned_minimum_days": 180,
    }


def _gates(**overrides: Any) -> dict[str, Any]:
    operands = _passing_operands()
    operands.update(overrides)
    return evaluation.evaluate_final_gates(**operands)


def test_complete_synthetic_evaluation_publishes_matched_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _synthetic_request(tmp_path)
    writes: list[str] = []
    write = evaluation_store._write_fsynced_at

    def observe_write(descriptor: int, name: str, raw: bytes) -> Any:
        writes.append(name)
        return write(descriptor, name, raw)

    monkeypatch.setattr(evaluation_store, "_write_fsynced_at", observe_write)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    final_dir = request.evaluation_root / request.run_id
    claim_path = request.development_run_dir / holdout.CLAIM_NAME
    assert final_dir.is_dir()
    assert claim_path.is_file()
    completion_path = request.development_run_dir / "holdout_evaluation_completed.json"
    assert completion_path.is_file()
    claim_bytes = claim_path.read_bytes()
    claim = json.loads(claim_bytes)
    assert claim["schema_version"] == holdout.EVALUATION_CLAIM_SCHEMA
    assert claim["run_id"] == request.development_run_dir.name
    assert claim["evaluation_run_id"] == request.run_id
    assert artifact.files[holdout.CLAIM_NAME] == claim_bytes
    assert artifact.files["input_market_snapshot.csv"] == request.market_snapshot.path.read_bytes()
    assert artifact.files["input_article_snapshot.jsonl"] == b""
    assert artifact.files["input_score_snapshot.jsonl"] == b""
    assert {
        "holdout_predictions.csv",
        "metrics.json",
        "baseline_metrics.json",
        "cost_sensitivity.json",
    }.issubset(artifact.files)
    assert any(name.startswith("evaluation_models/") for name in artifact.files)
    assert any(name.startswith("trade_ledgers/") for name in artifact.files)
    assert any(name.startswith("equity_curves/") for name in artifact.files)
    assert (final_dir / "evaluation_manifest.json").is_file()
    assert writes[-1] == "evaluation_manifest.json"
    manifest = json.loads((final_dir / "evaluation_manifest.json").read_bytes())
    assert manifest["metadata"]["claim_sha256"] == sha256_bytes(claim_bytes)
    completion = json.loads(completion_path.read_bytes())
    assert completion["claim_sha256"] == sha256_bytes(claim_bytes)
    assert completion["evaluation_manifest_sha256"] == sha256_bytes(
        (final_dir / "evaluation_manifest.json").read_bytes()
    )
    metrics = json.loads(artifact.files["metrics.json"])
    assert metrics["research_verdict"] in {"PASS", "FAIL"}
    assert metrics["production_decision"] == "NO-GO"
    assert metrics["base"]["num_trades"] >= 50


@pytest.mark.parametrize(
    "snapshot_name",
    ("market_snapshot", "article_snapshot", "score_snapshot", "feature_snapshot"),
)
def test_preflight_corrupted_snapshot_rejected_without_consuming_claim(
    tmp_path: Path, snapshot_name: str
) -> None:
    request = _synthetic_request(tmp_path)
    snapshot = getattr(request, snapshot_name)
    snapshot.path.write_bytes(snapshot.path.read_bytes() + b"corrupt")
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()
    assert not (request.evaluation_root / request.run_id).exists()


@pytest.mark.parametrize("kind", ("article", "score"))
def test_newly_hashed_malformed_news_snapshot_fails_before_claim(tmp_path: Path, kind: str) -> None:
    raw = b'{"unexpected":1}\n'
    kwargs = {"article_raw": raw} if kind == "article" else {"score_raw": raw}
    request = _synthetic_request(tmp_path, **kwargs)
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


@pytest.mark.parametrize("with_gap", (False, True))
def test_nonempty_synthetic_news_and_verified_gap_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_gap: bool
) -> None:
    request = _verified_news_request(tmp_path, monkeypatch, with_gap=with_gap)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    assert artifact.files["input_article_snapshot.jsonl"]
    assert artifact.files["input_score_snapshot.jsonl"]
    assert artifact.files["input_aggregation_evidence.json"]
    assert artifact.files["input_news_parent.json"] == evaluation._news_parent_bytes(
        request.verified_news_parent
    )
    assert (
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )
        == artifact
    )
    decisions = {
        int(line.split(",", 1)[0])
        for line in artifact.files["holdout_predictions.csv"].decode().splitlines()[1:]
    }
    if with_gap:
        assert request.readiness.provider_outage_state == "VERIFIED_GAP"
        assert FIRST_HOLDOUT + 36 not in decisions
    else:
        assert request.readiness.provider_outage_state == "CLEAR"
        assert FIRST_HOLDOUT + 36 in decisions


def test_competing_scores_for_one_content_hash_fail_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _verified_news_request(tmp_path, monkeypatch, with_gap=False)
    original = validate_score_record(
        json.loads(request.score_snapshot.path.read_bytes().splitlines()[0])
    )
    competing_version = "v2"
    competing = validate_score_record(
        replace(
            original,
            sentiment_model_version=competing_version,
            score_id=derive_score_id(
                content_hash=original.content_hash,
                asset=original.asset,
                sentiment_model_id=original.sentiment_model_id,
                sentiment_model_version=competing_version,
                prompt_version=original.prompt_version,
                scoring_config_hash=original.scoring_config_hash,
            ),
        ).to_dict()
    )
    score_raw = b"".join(canonicalize(item.to_dict()) + b"\n" for item in (original, competing))
    request.score_snapshot.path.write_bytes(score_raw)
    score_ref = replace(request.score_snapshot, sha256=sha256_bytes(score_raw))
    request = replace(request, score_snapshot=score_ref)
    request = _reissue_readiness(
        request,
        aggregation_evidence=request.aggregation_evidence,
    )
    with pytest.raises(evaluation.EvaluationPreflightError, match="competing"):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_self_attested_news_hashes_without_immutable_m4_m5_parents_fail_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_verifier = evaluation._verified_news_parent_expectations
    request = _verified_news_request(tmp_path, monkeypatch, with_gap=False)
    # The article observation and score-response hashes in this fixture are
    # placeholders. Rehashed self-attestation must not replace M4/M5 replay.
    monkeypatch.setattr(evaluation, "_verified_news_parent_expectations", real_verifier)
    with pytest.raises(evaluation.EvaluationPreflightError, match="M4/M5|parent"):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_completion_write_failure_unpublishes_owned_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _synthetic_request(tmp_path)
    completion_called = False

    def fail_completion(*args: Any, **kwargs: Any) -> None:
        nonlocal completion_called
        completion_called = True
        raise OSError("injected durable completion write failure")

    monkeypatch.setattr(evaluation, "_mark_claim_completed", fail_completion)
    with pytest.raises(evaluation.EvaluationIntegrityError) as failed:
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert completion_called
    assert isinstance(failed.value.__cause__, OSError)
    assert (request.development_run_dir / holdout.CLAIM_NAME).is_file()
    public_run = request.evaluation_root / request.run_id
    assert not (public_run / "evaluation_manifest.json").exists()
    assert not (public_run / "metrics.json").exists()
    with pytest.raises(evaluation_store.EvaluationIntegrityError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


def test_leading_verified_provider_gap_keeps_first_retained_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _verified_news_request(tmp_path, monkeypatch, with_gap=True, gap_offset_hours=0)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    predictions = artifact.files["holdout_predictions.csv"].decode().splitlines()[1:]
    assert predictions
    first_retained_ordinal = int(predictions[0].split(",", 1)[0])
    assert first_retained_ordinal >= FIRST_HOLDOUT + 24
    assert (
        artifact.manifest["metadata"]["common_window"]["first_open_ordinal"]
        == first_retained_ordinal + 1
    )


def test_rehashed_provider_gap_decision_cannot_reenter_feature_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _verified_news_request(tmp_path, monkeypatch, with_gap=True)
    excluded = FIRST_HOLDOUT + 36
    lines = request.feature_snapshot.path.read_text(encoding="utf-8").splitlines()
    before = next(line for line in lines[1:] if line.startswith(f"{excluded - 1},"))
    forged = before.split(",")
    forged[0] = str(excluded)
    forged[1] = format_utc_timestamp(MARKET_START + (excluded + 1) * HOUR)
    insert_at = next(
        index
        for index, line in enumerate(lines[1:], start=1)
        if int(line.split(",", 1)[0]) > excluded
    )
    lines.insert(insert_at, ",".join(forged))
    feature_raw = ("\n".join(lines) + "\n").encode("utf-8")
    evidence = json.loads(request.aggregation_evidence.path.read_bytes())
    evidence["feature_snapshot_sha256"] = sha256_bytes(feature_raw)
    evidence["prepared_dataset_manifest"]["combined_feature_sha256"] = sha256_bytes(feature_raw)
    evidence.pop("parent_manifest_sha256")
    evidence["parent_manifest_sha256"] = canonical_sha256(evidence)
    evidence_ref = _snapshot(
        request.aggregation_evidence.path.parent,
        request.aggregation_evidence.path.name,
        canonicalize(evidence),
    )
    request = _reissue_readiness(
        request,
        feature_raw=feature_raw,
        aggregation_evidence=evidence_ref,
        provider_outage_state="VERIFIED_GAP",
    )
    with pytest.raises(evaluation.EvaluationIntegrityError, match="provider-gap decision"):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_self_hashed_gap_without_terminal_attempts_is_not_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _verified_news_request(tmp_path, monkeypatch, with_gap=True)
    evidence = json.loads(request.aggregation_evidence.path.read_bytes())
    original = evidence["terminal_gap_evidence"][0]
    forged = TerminalGapEvidence.create(
        interval_start=original["interval_start"],
        interval_end_exclusive=original["interval_end_exclusive"],
        expected_source_locator=original["expected_source_locator"],
        attempts=(),
        terminal_at=original["terminal_at"],
        protocol_config_sha256=PROTOCOL_SHA256,
    )
    evidence["terminal_gap_evidence"] = [forged.to_dict()]
    evidence.pop("parent_manifest_sha256")
    evidence["parent_manifest_sha256"] = canonical_sha256(evidence)
    ref = _snapshot(
        request.aggregation_evidence.path.parent,
        request.aggregation_evidence.path.name,
        canonicalize(evidence),
    )
    request = _reissue_readiness(
        request, aggregation_evidence=ref, provider_outage_state="VERIFIED_GAP"
    )
    with pytest.raises(evaluation.EvaluationPreflightError, match="cannot be replayed"):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_timestamp_gap_in_verified_market_bytes_rejected_before_claim(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path, market_timestamp_gap_at=FIRST_HOLDOUT + 20)
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_newly_hashed_market_prices_cannot_reuse_old_readiness_proof(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    lines = request.market_snapshot.path.read_text(encoding="utf-8").splitlines()
    values = lines[FIRST_HOLDOUT + 21].split(",")
    values[2] = "9999.0"
    lines[FIRST_HOLDOUT + 21] = ",".join(values)
    changed_market_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    request.market_snapshot.path.write_bytes(changed_market_bytes)
    changed_market_ref = replace(request.market_snapshot, sha256=sha256_bytes(changed_market_bytes))
    changed_inventory_sha256 = sha256_bytes(
        canonicalize(
            {
                "evaluation_run_id": request.run_id,
                "market_snapshot_sha256": changed_market_ref.sha256,
                "article_snapshot_sha256": request.article_snapshot.sha256,
                "score_snapshot_sha256": request.score_snapshot.sha256,
                "feature_snapshot_sha256": request.feature_snapshot.sha256,
            }
        )
    )
    changed_request = replace(
        request,
        market_snapshot=changed_market_ref,
        input_inventory_sha256=changed_inventory_sha256,
    )
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(changed_request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_truncated_hashed_feature_interval_fails_before_claim(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    lines = request.feature_snapshot.path.read_text(encoding="utf-8").splitlines()
    truncated = ("\n".join(lines[:-1]) + "\n").encode("utf-8")
    changed_request = _reissue_readiness(request, feature_raw=truncated)
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(changed_request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_tampered_hashed_technical_value_fails_after_claim(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    lines = request.feature_snapshot.path.read_text(encoding="utf-8").splitlines()
    fields = lines[8].split(",")
    body_return_index = 2 + COMBINED_COLUMNS.index("body_return")
    fields[body_return_index] = repr(float(fields[body_return_index]) + 0.001)
    lines[8] = ",".join(fields)
    changed_request = _reissue_readiness(
        request, feature_raw=("\n".join(lines) + "\n").encode("utf-8")
    )
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(changed_request)
    assert (request.development_run_dir / holdout.CLAIM_NAME).is_file()
    assert not (request.evaluation_root / request.run_id).exists()


def test_scheduled_exits_without_model_signals_fail_before_claim(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    shifted_decisions = tuple(FIRST_HOLDOUT + 1 + 24 * index for index in range(50))
    changed_request = _reissue_readiness(request, scheduled_decisions=shifted_decisions)
    assert changed_request.readiness.ready is True
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(changed_request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_preflight_missing_model_rejected_without_consuming_claim(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    broken_fit = replace(request.boundary.frozen_fit, augmented_model_bytes=b"")
    broken_boundary = replace(request.boundary, frozen_fit=broken_fit)
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(replace(request, boundary=broken_boundary))
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


@pytest.mark.parametrize(
    "boundary", ("manifest_read", "after_preflight", "claim_constructor", "claim_acquire")
)
def test_development_directory_swap_cannot_rebind_verified_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    request = _synthetic_request(tmp_path)
    original_run = request.development_run_dir.with_name("original-verified-development")
    swapped = False

    def swap() -> None:
        nonlocal swapped
        assert not swapped
        request.development_run_dir.rename(original_run)
        request.development_run_dir.mkdir()
        swapped = True

    if boundary == "manifest_read":
        original = evaluation._development_manifest

        def before_manifest(*args: Any, **kwargs: Any) -> Any:
            swap()
            return original(*args, **kwargs)

        monkeypatch.setattr(evaluation, "_development_manifest", before_manifest)
    elif boundary == "after_preflight":
        original = evaluation._preflight

        def after_preflight(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            swap()
            return result

        monkeypatch.setattr(evaluation, "_preflight", after_preflight)
    elif boundary == "claim_constructor":
        original = holdout.HoldoutClaimManager.__init__

        def before_constructor(*args: Any, **kwargs: Any) -> Any:
            swap()
            return original(*args, **kwargs)

        monkeypatch.setattr(holdout.HoldoutClaimManager, "__init__", before_constructor)
    else:
        original = holdout.HoldoutClaimManager.acquire

        def before_acquire(*args: Any, **kwargs: Any) -> Any:
            swap()
            return original(*args, **kwargs)

        monkeypatch.setattr(holdout.HoldoutClaimManager, "acquire", before_acquire)

    with pytest.raises(evaluation.EvaluationIntegrityError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert swapped
    assert list(request.development_run_dir.iterdir()) == []
    assert not (original_run / holdout.CLAIM_NAME).exists()
    assert not holdout._generation_registry_path().exists()
    assert not (request.evaluation_root / request.run_id).exists()


@pytest.mark.parametrize("name", tuple(evaluation_store.DEVELOPMENT_FILE_MAP))
def test_missing_frozen_development_artifact_fails_before_claim(tmp_path: Path, name: str) -> None:
    request = _synthetic_request(tmp_path)
    (request.development_run_dir / name).unlink()
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()
    assert not holdout._generation_registry_path().exists()


def test_get_requires_original_development_models_and_scaler_parameters(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    store = evaluation.EvaluationStore(request.evaluation_root)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    assert store.get(request.run_id, development_run_dir=request.development_run_dir) == artifact
    for name, retained in evaluation_store.DEVELOPMENT_FILE_MAP.items():
        source = request.development_run_dir / name
        raw = source.read_bytes()
        assert raw == artifact.files[retained]
        source.unlink()
        with pytest.raises(CryptoAIError):
            evaluation_store._verify_source_claim(request.development_run_dir, artifact)
        source.write_bytes(raw + b" ")
        with pytest.raises(CryptoAIError):
            evaluation_store._verify_source_claim(request.development_run_dir, artifact)
        source.write_bytes(raw)

    model_path = request.development_run_dir / "augmented_model.json"
    model_raw = model_path.read_bytes()
    model = json.loads(model_raw)
    model["scaler_mean"][0] += 1.0
    model_path.write_bytes(canonicalize(model))
    with pytest.raises(CryptoAIError):
        store.get(request.run_id, development_run_dir=request.development_run_dir)
    model_path.write_bytes(model_raw)
    model_path.unlink()
    with pytest.raises(CryptoAIError):
        store.get(request.run_id, development_run_dir=request.development_run_dir)


def test_preflight_unready_sample_rejected_without_consuming_claim(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path, scheduled_trades=49)
    assert request.readiness.ready is False
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_reduced_random_fixture_requires_explicit_noncompliant_mode(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(replace(request, reduced_fixture_mode=False))
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_reduced_random_fixture_is_labeled_and_cannot_claim_full_contract(tmp_path: Path) -> None:
    assert backtests.RANDOM_SIMULATIONS == 1_000
    assert (
        evaluation.SyntheticEvaluationRequest.__dataclass_fields__["random_simulations"].default
        == backtests.RANDOM_SIMULATIONS
    )
    request = _synthetic_request(tmp_path)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    metadata = artifact.manifest["metadata"]
    metrics = json.loads(artifact.files["metrics.json"])
    assert metadata["random_simulations"] == 1
    assert metadata["baseline_mode"] == "REDUCED_SYNTHETIC_FIXTURE"
    assert metrics["engineering_status"] == "REDUCED_SYNTHETIC_FIXTURE"
    assert metrics["production_decision"] == "NO-GO"

    forged_metadata = {**metadata, "baseline_mode": "FULL_1000"}
    forged_root = tmp_path / "forged-full-contract"
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(forged_root).publish(
            request.run_id, artifact.files, forged_metadata
        )
    assert not (forged_root / request.run_id).exists()


def test_retained_random_draws_and_scenario_summaries_crosslink(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    evidence = json.loads(artifact.files["trade_ledgers/random_exposure.json"])
    baseline = json.loads(artifact.files["baseline_metrics.json"])
    costs = json.loads(artifact.files["cost_sensitivity.json"])
    metrics = json.loads(artifact.files["metrics.json"])

    assert evidence["simulations"] == request.random_simulations
    assert evidence["seed_base"] == backtests.RANDOM_SEED
    assert len(evidence["draws_and_metrics"]) == request.random_simulations
    assert baseline["random_exposure"] == costs["random_exposure"]
    assert set(baseline["random_exposure"]) == set(backtests.SCENARIO_ORDER)

    decision_count = len(request.feature_snapshot.path.read_bytes().splitlines()) - 1
    for index, run in enumerate(evidence["draws_and_metrics"]):
        assert run["simulation"] == index
        assert run["seed"] == backtests.RANDOM_SEED + index
        draws = run["draws"]
        assert len(draws) == decision_count
        assert all(type(draw) is int and draw in (0, 1) for draw in draws)
        assert (
            draws
            == np.random.default_rng(run["seed"])
            .binomial(1, evidence["signal_probability"], decision_count)
            .tolist()
        )
        assert run["draws_sha256"] == sha256_bytes(canonicalize(draws))
        assert set(run["scenarios"]) == set(backtests.SCENARIO_ORDER)
        for scenario in backtests.SCENARIO_ORDER:
            retained = run["scenarios"][scenario]
            summary = baseline["random_exposure"][scenario]
            assert summary["simulations"] == request.random_simulations
            assert summary["seed_base"] == evidence["seed_base"]
            assert summary["signal_probability"] == evidence["signal_probability"]
            assert len(retained["ledger_sha256"]) == 64
            assert len(retained["equity_sha256"]) == 64
            for metric in ("total_return", "sharpe_ratio", "maximum_drawdown"):
                assert summary[metric]["median"] == retained["metrics"][metric]
                assert summary[metric]["p05"] == retained["metrics"][metric]
                assert summary[metric]["p95"] == retained["metrics"][metric]
            expected_fraction = float(
                retained["metrics"]["total_return"]
                >= costs["models"]["augmented"][scenario]["total_return"]
            )
            assert summary["fraction_return_at_least_model"] == expected_fraction
    assert metrics["base"] == costs["models"]["augmented"]["base"]
    ledger_records = [
        json.loads(line)
        for line in gzip.decompress(artifact.files["trade_ledgers/random_exposure.jsonl.gz"])
        .decode("utf-8")
        .splitlines()
    ]
    curve_records = [
        json.loads(line)
        for line in gzip.decompress(artifact.files["equity_curves/random_exposure.jsonl.gz"])
        .decode("utf-8")
        .splitlines()
    ]
    assert len(ledger_records) == len(curve_records) == 3 * request.random_simulations
    for index, scenario in enumerate(backtests.SCENARIO_ORDER):
        path = evidence["draws_and_metrics"][0]["scenarios"][scenario]
        assert ledger_records[index]["simulation"] == 0
        assert ledger_records[index]["scenario"] == scenario
        assert (
            sha256_bytes(canonicalize(ledger_records[index]["trade_ledger"]))
            == path["ledger_sha256"]
        )
        assert curve_records[index]["simulation"] == 0
        assert curve_records[index]["scenario"] == scenario
        assert (
            sha256_bytes(canonicalize(curve_records[index]["equity_curve"]))
            == path["equity_sha256"]
        )


def test_rehashed_random_trajectory_mutation_is_rejected_by_store(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    final_dir = request.evaluation_root / request.run_id
    relative_path = "equity_curves/random_exposure.jsonl.gz"
    path = final_dir / relative_path
    records = gzip.decompress(path.read_bytes()).splitlines()
    forged_record = json.loads(records[0])
    forged_record["simulation"] = 123
    records[0] = canonicalize(forged_record)
    forged_zip = gzip.compress(b"\n".join(records) + b"\n", mtime=0)
    path.write_bytes(forged_zip)
    manifest_path = final_dir / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][relative_path] = {
        "sha256": sha256_bytes(forged_zip),
        "size_bytes": len(forged_zip),
    }
    manifest_path.write_bytes(canonicalize(manifest))
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


def test_rehashed_random_trajectory_semantic_forgery_and_corrupt_gzip_fail_closed(
    tmp_path: Path,
) -> None:
    request = _synthetic_request(tmp_path)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    curve_name = "equity_curves/random_exposure.jsonl.gz"
    evidence_name = "trade_ledgers/random_exposure.json"
    evidence = json.loads(artifact.files[evidence_name])
    curve_records = [
        json.loads(line) for line in gzip.decompress(artifact.files[curve_name]).splitlines()
    ]
    for record in curve_records:
        record["equity_curve"] = []
        evidence["draws_and_metrics"][record["simulation"]]["scenarios"][record["scenario"]][
            "equity_sha256"
        ] = sha256_bytes(canonicalize([]))
    forged_files = dict(artifact.files)
    forged_files[curve_name] = gzip.compress(
        b"".join(canonicalize(record) + b"\n" for record in curve_records), mtime=0
    )
    forged_files[evidence_name] = canonicalize(evidence)
    forged_store = evaluation.EvaluationStore(tmp_path / "forged-trajectory")
    with pytest.raises(CryptoAIError):
        forged_store.publish(request.run_id, forged_files, artifact.manifest["metadata"])
    assert not (forged_store.root / request.run_id).exists()

    corrupt_files = dict(artifact.files)
    corrupt_files[curve_name] = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03\xff" + b"\x00" * 8
    corrupt_store = evaluation.EvaluationStore(tmp_path / "corrupt-gzip")
    with pytest.raises(CryptoAIError):
        corrupt_store.publish(request.run_id, corrupt_files, artifact.manifest["metadata"])
    assert not (corrupt_store.root / request.run_id).exists()


def test_rehashed_random_draw_mutation_is_rejected_by_store(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    final_dir = request.evaluation_root / request.run_id
    path = final_dir / "trade_ledgers/random_exposure.json"
    evidence = json.loads(path.read_bytes())
    run = evidence["draws_and_metrics"][0]
    run["draws"][0] = 1 - run["draws"][0]
    run["draws_sha256"] = sha256_bytes(canonicalize(run["draws"]))
    forged_evidence = canonicalize(evidence)
    path.write_bytes(forged_evidence)

    manifest_path = final_dir / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["trade_ledgers/random_exposure.json"] = {
        "sha256": sha256_bytes(forged_evidence),
        "size_bytes": len(forged_evidence),
    }
    manifest_path.write_bytes(canonicalize(manifest))
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


def test_opaque_outcome_prices_are_not_parsed_until_after_irreversible_claim(
    tmp_path: Path,
) -> None:
    request = _synthetic_request(tmp_path, market_open_token="synthetic-unreadable-outcome")
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert (request.development_run_dir / holdout.CLAIM_NAME).is_file()
    assert not (request.evaluation_root / request.run_id).exists()
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)


def test_consumed_claim_prevents_a_second_evaluation_even_after_success(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    engine = evaluation.OfflineEvaluationEngine()
    first = engine.evaluate(request)
    claim_before = (request.development_run_dir / holdout.CLAIM_NAME).read_bytes()
    manifest_before = (
        request.evaluation_root / request.run_id / "evaluation_manifest.json"
    ).read_bytes()
    with pytest.raises(CryptoAIError):
        engine.evaluate(request)
    assert (request.development_run_dir / holdout.CLAIM_NAME).read_bytes() == claim_before
    assert (
        request.evaluation_root / request.run_id / "evaluation_manifest.json"
    ).read_bytes() == manifest_before
    assert first.files["input_market_snapshot.csv"] == request.market_snapshot.path.read_bytes()


def test_development_fit_provenance_cannot_include_boundary_purge_labels(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    path = request.development_run_dir / "development_manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["fitted_max_decision_ordinal"] = CUTOFF + 1
    path.write_bytes(canonicalize(manifest))
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_independent_development_rows_replay_rejects_rehashed_labels(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    rows_path = request.development_rows.path
    rows = json.loads(rows_path.read_bytes())
    rows["rows"][0]["label"] = 1 - rows["rows"][0]["label"]
    altered = canonicalize(rows)
    rows_path.write_bytes(altered)
    manifest_path = request.development_run_dir / "development_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["development_rows_sha256"] = sha256_bytes(altered)
    manifest_path.write_bytes(canonicalize(manifest))
    request = replace(
        request, development_rows=replace(request.development_rows, sha256=sha256_bytes(altered))
    )
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_selected_cell_pair_rejects_wrong_verified_model_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _synthetic_request(tmp_path)
    original = evaluation._verified_model

    def mislabeled_family(raw: bytes, columns: tuple[str, ...]) -> Any:
        return replace(original(raw, columns), family="XGBClassifier")

    monkeypatch.setattr(evaluation, "_verified_model", mislabeled_family)
    with pytest.raises(evaluation.EvaluationPreflightError, match="model family"):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


def test_cloned_development_run_cannot_claim_same_research_generation(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    clone_parent = tmp_path / "clone-parent"
    clone_parent.mkdir()
    clone_dir = clone_parent / request.development_run_dir.name
    shutil.copytree(request.development_run_dir, clone_dir)
    (clone_dir / holdout.CLAIM_NAME).unlink()
    (clone_dir / "holdout_evaluation_completed.json").unlink()
    second_run_id = generate_run_id()
    inventory = {
        "evaluation_run_id": second_run_id,
        "market_snapshot_sha256": request.market_snapshot.sha256,
        "article_snapshot_sha256": request.article_snapshot.sha256,
        "score_snapshot_sha256": request.score_snapshot.sha256,
        "feature_snapshot_sha256": request.feature_snapshot.sha256,
    }
    cloned = replace(
        request,
        run_id=second_run_id,
        development_run_dir=clone_dir,
        development_market=replace(
            request.development_market, path=clone_dir / "development_market.csv"
        ),
        development_rows=replace(
            request.development_rows, path=clone_dir / "development_rows.json"
        ),
        input_inventory_sha256=sha256_bytes(canonicalize(inventory)),
    )
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(cloned)
    assert not (clone_dir / holdout.CLAIM_NAME).exists()
    assert not (request.evaluation_root / second_run_id).exists()


def test_cloned_development_directory_cannot_rebind_completed_evaluation(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    clone_parent = tmp_path / "source-clone"
    clone_parent.mkdir()
    cloned_source = clone_parent / request.development_run_dir.name
    shutil.copytree(request.development_run_dir, cloned_source)
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=cloned_source
        )


def test_snapshot_symlink_rejected_before_claim(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    original = request.market_snapshot.path
    backup = original.with_name("market-regular.csv")
    original.rename(backup)
    original.symlink_to(backup)
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert not (request.development_run_dir / holdout.CLAIM_NAME).exists()


@pytest.mark.parametrize("mutation", ["payload", "unexpected_entry"])
def test_published_artifact_mutation_fails_descriptor_anchored_verification(
    tmp_path: Path, mutation: str
) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    final_dir = request.evaluation_root / request.run_id
    if mutation == "payload":
        path = final_dir / "metrics.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        (final_dir / "unmanifested.txt").write_bytes(b"injected")
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


def test_recomputed_manifest_metadata_forgery_rejected(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    manifest_path = request.evaluation_root / request.run_id / "evaluation_manifest.json"
    forged = json.loads(manifest_path.read_bytes())
    forged["metadata"]["augmented_model_sha256"] = "0" * 64
    manifest_path.write_bytes(canonicalize(forged))
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


def test_rehashed_control_curve_forgery_fails_semantic_replay(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    final_dir = request.evaluation_root / request.run_id
    curve_path = final_dir / "equity_curves/control_base.json"
    curve = json.loads(curve_path.read_bytes())
    curve.append(curve[-1])
    altered = canonicalize(curve)
    curve_path.write_bytes(altered)
    manifest_path = final_dir / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["equity_curves/control_base.json"] = {
        "sha256": sha256_bytes(altered),
        "size_bytes": len(altered),
    }
    updated_manifest = canonicalize(manifest)
    manifest_path.write_bytes(updated_manifest)
    completion_path = request.development_run_dir / "holdout_evaluation_completed.json"
    completion = json.loads(completion_path.read_bytes())
    completion["evaluation_manifest_sha256"] = sha256_bytes(updated_manifest)
    completion_path.write_bytes(canonicalize(completion))
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


def test_manifest_unlink_error_invalidates_post_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _synthetic_request(tmp_path)
    real_capture = evaluation_store._capture
    real_unlink = evaluation_store.os.unlink
    captures = 0
    manifest_unlink_attempts = 0

    def fail_post_rename(*args: Any, **kwargs: Any) -> Any:
        nonlocal captures
        captures += 1
        if captures == 3:
            raise evaluation_store.EvaluationIntegrityError("synthetic post-rename fault")
        return real_capture(*args, **kwargs)

    def fail_manifest_unlink(path: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal manifest_unlink_attempts
        if path == evaluation_store.MANIFEST_NAME and manifest_unlink_attempts == 0:
            manifest_unlink_attempts += 1
            raise OSError("synthetic unlink fault")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(evaluation_store, "_capture", fail_post_rename)
    monkeypatch.setattr(evaluation_store.os, "unlink", fail_manifest_unlink)
    with pytest.raises(CryptoAIError):
        evaluation.OfflineEvaluationEngine().evaluate(request)
    assert captures == 3
    assert manifest_unlink_attempts == 1
    public_dir = request.evaluation_root / request.run_id
    assert not (public_dir / evaluation_store.MANIFEST_NAME).exists()
    if public_dir.exists():
        assert list(public_dir.iterdir()) == []
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


def test_rewritten_retained_claim_fails_even_with_recomputed_manifest_hashes(
    tmp_path: Path,
) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    final_dir = request.evaluation_root / request.run_id
    claim_path = final_dir / holdout.CLAIM_NAME
    claim = json.loads(claim_path.read_bytes())
    claim["readiness"]["ready"] = False
    changed_claim = canonicalize(claim)
    claim_path.write_bytes(changed_claim)
    manifest_path = final_dir / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    changed_hash = sha256_bytes(changed_claim)
    manifest["files"][holdout.CLAIM_NAME] = {
        "sha256": changed_hash,
        "size_bytes": len(changed_claim),
    }
    manifest["metadata"]["claim_sha256"] = changed_hash
    manifest["metadata"]["claim_bytes_sha256"] = changed_hash
    manifest_path.write_bytes(canonicalize(manifest))
    with pytest.raises(CryptoAIError):
        evaluation.EvaluationStore(request.evaluation_root).get(
            request.run_id, development_run_dir=request.development_run_dir
        )


@pytest.mark.parametrize("mutation", ("tamper", "delete"))
def test_source_claim_change_fails_mandatory_development_binding(
    tmp_path: Path, mutation: str
) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    claim_path = request.development_run_dir / holdout.CLAIM_NAME
    if mutation == "tamper":
        claim_path.write_bytes(claim_path.read_bytes() + b" ")
    else:
        claim_path.unlink()
    store = evaluation.EvaluationStore(request.evaluation_root)
    with pytest.raises(CryptoAIError):
        store.get(request.run_id, development_run_dir=request.development_run_dir)


@pytest.mark.parametrize("mutation", ("tamper", "delete"))
def test_completion_change_fails_mandatory_development_binding(
    tmp_path: Path, mutation: str
) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    completion_path = request.development_run_dir / "holdout_evaluation_completed.json"
    if mutation == "tamper":
        completion = json.loads(completion_path.read_bytes())
        completion["evaluation_run_id"] = "other-evaluation"
        completion_path.write_bytes(canonicalize(completion))
    else:
        completion_path.unlink()
    store = evaluation.EvaluationStore(request.evaluation_root)
    with pytest.raises(CryptoAIError):
        store.get(request.run_id, development_run_dir=request.development_run_dir)


def test_rehashed_metrics_gate_forgery_fails_completed_source_binding(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    evaluation.OfflineEvaluationEngine().evaluate(request)
    final_dir = request.evaluation_root / request.run_id
    metrics_path = final_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_bytes())
    metrics["research_verdict"] = "FAIL" if metrics["research_verdict"] == "PASS" else "PASS"
    gates = metrics["final_gates"]
    gates["research_verdict"] = metrics["research_verdict"]
    gates["gates"]["completed_trades"] = not gates["gates"]["completed_trades"]
    forged_metrics = canonicalize(metrics)
    metrics_path.write_bytes(forged_metrics)
    manifest_path = final_dir / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["metrics.json"] = {
        "sha256": sha256_bytes(forged_metrics),
        "size_bytes": len(forged_metrics),
    }
    manifest_path.write_bytes(canonicalize(manifest))

    store = evaluation.EvaluationStore(request.evaluation_root)
    with pytest.raises(CryptoAIError):
        store.get(request.run_id)
    with pytest.raises(CryptoAIError):
        store.get(request.run_id, development_run_dir=request.development_run_dir)


def test_standalone_publish_requires_every_deterministic_and_random_path(tmp_path: Path) -> None:
    request = _synthetic_request(tmp_path)
    artifact = evaluation.OfflineEvaluationEngine().evaluate(request)
    required_paths = (
        "trade_ledgers/cash_base.json",
        "equity_curves/ema_9_21_high.json",
        "trade_ledgers/random_exposure.json",
        "trade_ledgers/random_exposure.jsonl.gz",
        "equity_curves/random_exposure.jsonl.gz",
    )
    for index, missing_path in enumerate(required_paths):
        files = dict(artifact.files)
        files.pop(missing_path)
        store_root = tmp_path / f"standalone-{index}"
        store = evaluation.EvaluationStore(store_root)
        with pytest.raises(CryptoAIError):
            store.publish(request.run_id, files, artifact.manifest["metadata"])
        assert not (store_root / request.run_id).exists()


def test_predeclared_final_gates_pass_only_as_a_conjunction() -> None:
    result = _gates()
    assert result["research_verdict"] == "PASS"
    assert result["production_decision"] == "NO-GO"
    assert result["gates"]
    assert all(value is True for value in result["gates"].values())


@pytest.mark.parametrize(
    "field,value",
    [
        ("total_return", 0.0),
        ("total_return", -0.001),
        ("sharpe_ratio", 0.0),
        ("sharpe_ratio", None),
        ("profit_factor", 1.05),
        ("profit_factor", None),
        ("maximum_drawdown", -0.200001),
        ("num_trades", 49),
    ],
)
def test_strict_strategy_gate_boundaries_fail(field: str, value: Any) -> None:
    operands = _passing_operands()
    operands["augmented_metrics"][field] = value
    if field == "num_trades":
        operands["augmented_ledger"] = operands["augmented_ledger"][:value]
    result = evaluation.evaluate_final_gates(**operands)
    assert result["research_verdict"] == "FAIL"
    assert result["production_decision"] == "NO-GO"
    assert any(passed is False for passed in result["gates"].values())


@pytest.mark.parametrize(
    "augmented_return,control_return,cash_return",
    [(0.06, 0.06, 0.0), (0.10, 0.06, 0.10), (0.0, -0.01, 0.0)],
)
def test_return_must_strictly_beat_both_matched_comparators(
    augmented_return: float, control_return: float, cash_return: float
) -> None:
    operands = _passing_operands()
    operands["augmented_metrics"]["total_return"] = augmented_return
    operands["control_metrics"]["total_return"] = control_return
    operands["cash_metrics"]["total_return"] = cash_return
    assert evaluation.evaluate_final_gates(**operands)["research_verdict"] == "FAIL"


def test_drawdown_magnitude_and_two_percentage_point_boundary() -> None:
    operands = _passing_operands()
    operands["augmented_metrics"]["maximum_drawdown"] = -0.20
    operands["control_metrics"]["maximum_drawdown"] = -0.18
    assert evaluation.evaluate_final_gates(**operands)["research_verdict"] == "PASS"
    operands["control_metrics"]["maximum_drawdown"] = -0.179
    assert evaluation.evaluate_final_gates(**operands)["research_verdict"] == "FAIL"


@pytest.mark.parametrize(
    "elapsed,minimum",
    [(179.999, 180), (180.0, 181)],
)
def test_frozen_readiness_duration_cannot_be_relaxed(elapsed: float, minimum: int) -> None:
    assert _gates(elapsed_days=elapsed, planned_minimum_days=minimum)["research_verdict"] == "FAIL"


def test_invalid_planned_duration_is_a_precondition_failure() -> None:
    with pytest.raises(CryptoAIError):
        _gates(planned_minimum_days=179)


def test_rolling_concentration_uses_utc_calendar_half_open_windows() -> None:
    # Day 30 belongs to the next window, not the [day 0, day 30) window.
    augmented = _ledger([(0, 4.0), (30, 3.0), (60, 3.0)])
    control = _ledger([(0, 0.0), (30, 0.0), (60, 0.0)])
    assert evaluation.rolling_incremental_concentration(augmented, control) == pytest.approx(0.4)
    filler_start = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(days=90)
    augmented.extend(
        {
            "exit_timestamp": format_utc_timestamp(filler_start + timedelta(hours=hour)),
            "pnl": 0.0,
        }
        for hour in range(47)
    )
    result = _gates(augmented_ledger=augmented, control_ledger=control)
    assert result["research_verdict"] == "PASS"
    augmented[0]["pnl"] = 4.001
    assert evaluation.rolling_incremental_concentration(augmented, control) > 0.4
    assert _gates(augmented_ledger=augmented, control_ledger=control)["research_verdict"] == "FAIL"


def test_rolling_incremental_pnl_uses_exit_hour_union_and_control_only_losses() -> None:
    augmented = _ledger([(0, 10.0), (31, 10.0)])
    control = _ledger([(0, 4.0), (15, -5.0), (31, 10.0)])
    # p(day 0)=6, p(day 15)=5, p(day 31)=0; the first 30-day window owns 100%.
    assert evaluation.rolling_incremental_concentration(augmented, control) == pytest.approx(1.0)


def test_rolling_concentration_handles_maximum_utc_calendar_date() -> None:
    augmented = [{"exit_timestamp": "9999-12-31T23:00:00Z", "pnl": 1.0}]
    assert evaluation.rolling_incremental_concentration(augmented, []) == 1.0


def test_rolling_concentration_rejects_finite_input_with_overflowing_sum() -> None:
    augmented = _ledger([(0, 1e308), (31, 1e308)])
    with pytest.raises(CryptoAIError):
        evaluation.rolling_incremental_concentration(augmented, [])


def test_final_trade_gate_rejects_forged_completed_count() -> None:
    operands = _passing_operands()
    operands["augmented_ledger"] = operands["augmented_ledger"][:3]
    with pytest.raises(CryptoAIError):
        evaluation.evaluate_final_gates(**operands)


@pytest.mark.parametrize(
    "augmented,control",
    [(_ledger([]), _ledger([])), (_ledger([(0, -1.0)]), _ledger([(0, 0.0)]))],
)
def test_zero_positive_incremental_pnl_is_not_a_passing_ratio(
    augmented: list[dict[str, Any]], control: list[dict[str, Any]]
) -> None:
    assert evaluation.rolling_incremental_concentration(augmented, control) is None
    operands = _passing_operands()
    operands["augmented_ledger"] = augmented
    operands["control_ledger"] = control
    operands["augmented_metrics"]["num_trades"] = len(augmented)
    assert evaluation.evaluate_final_gates(**operands)["research_verdict"] == "FAIL"
