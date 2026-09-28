"""Offline adversarial tests for the Milestone 8 holdout information firewall."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.phase2 import holdout
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp

OOF_START = datetime(2025, 1, 1, tzinfo=UTC)
MARKET_START = datetime(2025, 12, 1, tzinfo=UTC)
DAY = timedelta(days=1)
HOUR = timedelta(hours=1)
HOLDOUT_START = MARKET_START + 106 * HOUR
PLAN_FROZEN_AT = HOLDOUT_START - HOUR
CUTOFF = 100
PURGE = (101, 102, 103, 104, 105)
CLAIM_HASHES = {
    "protocol_sha256": "1" * 64,
    "input_inventory_sha256": "2" * 64,
    "code_commit": "3" * 40,
    "dependency_lock_sha256": "4" * 64,
}
OUTCOME_NAMES = (
    "forward_return",
    "label",
    "pnl",
    "equity_curve",
    "sharpe",
    "drawdown",
    "profit_factor",
    "hit_rate",
    "benchmark_return",
    "prediction_value",
    "trade_timestamp",
)
PUBLIC_REPORT_FIELDS = {
    "ready",
    "elapsed_days",
    "trade_threshold_met",
    "provider_outage_state",
    "planned_minimum_days",
    "synthetic",
}


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Milestone 8 tests must remain wholly offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture(scope="module")
def oof_spans() -> tuple[holdout.OofSpan, ...]:
    # Each closed validation span is [first decision - 1h, last decision).
    return tuple(
        holdout.OofSpan(
            first_test_decision_at=OOF_START + 36 * index * DAY + HOUR,
            last_test_decision_at=OOF_START + 36 * (index + 1) * DAY,
        )
        for index in range(5)
    )


@pytest.fixture(scope="module")
def plan(oof_spans: tuple[holdout.OofSpan, ...]) -> holdout.ReadinessPlan:
    return holdout.ZeroOutcomeReadinessInspector.plan(oof_spans, 50, frozen_at=PLAN_FROZEN_AT)


@pytest.fixture(scope="module")
def candles() -> tuple[holdout.ClosedCandle, ...]:
    return tuple(
        holdout.ClosedCandle(
            ordinal=ordinal,
            opened_at=MARKET_START + ordinal * HOUR,
            closed_at=MARKET_START + (ordinal + 1) * HOUR,
        )
        for ordinal in range(73, CUTOFF + 6 + 180 * 24)
    )


@pytest.fixture(scope="module")
def scheduled_exits() -> tuple[holdout.ScheduledExit, ...]:
    return tuple(
        holdout.ScheduledExit(
            decision_ordinal=CUTOFF + 1 + 24 * (index + 1),
            ordinal=CUTOFF + 6 + 24 * (index + 1),
            exit_at=HOLDOUT_START + 24 * (index + 1) * HOUR,
            policy_sha256=CLAIM_HASHES["protocol_sha256"],
        )
        for index in range(50)
    )


@pytest.fixture
def snapshots() -> tuple[holdout.RawSnapshot, ...]:
    # Deliberately hostile contents: the inspector may hash but must not parse
    # or surface an outcome-looking field carried by opaque raw input bytes.
    raw = b'{"synthetic":true,"forward_return":999,"pnl":888}'
    return (holdout.RawSnapshot(raw, hashlib.sha256(raw).hexdigest()),)


@pytest.fixture
def readiness_kwargs(
    plan: holdout.ReadinessPlan,
    candles: tuple[holdout.ClosedCandle, ...],
    snapshots: tuple[holdout.RawSnapshot, ...],
    scheduled_exits: tuple[holdout.ScheduledExit, ...],
) -> dict[str, Any]:
    return {
        "plan": plan,
        "market_first_ordinal": 73,
        "first_holdout_ordinal": CUTOFF + 6,
        "market_last_ordinal": CUTOFF + 6 + 180 * 24 - 1,
        "candles": candles,
        "raw_snapshots": snapshots,
        "provider_outage_state": "CLEAR",
        "provider_gap_exclusions_verified": True,
        "holdout_started_at": HOLDOUT_START,
        "inspected_at": HOLDOUT_START + 180 * DAY,
        "scheduled_exits": scheduled_exits,
        "frozen_policy_sha256": CLAIM_HASHES["protocol_sha256"],
        "synthetic": True,
    }


@pytest.fixture
def readiness(readiness_kwargs: dict[str, Any]) -> holdout.ReadinessReport:
    return holdout.ZeroOutcomeReadinessInspector.inspect(**readiness_kwargs)


@pytest.fixture
def frozen_fit() -> holdout.FrozenFitProof:
    frozen_at = MARKET_START + 105 * HOUR + timedelta(minutes=30)
    augmented_model_bytes = b"synthetic-C-model-artifact-v1"
    control_model_bytes = b"synthetic-A-model-artifact-v1"
    shared_rows_sha256 = "a" * 64
    augmented_model_sha256 = sha256_bytes(augmented_model_bytes)
    control_model_sha256 = sha256_bytes(control_model_bytes)
    fit_manifest_bytes = canonicalize(
        {
            "schema_version": "phase2-synthetic-frozen-fit-v1",
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
            "frozen_at_utc": format_utc_timestamp(frozen_at),
        }
    )
    return holdout.FrozenFitProof(
        selected_augmented_cell="C",
        matched_control_cell="A",
        development_cutoff_ordinal=CUTOFF,
        shared_labeled_rows_sha256=shared_rows_sha256,
        augmented_model_sha256=augmented_model_sha256,
        control_model_sha256=control_model_sha256,
        fit_manifest_sha256=sha256_bytes(fit_manifest_bytes),
        augmented_fit_count=1,
        control_fit_count=1,
        frozen_at=frozen_at,
        fit_manifest_bytes=fit_manifest_bytes,
        augmented_model_bytes=augmented_model_bytes,
        control_model_bytes=control_model_bytes,
        synthetic=True,
    )


@pytest.fixture
def labels() -> tuple[holdout.DevelopmentLabel, ...]:
    return (
        holdout.DevelopmentLabel(99, 104, MARKET_START + 104 * HOUR),
        holdout.DevelopmentLabel(100, 105, MARKET_START + 105 * HOUR),
    )


@pytest.fixture
def boundary_kwargs(
    labels: tuple[holdout.DevelopmentLabel, ...], frozen_fit: holdout.FrozenFitProof
) -> dict[str, Any]:
    return {
        "development_cutoff_ordinal": CUTOFF,
        "development_labels": labels,
        "purge_ordinals": PURGE,
        "first_holdout_ordinal": 106,
        "first_holdout_decision_at": MARKET_START + 107 * HOUR,
        "frozen_fit": frozen_fit,
    }


@pytest.fixture
def boundary(boundary_kwargs: dict[str, Any]) -> holdout.BoundaryPurgePlan:
    return holdout.BoundaryPurgeManager.validate(**boundary_kwargs)


def test_exact_five_span_union_and_planned_duration(
    oof_spans: tuple[holdout.OofSpan, ...], plan: holdout.ReadinessPlan
) -> None:
    assert plan.oof_elapsed_days == 180
    assert plan.planned_minimum_days == 180
    lower_rate = holdout.ZeroOutcomeReadinessInspector.plan(oof_spans, 40, frozen_at=PLAN_FROZEN_AT)
    assert lower_rate.planned_minimum_days == 225
    overlapping = tuple(
        holdout.OofSpan(
            first_test_decision_at=OOF_START + index * DAY + HOUR,
            last_test_decision_at=OOF_START + (index + 2) * DAY,
        )
        for index in range(5)
    )
    overlap_plan = holdout.ZeroOutcomeReadinessInspector.plan(
        overlapping, 1, frozen_at=PLAN_FROZEN_AT
    )
    assert overlap_plan.oof_elapsed_days == 6
    assert overlap_plan.planned_minimum_days == 300


@pytest.mark.parametrize(
    "elapsed_days,trades,expected_days",
    [(39, 5, 390), (67, 5, 670), (39, 7, 279)],
)
def test_planned_days_use_exact_rational_ceiling(
    elapsed_days: int, trades: int, expected_days: int
) -> None:
    bounds = (0, 1, 2, 3, 4, elapsed_days)
    spans = tuple(
        holdout.OofSpan(
            first_test_decision_at=OOF_START + bounds[index] * DAY + HOUR,
            last_test_decision_at=OOF_START + bounds[index + 1] * DAY,
        )
        for index in range(5)
    )
    plan = holdout.ZeroOutcomeReadinessInspector.plan(spans, trades, frozen_at=PLAN_FROZEN_AT)
    assert plan.oof_elapsed_days == elapsed_days
    assert plan.planned_minimum_days == expected_days


def test_readiness_plan_frozen_at_or_after_collection_start_is_rejected(
    readiness_kwargs: dict[str, Any], oof_spans: tuple[holdout.OofSpan, ...]
) -> None:
    late_plan = holdout.ZeroOutcomeReadinessInspector.plan(oof_spans, 50, frozen_at=HOLDOUT_START)
    with pytest.raises(holdout.HoldoutReadinessError):
        holdout.ZeroOutcomeReadinessInspector.inspect(**dict(readiness_kwargs, plan=late_plan))


@pytest.mark.parametrize("count", [0, -1, True, 1.0, float("nan"), float("inf")])
def test_oof_nonpositive_or_noninteger_trade_counts_fail_closed(
    oof_spans: tuple[holdout.OofSpan, ...], count: object
) -> None:
    with pytest.raises(holdout.HoldoutError):
        holdout.ZeroOutcomeReadinessInspector.plan(oof_spans, count, frozen_at=PLAN_FROZEN_AT)


@pytest.mark.parametrize("mutation", ["too_few", "reversed", "naive", "zero_duration"])
def test_oof_span_contract_fails_closed(
    oof_spans: tuple[holdout.OofSpan, ...], mutation: str
) -> None:
    spans = list(oof_spans)
    if mutation == "too_few":
        spans.pop()
    elif mutation == "reversed":
        spans[0] = replace(spans[0], first_test_decision_at=spans[0].last_test_decision_at + HOUR)
    elif mutation == "naive":
        spans[0] = replace(spans[0], first_test_decision_at=datetime(2025, 1, 1, 1))
    else:
        spans[0] = replace(spans[0], last_test_decision_at=spans[0].first_test_decision_at - HOUR)
    with pytest.raises(holdout.HoldoutError):
        holdout.ZeroOutcomeReadinessInspector.plan(spans, 50, frozen_at=PLAN_FROZEN_AT)


def test_oof_timestamp_underflow_is_project_specific(
    oof_spans: tuple[holdout.OofSpan, ...],
) -> None:
    spans = list(oof_spans)
    minimum_utc = datetime.min.replace(tzinfo=UTC)
    spans[0] = holdout.OofSpan(minimum_utc, minimum_utc)
    with pytest.raises(holdout.HoldoutInputError):
        holdout.ZeroOutcomeReadinessInspector.plan(spans, 50, frozen_at=PLAN_FROZEN_AT)


def test_ready_report_contains_only_allowlisted_operational_fields(
    readiness: holdout.ReadinessReport,
) -> None:
    public = readiness.to_dict()
    assert set(public) == PUBLIC_REPORT_FIELDS
    assert public["ready"] is True
    assert public["elapsed_days"] == 180
    assert public["trade_threshold_met"] is True
    assert public["planned_minimum_days"] == 180
    text = json.dumps(public).lower()
    assert all(name not in text for name in OUTCOME_NAMES)
    assert "999" not in text and "888" not in text
    assert "scheduled_exits" not in text
    assert "exit_at" not in text


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "open", "inauthentic", "wrong_hour"])
def test_candle_continuity_and_closure_fail_closed(
    readiness_kwargs: dict[str, Any], mutation: str
) -> None:
    candles = list(readiness_kwargs["candles"])
    if mutation == "missing":
        candles.pop(200)
    elif mutation == "duplicate":
        candles.insert(200, candles[200])
    elif mutation == "open":
        candles[200] = replace(candles[200], closed=False)
    elif mutation == "inauthentic":
        candles[200] = replace(candles[200], authentic=False)
    else:
        candles[200] = replace(candles[200], closed_at=candles[200].closed_at + HOUR)
    with pytest.raises(holdout.HoldoutError):
        holdout.ZeroOutcomeReadinessInspector.inspect(
            **dict(readiness_kwargs, candles=tuple(candles))
        )


@pytest.mark.parametrize("holdout_hours", [1, 1201])
def test_stale_candle_prefix_never_reports_180_day_readiness(
    readiness_kwargs: dict[str, Any], holdout_hours: int
) -> None:
    evidence = dict(readiness_kwargs)
    evidence["candles"] = evidence["candles"][: 33 + holdout_hours]
    evidence["market_last_ordinal"] = CUTOFF + 6 + holdout_hours - 1
    if holdout_hours == 1:
        evidence["scheduled_exits"] = ()
    with pytest.raises(holdout.HoldoutReadinessError, match="inspection time"):
        holdout.ZeroOutcomeReadinessInspector.inspect(**evidence)


def test_missing_technical_warmup_blocks_readiness(readiness_kwargs: dict[str, Any]) -> None:
    evidence = dict(readiness_kwargs)
    evidence["market_first_ordinal"] += 1
    evidence["candles"] = evidence["candles"][1:]
    with pytest.raises(holdout.HoldoutReadinessError):
        holdout.ZeroOutcomeReadinessInspector.inspect(**evidence)


def test_raw_snapshot_hash_is_checked_before_readiness(readiness_kwargs: dict[str, Any]) -> None:
    snapshot = readiness_kwargs["raw_snapshots"][0]
    changed = replace(snapshot, raw_bytes=snapshot.raw_bytes + b" ")
    with pytest.raises(holdout.HoldoutError):
        holdout.ZeroOutcomeReadinessInspector.inspect(
            **dict(readiness_kwargs, raw_snapshots=(changed,))
        )


@pytest.mark.parametrize("source", ["candle", "snapshot"])
def test_real_tagged_operational_evidence_is_rejected(
    readiness_kwargs: dict[str, Any], source: str
) -> None:
    evidence = dict(readiness_kwargs)
    if source == "candle":
        candles = list(evidence["candles"])
        candles[200] = replace(candles[200], synthetic=False)
        evidence["candles"] = tuple(candles)
    else:
        evidence["raw_snapshots"] = (replace(evidence["raw_snapshots"][0], synthetic=False),)
    with pytest.raises(holdout.HoldoutAuthorizationError):
        holdout.ZeroOutcomeReadinessInspector.inspect(**evidence)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "wrong_time",
        "wrong_policy",
        "wrong_horizon",
        "pre_holdout_decision",
        "out_of_market_range",
        "overlapping_position",
        "real_tag",
    ],
)
def test_scheduled_exits_require_unique_causal_frozen_policy_metadata(
    readiness_kwargs: dict[str, Any], mutation: str
) -> None:
    evidence = dict(readiness_kwargs)
    exits = list(evidence["scheduled_exits"])
    if mutation == "duplicate":
        exits.append(exits[0])
    elif mutation == "wrong_time":
        exits[0] = replace(exits[0], exit_at=exits[0].exit_at + HOUR)
    elif mutation == "wrong_policy":
        exits[0] = replace(exits[0], policy_sha256="f" * 64)
    elif mutation == "wrong_horizon":
        exits[0] = replace(exits[0], decision_ordinal=exits[0].decision_ordinal - 1)
    elif mutation == "pre_holdout_decision":
        ordinal = CUTOFF + 5 + 5
        exits[0] = replace(
            exits[0],
            decision_ordinal=CUTOFF + 5,
            ordinal=ordinal,
            exit_at=MARKET_START + ordinal * HOUR,
        )
    elif mutation == "out_of_market_range":
        ordinal = evidence["market_last_ordinal"] + 1
        exits[-1] = replace(
            exits[-1],
            decision_ordinal=ordinal - 5,
            ordinal=ordinal,
            exit_at=MARKET_START + ordinal * HOUR,
        )
    elif mutation == "overlapping_position":
        ordinal = exits[0].ordinal + 1
        exits[1] = replace(
            exits[1],
            decision_ordinal=ordinal - 5,
            ordinal=ordinal,
            exit_at=MARKET_START + ordinal * HOUR,
        )
    else:
        exits[0] = replace(exits[0], synthetic=False)
    evidence["scheduled_exits"] = tuple(exits)
    with pytest.raises(holdout.HoldoutError):
        holdout.ZeroOutcomeReadinessInspector.inspect(**evidence)


@pytest.mark.parametrize("outage,verified", [("UNRESOLVED", False), ("VERIFIED_GAP", False)])
def test_provider_outage_without_verified_exclusion_fails_closed(
    readiness_kwargs: dict[str, Any], outage: str, verified: bool
) -> None:
    with pytest.raises(holdout.HoldoutError):
        holdout.ZeroOutcomeReadinessInspector.inspect(
            **dict(
                readiness_kwargs,
                provider_outage_state=outage,
                provider_gap_exclusions_verified=verified,
            )
        )


def test_verified_gap_has_explicit_operational_state(readiness_kwargs: dict[str, Any]) -> None:
    report = holdout.ZeroOutcomeReadinessInspector.inspect(
        **dict(readiness_kwargs, provider_outage_state="VERIFIED_GAP")
    )
    assert report.to_dict()["provider_outage_state"] == "VERIFIED_GAP"


@pytest.mark.parametrize("reason,expected_threshold", [("trades", False), ("days", True)])
def test_insufficient_trades_or_calendar_days_are_not_ready(
    readiness_kwargs: dict[str, Any], reason: str, expected_threshold: bool
) -> None:
    evidence = dict(readiness_kwargs)
    if reason == "trades":
        evidence["scheduled_exits"] = evidence["scheduled_exits"][:-1]
    else:
        evidence["inspected_at"] = HOLDOUT_START + 179 * DAY
        evidence["candles"] = evidence["candles"][: 33 + 179 * 24]
        evidence["market_last_ordinal"] = CUTOFF + 6 + 179 * 24 - 1
    report = holdout.ZeroOutcomeReadinessInspector.inspect(**evidence)
    public = report.to_dict()
    assert public["ready"] is False
    assert public["trade_threshold_met"] is expected_threshold
    assert set(public) == PUBLIC_REPORT_FIELDS


def test_slower_oof_rate_extends_wait_even_at_180_days(
    readiness_kwargs: dict[str, Any], oof_spans: tuple[holdout.OofSpan, ...]
) -> None:
    slower_plan = holdout.ZeroOutcomeReadinessInspector.plan(
        oof_spans, 40, frozen_at=PLAN_FROZEN_AT
    )
    report = holdout.ZeroOutcomeReadinessInspector.inspect(
        **dict(readiness_kwargs, plan=slower_plan)
    )
    assert report.to_dict()["planned_minimum_days"] == 225
    assert report.to_dict()["ready"] is False


@pytest.mark.parametrize("value", [False, None, "true", 1])
def test_readiness_rejects_nonsynthetic_mode(
    readiness_kwargs: dict[str, Any], value: object
) -> None:
    with pytest.raises(holdout.HoldoutError):
        holdout.ZeroOutcomeReadinessInspector.inspect(**dict(readiness_kwargs, synthetic=value))


def test_exact_original_row_purge_and_strict_label_exit(
    boundary: holdout.BoundaryPurgePlan,
) -> None:
    assert boundary.development_cutoff_ordinal == CUTOFF
    assert tuple(boundary.purge_ordinals) == PURGE
    assert boundary.first_holdout_ordinal == CUTOFF + 6


@pytest.mark.parametrize(
    "purge,first",
    [
        ((101, 102, 103, 104), 106),
        ((101, 102, 103, 104, 106), 107),
        ((101, 102, 103, 104, 104), 106),
        ((102, 103, 104, 105, 106), 107),
        ((101, 103, 104, 105, 106), 107),
    ],
)
def test_purge_is_exact_original_five_ordinals_not_retained_row_count(
    boundary_kwargs: dict[str, Any], purge: tuple[int, ...], first: int
) -> None:
    with pytest.raises(holdout.HoldoutError):
        holdout.BoundaryPurgeManager.validate(
            **dict(boundary_kwargs, purge_ordinals=purge, first_holdout_ordinal=first)
        )


@pytest.mark.parametrize(
    "mutation", ["decision_in_purge", "exit_in_holdout", "late_exit", "bad_horizon"]
)
def test_development_labels_never_touch_holdout_prices(
    boundary_kwargs: dict[str, Any], mutation: str
) -> None:
    labels = list(boundary_kwargs["development_labels"])
    if mutation == "decision_in_purge":
        labels.append(holdout.DevelopmentLabel(101, 106, MARKET_START + 106 * HOUR))
    elif mutation == "exit_in_holdout":
        labels[-1] = replace(labels[-1], exit_ordinal=106)
    elif mutation == "late_exit":
        labels[-1] = replace(labels[-1], exit_at=boundary_kwargs["first_holdout_decision_at"])
    else:
        labels[-1] = replace(labels[-1], exit_ordinal=104)
    with pytest.raises(holdout.HoldoutError):
        holdout.BoundaryPurgeManager.validate(**dict(boundary_kwargs, development_labels=labels))


@pytest.mark.parametrize("mutation", ["early_cutoff_exit", "late_prior_exit"])
def test_development_exit_timestamp_must_match_original_hourly_ordinal(
    boundary_kwargs: dict[str, Any], mutation: str
) -> None:
    labels = list(boundary_kwargs["development_labels"])
    if mutation == "early_cutoff_exit":
        labels[-1] = replace(labels[-1], exit_at=labels[-1].exit_at - HOUR)
    else:
        labels[0] = replace(labels[0], exit_at=labels[0].exit_at + timedelta(minutes=30))
    # Both altered exits remain strictly before the first holdout decision;
    # chronology alone cannot detect their wrong original-hour mapping.
    assert max(label.exit_at for label in labels) < boundary_kwargs["first_holdout_decision_at"]
    with pytest.raises(holdout.HoldoutBoundaryError):
        holdout.BoundaryPurgeManager.validate(**dict(boundary_kwargs, development_labels=labels))


@pytest.mark.parametrize(
    "mutation",
    [
        "second_augmented_fit",
        "second_control_fit",
        "wrong_cutoff",
        "post_holdout_freeze",
        "wrong_pair",
        "bad_hash",
        "real_fit",
    ],
)
def test_frozen_pair_is_single_fit_preclaim_and_development_bound(
    boundary_kwargs: dict[str, Any], mutation: str
) -> None:
    fit = boundary_kwargs["frozen_fit"]
    if mutation == "second_augmented_fit":
        fit = replace(fit, augmented_fit_count=2)
    elif mutation == "second_control_fit":
        fit = replace(fit, control_fit_count=2)
    elif mutation == "wrong_cutoff":
        fit = replace(fit, development_cutoff_ordinal=CUTOFF - 1)
    elif mutation == "post_holdout_freeze":
        fit = replace(fit, frozen_at=boundary_kwargs["first_holdout_decision_at"])
    elif mutation == "wrong_pair":
        fit = replace(fit, matched_control_cell="B")
    elif mutation == "bad_hash":
        fit = replace(fit, augmented_model_sha256="not-a-sha256")
    else:
        fit = replace(fit, synthetic=False)
    with pytest.raises(holdout.HoldoutError):
        holdout.BoundaryPurgeManager.validate(**dict(boundary_kwargs, frozen_fit=fit))


@pytest.mark.parametrize(
    "mutation",
    ["augmented_bytes", "control_bytes", "noncanonical_manifest", "forged_manifest_count"],
)
def test_frozen_fit_exact_bytes_and_manifest_semantics_are_verified(
    boundary_kwargs: dict[str, Any], mutation: str
) -> None:
    fit = boundary_kwargs["frozen_fit"]
    if mutation == "augmented_bytes":
        fit = replace(fit, augmented_model_bytes=fit.augmented_model_bytes + b"tamper")
    elif mutation == "control_bytes":
        fit = replace(fit, control_model_bytes=fit.control_model_bytes + b"tamper")
    elif mutation == "noncanonical_manifest":
        altered = fit.fit_manifest_bytes + b" "
        fit = replace(
            fit,
            fit_manifest_bytes=altered,
            fit_manifest_sha256=sha256_bytes(altered),
        )
    else:
        altered_value = json.loads(fit.fit_manifest_bytes)
        altered_value["augmented_fit_count"] = 2
        altered = canonicalize(altered_value)
        fit = replace(
            fit,
            fit_manifest_bytes=altered,
            fit_manifest_sha256=sha256_bytes(altered),
        )
    with pytest.raises(holdout.HoldoutError):
        holdout.BoundaryPurgeManager.validate(**dict(boundary_kwargs, frozen_fit=fit))


def test_claim_is_canonical_exclusive_and_binds_frozen_proofs(
    tmp_path: Path, boundary: holdout.BoundaryPurgePlan, readiness: holdout.ReadinessReport
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    manager = holdout.HoldoutClaimManager(run_dir)
    manager.acquire(boundary=boundary, readiness=readiness, **CLAIM_HASHES)
    path = run_dir / "holdout_evaluation_claim.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == canonicalize(value)
    assert all(digest in raw.decode() for digest in CLAIM_HASHES.values())
    assert boundary.frozen_fit.augmented_model_sha256 in raw.decode()
    assert boundary.frozen_fit.control_model_sha256 in raw.decode()
    assert b"planned_minimum_days" in raw
    assert b"purge_ordinals" in raw
    assert b"development_cutoff_ordinal" in raw
    with pytest.raises(holdout.HoldoutError):
        manager.acquire(boundary=boundary, readiness=readiness, **CLAIM_HASHES)
    assert path.read_bytes() == raw


def test_failed_evaluation_cannot_retry_consumed_claim(
    tmp_path: Path, boundary: holdout.BoundaryPurgePlan, readiness: holdout.ReadinessReport
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    manager = holdout.HoldoutClaimManager(run_dir)
    with pytest.raises(TimeoutError):
        manager.acquire(boundary=boundary, readiness=readiness, **CLAIM_HASHES)
        raise TimeoutError("synthetic evaluator interruption after claim")
    assert (run_dir / "holdout_evaluation_claim.json").is_file()
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=readiness, **CLAIM_HASHES
        )


def test_claim_race_has_one_winner_and_no_replacement(
    tmp_path: Path, boundary: holdout.BoundaryPurgePlan, readiness: holdout.ReadinessReport
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()

    def contender(_: int) -> bool:
        try:
            holdout.HoldoutClaimManager(run_dir).acquire(
                boundary=boundary, readiness=readiness, **CLAIM_HASHES
            )
        except holdout.HoldoutError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(contender, range(8)))
    assert results.count(True) == 1
    assert results.count(False) == 7
    assert (run_dir / "holdout_evaluation_claim.json").is_file()


def test_preexisting_claim_symlink_is_never_followed_or_replaced(
    tmp_path: Path, boundary: holdout.BoundaryPurgePlan, readiness: holdout.ReadinessReport
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    target = tmp_path / "unrelated-synthetic-bytes"
    target.write_bytes(b"preserve")
    (run_dir / "holdout_evaluation_claim.json").symlink_to(target)
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=readiness, **CLAIM_HASHES
        )
    assert target.read_bytes() == b"preserve"


def test_claim_fsyncs_file_then_parent_directory(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    events: list[str] = []
    original_fsync = os.fsync

    def observed_fsync(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        events.append("file" if stat.S_ISREG(mode) else "directory")
        original_fsync(fd)

    monkeypatch.setattr(holdout.os, "fsync", observed_fsync)
    holdout.HoldoutClaimManager(run_dir).acquire(
        boundary=boundary, readiness=readiness, **CLAIM_HASHES
    )
    assert events[:2] == ["file", "directory"]


def test_fsync_failure_still_consumes_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    original_fsync = os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("synthetic directory fsync failure")
        original_fsync(fd)

    monkeypatch.setattr(holdout.os, "fsync", fail_directory_fsync)
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=readiness, **CLAIM_HASHES
        )
    assert (run_dir / "holdout_evaluation_claim.json").exists()
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=readiness, **CLAIM_HASHES
        )


@pytest.mark.parametrize("field", tuple(CLAIM_HASHES))
def test_bad_claim_binding_fails_before_creating_a_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
    field: str,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    invalid = dict(CLAIM_HASHES, **{field: "not-a-hash"})
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=readiness, **invalid
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "second_fit",
        "bad_model_hash",
        "wrong_purge",
        "late_first_holdout",
        "leaking_exit",
        "wrong_cutoff",
    ],
)
def test_forged_boundary_plan_cannot_acquire_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
    mutation: str,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    if mutation == "second_fit":
        forged = replace(
            boundary,
            frozen_fit=replace(boundary.frozen_fit, augmented_fit_count=2),
        )
    elif mutation == "bad_model_hash":
        forged = replace(
            boundary,
            frozen_fit=replace(boundary.frozen_fit, control_model_sha256="f" * 63),
        )
    elif mutation == "wrong_purge":
        forged = replace(boundary, purge_ordinals=(101, 102, 103, 104, 106))
    elif mutation == "late_first_holdout":
        forged = replace(boundary, first_holdout_ordinal=107)
    elif mutation == "leaking_exit":
        forged = replace(
            boundary,
            last_development_exit_at=boundary.first_holdout_decision_at,
        )
    else:
        forged = replace(boundary, development_cutoff_ordinal=CUTOFF - 1)
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=forged, readiness=readiness, **CLAIM_HASHES
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()


@pytest.mark.parametrize(
    "mutation", ["ready_flag", "trade_boolean", "planned_days", "plan_hash", "elapsed_days"]
)
def test_forged_readiness_report_cannot_acquire_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
    readiness_kwargs: dict[str, Any],
    mutation: str,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    if mutation == "ready_flag":
        not_ready = holdout.ZeroOutcomeReadinessInspector.inspect(
            **dict(readiness_kwargs, scheduled_exits=readiness_kwargs["scheduled_exits"][:-1])
        )
        forged = replace(not_ready, ready=True, trade_threshold_met=True)
    elif mutation == "trade_boolean":
        forged = replace(readiness, trade_threshold_met=False)
    elif mutation == "planned_days":
        forged = replace(readiness, planned_minimum_days=179)
    elif mutation == "plan_hash":
        forged = replace(readiness, plan_sha256="f" * 64)
    else:
        forged = replace(readiness, elapsed_days=181.0)
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=forged, **CLAIM_HASHES
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()


def test_in_place_mutation_of_issued_not_ready_report_cannot_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness_kwargs: dict[str, Any],
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    not_ready = holdout.ZeroOutcomeReadinessInspector.inspect(
        **dict(readiness_kwargs, scheduled_exits=readiness_kwargs["scheduled_exits"][:-1])
    )
    assert not_ready.to_dict()["ready"] is False
    # Frozen dataclasses are not a security boundary: object.__setattr__ can
    # alter the same identity kept in an issuance registry.
    object.__setattr__(not_ready, "ready", True)
    object.__setattr__(not_ready, "trade_threshold_met", True)
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=not_ready, **CLAIM_HASHES
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()


def test_in_place_mutation_of_validated_boundary_cannot_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    object.__setattr__(boundary, "first_holdout_ordinal", boundary.first_holdout_ordinal + 1)
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=readiness, **CLAIM_HASHES
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()


@pytest.mark.parametrize("payload", ["model", "manifest"])
def test_in_place_mutation_of_frozen_artifact_bytes_cannot_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
    payload: str,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    if payload == "model":
        object.__setattr__(
            boundary.frozen_fit,
            "augmented_model_bytes",
            boundary.frozen_fit.augmented_model_bytes + b"tamper",
        )
    else:
        object.__setattr__(
            boundary.frozen_fit,
            "fit_manifest_bytes",
            boundary.frozen_fit.fit_manifest_bytes + b"tamper",
        )
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=readiness, **CLAIM_HASHES
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()


def test_symlinked_run_directory_ancestor_is_rejected(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness: holdout.ReadinessReport,
) -> None:
    real_parent = tmp_path / "real-parent"
    run_dir = real_parent / "synthetic-evaluation-run"
    run_dir.mkdir(parents=True)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(alias_parent / run_dir.name).acquire(
            boundary=boundary, readiness=readiness, **CLAIM_HASHES
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()


def test_non_temp_production_tree_is_outside_offline_claim_authority() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    with pytest.raises(holdout.HoldoutAuthorizationError):
        holdout.HoldoutClaimManager(repo_root)
    assert not (repo_root / "holdout_evaluation_claim.json").exists()


def test_not_ready_cannot_claim_or_read_synthetic_outcome_file(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness_kwargs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    outcome = run_dir / "future_returns.json"
    outcome.write_bytes(b'{"forward_return":999}')
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == outcome:
            raise AssertionError("outcome-bearing bytes were read before claim")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    not_ready = holdout.ZeroOutcomeReadinessInspector.inspect(
        **dict(readiness_kwargs, scheduled_exits=readiness_kwargs["scheduled_exits"][:-1])
    )
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=not_ready, **CLAIM_HASHES
        )
    assert not (run_dir / "holdout_evaluation_claim.json").exists()
    assert original_read_bytes(outcome) == b'{"forward_return":999}'


@pytest.mark.parametrize("change", ["ordinal", "time"])
def test_claim_rejects_readiness_from_another_synthetic_window(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness_kwargs: dict[str, Any],
    change: str,
) -> None:
    evidence = dict(readiness_kwargs)
    if change == "ordinal":
        evidence["market_first_ordinal"] += 1
        evidence["first_holdout_ordinal"] += 1
        evidence["market_last_ordinal"] += 1
        evidence["candles"] = tuple(
            replace(candle, ordinal=candle.ordinal + 1) for candle in evidence["candles"]
        )
        evidence["scheduled_exits"] = tuple(
            replace(
                exit_record,
                decision_ordinal=exit_record.decision_ordinal + 1,
                ordinal=exit_record.ordinal + 1,
            )
            for exit_record in evidence["scheduled_exits"]
        )
    else:
        evidence["holdout_started_at"] += HOUR
        evidence["inspected_at"] += HOUR
        evidence["candles"] = tuple(
            replace(candle, opened_at=candle.opened_at + HOUR, closed_at=candle.closed_at + HOUR)
            for candle in evidence["candles"]
        )
        evidence["scheduled_exits"] = tuple(
            replace(exit_record, exit_at=exit_record.exit_at + HOUR)
            for exit_record in evidence["scheduled_exits"]
        )
    report = holdout.ZeroOutcomeReadinessInspector.inspect(**evidence)
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    with pytest.raises(holdout.HoldoutError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=report, **CLAIM_HASHES
        )
    assert not (run_dir / holdout.CLAIM_NAME).exists()


def test_oof_span_touching_purge_cannot_support_holdout_claim(
    tmp_path: Path,
    boundary: holdout.BoundaryPurgePlan,
    readiness_kwargs: dict[str, Any],
    oof_spans: tuple[holdout.OofSpan, ...],
) -> None:
    purge_decision_at = boundary.first_holdout_decision_at - 5 * HOUR
    spans = list(oof_spans)
    spans[-1] = holdout.OofSpan(
        first_test_decision_at=purge_decision_at - DAY + HOUR,
        last_test_decision_at=purge_decision_at,
    )
    plan = holdout.ZeroOutcomeReadinessInspector.plan(spans, 50, frozen_at=PLAN_FROZEN_AT)
    report = holdout.ZeroOutcomeReadinessInspector.inspect(**dict(readiness_kwargs, plan=plan))
    assert report.to_dict()["ready"] is True
    run_dir = tmp_path / "synthetic-evaluation-run"
    run_dir.mkdir()
    with pytest.raises(holdout.HoldoutClaimError):
        holdout.HoldoutClaimManager(run_dir).acquire(
            boundary=boundary, readiness=report, **CLAIM_HASHES
        )
    assert not (run_dir / holdout.CLAIM_NAME).exists()


def test_readiness_rejects_candles_from_a_different_calendar_window(
    readiness_kwargs: dict[str, Any],
) -> None:
    shifted_candles = tuple(
        replace(candle, opened_at=candle.opened_at + HOUR, closed_at=candle.closed_at + HOUR)
        for candle in readiness_kwargs["candles"]
    )
    with pytest.raises(holdout.HoldoutReadinessError):
        holdout.ZeroOutcomeReadinessInspector.inspect(
            **dict(
                readiness_kwargs,
                candles=shifted_candles,
                inspected_at=HOLDOUT_START + 180 * DAY + HOUR,
            )
        )
