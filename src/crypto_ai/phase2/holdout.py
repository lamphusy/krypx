"""Offline synthetic Milestone 8 claim, readiness, and boundary-purge contracts.

This module has no model-fitting, transport, market-price, return, or evaluation
path. Its only publication is an irreversible claim in a temporary synthetic run.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment.canonical import MAX_SAFE_INTEGER, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp
from crypto_ai.sentiment.storage import (
    _fsync_directory_descriptor,
    _open_directory_path,
    _read_regular_file_at_once,
    _require_descriptor_relative_mutations,
    _stat_identity,
)

SPECIFICATION_ID = "phase2-milestone8-future-holdout-v1"
CLAIM_SCHEMA = "phase2-synthetic-holdout-claim-v1"
FIT_SCHEMA = "phase2-synthetic-frozen-fit-v1"
CLAIM_NAME = "holdout_evaluation_claim.json"
OOF_FOLDS = 5
PURGE_ROWS = 5
HORIZON = 4
MINIMUM_DAYS = 180
MINIMUM_TRADES = 50
ONE_HOUR = timedelta(hours=1)
MICROSECONDS_PER_DAY = 86_400_000_000
TECHNICAL_CONTEXT_ROWS = 34
HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


class HoldoutError(CryptoAIError):
    """The offline synthetic holdout contract failed."""


class HoldoutInputError(HoldoutError):
    """An input violates the frozen Milestone 8 schema."""


class HoldoutReadinessError(HoldoutError):
    """Operational readiness is incomplete or cannot be verified."""


class HoldoutBoundaryError(HoldoutError):
    """The development/holdout boundary is not causally separated."""


class HoldoutClaimError(HoldoutError):
    """An exclusive claim cannot be created or has already been consumed."""


class HoldoutAuthorizationError(HoldoutError):
    """The request exceeds the synthetic-only implementation authority."""


def _sha256(value: object) -> bool:
    return type(value) is str and HASH_PATTERN.fullmatch(value) is not None


def _instant(value: object, *, field: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise HoldoutInputError(f"{field} must be an aware UTC datetime")
    return value


def _ordinal(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0 or value > MAX_SAFE_INTEGER:
        raise HoldoutInputError(f"{field} must be a non-negative original market ordinal")
    return value


@dataclass(frozen=True, slots=True)
class OofSpan:
    first_test_decision_at: datetime
    last_test_decision_at: datetime


@dataclass(frozen=True, slots=True)
class ReadinessPlan:
    oof_spans: tuple[OofSpan, ...]
    oof_completed_trades: int
    frozen_at: datetime
    oof_elapsed_days: float
    trade_rate_per_day: float
    planned_minimum_days: int
    sha256: str
    synthetic: bool = True


@dataclass(frozen=True, slots=True)
class ClosedCandle:
    ordinal: int
    opened_at: datetime
    closed_at: datetime
    closed: bool = True
    authentic: bool = True
    synthetic: bool = True


@dataclass(frozen=True, slots=True)
class RawSnapshot:
    raw_bytes: bytes
    expected_sha256: str
    synthetic: bool = True


@dataclass(frozen=True, slots=True)
class ScheduledExit:
    """Outcome-free proof that a frozen-policy position reached its exit open."""

    decision_ordinal: int
    ordinal: int
    exit_at: datetime
    policy_sha256: str
    synthetic: bool = True


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class ReadinessReport:
    ready: bool
    elapsed_days: float
    trade_threshold_met: bool
    provider_outage_state: str
    planned_minimum_days: int
    synthetic: bool
    plan_sha256: str
    last_oof_decision_at: datetime
    market_first_ordinal: int
    first_holdout_ordinal: int
    market_last_ordinal: int
    frozen_policy_sha256: str
    holdout_started_at: datetime
    inspected_at: datetime
    operational_evidence_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Expose only the frozen, outcome-free operational reporting surface."""
        return {
            "ready": self.ready,
            "elapsed_days": self.elapsed_days,
            "trade_threshold_met": self.trade_threshold_met,
            "provider_outage_state": self.provider_outage_state,
            "planned_minimum_days": self.planned_minimum_days,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentLabel:
    decision_ordinal: int
    exit_ordinal: int
    exit_at: datetime


@dataclass(frozen=True, slots=True)
class FrozenFitProof:
    """Synthetic-only frozen-fit declaration; never a real fitted-model permit."""

    selected_augmented_cell: str
    matched_control_cell: str
    development_cutoff_ordinal: int
    shared_labeled_rows_sha256: str
    augmented_model_sha256: str
    control_model_sha256: str
    fit_manifest_sha256: str
    augmented_fit_count: int
    control_fit_count: int
    frozen_at: datetime
    fit_manifest_bytes: bytes
    augmented_model_bytes: bytes
    control_model_bytes: bytes
    synthetic: bool = True


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class BoundaryPurgePlan:
    development_cutoff_ordinal: int
    purge_ordinals: tuple[int, int, int, int, int]
    first_holdout_ordinal: int
    first_holdout_decision_at: datetime
    last_development_exit_at: datetime
    frozen_fit: FrozenFitProof
    development_labels: tuple[DevelopmentLabel, ...]
    synthetic: bool = True


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    path: Path
    sha256: str
    raw_bytes: bytes


_issued_readiness_reports: weakref.WeakKeyDictionary[ReadinessReport, tuple[object, ...]] = (
    weakref.WeakKeyDictionary()
)
_issued_boundaries: weakref.WeakKeyDictionary[BoundaryPurgePlan, tuple[object, ...]] = (
    weakref.WeakKeyDictionary()
)


def _readiness_snapshot(report: ReadinessReport) -> tuple[object, ...]:
    return (
        report.ready,
        report.elapsed_days,
        report.trade_threshold_met,
        report.provider_outage_state,
        report.planned_minimum_days,
        report.synthetic,
        report.plan_sha256,
        report.last_oof_decision_at,
        report.market_first_ordinal,
        report.first_holdout_ordinal,
        report.market_last_ordinal,
        report.frozen_policy_sha256,
        report.holdout_started_at,
        report.inspected_at,
        report.operational_evidence_sha256,
    )


def _boundary_snapshot(plan: BoundaryPurgePlan) -> tuple[object, ...]:
    fit = plan.frozen_fit
    return (
        plan.development_cutoff_ordinal,
        plan.purge_ordinals,
        plan.first_holdout_ordinal,
        plan.first_holdout_decision_at,
        plan.last_development_exit_at,
        tuple(
            (label.decision_ordinal, label.exit_ordinal, label.exit_at)
            for label in plan.development_labels
        ),
        plan.synthetic,
        fit.selected_augmented_cell,
        fit.matched_control_cell,
        fit.development_cutoff_ordinal,
        fit.shared_labeled_rows_sha256,
        fit.augmented_model_sha256,
        fit.control_model_sha256,
        fit.fit_manifest_sha256,
        fit.augmented_fit_count,
        fit.control_fit_count,
        fit.frozen_at,
        fit.synthetic,
        sha256_bytes(fit.fit_manifest_bytes),
        sha256_bytes(fit.augmented_model_bytes),
        sha256_bytes(fit.control_model_bytes),
    )


class ZeroOutcomeReadinessInspector:
    """Inspect only allowlisted operational metadata; never prices or outcomes."""

    @staticmethod
    def plan(
        oof_spans: Sequence[OofSpan], oof_completed_trades: int, *, frozen_at: datetime
    ) -> ReadinessPlan:
        if (
            not isinstance(oof_spans, Sequence)
            or isinstance(oof_spans, (str, bytes, bytearray))
            or len(oof_spans) != OOF_FOLDS
            or any(type(span) is not OofSpan for span in oof_spans)
        ):
            raise HoldoutInputError("exactly five synthetic OOF spans are required")
        if (
            type(oof_completed_trades) is not int
            or not 0 < oof_completed_trades <= MAX_SAFE_INTEGER
        ):
            raise HoldoutReadinessError("development OOF completed trades must be positive")
        frozen = _instant(frozen_at, field="readiness plan freeze")

        intervals: list[tuple[datetime, datetime]] = []
        for span in oof_spans:
            first = _instant(span.first_test_decision_at, field="first OOF decision")
            last = _instant(span.last_test_decision_at, field="last OOF decision")
            if last < first:
                raise HoldoutInputError("OOF span end precedes its first decision")
            if last >= frozen:
                raise HoldoutReadinessError("readiness plan must freeze after OOF development")
            try:
                intervals.append((first - ONE_HOUR, last))
            except OverflowError as exc:
                raise HoldoutInputError(
                    "OOF span start is outside the UTC timestamp range"
                ) from exc
        intervals.sort()
        merged: list[tuple[datetime, datetime]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        elapsed_microseconds = sum(
            (duration.days * 86400 + duration.seconds) * 1_000_000 + duration.microseconds
            for start, end in merged
            for duration in (end - start,)
        )
        elapsed_days = elapsed_microseconds / MICROSECONDS_PER_DAY
        if not math.isfinite(elapsed_days) or elapsed_days <= 0:
            raise HoldoutReadinessError("OOF elapsed calendar days must be positive and finite")
        q = oof_completed_trades / elapsed_days
        if not math.isfinite(q) or q <= 0:
            raise HoldoutReadinessError("OOF completed-trade rate must be positive and finite")
        denominator = oof_completed_trades * MICROSECONDS_PER_DAY
        required_days = max(
            MINIMUM_DAYS,
            (MINIMUM_TRADES * elapsed_microseconds + denominator - 1) // denominator,
        )
        if required_days > MAX_SAFE_INTEGER:
            raise HoldoutReadinessError("planned duration exceeds the canonical integer range")
        spans = tuple(oof_spans)
        identity = {
            "specification_id": SPECIFICATION_ID,
            "oof_spans": [
                {
                    "first_test_decision_at": format_utc_timestamp(span.first_test_decision_at),
                    "last_test_decision_at": format_utc_timestamp(span.last_test_decision_at),
                }
                for span in spans
            ],
            "oof_completed_trades": oof_completed_trades,
            "frozen_at": format_utc_timestamp(frozen),
            "oof_elapsed_days": elapsed_days,
            "trade_rate_per_day": q,
            "planned_minimum_days": required_days,
        }
        try:
            digest = sha256_bytes(canonicalize(identity))
        except (CryptoAIError, TypeError, ValueError) as exc:
            raise HoldoutInputError("OOF readiness plan is not canonicalizable") from exc
        return ReadinessPlan(
            spans, oof_completed_trades, frozen, elapsed_days, q, required_days, digest
        )

    @staticmethod
    def inspect(
        *,
        plan: ReadinessPlan,
        market_first_ordinal: int,
        first_holdout_ordinal: int,
        market_last_ordinal: int,
        candles: Sequence[ClosedCandle],
        raw_snapshots: Sequence[RawSnapshot],
        provider_outage_state: str,
        provider_gap_exclusions_verified: bool,
        holdout_started_at: datetime,
        inspected_at: datetime,
        scheduled_exits: Sequence[ScheduledExit],
        frozen_policy_sha256: str,
        synthetic: bool = True,
    ) -> ReadinessReport:
        if synthetic is not True or type(plan) is not ReadinessPlan or plan.synthetic is not True:
            raise HoldoutAuthorizationError("readiness is limited to synthetic fixtures")
        if plan != ZeroOutcomeReadinessInspector.plan(
            plan.oof_spans, plan.oof_completed_trades, frozen_at=plan.frozen_at
        ):
            raise HoldoutReadinessError("OOF readiness plan does not replay deterministically")
        first = _ordinal(market_first_ordinal, field="first market ordinal")
        first_holdout = _ordinal(first_holdout_ordinal, field="first holdout ordinal")
        last = _ordinal(market_last_ordinal, field="last market ordinal")
        if last < first_holdout or first_holdout - first < TECHNICAL_CONTEXT_ROWS - 1:
            raise HoldoutReadinessError("market context has an inverted ordinal range")
        if (
            not isinstance(candles, Sequence)
            or isinstance(candles, (str, bytes, bytearray))
            or len(candles) != last - first + 1
        ):
            raise HoldoutReadinessError("authentic hourly market context is incomplete")
        start = _instant(holdout_started_at, field="holdout start")
        now = _instant(inspected_at, field="inspection time")
        if now < start:
            raise HoldoutReadinessError("inspection precedes the prospective start")
        if plan.frozen_at >= start:
            raise HoldoutReadinessError("readiness plan was not frozen before accumulation")
        if not _sha256(frozen_policy_sha256):
            raise HoldoutInputError("frozen policy identity must be a SHA-256 digest")
        for index, candle in enumerate(candles):
            if type(candle) is not ClosedCandle or candle.synthetic is not True:
                raise HoldoutAuthorizationError("only synthetic closed-candle metadata is allowed")
            opened = _instant(candle.opened_at, field="market candle open")
            closed = _instant(candle.closed_at, field="market candle close")
            if (
                type(candle.ordinal) is not int
                or candle.ordinal != first + index
                or candle.closed is not True
                or candle.authentic is not True
                or opened.minute != 0
                or opened.second != 0
                or opened.microsecond != 0
                or closed - opened != ONE_HOUR
                or closed > now
            ):
                raise HoldoutReadinessError("market context is not 100% closed and authentic")
            if index and opened != candles[index - 1].opened_at + ONE_HOUR:
                raise HoldoutReadinessError("market candles are not hourly-contiguous")
        if candles[first_holdout - first].opened_at != start:
            raise HoldoutReadinessError("market context starts outside the holdout window")
        if now - candles[-1].closed_at < timedelta(0) or now - candles[-1].closed_at >= ONE_HOUR:
            raise HoldoutReadinessError("market context does not cover the inspection time")
        if not isinstance(scheduled_exits, Sequence) or isinstance(
            scheduled_exits, (str, bytes, bytearray)
        ):
            raise HoldoutInputError("scheduled exits must be an outcome-free sequence")
        seen_exits: set[int] = set()
        completed_positions: list[tuple[int, int]] = []
        for exit_record in scheduled_exits:
            if type(exit_record) is not ScheduledExit or exit_record.synthetic is not True:
                raise HoldoutAuthorizationError("only synthetic scheduled exits are allowed")
            exit_ordinal = _ordinal(exit_record.ordinal, field="scheduled exit ordinal")
            decision_ordinal = _ordinal(
                exit_record.decision_ordinal, field="scheduled decision ordinal"
            )
            exit_at = _instant(exit_record.exit_at, field="scheduled exit timestamp")
            if (
                exit_ordinal in seen_exits
                or decision_ordinal < first_holdout
                or exit_ordinal != decision_ordinal + HORIZON + 1
                or exit_ordinal < first_holdout + HORIZON + 1
                or exit_ordinal > last
                or exit_at != candles[exit_ordinal - first].opened_at
                or exit_at > now
                or exit_record.policy_sha256 != frozen_policy_sha256
            ):
                raise HoldoutReadinessError("scheduled exit is not bound to the frozen window")
            seen_exits.add(exit_ordinal)
            completed_positions.append((decision_ordinal, exit_ordinal))
        completed_positions.sort()
        if any(
            next_decision < previous_exit
            for (_, previous_exit), (next_decision, _) in zip(
                completed_positions, completed_positions[1:], strict=False
            )
        ):
            raise HoldoutReadinessError("scheduled exits contain overlapping positions")
        if (
            not isinstance(raw_snapshots, Sequence)
            or isinstance(raw_snapshots, (str, bytes, bytearray))
            or not raw_snapshots
        ):
            raise HoldoutReadinessError("raw snapshot evidence is missing")
        for snapshot in raw_snapshots:
            if type(snapshot) is not RawSnapshot or snapshot.synthetic is not True:
                raise HoldoutAuthorizationError("only synthetic raw snapshots are allowed")
            if (
                type(snapshot.raw_bytes) is not bytes
                or not _sha256(snapshot.expected_sha256)
                or sha256_bytes(snapshot.raw_bytes) != snapshot.expected_sha256
            ):
                raise HoldoutReadinessError("raw snapshot SHA-256 integrity failed")
        if type(provider_outage_state) is not str or provider_outage_state not in {
            "CLEAR",
            "VERIFIED_GAP",
            "UNRESOLVED",
        }:
            raise HoldoutInputError("unknown provider outage state")
        if type(provider_gap_exclusions_verified) is not bool:
            raise HoldoutInputError("provider-gap verification must be a boolean")
        if provider_outage_state == "UNRESOLVED" or (
            provider_outage_state == "VERIFIED_GAP" and not provider_gap_exclusions_verified
        ):
            raise HoldoutReadinessError("provider outage evidence is unresolved")
        elapsed_duration = now - start
        elapsed_microseconds = (
            elapsed_duration.days * 86400 + elapsed_duration.seconds
        ) * 1_000_000 + elapsed_duration.microseconds
        elapsed_days = elapsed_microseconds / MICROSECONDS_PER_DAY
        if not math.isfinite(elapsed_days):
            raise HoldoutReadinessError("elapsed calendar duration is non-finite")
        threshold_met = len(seen_exits) >= MINIMUM_TRADES
        ready = (
            elapsed_microseconds >= plan.planned_minimum_days * MICROSECONDS_PER_DAY
            and threshold_met
        )
        operational_evidence = {
            "specification_id": SPECIFICATION_ID,
            "synthetic": True,
            "readiness_plan_sha256": plan.sha256,
            "last_oof_decision_at": format_utc_timestamp(
                max(span.last_test_decision_at for span in plan.oof_spans)
            ),
            "market_first_ordinal": first,
            "first_holdout_ordinal": first_holdout,
            "market_last_ordinal": last,
            "frozen_policy_sha256": frozen_policy_sha256,
            "market_candles": [
                {
                    "ordinal": candle.ordinal,
                    "opened_at": format_utc_timestamp(candle.opened_at),
                    "closed_at": format_utc_timestamp(candle.closed_at),
                }
                for candle in candles
            ],
            "raw_snapshot_sha256": [snapshot.expected_sha256 for snapshot in raw_snapshots],
            "provider_outage_state": provider_outage_state,
            "provider_gap_exclusions_verified": provider_gap_exclusions_verified,
            "holdout_started_at": format_utc_timestamp(start),
            "inspected_at": format_utc_timestamp(now),
            "trade_threshold_met": threshold_met,
            "scheduled_exit_ordinals": sorted(seen_exits),
        }
        try:
            evidence_sha256 = sha256_bytes(canonicalize(operational_evidence))
        except (CryptoAIError, TypeError, ValueError) as exc:
            raise HoldoutReadinessError("operational evidence cannot be canonically bound") from exc
        report = ReadinessReport(
            ready,
            elapsed_days,
            threshold_met,
            provider_outage_state,
            plan.planned_minimum_days,
            True,
            plan.sha256,
            max(span.last_test_decision_at for span in plan.oof_spans),
            first,
            first_holdout,
            last,
            frozen_policy_sha256,
            start,
            now,
            evidence_sha256,
        )
        _issued_readiness_reports[report] = _readiness_snapshot(report)
        return report


class BoundaryPurgeManager:
    """Validate original-ordinal purge and synthetic frozen-fit chronology."""

    @staticmethod
    def validate(
        *,
        development_cutoff_ordinal: int,
        development_labels: Sequence[DevelopmentLabel],
        purge_ordinals: Sequence[int],
        first_holdout_ordinal: int,
        first_holdout_decision_at: datetime,
        frozen_fit: FrozenFitProof,
    ) -> BoundaryPurgePlan:
        cutoff = _ordinal(development_cutoff_ordinal, field="development cutoff")
        first_holdout = _ordinal(first_holdout_ordinal, field="first holdout ordinal")
        holdout_at = _instant(first_holdout_decision_at, field="first holdout decision")
        if holdout_at.minute or holdout_at.second or holdout_at.microsecond:
            raise HoldoutBoundaryError("first holdout decision must close an hourly candle")
        if first_holdout != cutoff + PURGE_ROWS + 1:
            raise HoldoutBoundaryError("first holdout decision must be d+6")
        expected_purge = tuple(range(cutoff + 1, cutoff + PURGE_ROWS + 1))
        if (
            not isinstance(purge_ordinals, Sequence)
            or isinstance(purge_ordinals, (str, bytes, bytearray))
            or tuple(purge_ordinals) != expected_purge
        ):
            raise HoldoutBoundaryError("the exact original-ordinal d+1..d+5 purge is required")
        if (
            not isinstance(development_labels, Sequence)
            or isinstance(development_labels, (str, bytes, bytearray))
            or not development_labels
            or any(type(label) is not DevelopmentLabel for label in development_labels)
        ):
            raise HoldoutBoundaryError("labeled development decisions are required")
        seen: set[int] = set()
        exits: list[datetime] = []
        for label in development_labels:
            decision = _ordinal(label.decision_ordinal, field="development decision")
            exit_ordinal = _ordinal(label.exit_ordinal, field="development label exit")
            exit_at = _instant(label.exit_at, field="development label exit timestamp")
            if decision in seen or decision > cutoff or exit_ordinal != decision + HORIZON + 1:
                raise HoldoutBoundaryError("development label ordinals are not unique and causal")
            expected_hours = first_holdout + 1 - exit_ordinal
            elapsed = holdout_at - exit_at
            elapsed_microseconds = (
                elapsed.days * 86400 + elapsed.seconds
            ) * 1_000_000 + elapsed.microseconds
            if elapsed_microseconds != expected_hours * 3_600_000_000:
                raise HoldoutBoundaryError("development label exit timestamp mismatches ordinal")
            seen.add(decision)
            exits.append(exit_at)
        if max(seen) != cutoff or max(exits) >= holdout_at:
            raise HoldoutBoundaryError("development labels do not exit strictly before holdout")
        if type(frozen_fit) is not FrozenFitProof or frozen_fit.synthetic is not True:
            raise HoldoutAuthorizationError("only synthetic frozen-fit proof is supported")
        if (
            frozen_fit.development_cutoff_ordinal != cutoff
            or (frozen_fit.selected_augmented_cell, frozen_fit.matched_control_cell)
            not in {("C", "A"), ("D", "B")}
            or type(frozen_fit.augmented_fit_count) is not int
            or frozen_fit.augmented_fit_count != 1
            or type(frozen_fit.control_fit_count) is not int
            or frozen_fit.control_fit_count != 1
            or any(
                not _sha256(value)
                for value in (
                    frozen_fit.shared_labeled_rows_sha256,
                    frozen_fit.augmented_model_sha256,
                    frozen_fit.control_model_sha256,
                    frozen_fit.fit_manifest_sha256,
                )
            )
        ):
            raise HoldoutBoundaryError("matched single-fit artifacts are not frozen at cutoff")
        frozen_at = _instant(frozen_fit.frozen_at, field="single-fit freeze timestamp")
        if frozen_at >= holdout_at or frozen_at < max(exits):
            raise HoldoutBoundaryError(
                "single-fit freeze must follow development exits and precede holdout"
            )
        if (
            type(frozen_fit.fit_manifest_bytes) is not bytes
            or type(frozen_fit.augmented_model_bytes) is not bytes
            or type(frozen_fit.control_model_bytes) is not bytes
            or not frozen_fit.augmented_model_bytes
            or not frozen_fit.control_model_bytes
            or sha256_bytes(frozen_fit.augmented_model_bytes) != frozen_fit.augmented_model_sha256
            or sha256_bytes(frozen_fit.control_model_bytes) != frozen_fit.control_model_sha256
        ):
            raise HoldoutBoundaryError("frozen synthetic model bytes do not match their hashes")
        expected_manifest = {
            "schema_version": FIT_SCHEMA,
            "specification_id": SPECIFICATION_ID,
            "synthetic": True,
            "selected_augmented_cell": frozen_fit.selected_augmented_cell,
            "matched_control_cell": frozen_fit.matched_control_cell,
            "development_cutoff_ordinal": cutoff,
            "shared_labeled_rows_sha256": frozen_fit.shared_labeled_rows_sha256,
            "augmented_model_sha256": frozen_fit.augmented_model_sha256,
            "control_model_sha256": frozen_fit.control_model_sha256,
            "augmented_fit_count": 1,
            "control_fit_count": 1,
            "frozen_at_utc": format_utc_timestamp(frozen_at),
        }
        try:
            expected_bytes = canonicalize(expected_manifest)
        except (CryptoAIError, TypeError, ValueError) as exc:
            raise HoldoutBoundaryError("frozen-fit manifest is not canonicalizable") from exc
        if (
            frozen_fit.fit_manifest_bytes != expected_bytes
            or sha256_bytes(frozen_fit.fit_manifest_bytes) != frozen_fit.fit_manifest_sha256
        ):
            raise HoldoutBoundaryError("frozen-fit manifest bytes do not bind the model pair")
        boundary = BoundaryPurgePlan(
            cutoff,
            expected_purge,
            first_holdout,
            holdout_at,
            max(exits),
            frozen_fit,
            tuple(development_labels),
        )
        _issued_boundaries[boundary] = _boundary_snapshot(boundary)
        return boundary


class HoldoutClaimManager:
    """Create one durable, no-replace claim in an existing synthetic temp run."""

    def __init__(self, run_dir: Path):
        if not isinstance(run_dir, Path) or not run_dir.is_absolute():
            raise HoldoutInputError("claim run directory must be an absolute Path")
        temp_root = Path(tempfile.gettempdir()).resolve()
        try:
            resolved = run_dir.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise HoldoutClaimError("synthetic claim run directory is unavailable") from exc
        if not resolved.is_relative_to(temp_root) or resolved == temp_root:
            raise HoldoutAuthorizationError("claims are limited to temporary synthetic runs")
        try:
            descriptor = _open_directory_path(run_dir, description="synthetic holdout run")
            self._identity = _stat_identity(os.fstat(descriptor))
            os.close(descriptor)
        except (CryptoAIError, OSError) as exc:
            raise HoldoutClaimError("synthetic claim run directory is unsafe") from exc
        self._run_dir = resolved

    def acquire(
        self,
        *,
        boundary: BoundaryPurgePlan,
        readiness: ReadinessReport,
        protocol_sha256: str,
        input_inventory_sha256: str,
        code_commit: str,
        dependency_lock_sha256: str,
    ) -> ClaimRecord:
        if type(boundary) is not BoundaryPurgePlan or boundary.synthetic is not True:
            raise HoldoutClaimError("validated synthetic boundary evidence is required")
        try:
            if boundary not in _issued_boundaries or _issued_boundaries[
                boundary
            ] != _boundary_snapshot(boundary):
                raise HoldoutClaimError("boundary evidence changed after validation")
        except (TypeError, AttributeError, CryptoAIError) as exc:
            raise HoldoutClaimError("boundary evidence failed pre-claim verification") from exc
        if (
            type(readiness) is not ReadinessReport
            or readiness not in _issued_readiness_reports
            or _issued_readiness_reports[readiness] != _readiness_snapshot(readiness)
            or readiness.synthetic is not True
            or readiness.ready is not True
            or readiness.trade_threshold_met is not True
            or type(readiness.elapsed_days) is not float
            or not math.isfinite(readiness.elapsed_days)
            or type(readiness.planned_minimum_days) is not int
            or readiness.elapsed_days < readiness.planned_minimum_days
            or not _sha256(readiness.plan_sha256)
            or not _sha256(readiness.operational_evidence_sha256)
        ):
            raise HoldoutClaimError("verified zero-outcome readiness is required")
        if (
            readiness.first_holdout_ordinal != boundary.first_holdout_ordinal
            or readiness.holdout_started_at + ONE_HOUR != boundary.first_holdout_decision_at
            or readiness.frozen_policy_sha256 != protocol_sha256
            or readiness.inspected_at <= boundary.frozen_fit.frozen_at
            or readiness.last_oof_decision_at
            > boundary.first_holdout_decision_at - PURGE_ROWS * ONE_HOUR - ONE_HOUR
        ):
            raise HoldoutClaimError("readiness and boundary refer to different holdout windows")
        if (
            any(
                not _sha256(value)
                for value in (protocol_sha256, input_inventory_sha256, dependency_lock_sha256)
            )
            or type(code_commit) is not str
            or COMMIT_PATTERN.fullmatch(code_commit) is None
        ):
            raise HoldoutInputError("claim parent hashes or code commit are invalid")
        fit = boundary.frozen_fit
        if type(fit) is not FrozenFitProof or fit.synthetic is not True:
            raise HoldoutClaimError("synthetic single-fit proof is required")
        payload = {
            "schema_version": CLAIM_SCHEMA,
            "specification_id": SPECIFICATION_ID,
            "synthetic": True,
            "run_id": self._run_dir.name,
            "protocol_sha256": protocol_sha256,
            "input_inventory_sha256": input_inventory_sha256,
            "code_commit": code_commit,
            "dependency_lock_sha256": dependency_lock_sha256,
            "development_cutoff_ordinal": boundary.development_cutoff_ordinal,
            "purge_ordinals": list(boundary.purge_ordinals),
            "first_holdout_ordinal": boundary.first_holdout_ordinal,
            "first_holdout_decision_at": format_utc_timestamp(boundary.first_holdout_decision_at),
            "last_development_exit_at": format_utc_timestamp(boundary.last_development_exit_at),
            "selected_augmented_cell": fit.selected_augmented_cell,
            "matched_control_cell": fit.matched_control_cell,
            "shared_labeled_rows_sha256": fit.shared_labeled_rows_sha256,
            "augmented_model_sha256": fit.augmented_model_sha256,
            "control_model_sha256": fit.control_model_sha256,
            "fit_manifest_sha256": fit.fit_manifest_sha256,
            "frozen_at": format_utc_timestamp(fit.frozen_at),
            "readiness_plan_sha256": readiness.plan_sha256,
            "frozen_policy_sha256": readiness.frozen_policy_sha256,
            "operational_evidence_sha256": readiness.operational_evidence_sha256,
            "readiness": readiness.to_dict(),
            "claimed_at_utc": format_utc_timestamp(datetime.now(UTC)),
        }
        try:
            raw = canonicalize(payload)
            _require_descriptor_relative_mutations()
            run_descriptor = _open_directory_path(
                self._run_dir,
                description="synthetic holdout run",
                expected_identity=self._identity,
            )
        except (CryptoAIError, OSError, ValueError, TypeError) as exc:
            raise HoldoutClaimError("claim preparation failed closed") from exc
        file_descriptor: int | None = None
        created = False
        try:
            try:
                os.stat(CLAIM_NAME, dir_fd=run_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise HoldoutClaimError("holdout claim already exists and is consumed")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            file_descriptor = os.open(CLAIM_NAME, flags, 0o600, dir_fd=run_descriptor)
            created = True
            with os.fdopen(file_descriptor, "wb") as handle:
                file_descriptor = None
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory_descriptor(run_descriptor, description="holdout claim parent")
            captured, _ = _read_regular_file_at_once(
                run_descriptor, CLAIM_NAME, description="exclusive synthetic holdout claim"
            )
            if captured != raw:
                raise HoldoutClaimError("claim bytes changed during exclusive publication")
            current_descriptor = _open_directory_path(
                self._run_dir,
                description="synthetic holdout run after claim",
                expected_identity=self._identity,
            )
            os.close(current_descriptor)
            return ClaimRecord(self._run_dir / CLAIM_NAME, sha256_bytes(raw), raw)
        except HoldoutClaimError:
            raise
        except FileExistsError as exc:
            raise HoldoutClaimError("holdout claim already exists and is consumed") from exc
        except (CryptoAIError, OSError, ValueError, TypeError) as exc:
            raise HoldoutClaimError("exclusive holdout claim failed closed") from exc
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if created:
                try:
                    os.fsync(run_descriptor)
                except OSError:
                    pass  # The claim still exists and must never be retried.
            os.close(run_descriptor)
