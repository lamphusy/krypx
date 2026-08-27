"""Offline GDELT Global Similarity Graph ingestion and normalization.

This module intentionally contains no HTTP client and performs no I/O outside the
caller-supplied immutable store. Retrieval plans and retry decisions are pure data;
only separately authorized code may execute them in a future batch.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

import idna

from crypto_ai.exceptions import (
    ArticleValidationError,
    CanonicalizationError,
    NormalizationIntegrityError,
    ProviderIngestionError,
    PublicationCollisionError,
    SentimentStorageError,
)
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import (
    ArticleRecord,
    derive_article_id,
    derive_article_version_id,
    derive_content_hash,
    derive_duplicate_group_id,
    derive_version_fingerprint,
    format_utc_timestamp,
    parse_utc_timestamp,
    validate_article_collection,
    validate_article_record,
)
from crypto_ai.sentiment.storage import PUBLICATION_SCHEMA_VERSION, ContentAddressedStore

PROVIDER_ID = "gdelt_gsg"
PARSER_VERSION = "gdelt-gsg-jsonl-v1"
PARSER_POLICY_VERSION = "gdelt-gsg-parser-policy-v1"
SNAPSHOT_IDENTITY_VERSION = "gdelt-gsg-snapshot-v2"
NORMALIZER_VERSION = "gdelt-gsg-normalizer-v3"
NORMALIZER_STATE_VERSION = "gdelt-gsg-normalizer-state-v3"
CHRONOLOGY_VERSION = "gdelt-gsg-terminal-chronology-v2"
GAP_EVIDENCE_VERSION = "gdelt-gsg-terminal-gap-evidence-v1"
RETRY_POLICY_VERSION = "gdelt-gsg-retry-policy-v1"
URL_NORMALIZER_VERSION = "url-canonicalization-v1"
TEXT_NORMALIZER_VERSION = "text-normalization-v1"
LANGUAGE_MAP_VERSION = "language-map-v1"
EXPECTED_INTERVAL = timedelta(minutes=1)
PROVIDER_LAG = timedelta(minutes=30)
MAX_PLAN_INTERVALS = 10_080
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_JSON_LINES = 1_000_000
RIGHTS_SCOPE = "gdelt_gsg_english_btc_titles"
RIGHTS_APPROVAL_VERSION = "provider-rights-approval-v1"
GSG_ARCHIVE_ROOT = "https://data.gdeltproject.org/"

PRIMARY_EXCLUSION_PRECEDENCE = (
    "hash_mismatch",
    "malformed_record",
    "invalid_timestamp",
    "historical_backfill_without_availability",
    "undocumented_first_seen_semantics",
    "missing_first_seen",
    "revision_time_unknown",
    "missing_identity",
    "invalid_url_or_identifier",
    "missing_title_and_content",
    "unsupported_language",
    "asset_mismatch",
    "license_restricted",
    "provider_gap",
    "duplicate_unresolved",
)
_EXCLUSION_RANK = {reason: index for index, reason in enumerate(PRIMARY_EXCLUSION_PRECEDENCE)}
_STATE_DATA_FILES = (
    "approval.json",
    "articles.json",
    "chronology.json",
    "exclusions.json",
    "gap-evidence.json",
    "groups.json",
    "observation-links.json",
)

_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_INVALID_PERCENT_PATTERN = re.compile(r"%(?![0-9A-Fa-f]{2})")
_TRACKING_KEY_PATTERN = re.compile(r"(?:utm_.*|fbclid|gclid|mc_cid|mc_eid)\Z", re.IGNORECASE)
_POSITIVE_BTC_PATTERN = re.compile(r"(?iu)(?<!\w)(?:bitcoin|btc|xbt|satoshi nakamoto)(?!\w)")
_BTC_CITY_PATTERN = re.compile(r"(?iu)(?<!\w)btc city(?!\w)")
_UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
_PATH_SAFE = _UNRESERVED | frozenset("!$&'()*+,;=:@/")
_QUERY_SAFE = _UNRESERVED | frozenset("!$'()*+,;:@/?")
_SENSITIVE_KEYS = frozenset(
    {
        "api-key",
        "apikey",
        "api_key",
        "authorization",
        "credential",
        "key",
        "password",
        "secret",
        "signature",
        "token",
    }
)
_SAFE_SNAPSHOT_ERROR_CODES = frozenset(
    {
        "invalid_gzip",
        "invalid_json",
        "invalid_jsonl",
        "invalid_utf8",
        "malformed_snapshot",
        "resource_limit_exceeded",
    }
)
_SUPPORTED_GAP_ERROR_KINDS = frozenset({"network_transport_error", *_SAFE_SNAPSHOT_ERROR_CODES})
_COVERAGE_GAP_REASONS = frozenset({"non_retryable", "retry_exhausted", *_SAFE_SNAPSHOT_ERROR_CODES})
_RETRY_AFTER_HTTP_STATUSES = frozenset({429, 503})
MAX_RETRY_AFTER_SECONDS = 86_400.0


@dataclass(frozen=True, slots=True)
class RetrievalInterval:
    """One expected provider file; planning never performs retrieval."""

    filename_timestamp: str
    due_at: str
    relative_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """A bounded, deterministic one-minute schedule."""

    plan_id: str
    start_at: str
    end_at_exclusive: str
    intervals: tuple[RetrievalInterval, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_at_exclusive": self.end_at_exclusive,
            "intervals": [interval.to_dict() for interval in self.intervals],
            "plan_id": self.plan_id,
            "start_at": self.start_at,
        }


@dataclass(frozen=True, slots=True)
class RetryDecision:
    disposition: str
    delay_seconds: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class RightsApproval:
    """Immutable provider/scope/config-specific rights input; never network authority."""

    approval_id: str
    approval_kind: str
    approved: bool
    provider: str
    scope: str
    protocol_config_sha256: str
    authorized_fixture_sha256: tuple[str, ...]
    network_access_authorized: bool
    version: str = RIGHTS_APPROVAL_VERSION

    @classmethod
    def create(
        cls,
        *,
        approval_kind: str,
        approved: bool,
        provider: str,
        scope: str,
        protocol_config_sha256: str,
        authorized_fixture_sha256: Iterable[str] = (),
        network_access_authorized: bool = False,
    ) -> RightsApproval:
        fixture_hashes = tuple(sorted(set(authorized_fixture_sha256)))
        payload = {
            "approval_kind": approval_kind,
            "approved": approved,
            "authorized_fixture_sha256": list(fixture_hashes),
            "network_access_authorized": network_access_authorized,
            "protocol_config_sha256": protocol_config_sha256,
            "provider": provider,
            "scope": scope,
            "version": RIGHTS_APPROVAL_VERSION,
        }
        return cls(
            approval_id=canonical_sha256(payload),
            approval_kind=approval_kind,
            approved=approved,
            provider=provider,
            scope=scope,
            protocol_config_sha256=protocol_config_sha256,
            authorized_fixture_sha256=fixture_hashes,
            network_access_authorized=network_access_authorized,
        )

    @classmethod
    def synthetic_fixture_only(
        cls, *, protocol_config_sha256: str, raw_snapshot_sha256: Iterable[str]
    ) -> RightsApproval:
        """Authorize only explicitly hashed synthetic fixtures, never provider responses."""
        return cls.create(
            approval_kind="synthetic_fixture_only",
            approved=True,
            provider=PROVIDER_ID,
            scope=RIGHTS_SCOPE,
            protocol_config_sha256=protocol_config_sha256,
            authorized_fixture_sha256=raw_snapshot_sha256,
            network_access_authorized=False,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["authorized_fixture_sha256"] = list(self.authorized_fixture_sha256)
        return value

    def identity_payload(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("approval_id")
        return value

    def is_structurally_valid(self) -> bool:
        if (
            self.version != RIGHTS_APPROVAL_VERSION
            or not _is_sha256(self.approval_id)
            or not _is_sha256(self.protocol_config_sha256)
            or not isinstance(self.provider, str)
            or not isinstance(self.scope, str)
            or not isinstance(self.approval_kind, str)
            or self.approval_kind not in {"provider_rights", "synthetic_fixture_only"}
            or not isinstance(self.approved, bool)
            or not isinstance(self.network_access_authorized, bool)
            or not isinstance(self.authorized_fixture_sha256, tuple)
            or not all(_is_sha256(item) for item in self.authorized_fixture_sha256)
            or tuple(sorted(set(self.authorized_fixture_sha256))) != self.authorized_fixture_sha256
        ):
            return False
        try:
            return self.approval_id == canonical_sha256(self.identity_payload())
        except (CanonicalizationError, TypeError, ValueError, AttributeError):
            return False


@dataclass(frozen=True, slots=True)
class RawSnapshotReceipt:
    """Immutable audit receipt for exact compressed bytes."""

    snapshot_id: str
    filename_timestamp: str
    ingested_at: str
    raw_published_at: str
    raw_snapshot_sha256: str
    compressed_size_bytes: int
    source_locator: str
    collection_mode: str
    input_class: str
    max_compressed_bytes: int
    max_decompressed_bytes: int
    max_json_lines: int
    parser_policy_version: str = PARSER_POLICY_VERSION
    parser_version: str = PARSER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GSGObservation:
    provider_observation_id: str
    filename_timestamp: str
    raw_snapshot_sha256: str
    zero_based_line_number: int
    endpoint_side: str
    raw_url: object
    raw_title: object
    raw_language: object
    raw_provider_first_seen_at: object
    ingested_at: str
    raw_published_at: str
    collection_mode: str
    input_class: str


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    receipt: RawSnapshotReceipt
    state: str
    observations: tuple[GSGObservation, ...]
    json_line_count: int
    error_code: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "json_line_count": self.json_line_count,
            "observations": [asdict(observation) for observation in self.observations],
            "receipt": self.receipt.to_dict(),
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class ObservationLink:
    provider_observation_id: str
    article_version_id: str
    reused_existing_version: bool
    raw_snapshot_sha256: str
    input_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExcludedObservation:
    provider_observation_id: str
    reason: str
    diagnostic: str
    raw_snapshot_sha256: str
    input_class: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    articles: tuple[ArticleRecord, ...]
    observation_links: tuple[ObservationLink, ...]
    exclusions: tuple[ExcludedObservation, ...]
    protocol_config_sha256: str
    rights_approval_sha256: str
    semantic_sha256: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "articles": [article.to_dict() for article in self.articles],
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
            "normalizer_version": NORMALIZER_VERSION,
            "observation_links": [link.to_dict() for link in self.observation_links],
            "protocol_config_sha256": self.protocol_config_sha256,
            "rights_approval_sha256": self.rights_approval_sha256,
        }
        if include_hash:
            value["semantic_sha256"] = self.semantic_sha256
        return value


@dataclass(frozen=True, slots=True)
class GroupAnchor:
    duplicate_group_id: str
    anchor_article_id: str
    initial_first_seen_at: str
    dedup_fingerprint: str
    canonical_url: str
    source: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _CandidateVersion:
    observation: GSGObservation
    article_id: str
    version_fingerprint: str
    canonical_url: str
    source: str
    title: str
    language: str
    provider_first_seen_at: str | None
    first_seen_at: str
    content_hash: str
    failures: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ProviderGap:
    start_at: str
    end_at_exclusive: str
    duration_minutes: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    plan_id: str
    as_of: str
    expected_due_intervals: int
    complete_intervals: int
    zero_line_intervals: int
    pending_intervals: int
    unresolved_intervals: int
    gap_intervals: int
    retrieval_rate: float
    maximum_gap_minutes: int
    gaps: tuple[ProviderGap, ...]
    expected_schedule_sha256: str
    observed_receipts_sha256: str
    terminal_gap_evidence_sha256: str
    semantic_sha256: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "as_of": self.as_of,
            "complete_intervals": self.complete_intervals,
            "expected_due_intervals": self.expected_due_intervals,
            "expected_schedule_sha256": self.expected_schedule_sha256,
            "gap_intervals": self.gap_intervals,
            "gaps": [gap.to_dict() for gap in self.gaps],
            "maximum_gap_minutes": self.maximum_gap_minutes,
            "observed_receipts_sha256": self.observed_receipts_sha256,
            "pending_intervals": self.pending_intervals,
            "plan_id": self.plan_id,
            "retrieval_rate": self.retrieval_rate,
            "terminal_gap_evidence_sha256": self.terminal_gap_evidence_sha256,
            "unresolved_intervals": self.unresolved_intervals,
            "zero_line_intervals": self.zero_line_intervals,
        }
        if include_hash:
            value["semantic_sha256"] = self.semantic_sha256
        return value


@dataclass(frozen=True, slots=True)
class GapAttempt:
    """One synthetic, immutable attempt fact; this object performs no retrieval."""

    attempt_number: int
    attempted_at: str
    http_status: int | None
    error_kind: str | None
    retry_after_seconds: float | None
    retry_disposition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TerminalGapEvidence:
    """Canonical evidence that a planned interval reached a terminal provider gap."""

    evidence_id: str
    evidence_sha256: str
    provider: str
    scope: str
    interval_start: str
    interval_end_exclusive: str
    expected_source_locator: str
    collection_mode: str
    input_class: str
    network_access_authorized: bool
    observed_snapshot_id: str | None
    observed_raw_snapshot_sha256: str | None
    retry_policy_version: str
    attempt_count: int
    attempts: tuple[GapAttempt, ...]
    final_terminal_disposition: str
    terminal_at: str
    protocol_config_sha256: str
    version: str = GAP_EVIDENCE_VERSION

    @classmethod
    def create(
        cls,
        *,
        interval_start: str,
        interval_end_exclusive: str,
        expected_source_locator: str,
        attempts: Iterable[GapAttempt],
        terminal_at: str,
        protocol_config_sha256: str,
        provider: str = PROVIDER_ID,
        scope: str = RIGHTS_SCOPE,
        collection_mode: str = "prospective",
        input_class: str = "synthetic_fixture",
        network_access_authorized: bool = False,
        observed_snapshot_id: str | None = None,
        observed_raw_snapshot_sha256: str | None = None,
        retry_policy_version: str = RETRY_POLICY_VERSION,
        final_terminal_disposition: str = "non_retryable",
    ) -> TerminalGapEvidence:
        materialized_attempts = tuple(attempts)
        if not all(isinstance(attempt, GapAttempt) for attempt in materialized_attempts):
            raise ProviderIngestionError("gap evidence attempts must match the frozen schema")
        identity = {
            "attempt_count": len(materialized_attempts),
            "attempts": [attempt.to_dict() for attempt in materialized_attempts],
            "collection_mode": collection_mode,
            "expected_source_locator": expected_source_locator,
            "final_terminal_disposition": final_terminal_disposition,
            "input_class": input_class,
            "interval_end_exclusive": interval_end_exclusive,
            "interval_start": interval_start,
            "network_access_authorized": network_access_authorized,
            "observed_raw_snapshot_sha256": observed_raw_snapshot_sha256,
            "observed_snapshot_id": observed_snapshot_id,
            "protocol_config_sha256": protocol_config_sha256,
            "provider": provider,
            "retry_policy_version": retry_policy_version,
            "scope": scope,
            "terminal_at": terminal_at,
            "version": GAP_EVIDENCE_VERSION,
        }
        evidence_id = canonical_sha256(
            {
                "identity": identity,
                "identity_version": "gdelt-gsg-terminal-gap-identity-v1",
            }
        )
        evidence_sha256 = canonical_sha256({**identity, "evidence_id": evidence_id})
        return cls(
            evidence_id=evidence_id,
            evidence_sha256=evidence_sha256,
            provider=provider,
            scope=scope,
            interval_start=interval_start,
            interval_end_exclusive=interval_end_exclusive,
            expected_source_locator=expected_source_locator,
            collection_mode=collection_mode,
            input_class=input_class,
            network_access_authorized=network_access_authorized,
            observed_snapshot_id=observed_snapshot_id,
            observed_raw_snapshot_sha256=observed_raw_snapshot_sha256,
            retry_policy_version=retry_policy_version,
            attempt_count=len(materialized_attempts),
            attempts=materialized_attempts,
            final_terminal_disposition=final_terminal_disposition,
            terminal_at=terminal_at,
            protocol_config_sha256=protocol_config_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return value

    def identity_payload(self) -> dict[str, Any]:
        value = self.to_dict()
        value.pop("evidence_id")
        value.pop("evidence_sha256")
        return value

    def canonical_bytes(self) -> bytes:
        """Return the exact immutable RFC 8785 evidence envelope bytes."""
        return canonicalize(self.to_dict())


@dataclass(frozen=True, slots=True)
class TerminalInterval:
    """One immutable minute-level fact that advances normalization chronology."""

    start_at: str
    end_at_exclusive: str
    outcome: str
    snapshot_state: str
    snapshot_id: str | None
    raw_snapshot_sha256: str | None
    ingested_at: str | None
    raw_published_at: str | None
    terminal_at: str
    json_line_count: int | None
    collection_mode: str
    input_class: str
    gap_evidence_id: str | None
    gap_evidence_sha256: str | None
    terminal_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GSGRetryPolicy:
    """Pure retry classification: one initial attempt and at most two retries."""

    maximum_attempts = 3
    backoff_seconds = (2.0, 4.0)
    version = RETRY_POLICY_VERSION

    def decide(
        self,
        *,
        attempt: int,
        http_status: int | None = None,
        error_kind: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> RetryDecision:
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or not 1 <= attempt <= self.maximum_attempts
        ):
            raise ProviderIngestionError("attempt must be between 1 and 3")
        if http_status is not None and (
            not isinstance(http_status, int)
            or isinstance(http_status, bool)
            or not 100 <= http_status <= 599
        ):
            raise ProviderIngestionError("HTTP status must be an integer from 100 to 599")
        if error_kind is not None and (
            not isinstance(error_kind, str) or error_kind not in _SUPPORTED_GAP_ERROR_KINDS
        ):
            raise ProviderIngestionError("unsupported bounded provider error kind")
        if http_status is not None and error_kind is not None:
            raise ProviderIngestionError("provide either an HTTP status or an error kind")
        if http_status is None and error_kind is None:
            raise ProviderIngestionError("an HTTP status or bounded error kind is required")
        if retry_after_seconds is not None:
            if (
                not isinstance(retry_after_seconds, (int, float))
                or isinstance(retry_after_seconds, bool)
                or not math.isfinite(retry_after_seconds)
                or not 0 <= retry_after_seconds <= MAX_RETRY_AFTER_SECONDS
            ):
                raise ProviderIngestionError(
                    "Retry-After must be finite, nonnegative, and within the frozen cap"
                )
            if http_status not in _RETRY_AFTER_HTTP_STATUSES:
                raise ProviderIngestionError(
                    "Retry-After is not applicable to this provider outcome"
                )
        if http_status == 200 and error_kind is None:
            return RetryDecision("complete", None, "http_200")
        retryable = error_kind == "network_transport_error" or http_status in {408, 429}
        retryable = retryable or (http_status is not None and 500 <= http_status <= 599)
        reason = error_kind or (
            f"http_{http_status}" if http_status is not None else "unknown_error"
        )
        if retryable and attempt < self.maximum_attempts:
            base_delay = self.backoff_seconds[attempt - 1]
            delay = max(base_delay, retry_after_seconds or 0.0)
            return RetryDecision("retry", delay, reason)
        if retryable:
            return RetryDecision("gap", None, f"{reason}_attempts_exhausted")
        return RetryDecision("gap", None, reason)


class GSGAdapter:
    """Persist and parse caller-supplied GSG gzip bytes without network access."""

    def __init__(
        self,
        store: ContentAddressedStore,
        *,
        clock: Callable[[], datetime],
        max_compressed_bytes: int = MAX_COMPRESSED_BYTES,
        max_decompressed_bytes: int = MAX_DECOMPRESSED_BYTES,
        max_json_lines: int = MAX_JSON_LINES,
    ) -> None:
        bounds = (
            (max_compressed_bytes, MAX_COMPRESSED_BYTES, "max_compressed_bytes"),
            (max_decompressed_bytes, MAX_DECOMPRESSED_BYTES, "max_decompressed_bytes"),
            (max_json_lines, MAX_JSON_LINES, "max_json_lines"),
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= maximum
            for value, maximum, _ in bounds
        ):
            raise ProviderIngestionError(
                "adapter resource bounds must be positive integers within frozen maxima"
            )
        self.store = store
        self.clock = clock
        self.max_compressed_bytes = max_compressed_bytes
        self.max_decompressed_bytes = max_decompressed_bytes
        self.max_json_lines = max_json_lines

    def ingest_snapshot(
        self,
        raw_response_bytes: bytes,
        *,
        filename_timestamp: str,
        ingested_at: str,
        source_locator: str,
        collection_mode: str,
        input_class: str = "provider_response",
    ) -> SnapshotResult:
        """Snapshot exact bytes, preserve the first receipt, and parse deterministically."""
        if not isinstance(raw_response_bytes, bytes):
            raise ProviderIngestionError("raw response must be exact bytes")
        if len(raw_response_bytes) > self.max_compressed_bytes:
            raise ProviderIngestionError("compressed response exceeds the frozen byte cap")
        filename_time = _parse_minute_timestamp(filename_timestamp, "filename_timestamp")
        ingestion_time = _parse_required_timestamp(ingested_at, "ingested_at")
        if collection_mode not in {"prospective", "historical_backfill"}:
            raise ProviderIngestionError(
                "collection_mode must be prospective or historical_backfill"
            )
        if input_class not in {"provider_response", "synthetic_fixture"}:
            raise ProviderIngestionError(
                "input_class must be provider_response or synthetic_fixture"
            )

        digest = self.store.put_bytes(raw_response_bytes)
        snapshot_id = _derive_snapshot_id(
            collection_mode=collection_mode,
            filename_timestamp=format_utc_timestamp(filename_time),
            input_class=input_class,
            raw_snapshot_sha256=digest,
            max_compressed_bytes=self.max_compressed_bytes,
            max_decompressed_bytes=self.max_decompressed_bytes,
            max_json_lines=self.max_json_lines,
        )
        publication_id = f"gsg-snapshot-{snapshot_id}"
        existing = self._load_existing_receipt(snapshot_id)
        if existing is None:
            published_time = self.clock()
            if not isinstance(published_time, datetime) or published_time.tzinfo is None:
                raise ProviderIngestionError("adapter clock must return a timezone-aware datetime")
            published_time = published_time.astimezone(UTC)
            if published_time < ingestion_time:
                raise ProviderIngestionError("raw publication cannot precede final-byte ingestion")
            receipt = RawSnapshotReceipt(
                snapshot_id=snapshot_id,
                filename_timestamp=format_utc_timestamp(filename_time),
                ingested_at=format_utc_timestamp(ingestion_time),
                raw_published_at=format_utc_timestamp(published_time),
                raw_snapshot_sha256=digest,
                compressed_size_bytes=len(raw_response_bytes),
                source_locator=redact_url(source_locator),
                collection_mode=collection_mode,
                input_class=input_class,
                max_compressed_bytes=self.max_compressed_bytes,
                max_decompressed_bytes=self.max_decompressed_bytes,
                max_json_lines=self.max_json_lines,
            )
            try:
                self.store.publish_bundle(
                    publication_id,
                    {"receipt.json": canonicalize(receipt.to_dict())},
                    metadata={
                        "provider": PROVIDER_ID,
                        "raw_object_sha256": digest,
                        "scope": "offline_adapter_input",
                    },
                )
            except PublicationCollisionError:
                existing = self._load_existing_receipt(snapshot_id)
                if existing is None:
                    raise
            else:
                existing = receipt
        receipt = existing
        if receipt is None:  # pragma: no cover - defensive after collision handling
            raise ProviderIngestionError("snapshot receipt publication did not become visible")
        if (
            receipt.snapshot_id != snapshot_id
            or receipt.raw_snapshot_sha256 != digest
            or receipt.filename_timestamp != format_utc_timestamp(filename_time)
            or receipt.compressed_size_bytes != len(raw_response_bytes)
            or receipt.collection_mode != collection_mode
            or receipt.input_class != input_class
            or receipt.max_compressed_bytes != self.max_compressed_bytes
            or receipt.max_decompressed_bytes != self.max_decompressed_bytes
            or receipt.max_json_lines != self.max_json_lines
            or receipt.parser_policy_version != PARSER_POLICY_VERSION
            or receipt.parser_version != PARSER_VERSION
        ):
            raise ProviderIngestionError("snapshot receipt identity collision")
        return self._parse_snapshot(receipt, raw_response_bytes)

    def _load_existing_receipt(self, snapshot_id: str) -> RawSnapshotReceipt | None:
        publication_id = f"gsg-snapshot-{snapshot_id}"
        publication = self.store.publications_root / publication_id
        if not publication.exists():
            return None
        try:
            receipt, _ = _load_verified_snapshot_receipt(
                self.store,
                snapshot_id,
                raw_object_cache={},
            )
        except (NormalizationIntegrityError, OSError) as exc:
            raise ProviderIngestionError("unable to load existing snapshot receipt") from exc
        return receipt

    def _parse_snapshot(
        self, receipt: RawSnapshotReceipt, raw_response_bytes: bytes
    ) -> SnapshotResult:
        try:
            decompressed = _bounded_gzip_decompress(
                raw_response_bytes, maximum_bytes=self.max_decompressed_bytes
            )
            text = decompressed.decode("utf-8", errors="strict")
            raw_lines = text.splitlines()
            if len(raw_lines) > self.max_json_lines:
                raise ProviderIngestionError("JSON line count exceeds the frozen cap")
            observations: list[GSGObservation] = []
            for line_number, line in enumerate(raw_lines):
                if not line.strip():
                    raise ProviderIngestionError("blank JSONL records are invalid")
                record = json.loads(line, parse_constant=_reject_json_constant)
                if not isinstance(record, dict):
                    raise ProviderIngestionError("each GSG line must be a JSON object")
                for side in ("from", "to"):
                    observations.append(
                        _observation_from_endpoint(receipt, line_number, side, record)
                    )
        except (
            EOFError,
            gzip.BadGzipFile,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            ProviderIngestionError,
        ) as exc:
            return SnapshotResult(
                receipt=receipt,
                state="invalid",
                observations=(),
                json_line_count=0,
                error_code=_safe_error_code(exc),
            )
        return SnapshotResult(
            receipt=receipt,
            state="complete",
            observations=tuple(observations),
            json_line_count=len(raw_lines),
            error_code=None,
        )


class GSGNormalizer:
    """Transactional causal normalizer with immutable export and verified hydration."""

    def __init__(
        self,
        *,
        protocol_config_sha256: str,
        rights_approval: RightsApproval | None = None,
    ) -> None:
        if not _is_sha256(protocol_config_sha256):
            raise ProviderIngestionError("protocol_config_sha256 must be lowercase SHA-256")
        if rights_approval is not None and not isinstance(rights_approval, RightsApproval):
            raise ProviderIngestionError("rights_approval must be an immutable RightsApproval")
        self._protocol_config_sha256 = protocol_config_sha256
        self._rights_approval = rights_approval
        self._versions_by_fingerprint: dict[tuple[str, str], ArticleRecord] = {}
        self._versions_by_id: dict[str, ArticleRecord] = {}
        self._links: dict[str, ObservationLink] = {}
        self._exclusions: dict[str, ExcludedObservation] = {}
        self._groups_by_id: dict[str, GroupAnchor] = {}
        self._article_groups: dict[str, str] = {}
        self._terminal_intervals: tuple[TerminalInterval, ...] = ()
        self._next_expected_interval_start: str | None = None
        self._closed_availability_through: str | None = None
        self._gap_evidence_by_id: dict[str, TerminalGapEvidence] = {}

    @property
    def protocol_config_sha256(self) -> str:
        """Return the immutable protocol identity governing this state generation."""
        return self._protocol_config_sha256

    @property
    def rights_approval(self) -> RightsApproval | None:
        """Return the immutable rights input governing this state generation."""
        return self._rights_approval

    @property
    def rights_approval_sha256(self) -> str:
        payload = self.rights_approval.to_dict() if self.rights_approval is not None else None
        return canonical_sha256(payload)

    @property
    def next_expected_interval_start(self) -> str | None:
        """Exclusive end of the last terminally normalized interval."""
        return self._next_expected_interval_start

    @property
    def terminal_intervals(self) -> tuple[TerminalInterval, ...]:
        """Return the immutable, contiguous terminal interval ledger."""
        return self._terminal_intervals

    @property
    def closed_availability_through(self) -> str | None:
        """Exclusive microsecond after the latest committed terminal availability event."""
        return self._closed_availability_through

    @property
    def terminal_gap_evidence(self) -> tuple[TerminalGapEvidence, ...]:
        """Return accepted gap evidence in canonical interval order."""
        return tuple(
            sorted(
                self._gap_evidence_by_id.values(),
                key=lambda item: (item.interval_start, item.evidence_id),
            )
        )

    def normalize(
        self,
        snapshots: Iterable[SnapshotResult],
        *,
        retrieval_plan: RetrievalPlan,
        terminal_as_of: str,
        gap_evidence: Iterable[TerminalGapEvidence] = (),
    ) -> NormalizationResult:
        """Validate a whole terminal batch, then commit one order-independent state change."""
        materialized_snapshots = tuple(snapshots)
        materialized_gap_evidence = tuple(gap_evidence)
        self._validate_chronology(
            self._terminal_intervals,
            self._next_expected_interval_start,
            self._closed_availability_through,
            self._gap_evidence_by_id,
        )
        self._validate_plan_chronology(retrieval_plan)
        coverage = build_coverage_report(
            retrieval_plan,
            materialized_snapshots,
            as_of=terminal_as_of,
            gap_evidence=materialized_gap_evidence,
            protocol_config_sha256=self.protocol_config_sha256,
        )
        if coverage.pending_intervals:
            raise ProviderIngestionError(
                "normalization watermark is not terminal; planned intervals remain pending"
            )
        if coverage.unresolved_intervals:
            raise ProviderIngestionError(
                "normalization watermark is not terminal; intervals lack verified gap evidence"
            )
        appended_intervals, accepted_gap_evidence, next_availability_boundary = (
            _terminal_interval_facts(
                retrieval_plan,
                materialized_snapshots,
                materialized_gap_evidence,
                protocol_config_sha256=self.protocol_config_sha256,
                terminal_as_of=terminal_as_of,
                prior_closed_availability_through=self._closed_availability_through,
            )
        )
        next_terminal_intervals = self._terminal_intervals + appended_intervals
        next_watermark = retrieval_plan.end_at_exclusive
        next_gap_evidence = dict(self._gap_evidence_by_id)
        for evidence in accepted_gap_evidence:
            if evidence.evidence_id in next_gap_evidence:
                raise NormalizationIntegrityError(
                    "terminal gap evidence identity is already committed"
                )
            next_gap_evidence[evidence.evidence_id] = evidence
        self._validate_chronology(
            next_terminal_intervals,
            next_watermark,
            next_availability_boundary,
            next_gap_evidence,
        )
        observations = sorted(
            (
                observation
                for snapshot in materialized_snapshots
                if snapshot.state == "complete"
                for observation in snapshot.observations
            ),
            key=lambda item: item.provider_observation_id,
        )

        next_versions = dict(self._versions_by_id)
        next_fingerprints = dict(self._versions_by_fingerprint)
        next_links = dict(self._links)
        next_exclusions = dict(self._exclusions)
        next_groups = dict(self._groups_by_id)
        next_article_groups = dict(self._article_groups)
        batch_links: dict[str, ObservationLink] = {}
        batch_exclusions: dict[str, ExcludedObservation] = {}
        touched_versions: dict[str, ArticleRecord] = {}
        candidates: list[_CandidateVersion] = []

        for observation in observations:
            observation_id = observation.provider_observation_id
            existing_link = self._links.get(observation_id)
            if existing_link is not None:
                if (
                    existing_link.raw_snapshot_sha256 != observation.raw_snapshot_sha256
                    or existing_link.input_class != observation.input_class
                ):
                    raise NormalizationIntegrityError(
                        "repeated observation identity has conflicting provenance class"
                    )
                batch_links[observation_id] = existing_link
                touched_versions[existing_link.article_version_id] = self._versions_by_id[
                    existing_link.article_version_id
                ]
                continue
            existing_exclusion = self._exclusions.get(observation_id)
            if existing_exclusion is not None:
                if (
                    existing_exclusion.raw_snapshot_sha256 != observation.raw_snapshot_sha256
                    or existing_exclusion.input_class != observation.input_class
                ):
                    raise NormalizationIntegrityError(
                        "excluded observation identity has conflicting provenance class"
                    )
                batch_exclusions[observation_id] = existing_exclusion
                continue
            try:
                candidates.append(self._materialize_candidate(observation))
            except _ObservationExcluded as exc:
                exclusion = ExcludedObservation(
                    observation_id,
                    exc.reason,
                    exc.diagnostic,
                    observation.raw_snapshot_sha256,
                    observation.input_class,
                )
                batch_exclusions[observation_id] = exclusion
                next_exclusions[observation_id] = exclusion

        existing_by_article_time: dict[tuple[str, str], set[str]] = {}
        for (article_id, fingerprint), record in self._versions_by_fingerprint.items():
            if record.first_seen_at is None:  # pragma: no cover - validated articles
                raise NormalizationIntegrityError("stored article is missing first_seen_at")
            existing_by_article_time.setdefault((article_id, record.first_seen_at), set()).add(
                fingerprint
            )
        for candidate in candidates:
            existing_fingerprints = existing_by_article_time.get(
                (candidate.article_id, candidate.first_seen_at), set()
            )
            if existing_fingerprints and candidate.version_fingerprint not in existing_fingerprints:
                raise NormalizationIntegrityError(
                    "same-time revision conflicts with immutable prior state; no state was changed"
                )

        incoming_by_article_time: dict[tuple[str, str], list[_CandidateVersion]] = {}
        for candidate in candidates:
            incoming_by_article_time.setdefault(
                (candidate.article_id, candidate.first_seen_at), []
            ).append(candidate)
        conflicted_observations: set[str] = set()
        for same_time_candidates in incoming_by_article_time.values():
            fingerprints = {item.version_fingerprint for item in same_time_candidates}
            if len(fingerprints) <= 1:
                continue
            for candidate in same_time_candidates:
                observation_id = candidate.observation.provider_observation_id
                conflicted_observations.add(observation_id)
                reason, diagnostic = _primary_exclusion(
                    (
                        *candidate.failures,
                        (
                            "revision_time_unknown",
                            "all same-time conflicting fingerprints are excluded "
                            "before state mutation",
                        ),
                    )
                )
                exclusion = ExcludedObservation(
                    observation_id,
                    reason,
                    diagnostic,
                    candidate.observation.raw_snapshot_sha256,
                    candidate.observation.input_class,
                )
                batch_exclusions[observation_id] = exclusion
                next_exclusions[observation_id] = exclusion
        candidates = [
            candidate
            for candidate in candidates
            if candidate.observation.provider_observation_id not in conflicted_observations
        ]
        eligible_candidates: list[_CandidateVersion] = []
        for candidate in candidates:
            if candidate.failures:
                reason, diagnostic = _primary_exclusion(candidate.failures)
                observation_id = candidate.observation.provider_observation_id
                exclusion = ExcludedObservation(
                    observation_id,
                    reason,
                    diagnostic,
                    candidate.observation.raw_snapshot_sha256,
                    candidate.observation.input_class,
                )
                batch_exclusions[observation_id] = exclusion
                next_exclusions[observation_id] = exclusion
            else:
                eligible_candidates.append(candidate)
        candidates = eligible_candidates

        candidates_by_version: dict[tuple[str, str], list[_CandidateVersion]] = {}
        for candidate in candidates:
            candidates_by_version.setdefault(
                (candidate.article_id, candidate.version_fingerprint), []
            ).append(candidate)

        new_version_candidates: dict[tuple[str, str], list[_CandidateVersion]] = {}
        for key, version_candidates in candidates_by_version.items():
            version_candidates.sort(
                key=lambda item: (
                    item.first_seen_at,
                    item.article_id,
                    item.observation.provider_observation_id,
                )
            )
            existing = self._versions_by_fingerprint.get(key)
            if existing is not None:
                if existing.first_seen_at is None:  # pragma: no cover - validated articles
                    raise NormalizationIntegrityError("stored article is missing first_seen_at")
                if version_candidates[0].first_seen_at < existing.first_seen_at:
                    raise NormalizationIntegrityError(
                        "an earlier repeat would change immutable first_seen_at; batch rejected"
                    )
                for candidate in version_candidates:
                    link = ObservationLink(
                        candidate.observation.provider_observation_id,
                        existing.article_version_id,
                        True,
                        candidate.observation.raw_snapshot_sha256,
                        candidate.observation.input_class,
                    )
                    batch_links[link.provider_observation_id] = link
                    next_links[link.provider_observation_id] = link
                touched_versions[existing.article_version_id] = existing
            else:
                new_version_candidates[key] = version_candidates

        new_article_initial: dict[str, _CandidateVersion] = {}
        for version_candidates in new_version_candidates.values():
            candidate = version_candidates[0]
            if candidate.article_id in next_article_groups:
                continue
            prior = new_article_initial.get(candidate.article_id)
            if prior is None or (
                candidate.first_seen_at,
                candidate.article_id,
                candidate.version_fingerprint,
            ) < (prior.first_seen_at, prior.article_id, prior.version_fingerprint):
                new_article_initial[candidate.article_id] = candidate

        for candidate in sorted(
            new_article_initial.values(), key=lambda item: (item.first_seen_at, item.article_id)
        ):
            group_id = self._causal_group_for_candidate(candidate, next_groups)
            if group_id is None:
                group_id = derive_duplicate_group_id(candidate.article_id)
                next_groups[group_id] = GroupAnchor(
                    duplicate_group_id=group_id,
                    anchor_article_id=candidate.article_id,
                    initial_first_seen_at=candidate.first_seen_at,
                    dedup_fingerprint=_dedup_fingerprint(candidate.title, candidate.language),
                    canonical_url=candidate.canonical_url,
                    source=candidate.source,
                )
            next_article_groups[candidate.article_id] = group_id

        for key, version_candidates in sorted(
            new_version_candidates.items(),
            key=lambda item: (
                item[1][0].first_seen_at,
                item[1][0].article_id,
                item[0][1],
            ),
        ):
            representative = version_candidates[0]
            group_id = next_article_groups.get(representative.article_id)
            if group_id is None:
                raise NormalizationIntegrityError("new logical article has no causal group")
            record = self._article_from_candidate(representative, group_id)
            next_versions[record.article_version_id] = record
            next_fingerprints[key] = record
            touched_versions[record.article_version_id] = record
            representative_id = representative.observation.provider_observation_id
            for candidate in version_candidates:
                link = ObservationLink(
                    provider_observation_id=candidate.observation.provider_observation_id,
                    article_version_id=record.article_version_id,
                    reused_existing_version=(
                        candidate.observation.provider_observation_id != representative_id
                    ),
                    raw_snapshot_sha256=candidate.observation.raw_snapshot_sha256,
                    input_class=candidate.observation.input_class,
                )
                batch_links[link.provider_observation_id] = link
                next_links[link.provider_observation_id] = link

        self._validate_state_components(
            next_versions,
            next_fingerprints,
            next_links,
            next_exclusions,
            next_groups,
            next_article_groups,
        )
        self._validate_rights_bound_state(next_versions, next_links)
        self._validate_chronology_state_relationships(
            next_terminal_intervals,
            next_versions,
            next_links,
            next_exclusions,
        )
        result = self._result(touched_versions, batch_links, batch_exclusions)
        self._versions_by_id = next_versions
        self._versions_by_fingerprint = next_fingerprints
        self._links = next_links
        self._exclusions = next_exclusions
        self._groups_by_id = next_groups
        self._article_groups = next_article_groups
        self._terminal_intervals = next_terminal_intervals
        self._next_expected_interval_start = next_watermark
        self._closed_availability_through = next_availability_boundary
        self._gap_evidence_by_id = next_gap_evidence
        return result

    def _validate_plan_chronology(self, retrieval_plan: RetrievalPlan) -> None:
        """Require an exact next plan; overlap and replay are intentionally unsupported."""
        _validate_retrieval_plan(retrieval_plan)
        watermark = self._next_expected_interval_start
        if watermark is None:
            return
        plan_start = _parse_minute_timestamp(retrieval_plan.start_at, "plan.start_at")
        watermark_time = _parse_minute_timestamp(watermark, "next_expected_interval_start")
        if plan_start < watermark_time:
            raise NormalizationIntegrityError(
                "retrieval plan regresses or overlaps the terminal chronology"
            )
        if plan_start > watermark_time:
            raise NormalizationIntegrityError(
                "retrieval plan leaves an unrecorded interval before its start"
            )

    def _validate_chronology(
        self,
        intervals: Sequence[TerminalInterval],
        watermark: str | None,
        availability_boundary: str | None,
        gap_evidence: Mapping[str, TerminalGapEvidence],
    ) -> None:
        previous_end: datetime | None = None
        previous_raw_published_at: datetime | None = None
        previous_terminal_at: datetime | None = None
        referenced_gap_evidence: set[str] = set()
        for position, interval in enumerate(intervals):
            if not isinstance(interval, TerminalInterval):
                raise NormalizationIntegrityError(
                    "terminal chronology contains an invalid interval record"
                )
            start = _parse_chronology_minute(interval.start_at, "terminal interval start")
            end = _parse_chronology_minute(interval.end_at_exclusive, "terminal interval end")
            if end - start != EXPECTED_INTERVAL:
                raise NormalizationIntegrityError(
                    "terminal chronology intervals must cover exactly one minute"
                )
            if previous_end is not None and start != previous_end:
                raise NormalizationIntegrityError(
                    "terminal chronology is duplicated, overlapping, or discontinuous"
                )
            if (
                not isinstance(interval.collection_mode, str)
                or interval.collection_mode not in {"prospective", "historical_backfill"}
                or not isinstance(interval.input_class, str)
                or interval.input_class not in {"provider_response", "synthetic_fixture"}
            ):
                raise NormalizationIntegrityError("terminal interval provenance class is invalid")
            if not isinstance(interval.snapshot_state, str):
                raise NormalizationIntegrityError("terminal interval snapshot state is invalid")
            if interval.snapshot_state in {"complete", "invalid"}:
                if (
                    not _is_sha256(interval.snapshot_id)
                    or not _is_sha256(interval.raw_snapshot_sha256)
                    or interval.ingested_at is None
                    or interval.raw_published_at is None
                ):
                    raise NormalizationIntegrityError(
                        "delivered terminal interval snapshot identity is invalid"
                    )
                ingested_at = _parse_chronology_instant(
                    interval.ingested_at, "terminal interval ingested_at"
                )
                raw_published_at = _parse_chronology_instant(
                    interval.raw_published_at,
                    "terminal interval raw_published_at",
                )
                if raw_published_at < ingested_at:
                    raise NormalizationIntegrityError(
                        "terminal snapshot publication precedes ingestion"
                    )
                if previous_terminal_at is not None and raw_published_at <= previous_terminal_at:
                    raise NormalizationIntegrityError(
                        "raw snapshot publication crosses the closed causal availability boundary"
                    )
                if (
                    previous_raw_published_at is not None
                    and raw_published_at <= previous_raw_published_at
                ):
                    raise NormalizationIntegrityError(
                        "raw snapshot publication timestamps must strictly increase"
                    )
                previous_raw_published_at = raw_published_at
            terminal_at = _parse_chronology_instant(
                interval.terminal_at, "terminal interval terminal_at"
            )
            if previous_terminal_at is not None and terminal_at <= previous_terminal_at:
                raise NormalizationIntegrityError(
                    "terminal availability timestamps must strictly increase"
                )
            if interval.outcome == "retrieved_and_normalized":
                if (
                    interval.snapshot_state != "complete"
                    or interval.raw_published_at != interval.terminal_at
                    or interval.gap_evidence_id is not None
                    or interval.gap_evidence_sha256 is not None
                    or not isinstance(interval.json_line_count, int)
                    or isinstance(interval.json_line_count, bool)
                    or interval.json_line_count < 0
                    or interval.json_line_count > MAX_JSON_LINES
                    or interval.terminal_reason is not None
                ):
                    raise NormalizationIntegrityError(
                        "successful terminal interval facts are contradictory"
                    )
            elif interval.outcome == "provider_gap":
                if interval.snapshot_state == "missing":
                    if (
                        interval.snapshot_id is not None
                        or interval.raw_snapshot_sha256 is not None
                        or interval.ingested_at is not None
                        or interval.raw_published_at is not None
                        or interval.json_line_count is not None
                        or not _is_sha256(interval.gap_evidence_id)
                        or not _is_sha256(interval.gap_evidence_sha256)
                        or interval.terminal_reason != "verified_terminal_gap_evidence"
                    ):
                        raise NormalizationIntegrityError(
                            "missing terminal interval has snapshot facts"
                        )
                    evidence = gap_evidence.get(interval.gap_evidence_id)
                    if evidence is None:
                        raise NormalizationIntegrityError(
                            "missing terminal interval lacks immutable gap evidence"
                        )
                    if evidence.evidence_sha256 != interval.gap_evidence_sha256:
                        raise NormalizationIntegrityError(
                            "terminal interval gap evidence hash mismatch"
                        )
                    if evidence.terminal_at != interval.terminal_at:
                        raise NormalizationIntegrityError(
                            "terminal interval time does not match its gap evidence"
                        )
                    _validate_terminal_gap_evidence(
                        evidence,
                        interval_start=interval.start_at,
                        interval_end_exclusive=interval.end_at_exclusive,
                        expected_source_locator=_expected_source_locator_at(interval.start_at),
                        protocol_config_sha256=self.protocol_config_sha256,
                        terminal_as_of=None,
                    )
                    if (
                        evidence.collection_mode != interval.collection_mode
                        or evidence.input_class != interval.input_class
                    ):
                        raise NormalizationIntegrityError(
                            "terminal gap evidence provenance does not match its interval"
                        )
                    referenced_gap_evidence.add(evidence.evidence_id)
                elif interval.snapshot_state == "invalid":
                    if (
                        not _is_sha256(interval.raw_snapshot_sha256)
                        or not _is_sha256(interval.gap_evidence_id)
                        or not _is_sha256(interval.gap_evidence_sha256)
                        or not isinstance(interval.json_line_count, int)
                        or isinstance(interval.json_line_count, bool)
                        or interval.json_line_count < 0
                        or interval.json_line_count > MAX_JSON_LINES
                        or not isinstance(interval.terminal_reason, str)
                        or interval.terminal_reason not in _SAFE_SNAPSHOT_ERROR_CODES
                    ):
                        raise NormalizationIntegrityError(
                            "invalid terminal interval has malformed snapshot facts"
                        )
                    evidence = gap_evidence.get(interval.gap_evidence_id)
                    if (
                        evidence is None
                        or evidence.evidence_sha256 != interval.gap_evidence_sha256
                        or evidence.terminal_at != interval.terminal_at
                        or evidence.observed_snapshot_id != interval.snapshot_id
                        or evidence.observed_raw_snapshot_sha256 != interval.raw_snapshot_sha256
                    ):
                        raise NormalizationIntegrityError(
                            "invalid terminal interval lacks matching gap evidence"
                        )
                    _validate_terminal_gap_evidence(
                        evidence,
                        interval_start=interval.start_at,
                        interval_end_exclusive=interval.end_at_exclusive,
                        expected_source_locator=_expected_source_locator_at(interval.start_at),
                        protocol_config_sha256=self.protocol_config_sha256,
                        terminal_as_of=None,
                    )
                    if (
                        evidence.attempts[-1].error_kind != interval.terminal_reason
                        or evidence.collection_mode != interval.collection_mode
                        or evidence.input_class != interval.input_class
                        or terminal_at
                        < _parse_chronology_instant(
                            interval.raw_published_at,
                            "terminal interval raw_published_at",
                        )
                    ):
                        raise NormalizationIntegrityError(
                            "invalid terminal gap evidence contradicts its raw receipt"
                        )
                    referenced_gap_evidence.add(evidence.evidence_id)
                else:
                    raise NormalizationIntegrityError(
                        "provider gap has an unsupported snapshot state"
                    )
                if not isinstance(interval.terminal_reason, str) or not interval.terminal_reason:
                    raise NormalizationIntegrityError(
                        "provider gap must record one terminal reason"
                    )
            else:
                raise NormalizationIntegrityError(
                    "terminal chronology contains an unsupported outcome"
                )
            previous_terminal_at = terminal_at
            previous_end = end
            if position and intervals[position - 1].start_at >= interval.start_at:
                raise NormalizationIntegrityError(
                    "terminal chronology is not in canonical interval order"
                )
        if set(gap_evidence) != referenced_gap_evidence:
            raise NormalizationIntegrityError(
                "terminal gap evidence inventory is unreferenced or incomplete"
            )
        if not intervals:
            if watermark is not None or availability_boundary is not None or gap_evidence:
                raise NormalizationIntegrityError(
                    "empty terminal chronology has unjustified durable state"
                )
            return
        watermark_time = _parse_chronology_minute(watermark, "next_expected_interval_start")
        if previous_end != watermark_time:
            raise NormalizationIntegrityError(
                "terminal watermark is not justified by interval facts"
            )
        boundary_time = _parse_chronology_instant(
            availability_boundary,
            "closed_availability_through",
        )
        if (
            previous_terminal_at is None
            or _exclusive_boundary_after(previous_terminal_at) != boundary_time
        ):
            raise NormalizationIntegrityError(
                "causal availability boundary is not justified by terminal facts"
            )

    @staticmethod
    def _validate_chronology_state_relationships(
        intervals: Sequence[TerminalInterval],
        versions: Mapping[str, ArticleRecord],
        links: Mapping[str, ObservationLink],
        exclusions: Mapping[str, ExcludedObservation],
    ) -> None:
        delivered = {
            (interval.start_at, interval.raw_snapshot_sha256): interval
            for interval in intervals
            if interval.snapshot_state == "complete"
        }
        observation_positions: dict[tuple[str, str], set[tuple[int, str]]] = {}
        for observation_id, value in {**links, **exclusions}.items():
            start_at, raw_hash, line_number, endpoint_side = _observation_provenance(observation_id)
            interval = delivered.get((start_at, raw_hash))
            if interval is None or value.input_class != interval.input_class:
                raise NormalizationIntegrityError(
                    "observation provenance has no matching complete terminal interval"
                )
            if interval.json_line_count is None or line_number >= interval.json_line_count:
                raise NormalizationIntegrityError(
                    "observation line lies outside its terminal snapshot"
                )
            key = (start_at, raw_hash)
            positions = observation_positions.setdefault(key, set())
            position = (line_number, endpoint_side)
            if position in positions:
                raise NormalizationIntegrityError(
                    "terminal snapshot contains duplicate observation provenance"
                )
            positions.add(position)
        for key, interval in delivered.items():
            if len(observation_positions.get(key, set())) != 2 * (interval.json_line_count or 0):
                raise NormalizationIntegrityError(
                    "terminal snapshot observations are not fully accounted for"
                )
        for article in versions.values():
            start_at, raw_hash, _, _ = _observation_provenance(article.provider_observation_id)
            interval = delivered.get((start_at, raw_hash))
            if (
                interval is None
                or article.raw_snapshot_sha256 != interval.raw_snapshot_sha256
                or article.first_seen_at != interval.raw_published_at
                or article.ingested_at != interval.ingested_at
            ):
                raise NormalizationIntegrityError(
                    "article availability is inconsistent with terminal chronology"
                )

    def export_state_files(self) -> dict[str, bytes]:
        """Return the complete deterministic state as canonical, transitively hashed files."""
        if self.rights_approval is not None and not self.rights_approval.is_structurally_valid():
            raise NormalizationIntegrityError("cannot persist a structurally invalid approval")
        self._validate_state_components(
            self._versions_by_id,
            self._versions_by_fingerprint,
            self._links,
            self._exclusions,
            self._groups_by_id,
            self._article_groups,
        )
        self._validate_rights_bound_state(self._versions_by_id, self._links)
        self._validate_chronology(
            self._terminal_intervals,
            self._next_expected_interval_start,
            self._closed_availability_through,
            self._gap_evidence_by_id,
        )
        self._validate_chronology_state_relationships(
            self._terminal_intervals,
            self._versions_by_id,
            self._links,
            self._exclusions,
        )
        data_files = {
            "approval.json": canonicalize(
                self.rights_approval.to_dict() if self.rights_approval is not None else None
            ),
            "articles.json": canonicalize(
                [
                    item.to_dict()
                    for item in sorted(
                        self._versions_by_id.values(),
                        key=lambda value: value.article_version_id,
                    )
                ]
            ),
            "chronology.json": canonicalize(
                {
                    "closed_availability_through": self._closed_availability_through,
                    "next_expected_interval_start": self._next_expected_interval_start,
                    "terminal_intervals": [
                        interval.to_dict() for interval in self._terminal_intervals
                    ],
                    "version": CHRONOLOGY_VERSION,
                }
            ),
            "exclusions.json": canonicalize(
                [
                    item.to_dict()
                    for item in sorted(
                        self._exclusions.values(),
                        key=lambda value: value.provider_observation_id,
                    )
                ]
            ),
            "gap-evidence.json": canonicalize(
                [item.to_dict() for item in self.terminal_gap_evidence]
            ),
            "groups.json": canonicalize(
                [
                    item.to_dict()
                    for item in sorted(
                        self._groups_by_id.values(),
                        key=lambda value: value.duplicate_group_id,
                    )
                ]
            ),
            "observation-links.json": canonicalize(
                [
                    item.to_dict()
                    for item in sorted(
                        self._links.values(),
                        key=lambda value: value.provider_observation_id,
                    )
                ]
            ),
        }
        descriptors = {
            name: {"sha256": sha256_bytes(data), "size_bytes": len(data)}
            for name, data in sorted(data_files.items())
        }
        state_identity = {
            "files": descriptors,
            "normalizer_version": NORMALIZER_VERSION,
            "protocol_config_sha256": self.protocol_config_sha256,
            "provider": PROVIDER_ID,
            "rights_approval_sha256": self.rights_approval_sha256,
            "schema_version": NORMALIZER_STATE_VERSION,
        }
        state_index = {**state_identity, "state_sha256": canonical_sha256(state_identity)}
        return {**data_files, "state.json": canonicalize(state_index)}

    def publish_state(self, store: ContentAddressedStore, state_name: str) -> Path:
        """Publish one complete immutable state; collisions are never overwritten."""
        files = self.export_state_files()
        state_index = _parse_canonical_json_buffer(files["state.json"], "state index")
        if not isinstance(state_index, dict):  # pragma: no cover - constructed above
            raise NormalizationIntegrityError("state index must be an object")
        return store.publish_bundle(
            f"gsg-normalizer-state-{state_name}",
            files,
            metadata={
                "protocol_config_sha256": self.protocol_config_sha256,
                "provider": PROVIDER_ID,
                "rights_approval_sha256": self.rights_approval_sha256,
                "state_sha256": state_index["state_sha256"],
            },
        )

    @classmethod
    def hydrate(cls, store: ContentAddressedStore, publication_id: str) -> GSGNormalizer:
        """Construct state only from one-read buffers verified by the publication manifest."""
        try:
            verified = store.read_publication(publication_id)
        except SentimentStorageError as exc:
            raise NormalizationIntegrityError(
                f"normalizer state publication failed verification: {publication_id}"
            ) from exc
        expected_files = set(_STATE_DATA_FILES) | {"state.json"}
        if set(verified.files) != expected_files:
            raise NormalizationIntegrityError("normalizer state file inventory is incomplete")
        state_index = _parse_canonical_json_buffer(verified.files["state.json"], "state index")
        if not isinstance(state_index, dict):
            raise NormalizationIntegrityError("state index must be an object")
        required_index_fields = {
            "files",
            "normalizer_version",
            "protocol_config_sha256",
            "provider",
            "rights_approval_sha256",
            "schema_version",
            "state_sha256",
        }
        if set(state_index) != required_index_fields:
            raise NormalizationIntegrityError("state index fields do not match the contract")
        state_identity = dict(state_index)
        state_sha256 = state_identity.pop("state_sha256")
        if state_sha256 != canonical_sha256(state_identity):
            raise NormalizationIntegrityError("normalizer state identity hash mismatch")
        if (
            state_index["schema_version"] != NORMALIZER_STATE_VERSION
            or state_index["normalizer_version"] != NORMALIZER_VERSION
            or state_index["provider"] != PROVIDER_ID
        ):
            raise NormalizationIntegrityError("normalizer state version or provider mismatch")
        expected_manifest_metadata = {
            "protocol_config_sha256": state_index["protocol_config_sha256"],
            "provider": PROVIDER_ID,
            "rights_approval_sha256": state_index["rights_approval_sha256"],
            "state_sha256": state_index["state_sha256"],
        }
        if verified.manifest.get("metadata") != expected_manifest_metadata:
            raise NormalizationIntegrityError(
                "normalizer state manifest metadata does not match its state index"
            )
        descriptors = state_index["files"]
        if not isinstance(descriptors, dict) or set(descriptors) != set(_STATE_DATA_FILES):
            raise NormalizationIntegrityError("state referenced-file inventory is invalid")
        for name in _STATE_DATA_FILES:
            descriptor = descriptors.get(name)
            data = verified.files[name]
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"sha256", "size_bytes"}
                or descriptor["sha256"] != sha256_bytes(data)
                or descriptor["size_bytes"] != len(data)
            ):
                raise NormalizationIntegrityError(f"state transitive hash mismatch: {name}")

        approval_payload = _parse_canonical_json_buffer(
            verified.files["approval.json"], "rights approval"
        )
        approval = _rights_approval_from_payload(approval_payload)
        protocol_hash = state_index["protocol_config_sha256"]
        normalizer = cls(
            protocol_config_sha256=protocol_hash,
            rights_approval=approval,
        )
        if normalizer.rights_approval_sha256 != state_index["rights_approval_sha256"]:
            raise NormalizationIntegrityError("state rights-approval hash mismatch")

        articles_payload = _require_json_array(verified.files["articles.json"], "state articles")
        exclusions_payload = _require_json_array(
            verified.files["exclusions.json"], "state exclusions"
        )
        groups_payload = _require_json_array(verified.files["groups.json"], "state groups")
        links_payload = _require_json_array(
            verified.files["observation-links.json"], "state observation links"
        )
        chronology_payload = _parse_canonical_json_buffer(
            verified.files["chronology.json"], "state chronology"
        )
        if not isinstance(chronology_payload, dict) or set(chronology_payload) != {
            "closed_availability_through",
            "next_expected_interval_start",
            "terminal_intervals",
            "version",
        }:
            raise NormalizationIntegrityError(
                "persisted chronology fields do not match the contract"
            )
        if chronology_payload["version"] != CHRONOLOGY_VERSION:
            raise NormalizationIntegrityError("persisted chronology version is unsupported")
        raw_terminal_intervals = chronology_payload["terminal_intervals"]
        if not isinstance(raw_terminal_intervals, list):
            raise NormalizationIntegrityError("persisted terminal intervals must be an array")
        gap_evidence_payload = _require_json_array(
            verified.files["gap-evidence.json"], "state gap evidence"
        )
        try:
            articles = [validate_article_record(item) for item in articles_payload]
            exclusions = [
                _dataclass_from_exact_mapping(ExcludedObservation, item, "state exclusion")
                for item in exclusions_payload
            ]
            groups = [
                _dataclass_from_exact_mapping(GroupAnchor, item, "state group")
                for item in groups_payload
            ]
            links = [
                _dataclass_from_exact_mapping(ObservationLink, item, "state observation link")
                for item in links_payload
            ]
            terminal_intervals = [
                _dataclass_from_exact_mapping(TerminalInterval, item, "state terminal interval")
                for item in raw_terminal_intervals
            ]
            terminal_gap_evidence = [
                _terminal_gap_evidence_from_payload(item) for item in gap_evidence_payload
            ]
        except (ArticleValidationError, TypeError, ValueError) as exc:
            raise NormalizationIntegrityError("normalizer state payload validation failed") from exc

        normalizer._terminal_intervals = tuple(terminal_intervals)
        normalizer._next_expected_interval_start = chronology_payload[
            "next_expected_interval_start"
        ]
        normalizer._closed_availability_through = chronology_payload["closed_availability_through"]
        normalizer._gap_evidence_by_id = _unique_by_field(
            terminal_gap_evidence,
            "evidence_id",
            "state terminal gap evidence",
        )
        normalizer._validate_chronology(
            normalizer._terminal_intervals,
            normalizer._next_expected_interval_start,
            normalizer._closed_availability_through,
            normalizer._gap_evidence_by_id,
        )

        for article in articles:
            normalizer._versions_by_id[article.article_version_id] = article
            fingerprint = _fingerprint_for_article(article)
            normalizer._versions_by_fingerprint[(article.article_id, fingerprint)] = article
            if article.duplicate_group_id is None:  # pragma: no cover - eligible contract
                raise NormalizationIntegrityError("persisted article has no duplicate group")
            prior_group = normalizer._article_groups.setdefault(
                article.article_id, article.duplicate_group_id
            )
            if prior_group != article.duplicate_group_id:
                raise NormalizationIntegrityError("one article belongs to conflicting groups")
        normalizer._links = _unique_by_field(
            links, "provider_observation_id", "state observation link"
        )
        normalizer._exclusions = _unique_by_field(
            exclusions, "provider_observation_id", "state exclusion"
        )
        normalizer._groups_by_id = _unique_by_field(groups, "duplicate_group_id", "state group")

        raw_object_cache: dict[str, bytes] = {}
        parsed_snapshots: dict[str, SnapshotResult] = {}
        parsers: dict[tuple[int, int, int], GSGAdapter] = {}
        for interval in normalizer._terminal_intervals:
            if interval.snapshot_id is None:
                continue
            receipt, raw_bytes = _load_verified_snapshot_receipt(
                store,
                interval.snapshot_id,
                raw_object_cache=raw_object_cache,
            )
            if (
                receipt.filename_timestamp != interval.start_at
                or receipt.raw_snapshot_sha256 != interval.raw_snapshot_sha256
                or receipt.ingested_at != interval.ingested_at
                or receipt.raw_published_at != interval.raw_published_at
                or receipt.collection_mode != interval.collection_mode
                or receipt.input_class != interval.input_class
            ):
                raise NormalizationIntegrityError(
                    "terminal interval does not match its immutable raw receipt"
                )
            if interval.snapshot_state == "invalid":
                evidence = normalizer._gap_evidence_by_id.get(interval.gap_evidence_id or "")
                if evidence is None or evidence.expected_source_locator != receipt.source_locator:
                    raise NormalizationIntegrityError(
                        "invalid snapshot receipt does not match its expected gap locator"
                    )
            parser_key = (
                receipt.max_compressed_bytes,
                receipt.max_decompressed_bytes,
                receipt.max_json_lines,
            )
            parser = parsers.get(parser_key)
            if parser is None:
                parser = GSGAdapter(
                    store,
                    clock=lambda: datetime(1970, 1, 1, tzinfo=UTC),
                    max_compressed_bytes=receipt.max_compressed_bytes,
                    max_decompressed_bytes=receipt.max_decompressed_bytes,
                    max_json_lines=receipt.max_json_lines,
                )
                parsers[parser_key] = parser
            parsed_snapshot = parser._parse_snapshot(receipt, raw_bytes)
            if (
                parsed_snapshot.state != interval.snapshot_state
                or parsed_snapshot.json_line_count != interval.json_line_count
                or parsed_snapshot.error_code
                != (interval.terminal_reason if interval.snapshot_state == "invalid" else None)
            ):
                raise NormalizationIntegrityError(
                    "terminal interval does not match deterministic raw-byte parsing"
                )
            parsed_snapshots[interval.start_at] = parsed_snapshot

        normalizer._validate_state_components(
            normalizer._versions_by_id,
            normalizer._versions_by_fingerprint,
            normalizer._links,
            normalizer._exclusions,
            normalizer._groups_by_id,
            normalizer._article_groups,
        )
        normalizer._validate_rights_bound_state(normalizer._versions_by_id, normalizer._links)
        normalizer._validate_chronology_state_relationships(
            normalizer._terminal_intervals,
            normalizer._versions_by_id,
            normalizer._links,
            normalizer._exclusions,
        )
        replayed = cls(
            protocol_config_sha256=protocol_hash,
            rights_approval=approval,
        )
        try:
            for offset in range(0, len(normalizer._terminal_intervals), MAX_PLAN_INTERVALS):
                replay_intervals = normalizer._terminal_intervals[
                    offset : offset + MAX_PLAN_INTERVALS
                ]
                first_interval = replay_intervals[0]
                last_interval = replay_intervals[-1]
                plan = plan_retrieval(
                    first_interval.start_at,
                    last_interval.end_at_exclusive,
                )
                terminal_time = _parse_chronology_instant(
                    last_interval.terminal_at, "terminal interval terminal_at"
                )
                last_interval_start = _parse_chronology_minute(
                    last_interval.start_at, "terminal interval start"
                )
                terminal_as_of = format_utc_timestamp(
                    max(terminal_time, last_interval_start + PROVIDER_LAG)
                )
                replay_evidence = tuple(
                    normalizer._gap_evidence_by_id[interval.gap_evidence_id]
                    for interval in replay_intervals
                    if interval.gap_evidence_id is not None
                )
                replay_snapshots = tuple(
                    parsed_snapshots[interval.start_at]
                    for interval in replay_intervals
                    if interval.snapshot_id is not None
                )
                replayed.normalize(
                    replay_snapshots,
                    retrieval_plan=plan,
                    terminal_as_of=terminal_as_of,
                    gap_evidence=replay_evidence,
                )
        except ProviderIngestionError as exc:
            raise NormalizationIntegrityError(
                "persisted state failed deterministic raw-byte replay"
            ) from exc
        if replayed.export_state_files() != verified.files:
            raise NormalizationIntegrityError(
                "deterministic raw-byte replay does not reproduce the canonical state files"
            )
        return replayed

    def _materialize_candidate(self, observation: GSGObservation) -> _CandidateVersion:
        failures: list[tuple[str, str]] = []
        if observation.collection_mode == "historical_backfill":
            failures.append(
                (
                    "historical_backfill_without_availability",
                    "historical GSG observations are audit-only",
                )
            )
        try:
            first_seen = format_utc_timestamp(
                _parse_required_timestamp(observation.raw_published_at, "raw_published_at")
            )
            ingested = _parse_required_timestamp(observation.ingested_at, "ingested_at")
            if _parse_required_timestamp(first_seen, "first_seen_at") < ingested:
                raise ValueError("first_seen_at precedes ingested_at")
        except (ProviderIngestionError, ValueError) as exc:
            failures.append(("invalid_timestamp", str(exc)))
            first_seen = observation.raw_published_at

        canonical_url: str | None = None
        if not isinstance(observation.raw_url, str) or not observation.raw_url.strip():
            failures.append(("invalid_url_or_identifier", "endpoint URL is missing"))
        else:
            try:
                canonical_url = canonicalize_url(observation.raw_url)
            except _ObservationExcluded as exc:
                failures.append((exc.reason, exc.diagnostic))

        title: str | None = None
        if not isinstance(observation.raw_title, str):
            failures.append(("missing_title_and_content", "endpoint title is missing"))
        else:
            title = normalize_title(observation.raw_title)
            if not title:
                failures.append(("missing_title_and_content", "normalized title is blank"))

        language: str | None = None
        if not isinstance(observation.raw_language, str):
            failures.append(("unsupported_language", "endpoint language is missing"))
        else:
            try:
                language = normalize_gsg_language(observation.raw_language)
            except _ObservationExcluded as exc:
                failures.append((exc.reason, exc.diagnostic))

        provider_first_seen_at: str | None = None
        try:
            provider_first_seen_at = normalize_provider_timestamp(
                observation.raw_provider_first_seen_at
            )
        except _ObservationExcluded as exc:
            failures.append((exc.reason, exc.diagnostic))

        if title is not None and title and not selects_direct_btc(title):
            failures.append(("asset_mismatch", "title does not match direct BTC selector"))
        if not self._rights_apply(observation):
            failures.append(
                (
                    "license_restricted",
                    "no valid provider/scope/config-specific rights approval applies",
                )
            )
        blocking_reasons = {
            "hash_mismatch",
            "malformed_record",
            "invalid_timestamp",
            "missing_first_seen",
            "missing_identity",
            "invalid_url_or_identifier",
            "missing_title_and_content",
            "unsupported_language",
        }
        if any(reason in blocking_reasons for reason, _ in failures):
            reason, diagnostic = _primary_exclusion(failures)
            raise _ObservationExcluded(reason, diagnostic)
        if canonical_url is None or title is None or language is None:
            raise _ObservationExcluded("malformed_record", "validated candidate is incomplete")
        source = urlsplit(canonical_url).hostname
        if source is None:
            raise _ObservationExcluded("invalid_url_or_identifier", "canonical URL has no host")
        try:
            article_id = derive_article_id(PROVIDER_ID, None, canonical_url)
            content_hash = derive_content_hash(
                asset="BTC", content=None, language=language, source=source, title=title
            )
            fingerprint = derive_version_fingerprint(
                article_id=article_id,
                content_hash=content_hash,
                content=None,
                language=language,
                provider=PROVIDER_ID,
                source=source,
                title=title,
            )
        except (ArticleValidationError, TypeError, ValueError) as exc:
            raise _ObservationExcluded("missing_identity", str(exc)) from exc
        return _CandidateVersion(
            observation=observation,
            article_id=article_id,
            version_fingerprint=fingerprint,
            canonical_url=canonical_url,
            source=source,
            title=title,
            language=language,
            provider_first_seen_at=provider_first_seen_at,
            first_seen_at=first_seen,
            content_hash=content_hash,
            failures=tuple(failures),
        )

    def _rights_apply(self, observation: GSGObservation) -> bool:
        approval = self.rights_approval
        if approval is None or not approval.is_structurally_valid():
            return False
        if (
            not approval.approved
            or approval.network_access_authorized
            or approval.provider != PROVIDER_ID
            or approval.scope != RIGHTS_SCOPE
            or approval.protocol_config_sha256 != self.protocol_config_sha256
        ):
            return False
        if approval.approval_kind == "synthetic_fixture_only":
            return (
                observation.input_class == "synthetic_fixture"
                and observation.raw_snapshot_sha256 in approval.authorized_fixture_sha256
            )
        if approval.approval_kind == "provider_rights":
            return (
                observation.input_class == "provider_response"
                and not approval.authorized_fixture_sha256
            )
        return False

    @staticmethod
    def _causal_group_for_candidate(
        candidate: _CandidateVersion, groups: Mapping[str, GroupAnchor]
    ) -> str | None:
        first_seen = _parse_required_timestamp(candidate.first_seen_at, "initial_first_seen_at")
        fingerprint = _dedup_fingerprint(candidate.title, candidate.language)
        matches: list[GroupAnchor] = []
        for group in groups.values():
            if group.dedup_fingerprint != fingerprint:
                continue
            anchor_time = _parse_required_timestamp(
                group.initial_first_seen_at, "group.initial_first_seen_at"
            )
            if not timedelta(0) <= first_seen - anchor_time <= timedelta(hours=72):
                continue
            if candidate.canonical_url != group.canonical_url or candidate.source != group.source:
                matches.append(group)
        if not matches:
            return None
        return min(
            matches,
            key=lambda item: (item.initial_first_seen_at, item.anchor_article_id),
        ).duplicate_group_id

    @staticmethod
    def _article_from_candidate(
        candidate: _CandidateVersion, duplicate_group_id: str
    ) -> ArticleRecord:
        version_id = derive_article_version_id(
            article_id=candidate.article_id,
            first_seen_at=candidate.first_seen_at,
            language=candidate.language,
            content_hash=candidate.content_hash,
        )
        value = {
            "article_id": candidate.article_id,
            "article_version_id": version_id,
            "provider": PROVIDER_ID,
            "provider_article_id": None,
            "provider_observation_id": candidate.observation.provider_observation_id,
            "source": candidate.source,
            "canonical_url": candidate.canonical_url,
            "title": candidate.title,
            "content": None,
            "language": candidate.language,
            "published_at": None,
            "provider_first_seen_at": candidate.provider_first_seen_at,
            "first_seen_at": candidate.first_seen_at,
            "ingested_at": candidate.observation.ingested_at,
            "provider_updated_at": None,
            "asset": "BTC",
            "content_hash": candidate.content_hash,
            "raw_snapshot_sha256": candidate.observation.raw_snapshot_sha256,
            "point_in_time_eligible": True,
            "exclusion_reason": None,
            "duplicate_group_id": duplicate_group_id,
        }
        try:
            return validate_article_record(value)
        except ArticleValidationError as exc:
            raise NormalizationIntegrityError("candidate failed final article validation") from exc

    def _result(
        self,
        touched_versions: Mapping[str, ArticleRecord],
        links: Mapping[str, ObservationLink],
        exclusions: Mapping[str, ExcludedObservation],
    ) -> NormalizationResult:
        articles_tuple = tuple(
            sorted(
                touched_versions.values(),
                key=lambda item: (
                    item.first_seen_at or "",
                    item.article_id,
                    item.article_version_id,
                ),
            )
        )
        links_tuple = tuple(sorted(links.values(), key=lambda item: item.provider_observation_id))
        exclusions_tuple = tuple(
            sorted(exclusions.values(), key=lambda item: item.provider_observation_id)
        )
        hash_payload = {
            "articles": [article.to_dict() for article in articles_tuple],
            "exclusions": [exclusion.to_dict() for exclusion in exclusions_tuple],
            "normalizer_version": NORMALIZER_VERSION,
            "observation_links": [link.to_dict() for link in links_tuple],
            "protocol_config_sha256": self.protocol_config_sha256,
            "rights_approval_sha256": self.rights_approval_sha256,
        }
        return NormalizationResult(
            articles=articles_tuple,
            observation_links=links_tuple,
            exclusions=exclusions_tuple,
            protocol_config_sha256=self.protocol_config_sha256,
            rights_approval_sha256=self.rights_approval_sha256,
            semantic_sha256=canonical_sha256(hash_payload),
        )

    @staticmethod
    def _validate_state_components(
        versions: Mapping[str, ArticleRecord],
        fingerprints: Mapping[tuple[str, str], ArticleRecord],
        links: Mapping[str, ObservationLink],
        exclusions: Mapping[str, ExcludedObservation],
        groups: Mapping[str, GroupAnchor],
        article_groups: Mapping[str, str],
    ) -> None:
        if set(links) & set(exclusions):
            raise NormalizationIntegrityError("an observation is both linked and excluded")
        if len(versions) != len(fingerprints):
            raise NormalizationIntegrityError("version and fingerprint indexes disagree")
        article_times: dict[tuple[str, str], set[str]] = {}
        initial_records: dict[str, ArticleRecord] = {}
        for version_id, article in versions.items():
            if version_id != article.article_version_id:
                raise NormalizationIntegrityError("version index key mismatch")
            fingerprint = _fingerprint_for_article(article)
            if fingerprints.get((article.article_id, fingerprint)) != article:
                raise NormalizationIntegrityError("fingerprint index mismatch")
            if article.first_seen_at is None or article.duplicate_group_id is None:
                raise NormalizationIntegrityError("eligible state article is incomplete")
            article_times.setdefault((article.article_id, article.first_seen_at), set()).add(
                fingerprint
            )
            prior = initial_records.get(article.article_id)
            if prior is None or (article.first_seen_at, article.article_version_id) < (
                prior.first_seen_at or "",
                prior.article_version_id,
            ):
                initial_records[article.article_id] = article
            if article_groups.get(article.article_id) != article.duplicate_group_id:
                raise NormalizationIntegrityError("article-to-group index mismatch")
        if any(len(items) > 1 for items in article_times.values()):
            raise NormalizationIntegrityError("persisted state contains same-time revisions")
        if set(article_groups) != set(initial_records):
            raise NormalizationIntegrityError("article group index is incomplete")
        representative_links: dict[str, str] = {}
        for observation_id, link in links.items():
            if observation_id != link.provider_observation_id:
                raise NormalizationIntegrityError("observation-link index key mismatch")
            if not _is_sha256(link.article_version_id) or link.article_version_id not in versions:
                raise NormalizationIntegrityError("observation link references a missing version")
            if (
                not _is_sha256(link.raw_snapshot_sha256)
                or _raw_hash_from_observation_id(observation_id) != link.raw_snapshot_sha256
                or not isinstance(link.input_class, str)
                or link.input_class not in {"provider_response", "synthetic_fixture"}
                or not isinstance(link.reused_existing_version, bool)
            ):
                raise NormalizationIntegrityError("observation link provenance is invalid")
            if not link.reused_existing_version:
                if link.article_version_id in representative_links:
                    raise NormalizationIntegrityError(
                        "article version has multiple representative observations"
                    )
                representative_links[link.article_version_id] = observation_id
        for observation_id, exclusion in exclusions.items():
            if observation_id != exclusion.provider_observation_id:
                raise NormalizationIntegrityError("exclusion index key mismatch")
            if (
                not isinstance(exclusion.reason, str)
                or exclusion.reason not in _EXCLUSION_RANK
                or not isinstance(exclusion.diagnostic, str)
            ):
                raise NormalizationIntegrityError("persisted exclusion reason is unknown")
            if (
                not _is_sha256(exclusion.raw_snapshot_sha256)
                or _raw_hash_from_observation_id(observation_id) != exclusion.raw_snapshot_sha256
                or not isinstance(exclusion.input_class, str)
                or exclusion.input_class not in {"provider_response", "synthetic_fixture"}
            ):
                raise NormalizationIntegrityError("exclusion provenance is invalid")
        for article in versions.values():
            first_link = links.get(article.provider_observation_id)
            if (
                first_link is None
                or first_link.article_version_id != article.article_version_id
                or first_link.raw_snapshot_sha256 != article.raw_snapshot_sha256
                or first_link.reused_existing_version
                or representative_links.get(article.article_version_id)
                != article.provider_observation_id
            ):
                raise NormalizationIntegrityError(
                    "article first observation is not bound to its provenance link"
                )
        used_group_ids = set(article_groups.values())
        if set(groups) != used_group_ids:
            raise NormalizationIntegrityError("group anchors do not match used groups")
        for group_id, group in groups.items():
            if (
                not _is_sha256(group_id)
                or group_id != group.duplicate_group_id
                or not _is_sha256(group.anchor_article_id)
                or not _is_sha256(group.dedup_fingerprint)
                or not isinstance(group.initial_first_seen_at, str)
                or not isinstance(group.canonical_url, str)
                or not isinstance(group.source, str)
            ):
                raise NormalizationIntegrityError("group index key mismatch")
            if derive_duplicate_group_id(group.anchor_article_id) != group_id:
                raise NormalizationIntegrityError("permanent group anchor identity mismatch")
            anchor = initial_records.get(group.anchor_article_id)
            if anchor is None:
                raise NormalizationIntegrityError("group anchor article is missing")
            if (
                anchor.first_seen_at != group.initial_first_seen_at
                or anchor.canonical_url != group.canonical_url
                or anchor.source != group.source
                or anchor.title is None
                or _dedup_fingerprint(anchor.title, anchor.language) != group.dedup_fingerprint
            ):
                raise NormalizationIntegrityError("group anchor fields are inconsistent")

    def _validate_rights_bound_state(
        self,
        versions: Mapping[str, ArticleRecord],
        links: Mapping[str, ObservationLink],
    ) -> None:
        if not versions:
            return
        approval = self.rights_approval
        if (
            approval is None
            or not approval.is_structurally_valid()
            or not approval.approved
            or approval.network_access_authorized
            or approval.provider != PROVIDER_ID
            or approval.scope != RIGHTS_SCOPE
            or approval.protocol_config_sha256 != self.protocol_config_sha256
        ):
            raise NormalizationIntegrityError("eligible state lacks applicable rights approval")
        if approval.approval_kind == "synthetic_fixture_only":
            allowed = set(approval.authorized_fixture_sha256)
            referenced = {item.raw_snapshot_sha256 for item in versions.values()}
            referenced.update(item.raw_snapshot_sha256 for item in links.values())
            if not referenced <= allowed or any(
                item.input_class != "synthetic_fixture" for item in links.values()
            ):
                raise NormalizationIntegrityError(
                    "synthetic state references bytes outside its fixture-only approval"
                )
        elif (
            approval.approval_kind != "provider_rights"
            or approval.authorized_fixture_sha256
            or any(item.input_class != "provider_response" for item in links.values())
        ):
            raise NormalizationIntegrityError("eligible state rights approval kind is invalid")


class _ObservationExcluded(Exception):
    def __init__(self, reason: str, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.reason = reason
        self.diagnostic = diagnostic


def _derive_snapshot_id(
    *,
    collection_mode: str,
    filename_timestamp: str,
    input_class: str,
    raw_snapshot_sha256: str,
    max_compressed_bytes: int,
    max_decompressed_bytes: int,
    max_json_lines: int,
) -> str:
    return canonical_sha256(
        {
            "collection_mode": collection_mode,
            "filename_timestamp": filename_timestamp,
            "input_class": input_class,
            "parser_policy": {
                "max_compressed_bytes": max_compressed_bytes,
                "max_decompressed_bytes": max_decompressed_bytes,
                "max_json_lines": max_json_lines,
                "parser_policy_version": PARSER_POLICY_VERSION,
                "parser_version": PARSER_VERSION,
            },
            "raw_snapshot_sha256": raw_snapshot_sha256,
            "version": SNAPSHOT_IDENTITY_VERSION,
        }
    )


def plan_retrieval(
    start_at: str,
    end_at_exclusive: str,
    *,
    maximum_intervals: int = MAX_PLAN_INTERVALS,
) -> RetrievalPlan:
    """Create a closed-open, minute-aligned, explicitly bounded retrieval schedule."""
    if not 1 <= maximum_intervals <= MAX_PLAN_INTERVALS:
        raise ProviderIngestionError(
            f"maximum_intervals must be between 1 and {MAX_PLAN_INTERVALS}"
        )
    start = _parse_minute_timestamp(start_at, "start_at")
    end = _parse_minute_timestamp(end_at_exclusive, "end_at_exclusive")
    if end <= start:
        raise ProviderIngestionError("retrieval end must be after start")
    count = int((end - start) / EXPECTED_INTERVAL)
    if count > maximum_intervals:
        raise ProviderIngestionError("retrieval plan exceeds the explicit interval cap")
    intervals: list[RetrievalInterval] = []
    for offset in range(count):
        timestamp = start + (offset * EXPECTED_INTERVAL)
        filename_timestamp = format_utc_timestamp(timestamp)
        intervals.append(
            RetrievalInterval(
                filename_timestamp=filename_timestamp,
                due_at=format_utc_timestamp(timestamp + PROVIDER_LAG),
                relative_path=f"gdeltv3/gsg/{timestamp:%Y%m%d%H%M%S}.gsg.json.gz",
            )
        )
    identity = {
        "end_at_exclusive": format_utc_timestamp(end),
        "expected_interval_seconds": 60,
        "intervals": [interval.to_dict() for interval in intervals],
        "provider_lag_seconds": 1800,
        "start_at": format_utc_timestamp(start),
        "version": "gdelt-gsg-retrieval-plan-v1",
    }
    return RetrievalPlan(
        plan_id=canonical_sha256(identity),
        start_at=format_utc_timestamp(start),
        end_at_exclusive=format_utc_timestamp(end),
        intervals=tuple(intervals),
    )


def _validate_retrieval_plan(plan: RetrievalPlan) -> None:
    if not isinstance(plan, RetrievalPlan):
        raise NormalizationIntegrityError("retrieval plan does not match its contract")
    try:
        expected = plan_retrieval(plan.start_at, plan.end_at_exclusive)
    except ProviderIngestionError as exc:
        raise NormalizationIntegrityError("retrieval plan is malformed") from exc
    if plan != expected:
        raise NormalizationIntegrityError(
            "retrieval plan identity or expected-minute schedule is inconsistent"
        )


def expected_gsg_source_locator(interval: RetrievalInterval) -> str:
    """Return the frozen archive identity for one planned interval; performs no I/O."""
    if not isinstance(interval, RetrievalInterval):
        raise ProviderIngestionError("expected locator requires one retrieval interval")
    return f"{GSG_ARCHIVE_ROOT}{interval.relative_path}"


def _expected_source_locator(interval: RetrievalInterval) -> str:
    return expected_gsg_source_locator(interval)


def _expected_source_locator_at(interval_start: str) -> str:
    start = _parse_chronology_minute(interval_start, "gap interval start")
    plan = plan_retrieval(
        format_utc_timestamp(start),
        format_utc_timestamp(start + EXPECTED_INTERVAL),
    )
    return _expected_source_locator(plan.intervals[0])


def _validate_terminal_gap_evidence(
    evidence: TerminalGapEvidence,
    *,
    interval_start: str,
    interval_end_exclusive: str,
    expected_source_locator: str,
    protocol_config_sha256: str,
    terminal_as_of: str | None,
) -> None:
    if not isinstance(evidence, TerminalGapEvidence):
        raise NormalizationIntegrityError("terminal gap evidence has an invalid type")
    string_fields = (
        evidence.evidence_id,
        evidence.evidence_sha256,
        evidence.provider,
        evidence.scope,
        evidence.interval_start,
        evidence.interval_end_exclusive,
        evidence.expected_source_locator,
        evidence.collection_mode,
        evidence.input_class,
        evidence.retry_policy_version,
        evidence.final_terminal_disposition,
        evidence.terminal_at,
        evidence.protocol_config_sha256,
        evidence.version,
    )
    if not all(isinstance(value, str) and value for value in string_fields):
        raise NormalizationIntegrityError("terminal gap evidence scalar fields are invalid")
    if not _is_sha256(evidence.evidence_id) or not _is_sha256(evidence.evidence_sha256):
        raise NormalizationIntegrityError("terminal gap evidence identity fields are invalid")
    if not isinstance(evidence.attempts, tuple) or not all(
        isinstance(attempt, GapAttempt) for attempt in evidence.attempts
    ):
        raise NormalizationIntegrityError("terminal gap evidence attempts are not immutable")
    for attempt in evidence.attempts:
        if (
            not isinstance(attempt.attempted_at, str)
            or not attempt.attempted_at
            or not isinstance(attempt.retry_disposition, str)
            or not attempt.retry_disposition
        ):
            raise NormalizationIntegrityError("terminal gap attempt scalar fields are invalid")
    try:
        identity = evidence.identity_payload()
        expected_id = canonical_sha256(
            {
                "identity": identity,
                "identity_version": "gdelt-gsg-terminal-gap-identity-v1",
            }
        )
        expected_sha256 = canonical_sha256({**identity, "evidence_id": expected_id})
    except (CanonicalizationError, TypeError, ValueError, AttributeError) as exc:
        raise NormalizationIntegrityError(
            "terminal gap evidence cannot be canonically serialized"
        ) from exc
    if evidence.evidence_id != expected_id or evidence.evidence_sha256 != expected_sha256:
        raise NormalizationIntegrityError("terminal gap evidence identity or hash mismatch")
    if (
        evidence.version != GAP_EVIDENCE_VERSION
        or evidence.provider != PROVIDER_ID
        or evidence.scope != RIGHTS_SCOPE
        or evidence.protocol_config_sha256 != protocol_config_sha256
        or evidence.retry_policy_version != RETRY_POLICY_VERSION
        or evidence.interval_start != interval_start
        or evidence.interval_end_exclusive != interval_end_exclusive
        or evidence.expected_source_locator != expected_source_locator
        or evidence.collection_mode != "prospective"
        or evidence.input_class != "synthetic_fixture"
        or evidence.network_access_authorized is not False
    ):
        raise NormalizationIntegrityError(
            "terminal gap evidence context does not match the frozen offline contract"
        )
    start = _parse_chronology_minute(evidence.interval_start, "gap evidence interval start")
    end = _parse_chronology_minute(evidence.interval_end_exclusive, "gap evidence interval end")
    if end - start != EXPECTED_INTERVAL:
        raise NormalizationIntegrityError("terminal gap evidence interval is not one minute")
    if (evidence.observed_snapshot_id is None) != (
        evidence.observed_raw_snapshot_sha256 is None
    ) or (
        evidence.observed_snapshot_id is not None
        and (
            not _is_sha256(evidence.observed_snapshot_id)
            or not _is_sha256(evidence.observed_raw_snapshot_sha256)
        )
    ):
        raise NormalizationIntegrityError(
            "terminal gap evidence observed-snapshot binding is invalid"
        )
    if (
        not isinstance(evidence.attempt_count, int)
        or isinstance(evidence.attempt_count, bool)
        or evidence.attempt_count != len(evidence.attempts)
        or not 1 <= evidence.attempt_count <= GSGRetryPolicy.maximum_attempts
    ):
        raise NormalizationIntegrityError("terminal gap evidence attempt count is invalid")
    due_at = start + PROVIDER_LAG
    previous_attempt_at: datetime | None = None
    previous_delay: float | None = None
    policy = GSGRetryPolicy()
    final_decision: RetryDecision | None = None
    for expected_number, attempt in enumerate(evidence.attempts, start=1):
        if not isinstance(attempt, GapAttempt):
            raise NormalizationIntegrityError("terminal gap attempt has an invalid type")
        if attempt.attempt_number != expected_number:
            raise NormalizationIntegrityError(
                "terminal gap attempts are duplicated, skipped, or reordered"
            )
        attempted_at = _parse_chronology_instant(attempt.attempted_at, "gap attempt attempted_at")
        if attempted_at < due_at:
            raise NormalizationIntegrityError("terminal gap attempt precedes interval due time")
        if previous_attempt_at is not None:
            if attempted_at <= previous_attempt_at:
                raise NormalizationIntegrityError(
                    "terminal gap attempt timestamps must strictly increase"
                )
            if previous_delay is None or attempted_at < previous_attempt_at + timedelta(
                seconds=previous_delay
            ):
                raise NormalizationIntegrityError(
                    "terminal gap attempts violate the frozen retry delay"
                )
        try:
            decision = policy.decide(
                attempt=attempt.attempt_number,
                http_status=attempt.http_status,
                error_kind=attempt.error_kind,
                retry_after_seconds=attempt.retry_after_seconds,
            )
        except ProviderIngestionError as exc:
            raise NormalizationIntegrityError(
                "terminal gap attempt violates the retry policy"
            ) from exc
        if decision.disposition != attempt.retry_disposition:
            raise NormalizationIntegrityError(
                "terminal gap attempt disposition contradicts the retry policy"
            )
        if expected_number < evidence.attempt_count and decision.disposition != "retry":
            raise NormalizationIntegrityError(
                "terminal gap evidence continues after a terminal attempt"
            )
        previous_attempt_at = attempted_at
        previous_delay = decision.delay_seconds
        final_decision = decision
    if final_decision is None or final_decision.disposition != "gap":
        raise NormalizationIntegrityError("terminal gap evidence is not terminal")
    retryable_final = (
        evidence.attempts[-1].error_kind == "network_transport_error"
        or evidence.attempts[-1].http_status in {408, 429}
        or (
            evidence.attempts[-1].http_status is not None
            and 500 <= evidence.attempts[-1].http_status <= 599
        )
    )
    expected_terminal_disposition = "retry_exhausted" if retryable_final else "non_retryable"
    if evidence.final_terminal_disposition != expected_terminal_disposition or (
        retryable_final and evidence.attempt_count != policy.maximum_attempts
    ):
        raise NormalizationIntegrityError(
            "terminal gap evidence final disposition is contradictory"
        )
    if evidence.observed_snapshot_id is not None:
        if evidence.attempts[-1].error_kind not in _SAFE_SNAPSHOT_ERROR_CODES:
            raise NormalizationIntegrityError(
                "observed invalid snapshot lacks a bounded parse error"
            )
    elif evidence.attempts[-1].error_kind in _SAFE_SNAPSHOT_ERROR_CODES:
        raise NormalizationIntegrityError(
            "snapshot parse error lacks an observed raw snapshot binding"
        )
    terminal_at = _parse_chronology_instant(evidence.terminal_at, "gap evidence terminal_at")
    if previous_attempt_at is None or terminal_at < previous_attempt_at:
        raise NormalizationIntegrityError("terminal gap timestamp precedes the final attempt")
    if terminal_as_of is not None:
        as_of = _parse_chronology_instant(terminal_as_of, "terminal_as_of")
        if terminal_at > as_of or any(
            _parse_chronology_instant(attempt.attempted_at, "gap attempt attempted_at") > as_of
            for attempt in evidence.attempts
        ):
            raise NormalizationIntegrityError(
                "terminal gap evidence contains a future attempt or terminal timestamp"
            )


def _terminal_interval_facts(
    plan: RetrievalPlan,
    snapshots: Sequence[SnapshotResult],
    gap_evidence: Sequence[TerminalGapEvidence],
    *,
    protocol_config_sha256: str,
    terminal_as_of: str,
    prior_closed_availability_through: str | None,
) -> tuple[tuple[TerminalInterval, ...], tuple[TerminalGapEvidence, ...], str | None]:
    """Derive verified terminal facts without mutating normalizer state."""
    snapshot_by_timestamp = {
        snapshot.receipt.filename_timestamp: snapshot for snapshot in snapshots
    }
    evidence_by_interval: dict[str, TerminalGapEvidence] = {}
    evidence_ids: set[str] = set()
    for evidence in gap_evidence:
        if not isinstance(evidence, TerminalGapEvidence):
            raise NormalizationIntegrityError("terminal gap evidence does not match its contract")
        if not _is_sha256(evidence.evidence_id) or not isinstance(evidence.interval_start, str):
            raise NormalizationIntegrityError("terminal gap evidence identity is malformed")
        if evidence.evidence_id in evidence_ids:
            raise NormalizationIntegrityError("duplicate terminal gap evidence identity")
        if evidence.interval_start in evidence_by_interval:
            raise NormalizationIntegrityError("duplicate terminal gap evidence interval")
        evidence_ids.add(evidence.evidence_id)
        evidence_by_interval[evidence.interval_start] = evidence
    as_of_time = _parse_chronology_instant(terminal_as_of, "terminal_as_of")
    closed_through = (
        _parse_chronology_instant(
            prior_closed_availability_through,
            "closed_availability_through",
        )
        if prior_closed_availability_through is not None
        else None
    )
    facts: list[TerminalInterval] = []
    accepted_evidence: list[TerminalGapEvidence] = []
    for interval in plan.intervals:
        start = _parse_minute_timestamp(interval.filename_timestamp, "interval.filename_timestamp")
        end = format_utc_timestamp(start + EXPECTED_INTERVAL)
        snapshot = snapshot_by_timestamp.get(interval.filename_timestamp)
        evidence = evidence_by_interval.pop(interval.filename_timestamp, None)
        if snapshot is None:
            if evidence is None:
                raise NormalizationIntegrityError(
                    "missing snapshot requires verified immutable terminal gap evidence"
                )
            _validate_terminal_gap_evidence(
                evidence,
                interval_start=interval.filename_timestamp,
                interval_end_exclusive=end,
                expected_source_locator=_expected_source_locator(interval),
                protocol_config_sha256=protocol_config_sha256,
                terminal_as_of=terminal_as_of,
            )
            if (
                evidence.observed_snapshot_id is not None
                or evidence.observed_raw_snapshot_sha256 is not None
            ):
                raise NormalizationIntegrityError(
                    "missing-file gap evidence cannot reference a delivered snapshot"
                )
            terminal_time = _parse_chronology_instant(
                evidence.terminal_at, "gap evidence terminal_at"
            )
            if closed_through is not None and terminal_time < closed_through:
                raise NormalizationIntegrityError(
                    "terminal gap evidence regresses or equals closed causal availability"
                )
            closed_through = _exclusive_boundary_after(terminal_time)
            facts.append(
                TerminalInterval(
                    start_at=interval.filename_timestamp,
                    end_at_exclusive=end,
                    outcome="provider_gap",
                    snapshot_state="missing",
                    snapshot_id=None,
                    raw_snapshot_sha256=None,
                    ingested_at=None,
                    raw_published_at=None,
                    terminal_at=evidence.terminal_at,
                    json_line_count=None,
                    collection_mode=evidence.collection_mode,
                    input_class=evidence.input_class,
                    gap_evidence_id=evidence.evidence_id,
                    gap_evidence_sha256=evidence.evidence_sha256,
                    terminal_reason="verified_terminal_gap_evidence",
                )
            )
            accepted_evidence.append(evidence)
            continue
        _validate_snapshot_terminal_fact(snapshot, interval.filename_timestamp)
        published_at = _parse_chronology_instant(
            snapshot.receipt.raw_published_at, "receipt.raw_published_at"
        )
        if closed_through is not None and published_at < closed_through:
            raise NormalizationIntegrityError(
                "raw snapshot publication regresses or equals closed causal availability"
            )
        if snapshot.state == "complete":
            if evidence is not None:
                raise NormalizationIntegrityError(
                    "complete snapshot cannot consume terminal gap evidence"
                )
            terminal_time = published_at
            facts.append(
                TerminalInterval(
                    start_at=interval.filename_timestamp,
                    end_at_exclusive=end,
                    outcome="retrieved_and_normalized",
                    snapshot_state="complete",
                    snapshot_id=snapshot.receipt.snapshot_id,
                    raw_snapshot_sha256=snapshot.receipt.raw_snapshot_sha256,
                    ingested_at=snapshot.receipt.ingested_at,
                    raw_published_at=snapshot.receipt.raw_published_at,
                    terminal_at=snapshot.receipt.raw_published_at,
                    json_line_count=snapshot.json_line_count,
                    collection_mode=snapshot.receipt.collection_mode,
                    input_class=snapshot.receipt.input_class,
                    gap_evidence_id=None,
                    gap_evidence_sha256=None,
                    terminal_reason=None,
                )
            )
        else:
            if evidence is None:
                raise NormalizationIntegrityError(
                    "invalid snapshot requires verified immutable terminal gap evidence"
                )
            _validate_terminal_gap_evidence(
                evidence,
                interval_start=interval.filename_timestamp,
                interval_end_exclusive=end,
                expected_source_locator=_expected_source_locator(interval),
                protocol_config_sha256=protocol_config_sha256,
                terminal_as_of=terminal_as_of,
            )
            if (
                evidence.observed_snapshot_id != snapshot.receipt.snapshot_id
                or evidence.observed_raw_snapshot_sha256 != snapshot.receipt.raw_snapshot_sha256
                or evidence.attempts[-1].error_kind != snapshot.error_code
                or evidence.expected_source_locator != snapshot.receipt.source_locator
            ):
                raise NormalizationIntegrityError(
                    "invalid-snapshot gap evidence does not bind its raw receipt"
                )
            terminal_time = _parse_chronology_instant(
                evidence.terminal_at, "gap evidence terminal_at"
            )
            if terminal_time < published_at:
                raise NormalizationIntegrityError(
                    "invalid-snapshot gap terminal precedes raw publication"
                )
            facts.append(
                TerminalInterval(
                    start_at=interval.filename_timestamp,
                    end_at_exclusive=end,
                    outcome="provider_gap",
                    snapshot_state="invalid",
                    snapshot_id=snapshot.receipt.snapshot_id,
                    raw_snapshot_sha256=snapshot.receipt.raw_snapshot_sha256,
                    ingested_at=snapshot.receipt.ingested_at,
                    raw_published_at=snapshot.receipt.raw_published_at,
                    terminal_at=evidence.terminal_at,
                    json_line_count=snapshot.json_line_count,
                    collection_mode=snapshot.receipt.collection_mode,
                    input_class=snapshot.receipt.input_class,
                    gap_evidence_id=evidence.evidence_id,
                    gap_evidence_sha256=evidence.evidence_sha256,
                    terminal_reason=snapshot.error_code,
                )
            )
            accepted_evidence.append(evidence)
        if terminal_time > as_of_time:
            raise NormalizationIntegrityError(
                "terminal interval availability lies after terminal_as_of"
            )
        if closed_through is not None and terminal_time < closed_through:
            raise NormalizationIntegrityError(
                "terminal interval regresses or equals closed causal availability"
            )
        closed_through = _exclusive_boundary_after(terminal_time)
    if evidence_by_interval:
        raise NormalizationIntegrityError(
            "terminal gap evidence is outside the plan or was not consumed"
        )
    return (
        tuple(facts),
        tuple(accepted_evidence),
        format_utc_timestamp(closed_through) if closed_through is not None else None,
    )


def _validate_snapshot_terminal_fact(snapshot: SnapshotResult, interval_start: str) -> None:
    if not isinstance(snapshot, SnapshotResult) or not isinstance(
        snapshot.receipt, RawSnapshotReceipt
    ):
        raise NormalizationIntegrityError("terminal snapshot does not match its schema")
    if snapshot.receipt.filename_timestamp != interval_start:
        raise NormalizationIntegrityError(
            "snapshot receipt does not belong to its terminal interval"
        )
    if not _is_sha256(snapshot.receipt.snapshot_id) or not _is_sha256(
        snapshot.receipt.raw_snapshot_sha256
    ):
        raise NormalizationIntegrityError("terminal snapshot identity or raw hash is invalid")
    ingested_at = _parse_chronology_instant(snapshot.receipt.ingested_at, "receipt.ingested_at")
    raw_published_at = _parse_chronology_instant(
        snapshot.receipt.raw_published_at, "receipt.raw_published_at"
    )
    if raw_published_at < ingested_at:
        raise NormalizationIntegrityError("raw snapshot publication precedes ingestion")
    receipt = snapshot.receipt
    if (
        receipt.parser_version != PARSER_VERSION
        or receipt.parser_policy_version != PARSER_POLICY_VERSION
        or not isinstance(receipt.collection_mode, str)
        or receipt.collection_mode not in {"prospective", "historical_backfill"}
        or not isinstance(receipt.input_class, str)
        or receipt.input_class not in {"provider_response", "synthetic_fixture"}
        or not isinstance(receipt.max_compressed_bytes, int)
        or isinstance(receipt.max_compressed_bytes, bool)
        or not 1 <= receipt.max_compressed_bytes <= MAX_COMPRESSED_BYTES
        or not isinstance(receipt.max_decompressed_bytes, int)
        or isinstance(receipt.max_decompressed_bytes, bool)
        or not 1 <= receipt.max_decompressed_bytes <= MAX_DECOMPRESSED_BYTES
        or not isinstance(receipt.max_json_lines, int)
        or isinstance(receipt.max_json_lines, bool)
        or not 1 <= receipt.max_json_lines <= MAX_JSON_LINES
        or not isinstance(receipt.compressed_size_bytes, int)
        or isinstance(receipt.compressed_size_bytes, bool)
        or receipt.compressed_size_bytes < 0
        or receipt.compressed_size_bytes > receipt.max_compressed_bytes
        or not isinstance(receipt.source_locator, str)
        or not receipt.source_locator
    ):
        raise NormalizationIntegrityError("terminal snapshot receipt fields are invalid")
    expected_snapshot_id = _derive_snapshot_id(
        collection_mode=receipt.collection_mode,
        filename_timestamp=receipt.filename_timestamp,
        input_class=receipt.input_class,
        raw_snapshot_sha256=receipt.raw_snapshot_sha256,
        max_compressed_bytes=receipt.max_compressed_bytes,
        max_decompressed_bytes=receipt.max_decompressed_bytes,
        max_json_lines=receipt.max_json_lines,
    )
    if receipt.snapshot_id != expected_snapshot_id:
        raise NormalizationIntegrityError("terminal snapshot receipt identity is invalid")
    try:
        if redact_url(receipt.source_locator) != receipt.source_locator:
            raise NormalizationIntegrityError(
                "terminal snapshot receipt source locator is not credential-redacted"
            )
    except ProviderIngestionError as exc:
        raise NormalizationIntegrityError("terminal snapshot source locator is invalid") from exc
    if (
        not isinstance(snapshot.json_line_count, int)
        or isinstance(snapshot.json_line_count, bool)
        or snapshot.json_line_count < 0
        or snapshot.json_line_count > MAX_JSON_LINES
        or snapshot.json_line_count > receipt.max_json_lines
    ):
        raise NormalizationIntegrityError("terminal snapshot line count is invalid")
    if not isinstance(snapshot.observations, tuple):
        raise NormalizationIntegrityError("terminal snapshot observations are not immutable")
    if snapshot.state == "complete":
        if snapshot.error_code is not None:
            raise NormalizationIntegrityError(
                "complete terminal snapshot cannot contain an error code"
            )
        if len(snapshot.observations) != snapshot.json_line_count * 2:
            raise NormalizationIntegrityError(
                "complete terminal snapshot line and observation counts disagree"
            )
    elif snapshot.state == "invalid":
        if (
            snapshot.observations
            or not isinstance(snapshot.error_code, str)
            or snapshot.error_code not in _SAFE_SNAPSHOT_ERROR_CODES
        ):
            raise NormalizationIntegrityError("invalid terminal snapshot facts are contradictory")
    else:
        raise NormalizationIntegrityError("snapshot has an unsupported terminal state")
    observed_positions: set[tuple[int, str]] = set()
    for observation in snapshot.observations:
        if not isinstance(observation, GSGObservation):
            raise NormalizationIntegrityError("terminal snapshot observation schema is invalid")
        if (
            observation.filename_timestamp != interval_start
            or observation.raw_snapshot_sha256 != snapshot.receipt.raw_snapshot_sha256
            or observation.ingested_at != snapshot.receipt.ingested_at
            or observation.raw_published_at != snapshot.receipt.raw_published_at
            or observation.collection_mode != snapshot.receipt.collection_mode
            or observation.input_class != snapshot.receipt.input_class
        ):
            raise NormalizationIntegrityError(
                "terminal snapshot observations have inconsistent provenance"
            )
        if (
            not isinstance(observation.zero_based_line_number, int)
            or isinstance(observation.zero_based_line_number, bool)
            or not 0 <= observation.zero_based_line_number < snapshot.json_line_count
            or not isinstance(observation.endpoint_side, str)
            or observation.endpoint_side not in {"from", "to"}
        ):
            raise NormalizationIntegrityError("terminal snapshot observation position is invalid")
        expected_observation_id = (
            f"gdelt-gsg:{interval_start}:{receipt.raw_snapshot_sha256}:"
            f"{observation.zero_based_line_number}:{observation.endpoint_side}"
        )
        position = (observation.zero_based_line_number, observation.endpoint_side)
        if observation.provider_observation_id != expected_observation_id or (
            position in observed_positions
        ):
            raise NormalizationIntegrityError(
                "terminal snapshot observation identity is duplicated or inconsistent"
            )
        observed_positions.add(position)


def build_coverage_report(
    plan: RetrievalPlan,
    snapshots: Iterable[SnapshotResult],
    *,
    as_of: str,
    gap_evidence: Iterable[TerminalGapEvidence] = (),
    protocol_config_sha256: str | None = None,
) -> CoverageReport:
    """Report only evidence-backed gaps; caller omission remains unresolved."""
    _validate_retrieval_plan(plan)
    as_of_time = _parse_required_timestamp(as_of, "as_of")
    snapshot_by_timestamp: dict[str, SnapshotResult] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, SnapshotResult) or not isinstance(
            snapshot.receipt, RawSnapshotReceipt
        ):
            raise ProviderIngestionError("coverage snapshot does not match its contract")
        timestamp = snapshot.receipt.filename_timestamp
        if timestamp in snapshot_by_timestamp:
            raise ProviderIngestionError(f"duplicate snapshot receipt interval: {timestamp}")
        snapshot_by_timestamp[timestamp] = snapshot
    planned_timestamps = {interval.filename_timestamp for interval in plan.intervals}
    unplanned = sorted(set(snapshot_by_timestamp) - planned_timestamps)
    if unplanned:
        raise ProviderIngestionError(
            f"snapshot receipts fall outside the retrieval plan: {unplanned}"
        )
    evidence_by_interval: dict[str, TerminalGapEvidence] = {}
    evidence_ids: set[str] = set()
    for evidence in gap_evidence:
        if not isinstance(evidence, TerminalGapEvidence):
            raise ProviderIngestionError("gap evidence does not match its contract")
        if not _is_sha256(evidence.evidence_id) or not isinstance(evidence.interval_start, str):
            raise ProviderIngestionError("gap evidence identity is malformed")
        if evidence.evidence_id in evidence_ids or evidence.interval_start in evidence_by_interval:
            raise ProviderIngestionError("duplicate gap evidence identity or interval")
        evidence_ids.add(evidence.evidence_id)
        evidence_by_interval[evidence.interval_start] = evidence
    if evidence_by_interval and not _is_sha256(protocol_config_sha256):
        raise ProviderIngestionError(
            "protocol_config_sha256 is required when reporting terminal gap evidence"
        )

    due_count = 0
    complete_count = 0
    zero_line_count = 0
    pending_count = 0
    unresolved_count = 0
    gap_points: list[tuple[datetime, str]] = []
    observed_receipts: list[dict[str, Any]] = []
    accepted_gap_evidence: list[dict[str, str]] = []
    for interval in plan.intervals:
        due_at = _parse_required_timestamp(interval.due_at, "interval.due_at")
        snapshot = snapshot_by_timestamp.get(interval.filename_timestamp)
        evidence = evidence_by_interval.pop(interval.filename_timestamp, None)
        if snapshot is not None:
            try:
                _validate_snapshot_terminal_fact(snapshot, interval.filename_timestamp)
            except NormalizationIntegrityError as exc:
                raise ProviderIngestionError(
                    "coverage snapshot terminal facts are invalid"
                ) from exc
            published_at = _parse_required_timestamp(
                snapshot.receipt.raw_published_at, "receipt.raw_published_at"
            )
            if published_at > as_of_time:
                raise ProviderIngestionError("coverage report cannot include a future receipt")
            observed_receipts.append(
                {
                    "error_code": snapshot.error_code,
                    "filename_timestamp": interval.filename_timestamp,
                    "raw_snapshot_sha256": snapshot.receipt.raw_snapshot_sha256,
                    "state": snapshot.state,
                }
            )
        if due_at > as_of_time:
            if evidence is not None:
                raise ProviderIngestionError(
                    "terminal gap evidence cannot close an interval before its due time"
                )
            pending_count += 1
            continue
        due_count += 1
        interval_time = _parse_minute_timestamp(
            interval.filename_timestamp, "interval.filename_timestamp"
        )
        if snapshot is not None and snapshot.state == "complete":
            if evidence is not None:
                raise ProviderIngestionError(
                    "complete snapshot cannot also be reported as a provider gap"
                )
            complete_count += 1
            if snapshot.json_line_count == 0:
                zero_line_count += 1
        else:
            if evidence is None:
                unresolved_count += 1
                continue
            try:
                _validate_terminal_gap_evidence(
                    evidence,
                    interval_start=interval.filename_timestamp,
                    interval_end_exclusive=format_utc_timestamp(interval_time + EXPECTED_INTERVAL),
                    expected_source_locator=_expected_source_locator(interval),
                    protocol_config_sha256=protocol_config_sha256,
                    terminal_as_of=as_of,
                )
            except NormalizationIntegrityError as exc:
                raise ProviderIngestionError("coverage gap evidence failed verification") from exc
            if snapshot is None:
                if (
                    evidence.observed_snapshot_id is not None
                    or evidence.observed_raw_snapshot_sha256 is not None
                ):
                    raise ProviderIngestionError(
                        "missing gap evidence references an unexpected snapshot"
                    )
                reason = evidence.final_terminal_disposition
            else:
                if (
                    snapshot.state != "invalid"
                    or evidence.observed_snapshot_id != snapshot.receipt.snapshot_id
                    or evidence.observed_raw_snapshot_sha256 != snapshot.receipt.raw_snapshot_sha256
                    or evidence.attempts[-1].error_kind != snapshot.error_code
                    or evidence.expected_source_locator != snapshot.receipt.source_locator
                ):
                    raise ProviderIngestionError(
                        "invalid snapshot gap evidence does not match its receipt"
                    )
                reason = snapshot.error_code or "invalid_snapshot"
            gap_points.append((interval_time, reason))
            accepted_gap_evidence.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "evidence_sha256": evidence.evidence_sha256,
                }
            )
    if evidence_by_interval:
        raise ProviderIngestionError("gap evidence falls outside the due retrieval plan")
    gaps = _group_gap_points(gap_points)
    retrieval_rate = complete_count / due_count if due_count else 1.0
    expected_schedule = [interval.to_dict() for interval in plan.intervals]
    base_payload = {
        "as_of": format_utc_timestamp(as_of_time),
        "complete_intervals": complete_count,
        "expected_due_intervals": due_count,
        "expected_schedule_sha256": canonical_sha256(expected_schedule),
        "gap_intervals": len(gap_points),
        "gaps": [gap.to_dict() for gap in gaps],
        "maximum_gap_minutes": max((gap.duration_minutes for gap in gaps), default=0),
        "observed_receipts_sha256": canonical_sha256(observed_receipts),
        "pending_intervals": pending_count,
        "plan_id": plan.plan_id,
        "retrieval_rate": retrieval_rate,
        "terminal_gap_evidence_sha256": canonical_sha256(
            sorted(accepted_gap_evidence, key=lambda item: item["evidence_id"])
        ),
        "unresolved_intervals": unresolved_count,
        "zero_line_intervals": zero_line_count,
    }
    return CoverageReport(
        plan_id=plan.plan_id,
        as_of=format_utc_timestamp(as_of_time),
        expected_due_intervals=due_count,
        complete_intervals=complete_count,
        zero_line_intervals=zero_line_count,
        pending_intervals=pending_count,
        unresolved_intervals=unresolved_count,
        gap_intervals=len(gap_points),
        retrieval_rate=retrieval_rate,
        maximum_gap_minutes=base_payload["maximum_gap_minutes"],
        gaps=gaps,
        expected_schedule_sha256=base_payload["expected_schedule_sha256"],
        observed_receipts_sha256=base_payload["observed_receipts_sha256"],
        terminal_gap_evidence_sha256=base_payload["terminal_gap_evidence_sha256"],
        semantic_sha256=canonical_sha256(base_payload),
    )


def decisions_intersecting_gaps(
    report: CoverageReport, decision_times: Iterable[str], *, window_hours: int = 24
) -> tuple[str, ...]:
    """Return decisions that must be removed, never zero-filled, due to provider gaps."""
    if window_hours <= 0:
        raise ProviderIngestionError("gap window must be positive")
    gap_intervals = [
        (
            _parse_required_timestamp(gap.start_at, "gap.start_at"),
            _parse_required_timestamp(gap.end_at_exclusive, "gap.end_at_exclusive"),
        )
        for gap in report.gaps
    ]
    affected: list[str] = []
    for decision_text in decision_times:
        decision = _parse_required_timestamp(decision_text, "decision_at")
        window_start = decision - timedelta(hours=window_hours)
        if any(
            gap_start <= decision and gap_end > window_start for gap_start, gap_end in gap_intervals
        ):
            affected.append(format_utc_timestamp(decision))
    return tuple(sorted(set(affected)))


def publish_normalization(
    store: ContentAddressedStore,
    batch_id: str,
    result: NormalizationResult,
    coverage: CoverageReport,
) -> Path:
    """Publish deterministic normalized records with no-overwrite/idempotent semantics."""
    _validate_normalization_result_for_publication(result)
    _validate_coverage_report_for_publication(coverage)
    if coverage.pending_intervals or coverage.unresolved_intervals:
        raise ProviderIngestionError(
            "cannot publish normalization with nonterminal coverage intervals"
        )
    files = {
        "articles.jsonl": _canonical_json_lines(article.to_dict() for article in result.articles),
        "coverage.json": canonicalize(coverage.to_dict()),
        "exclusions.jsonl": _canonical_json_lines(
            exclusion.to_dict() for exclusion in result.exclusions
        ),
        "observation-links.jsonl": _canonical_json_lines(
            link.to_dict() for link in result.observation_links
        ),
        "summary.json": canonicalize(
            {
                "article_count": len(result.articles),
                "coverage_semantic_sha256": coverage.semantic_sha256,
                "exclusion_count": len(result.exclusions),
                "normalization_semantic_sha256": result.semantic_sha256,
                "normalizer_version": NORMALIZER_VERSION,
                "observation_link_count": len(result.observation_links),
                "protocol_config_sha256": result.protocol_config_sha256,
                "provider": PROVIDER_ID,
                "rights_approval_sha256": result.rights_approval_sha256,
            }
        ),
    }
    publication_id = f"gsg-normalized-{batch_id}"
    metadata = {
        "coverage_semantic_sha256": coverage.semantic_sha256,
        "normalization_semantic_sha256": result.semantic_sha256,
        "protocol_config_sha256": result.protocol_config_sha256,
        "provider": PROVIDER_ID,
        "rights_approval_sha256": result.rights_approval_sha256,
        "scope": "normalized_offline_adapter_batch",
    }
    expected_manifest = {
        "files": {
            name: {"sha256": sha256_bytes(data), "size_bytes": len(data)}
            for name, data in sorted(files.items())
        },
        "metadata": metadata,
        "publication_id": publication_id,
        "schema_version": PUBLICATION_SCHEMA_VERSION,
    }
    try:
        return store.publish_bundle(publication_id, files, metadata=metadata)
    except PublicationCollisionError as exc:
        try:
            verified = store.read_publication(publication_id)
            if verified.files == files and verified.manifest == expected_manifest:
                return store.publications_root / publication_id
        except SentimentStorageError:
            pass
        raise ProviderIngestionError(
            f"normalization batch ID collision with different bytes: {batch_id}"
        ) from exc


def _validate_normalization_result_for_publication(result: object) -> None:
    """Reject caller-forged normalization artifacts before immutable publication."""
    if not isinstance(result, NormalizationResult):
        raise NormalizationIntegrityError("normalization result does not match its contract")
    if not _is_sha256(result.protocol_config_sha256) or not _is_sha256(
        result.rights_approval_sha256
    ):
        raise NormalizationIntegrityError("normalization configuration identity is malformed")
    if not _is_sha256(result.semantic_sha256):
        raise NormalizationIntegrityError("normalization semantic hash is malformed")
    if not all(
        isinstance(value, tuple)
        for value in (result.articles, result.observation_links, result.exclusions)
    ):
        raise NormalizationIntegrityError("normalization collections must be immutable tuples")

    validated_articles: list[ArticleRecord] = []
    try:
        for article in result.articles:
            if not isinstance(article, ArticleRecord):
                raise NormalizationIntegrityError(
                    "normalization article does not match its contract"
                )
            validated = validate_article_record(article.to_dict())
            if validated != article:
                raise NormalizationIntegrityError("normalization article changed during validation")
            validated_articles.append(validated)
        validate_article_collection(validated_articles)
    except ArticleValidationError as exc:
        raise NormalizationIntegrityError("normalization article validation failed") from exc
    expected_articles = tuple(
        sorted(
            validated_articles,
            key=lambda item: (
                item.first_seen_at or "",
                item.article_id,
                item.article_version_id,
            ),
        )
    )
    if result.articles != expected_articles:
        raise NormalizationIntegrityError("normalization articles are not canonically ordered")

    version_ids = {article.article_version_id for article in result.articles}
    link_observations: set[str] = set()
    for link in result.observation_links:
        if (
            not isinstance(link, ObservationLink)
            or not isinstance(link.provider_observation_id, str)
            or not link.provider_observation_id.strip()
            or not _is_sha256(link.article_version_id)
            or link.article_version_id not in version_ids
            or not isinstance(link.reused_existing_version, bool)
            or not _is_sha256(link.raw_snapshot_sha256)
            or not isinstance(link.input_class, str)
            or link.input_class not in {"provider_response", "synthetic_fixture"}
            or link.provider_observation_id in link_observations
        ):
            raise NormalizationIntegrityError("normalization observation link is invalid")
        link_observations.add(link.provider_observation_id)
    if result.observation_links != tuple(
        sorted(result.observation_links, key=lambda item: item.provider_observation_id)
    ):
        raise NormalizationIntegrityError("normalization observation links are not ordered")

    exclusion_observations: set[str] = set()
    for exclusion in result.exclusions:
        if (
            not isinstance(exclusion, ExcludedObservation)
            or not isinstance(exclusion.provider_observation_id, str)
            or not exclusion.provider_observation_id.strip()
            or not isinstance(exclusion.reason, str)
            or exclusion.reason not in _EXCLUSION_RANK
            or not isinstance(exclusion.diagnostic, str)
            or not exclusion.diagnostic.strip()
            or not _is_sha256(exclusion.raw_snapshot_sha256)
            or not isinstance(exclusion.input_class, str)
            or exclusion.input_class not in {"provider_response", "synthetic_fixture"}
            or exclusion.provider_observation_id in exclusion_observations
        ):
            raise NormalizationIntegrityError("normalization exclusion is invalid")
        exclusion_observations.add(exclusion.provider_observation_id)
    if link_observations & exclusion_observations:
        raise NormalizationIntegrityError("normalization observation is linked and excluded")
    if result.exclusions != tuple(
        sorted(result.exclusions, key=lambda item: item.provider_observation_id)
    ):
        raise NormalizationIntegrityError("normalization exclusions are not ordered")

    try:
        expected_semantic_sha256 = canonical_sha256(result.to_dict(include_hash=False))
    except CanonicalizationError as exc:
        raise NormalizationIntegrityError("normalization result is not canonicalizable") from exc
    if result.semantic_sha256 != expected_semantic_sha256:
        raise NormalizationIntegrityError("normalization semantic hash mismatch")


def _validate_coverage_report_for_publication(coverage: object) -> None:
    """Validate coverage arithmetic, chronology, canonical order, and semantic identity."""
    if not isinstance(coverage, CoverageReport):
        raise NormalizationIntegrityError("coverage report does not match its contract")
    if not _is_sha256(coverage.plan_id):
        raise NormalizationIntegrityError("coverage plan identity is malformed")
    try:
        as_of = _parse_required_timestamp(coverage.as_of, "coverage.as_of")
    except ProviderIngestionError as exc:
        raise NormalizationIntegrityError("coverage as_of timestamp is invalid") from exc
    if format_utc_timestamp(as_of) != coverage.as_of:
        raise NormalizationIntegrityError("coverage as_of timestamp is not canonical UTC")

    counter_fields = (
        "expected_due_intervals",
        "complete_intervals",
        "zero_line_intervals",
        "pending_intervals",
        "unresolved_intervals",
        "gap_intervals",
        "maximum_gap_minutes",
    )
    for field in counter_fields:
        value = getattr(coverage, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NormalizationIntegrityError(f"coverage {field} must be a nonnegative integer")
    if (
        isinstance(coverage.retrieval_rate, bool)
        or not isinstance(coverage.retrieval_rate, (int, float))
        or not math.isfinite(coverage.retrieval_rate)
        or not 0.0 <= coverage.retrieval_rate <= 1.0
    ):
        raise NormalizationIntegrityError("coverage retrieval_rate must be finite in [0, 1]")
    for field in (
        "expected_schedule_sha256",
        "observed_receipts_sha256",
        "terminal_gap_evidence_sha256",
        "semantic_sha256",
    ):
        if not _is_sha256(getattr(coverage, field)):
            raise NormalizationIntegrityError(f"coverage {field} is malformed")
    if not isinstance(coverage.gaps, tuple):
        raise NormalizationIntegrityError("coverage gaps must be an immutable tuple")

    gap_minutes = 0
    maximum_gap = 0
    previous_end: datetime | None = None
    for gap in coverage.gaps:
        if not isinstance(gap, ProviderGap):
            raise NormalizationIntegrityError("coverage gap does not match its contract")
        try:
            start = _parse_minute_timestamp(gap.start_at, "coverage.gap.start_at")
            end = _parse_minute_timestamp(gap.end_at_exclusive, "coverage.gap.end_at_exclusive")
        except ProviderIngestionError as exc:
            raise NormalizationIntegrityError("coverage gap timestamp is invalid") from exc
        if (
            format_utc_timestamp(start) != gap.start_at
            or format_utc_timestamp(end) != gap.end_at_exclusive
            or end <= start
        ):
            raise NormalizationIntegrityError("coverage gap bounds are invalid")
        duration_seconds = int((end - start).total_seconds())
        if duration_seconds % int(EXPECTED_INTERVAL.total_seconds()):
            raise NormalizationIntegrityError("coverage gap is not minute aligned")
        expected_duration = duration_seconds // int(EXPECTED_INTERVAL.total_seconds())
        if (
            isinstance(gap.duration_minutes, bool)
            or not isinstance(gap.duration_minutes, int)
            or gap.duration_minutes != expected_duration
            or not isinstance(gap.reasons, tuple)
            or not gap.reasons
            or any(
                not isinstance(reason, str) or reason not in _COVERAGE_GAP_REASONS
                for reason in gap.reasons
            )
            or gap.reasons != tuple(sorted(set(gap.reasons)))
        ):
            raise NormalizationIntegrityError("coverage gap fields are invalid")
        if previous_end is not None and start <= previous_end:
            raise NormalizationIntegrityError("coverage gaps overlap or are not maximally grouped")
        previous_end = end
        gap_minutes += gap.duration_minutes
        maximum_gap = max(maximum_gap, gap.duration_minutes)

    if coverage.gap_intervals != gap_minutes:
        raise NormalizationIntegrityError("coverage gap interval count is inconsistent")
    if coverage.maximum_gap_minutes != maximum_gap:
        raise NormalizationIntegrityError("coverage maximum gap is inconsistent")
    if coverage.zero_line_intervals > coverage.complete_intervals:
        raise NormalizationIntegrityError("coverage zero-line count exceeds complete intervals")
    if coverage.expected_due_intervals != (
        coverage.complete_intervals + coverage.gap_intervals + coverage.unresolved_intervals
    ):
        raise NormalizationIntegrityError("coverage due interval arithmetic is inconsistent")
    expected_rate = (
        coverage.complete_intervals / coverage.expected_due_intervals
        if coverage.expected_due_intervals
        else 1.0
    )
    if coverage.retrieval_rate != expected_rate:
        raise NormalizationIntegrityError("coverage retrieval rate is inconsistent")

    try:
        expected_semantic_sha256 = canonical_sha256(coverage.to_dict(include_hash=False))
    except CanonicalizationError as exc:
        raise NormalizationIntegrityError("coverage report is not canonicalizable") from exc
    if coverage.semantic_sha256 != expected_semantic_sha256:
        raise NormalizationIntegrityError("coverage semantic hash mismatch")


def canonicalize_url(value: str) -> str:
    """Apply the frozen URL canonicalization rules used by the GSG adapter."""
    value = unicodedata.normalize("NFC", value.strip())
    if not value or _CONTROL_PATTERN.search(value) or _INVALID_PERCENT_PATTERN.search(value):
        raise _ObservationExcluded("invalid_url_or_identifier", "URL contains invalid characters")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise _ObservationExcluded("invalid_url_or_identifier", "URL parse failed") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise _ObservationExcluded("invalid_url_or_identifier", "URL must be absolute HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise _ObservationExcluded("invalid_url_or_identifier", "URL userinfo is forbidden")
    try:
        ascii_host = idna.encode(
            parsed.hostname, uts46=True, transitional=False, std3_rules=True
        ).decode("ascii")
    except idna.IDNAError as exc:
        raise _ObservationExcluded("invalid_url_or_identifier", "URL hostname is invalid") from exc
    host = ascii_host.lower()
    if ":" in host:
        host = f"[{host}]"
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = _normalize_percent_component(parsed.path or "/", _PATH_SAFE)
    path = _remove_dot_segments(path)
    query_pairs: list[tuple[str, str]] = []
    if parsed.query:
        for item in parsed.query.split("&"):
            raw_key, separator, raw_value = item.partition("=")
            key = _normalize_percent_component(raw_key, _QUERY_SAFE)
            value_part = _normalize_percent_component(raw_value, _QUERY_SAFE)
            tracking_key = unquote(key).casefold()
            if _TRACKING_KEY_PATTERN.fullmatch(tracking_key):
                continue
            query_pairs.append((key, value_part if separator else ""))
        query_pairs.sort(key=lambda pair: (pair[0].encode(), pair[1].encode()))
    query = "&".join(f"{key}={item}" for key, item in query_pairs)
    return urlunsplit((scheme, host, path or "/", query, ""))


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFC", value).replace("\x00", "")
    return " ".join(value.split())


def normalize_gsg_language(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).strip(" \t\r\n").casefold()
    if normalized != "english":
        raise _ObservationExcluded("unsupported_language", "GSG language is not English")
    return "en"


def normalize_provider_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _ObservationExcluded("invalid_timestamp", "provider first-seen value is not a string")
    if re.fullmatch(r"\d{14}", value):
        try:
            return format_utc_timestamp(
                datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            )
        except ValueError as exc:
            raise _ObservationExcluded(
                "invalid_timestamp", "provider first-seen value is invalid"
            ) from exc
    try:
        parsed = parse_utc_timestamp(value, field="provider_first_seen_at")
    except ValueError as exc:
        raise _ObservationExcluded(
            "invalid_timestamp", "provider first-seen value is invalid"
        ) from exc
    if parsed is None:  # pragma: no cover - nonnullable call
        return None
    return format_utc_timestamp(parsed)


def selects_direct_btc(title: str) -> bool:
    without_false_positive = _BTC_CITY_PATTERN.sub(" ", title)
    return _POSITIVE_BTC_PATTERN.search(without_false_positive) is not None


def redact_url(value: str) -> str:
    """Remove credentials and secret-like query values before durable logging."""
    if not isinstance(value, str):
        raise ProviderIngestionError("source locator must be a string")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ProviderIngestionError("source locator is invalid") from exc
    if not parsed.scheme or not hostname:
        raise ProviderIngestionError("source locator must be an absolute URL")
    host = hostname
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    query_parts: list[str] = []
    for part in parsed.query.split("&") if parsed.query else []:
        key, separator, item = part.partition("=")
        if _is_sensitive_key(key):
            item = "REDACTED"
            separator = "="
        query_parts.append(key + separator + item)
    return urlunsplit((parsed.scheme, host, parsed.path, "&".join(query_parts), ""))


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: "REDACTED" if _is_sensitive_key(key) else value for key, value in headers.items()}


def _observation_from_endpoint(
    receipt: RawSnapshotReceipt, line_number: int, side: str, record: Mapping[str, Any]
) -> GSGObservation:
    side_name = "from" if side == "from" else "to"
    observation_id = (
        f"gdelt-gsg:{receipt.filename_timestamp}:{receipt.raw_snapshot_sha256}:"
        f"{line_number}:{side_name}"
    )
    return GSGObservation(
        provider_observation_id=observation_id,
        filename_timestamp=receipt.filename_timestamp,
        raw_snapshot_sha256=receipt.raw_snapshot_sha256,
        zero_based_line_number=line_number,
        endpoint_side=side_name,
        raw_url=record.get(f"{side_name}Url"),
        raw_title=record.get(f"{side_name}Title"),
        raw_language=record.get(f"{side_name}Lang"),
        raw_provider_first_seen_at=record.get(f"{side_name}Date"),
        ingested_at=receipt.ingested_at,
        raw_published_at=receipt.raw_published_at,
        collection_mode=receipt.collection_mode,
        input_class=receipt.input_class,
    )


def _observation_sort_key(observation: GSGObservation) -> tuple[Any, ...]:
    return (
        _parse_required_timestamp(observation.raw_published_at, "raw_published_at"),
        _parse_minute_timestamp(observation.filename_timestamp, "filename_timestamp"),
        observation.raw_snapshot_sha256,
        observation.zero_based_line_number,
        observation.endpoint_side,
    )


def _bounded_gzip_decompress(value: bytes, *, maximum_bytes: int) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(value), mode="rb") as handle:
        decompressed = handle.read(maximum_bytes + 1)
    if len(decompressed) > maximum_bytes:
        raise ProviderIngestionError("decompressed response exceeds the frozen byte cap")
    return decompressed


def _canonical_json_lines(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonicalize(value) + b"\n" for value in values)


def _parse_required_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = parse_utc_timestamp(value, field=field)
    except ValueError as exc:
        raise ProviderIngestionError(str(exc)) from exc
    if parsed is None:  # pragma: no cover - nonnullable call
        raise ProviderIngestionError(f"{field} cannot be null")
    return parsed


def _parse_minute_timestamp(value: str, field: str) -> datetime:
    parsed = _parse_required_timestamp(value, field)
    if parsed.second or parsed.microsecond:
        raise ProviderIngestionError(f"{field} must be aligned to an exact UTC minute")
    return parsed


def _parse_chronology_minute(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise NormalizationIntegrityError(f"{field} must be a UTC timestamp string")
    try:
        parsed = _parse_minute_timestamp(value, field)
    except ProviderIngestionError as exc:
        raise NormalizationIntegrityError(str(exc)) from exc
    if format_utc_timestamp(parsed) != value:
        raise NormalizationIntegrityError(f"{field} must use canonical UTC RFC3339 representation")
    return parsed


def _parse_chronology_instant(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise NormalizationIntegrityError(f"{field} must be a UTC timestamp string")
    try:
        parsed = _parse_required_timestamp(value, field)
    except ProviderIngestionError as exc:
        raise NormalizationIntegrityError(str(exc)) from exc
    if format_utc_timestamp(parsed) != value:
        raise NormalizationIntegrityError(f"{field} must use canonical UTC RFC3339 representation")
    return parsed


def _exclusive_boundary_after(value: datetime) -> datetime:
    try:
        return value + timedelta(microseconds=1)
    except OverflowError as exc:
        raise NormalizationIntegrityError(
            "terminal availability cannot advance the exclusive boundary"
        ) from exc


def _normalize_percent_component(value: str, safe: frozenset[str]) -> str:
    result: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character == "%":
            byte = int(value[index + 1 : index + 3], 16)
            decoded = chr(byte)
            result.append(decoded if decoded in _UNRESERVED else f"%{byte:02X}")
            index += 3
            continue
        if character in safe:
            result.append(character)
        else:
            result.extend(f"%{byte:02X}" for byte in character.encode("utf-8"))
        index += 1
    return "".join(result)


def _remove_dot_segments(path: str) -> str:
    leading = path.startswith("/")
    trailing = path.endswith("/") or path.endswith("/.") or path.endswith("/..")
    segments: list[str] = []
    for segment in path.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if segments:
                segments.pop()
        else:
            segments.append(segment)
    result = ("/" if leading else "") + "/".join(segments)
    if trailing and result != "/":
        result += "/"
    return result or "/"


def _group_gap_points(points: Sequence[tuple[datetime, str]]) -> tuple[ProviderGap, ...]:
    if not points:
        return ()
    sorted_points = sorted(points)
    groups: list[list[tuple[datetime, str]]] = [[sorted_points[0]]]
    for point in sorted_points[1:]:
        if point[0] == groups[-1][-1][0] + EXPECTED_INTERVAL:
            groups[-1].append(point)
        else:
            groups.append([point])
    return tuple(
        ProviderGap(
            start_at=format_utc_timestamp(group[0][0]),
            end_at_exclusive=format_utc_timestamp(group[-1][0] + EXPECTED_INTERVAL),
            duration_minutes=len(group),
            reasons=tuple(sorted({reason for _, reason in group})),
        )
        for group in groups
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _primary_exclusion(failures: Iterable[tuple[str, str]]) -> tuple[str, str]:
    materialized = tuple(failures)
    try:
        primary = min(materialized, key=lambda item: _EXCLUSION_RANK[item[0]])
    except (KeyError, ValueError) as exc:
        raise NormalizationIntegrityError("unrecognized or empty exclusion set") from exc
    diagnostics = "; ".join(
        f"{reason}:{diagnostic}"
        for reason, diagnostic in sorted(
            materialized, key=lambda item: (_EXCLUSION_RANK[item[0]], item[0], item[1])
        )
    )
    return primary[0], diagnostics


def _dedup_fingerprint(title: str, language: str) -> str:
    return canonical_sha256(
        {
            "content": "",
            "language": language,
            "serialization_version": "dedup-fingerprint-v1",
            "title_casefold": title.casefold(),
        }
    )


def _fingerprint_for_article(article: ArticleRecord) -> str:
    return derive_version_fingerprint(
        article_id=article.article_id,
        content_hash=article.content_hash,
        content=article.content,
        language=article.language,
        provider=article.provider,
        source=article.source,
        title=article.title,
    )


def _parse_canonical_json_buffer(value: bytes, description: str) -> Any:
    try:
        payload = json.loads(value.decode("utf-8"), parse_constant=_reject_json_constant)
        canonical = canonicalize(payload)
    except (UnicodeError, json.JSONDecodeError, ValueError, CanonicalizationError) as exc:
        raise NormalizationIntegrityError(f"{description} is not valid strict JSON") from exc
    if canonical != value:
        raise NormalizationIntegrityError(f"{description} is not canonical RFC 8785 JSON")
    return payload


def _require_json_array(value: bytes, description: str) -> list[Any]:
    payload = _parse_canonical_json_buffer(value, description)
    if not isinstance(payload, list):
        raise NormalizationIntegrityError(f"{description} must be an array")
    return payload


def _rights_approval_from_payload(value: Any) -> RightsApproval | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise NormalizationIntegrityError("persisted rights approval must be an object or null")
    expected_fields = set(RightsApproval.__dataclass_fields__)
    if set(value) != expected_fields:
        raise NormalizationIntegrityError("persisted rights approval fields do not match")
    fixture_hashes = value.get("authorized_fixture_sha256")
    if not isinstance(fixture_hashes, list) or not all(
        isinstance(item, str) for item in fixture_hashes
    ):
        raise NormalizationIntegrityError("persisted fixture hash allowlist is invalid")
    try:
        approval = RightsApproval(**{**value, "authorized_fixture_sha256": tuple(fixture_hashes)})
    except TypeError as exc:
        raise NormalizationIntegrityError("persisted rights approval is invalid") from exc
    if not approval.is_structurally_valid():
        raise NormalizationIntegrityError("persisted rights approval identity is invalid")
    return approval


def _terminal_gap_evidence_from_payload(value: Any) -> TerminalGapEvidence:
    if not isinstance(value, dict) or set(value) != set(TerminalGapEvidence.__dataclass_fields__):
        raise NormalizationIntegrityError("persisted terminal gap evidence fields do not match")
    raw_attempts = value.get("attempts")
    if not isinstance(raw_attempts, list):
        raise NormalizationIntegrityError(
            "persisted terminal gap evidence attempts must be an array"
        )
    attempts = tuple(
        _dataclass_from_exact_mapping(GapAttempt, item, "state terminal gap attempt")
        for item in raw_attempts
    )
    try:
        return TerminalGapEvidence(**{**value, "attempts": attempts})
    except TypeError as exc:
        raise NormalizationIntegrityError("persisted terminal gap evidence is invalid") from exc


def _load_verified_snapshot_receipt(
    store: ContentAddressedStore,
    snapshot_id: str,
    *,
    raw_object_cache: dict[str, bytes],
) -> tuple[RawSnapshotReceipt, bytes]:
    publication_id = f"gsg-snapshot-{snapshot_id}"
    try:
        verified = store.read_publication(publication_id)
    except SentimentStorageError as exc:
        raise NormalizationIntegrityError(
            f"terminal raw receipt failed verification: {snapshot_id}"
        ) from exc
    if set(verified.files) != {"receipt.json"}:
        raise NormalizationIntegrityError(
            "terminal raw receipt publication has an invalid inventory"
        )
    payload = _parse_canonical_json_buffer(verified.files["receipt.json"], "terminal raw receipt")
    receipt = _dataclass_from_exact_mapping(RawSnapshotReceipt, payload, "terminal raw receipt")
    if verified.manifest.get("metadata") != {
        "provider": PROVIDER_ID,
        "raw_object_sha256": receipt.raw_snapshot_sha256,
        "scope": "offline_adapter_input",
    }:
        raise NormalizationIntegrityError("terminal raw receipt manifest metadata is invalid")
    try:
        filename_time = _parse_chronology_minute(
            receipt.filename_timestamp, "receipt.filename_timestamp"
        )
        ingested_at = _parse_chronology_instant(receipt.ingested_at, "receipt.ingested_at")
        raw_published_at = _parse_chronology_instant(
            receipt.raw_published_at, "receipt.raw_published_at"
        )
    except NormalizationIntegrityError:
        raise
    try:
        redacted_source_locator = redact_url(receipt.source_locator)
    except ProviderIngestionError as exc:
        raise NormalizationIntegrityError("terminal raw receipt source locator is invalid") from exc
    if (
        receipt.snapshot_id != snapshot_id
        or not _is_sha256(receipt.snapshot_id)
        or not _is_sha256(receipt.raw_snapshot_sha256)
        or receipt.parser_version != PARSER_VERSION
        or receipt.parser_policy_version != PARSER_POLICY_VERSION
        or not isinstance(receipt.collection_mode, str)
        or receipt.collection_mode not in {"prospective", "historical_backfill"}
        or not isinstance(receipt.input_class, str)
        or receipt.input_class not in {"provider_response", "synthetic_fixture"}
        or not isinstance(receipt.max_compressed_bytes, int)
        or isinstance(receipt.max_compressed_bytes, bool)
        or not 1 <= receipt.max_compressed_bytes <= MAX_COMPRESSED_BYTES
        or not isinstance(receipt.max_decompressed_bytes, int)
        or isinstance(receipt.max_decompressed_bytes, bool)
        or not 1 <= receipt.max_decompressed_bytes <= MAX_DECOMPRESSED_BYTES
        or not isinstance(receipt.max_json_lines, int)
        or isinstance(receipt.max_json_lines, bool)
        or not 1 <= receipt.max_json_lines <= MAX_JSON_LINES
        or not isinstance(receipt.source_locator, str)
        or not receipt.source_locator
        or redacted_source_locator != receipt.source_locator
        or not isinstance(receipt.compressed_size_bytes, int)
        or isinstance(receipt.compressed_size_bytes, bool)
        or receipt.compressed_size_bytes < 0
        or receipt.compressed_size_bytes > receipt.max_compressed_bytes
        or raw_published_at < ingested_at
    ):
        raise NormalizationIntegrityError("terminal raw receipt fields are invalid")
    expected_snapshot_id = _derive_snapshot_id(
        collection_mode=receipt.collection_mode,
        filename_timestamp=format_utc_timestamp(filename_time),
        input_class=receipt.input_class,
        raw_snapshot_sha256=receipt.raw_snapshot_sha256,
        max_compressed_bytes=receipt.max_compressed_bytes,
        max_decompressed_bytes=receipt.max_decompressed_bytes,
        max_json_lines=receipt.max_json_lines,
    )
    if expected_snapshot_id != receipt.snapshot_id:
        raise NormalizationIntegrityError("terminal raw receipt identity mismatch")
    raw_bytes = raw_object_cache.get(receipt.raw_snapshot_sha256)
    if raw_bytes is None:
        try:
            raw_bytes = store.get_bytes(receipt.raw_snapshot_sha256)
        except SentimentStorageError as exc:
            raise NormalizationIntegrityError(
                "state transitive raw object failed verification: " f"{receipt.raw_snapshot_sha256}"
            ) from exc
        raw_object_cache[receipt.raw_snapshot_sha256] = raw_bytes
    if len(raw_bytes) != receipt.compressed_size_bytes:
        raise NormalizationIntegrityError("terminal raw receipt byte count mismatch")
    return receipt, raw_bytes


def _dataclass_from_exact_mapping(class_type: type[Any], value: Any, description: str) -> Any:
    if not isinstance(value, dict) or set(value) != set(class_type.__dataclass_fields__):
        raise NormalizationIntegrityError(f"{description} fields do not match")
    return class_type(**value)


def _unique_by_field(values: Iterable[Any], field: str, description: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        key = getattr(value, field)
        if not isinstance(key, str) or key in result:
            raise NormalizationIntegrityError(f"duplicate or invalid {description} identity")
        result[key] = value
    return result


def _raw_hash_from_observation_id(value: str) -> str:
    _, raw_hash, _, _ = _observation_provenance(value)
    return raw_hash


def _observation_provenance(value: str) -> tuple[str, str, int, str]:
    match = re.fullmatch(r"gdelt-gsg:(.+Z):([0-9a-f]{64}):(\d+):(from|to)", value)
    if match is None:
        raise NormalizationIntegrityError("persisted provider observation ID is malformed")
    start = _parse_chronology_minute(match.group(1), "observation interval start")
    line_number = int(match.group(3))
    if line_number > MAX_JSON_LINES:
        raise NormalizationIntegrityError("persisted observation line number is invalid")
    return format_utc_timestamp(start), match.group(2), line_number, match.group(4)


def _is_sensitive_key(value: str) -> bool:
    normalized = unquote(value).strip().casefold()
    return normalized in _SENSITIVE_KEYS or normalized in {
        "access-token",
        "access_token",
        "x-api-key",
        "x_api_key",
        "client-secret",
        "client_secret",
        "sig",
    }


def _safe_error_code(error: BaseException) -> str:
    if isinstance(error, gzip.BadGzipFile):
        return "invalid_gzip"
    if isinstance(error, UnicodeError):
        return "invalid_utf8"
    if isinstance(error, json.JSONDecodeError):
        return "invalid_json"
    if isinstance(error, ProviderIngestionError):
        message = str(error)
        if "line count" in message or "byte cap" in message:
            return "resource_limit_exceeded"
        if "blank JSONL" in message:
            return "invalid_jsonl"
    return "malformed_snapshot"
