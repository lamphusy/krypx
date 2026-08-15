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
    validate_article_record,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

PROVIDER_ID = "gdelt_gsg"
PARSER_VERSION = "gdelt-gsg-jsonl-v1"
NORMALIZER_VERSION = "gdelt-gsg-normalizer-v2"
NORMALIZER_STATE_VERSION = "gdelt-gsg-normalizer-state-v2"
CHRONOLOGY_VERSION = "gdelt-gsg-terminal-chronology-v1"
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
        return (
            self.version == RIGHTS_APPROVAL_VERSION
            and _is_sha256(self.approval_id)
            and _is_sha256(self.protocol_config_sha256)
            and all(_is_sha256(item) for item in self.authorized_fixture_sha256)
            and tuple(sorted(set(self.authorized_fixture_sha256))) == self.authorized_fixture_sha256
            and self.approval_id == canonical_sha256(self.identity_payload())
            and self.approval_kind in {"provider_rights", "synthetic_fixture_only"}
            and isinstance(self.approved, bool)
            and isinstance(self.network_access_authorized, bool)
        )


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
    gap_intervals: int
    retrieval_rate: float
    maximum_gap_minutes: int
    gaps: tuple[ProviderGap, ...]
    expected_schedule_sha256: str
    observed_receipts_sha256: str
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
            "zero_line_intervals": self.zero_line_intervals,
        }
        if include_hash:
            value["semantic_sha256"] = self.semantic_sha256
        return value


@dataclass(frozen=True, slots=True)
class TerminalInterval:
    """One immutable minute-level fact that advances normalization chronology."""

    start_at: str
    end_at_exclusive: str
    outcome: str
    snapshot_state: str
    raw_snapshot_sha256: str | None
    json_line_count: int | None
    terminal_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GSGRetryPolicy:
    """Pure retry classification: one initial attempt and at most two retries."""

    maximum_attempts = 3
    backoff_seconds = (2.0, 4.0)

    def decide(
        self,
        *,
        attempt: int,
        http_status: int | None = None,
        error_kind: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> RetryDecision:
        if not 1 <= attempt <= self.maximum_attempts:
            raise ProviderIngestionError("attempt must be between 1 and 3")
        if retry_after_seconds is not None and (
            not math.isfinite(retry_after_seconds) or retry_after_seconds < 0
        ):
            raise ProviderIngestionError("Retry-After must be finite and nonnegative")
        if http_status is not None and error_kind is not None:
            raise ProviderIngestionError("provide either an HTTP status or an error kind")
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
        if min(max_compressed_bytes, max_decompressed_bytes, max_json_lines) <= 0:
            raise ProviderIngestionError("adapter resource bounds must be positive")
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
        snapshot_id = canonical_sha256(
            {
                "collection_mode": collection_mode,
                "filename_timestamp": format_utc_timestamp(filename_time),
                "input_class": input_class,
                "raw_snapshot_sha256": digest,
                "version": "gdelt-gsg-snapshot-v1",
            }
        )
        publication_id = f"gsg-snapshot-{snapshot_id}"
        existing = self._load_existing_receipt(publication_id)
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
                existing = self._load_existing_receipt(publication_id)
                if existing is None:
                    raise
            else:
                existing = receipt
        receipt = existing
        if receipt is None:  # pragma: no cover - defensive after collision handling
            raise ProviderIngestionError("snapshot receipt publication did not become visible")
        if (
            receipt.raw_snapshot_sha256 != digest
            or receipt.filename_timestamp != format_utc_timestamp(filename_time)
            or receipt.collection_mode != collection_mode
            or receipt.input_class != input_class
        ):
            raise ProviderIngestionError("snapshot receipt identity collision")
        return self._parse_snapshot(receipt, raw_response_bytes)

    def _load_existing_receipt(self, publication_id: str) -> RawSnapshotReceipt | None:
        publication = self.store.publications_root / publication_id
        if not publication.exists():
            return None
        try:
            verified = self.store.read_publication(publication_id)
            if set(verified.files) != {"receipt.json"}:
                raise ProviderIngestionError("snapshot publication has unexpected files")
            raw = verified.files["receipt.json"]
            payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
            if not isinstance(payload, dict) or canonicalize(payload) != raw:
                raise ProviderIngestionError("stored snapshot receipt is not canonical")
            receipt = RawSnapshotReceipt(**payload)
        except ProviderIngestionError:
            raise
        except (OSError, UnicodeError, TypeError, ValueError, SentimentStorageError) as exc:
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
        self.protocol_config_sha256 = protocol_config_sha256
        self.rights_approval = rights_approval
        self._versions_by_fingerprint: dict[tuple[str, str], ArticleRecord] = {}
        self._versions_by_id: dict[str, ArticleRecord] = {}
        self._links: dict[str, ObservationLink] = {}
        self._exclusions: dict[str, ExcludedObservation] = {}
        self._groups_by_id: dict[str, GroupAnchor] = {}
        self._article_groups: dict[str, str] = {}
        self._terminal_intervals: tuple[TerminalInterval, ...] = ()
        self._next_expected_interval_start: str | None = None

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

    def normalize(
        self,
        snapshots: Iterable[SnapshotResult],
        *,
        retrieval_plan: RetrievalPlan,
        terminal_as_of: str,
    ) -> NormalizationResult:
        """Validate a whole terminal batch, then commit one order-independent state change."""
        materialized_snapshots = tuple(snapshots)
        self._validate_chronology(self._terminal_intervals, self._next_expected_interval_start)
        self._validate_plan_chronology(retrieval_plan)
        coverage = build_coverage_report(
            retrieval_plan, materialized_snapshots, as_of=terminal_as_of
        )
        if coverage.pending_intervals:
            raise ProviderIngestionError(
                "normalization watermark is not terminal; planned intervals remain pending"
            )
        appended_intervals = _terminal_interval_facts(retrieval_plan, materialized_snapshots)
        next_terminal_intervals = self._terminal_intervals + appended_intervals
        next_watermark = retrieval_plan.end_at_exclusive
        self._validate_chronology(next_terminal_intervals, next_watermark)
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
        result = self._result(touched_versions, batch_links, batch_exclusions)
        self._versions_by_id = next_versions
        self._versions_by_fingerprint = next_fingerprints
        self._links = next_links
        self._exclusions = next_exclusions
        self._groups_by_id = next_groups
        self._article_groups = next_article_groups
        self._terminal_intervals = next_terminal_intervals
        self._next_expected_interval_start = next_watermark
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

    @staticmethod
    def _validate_chronology(intervals: Sequence[TerminalInterval], watermark: str | None) -> None:
        previous_end: datetime | None = None
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
            if interval.outcome == "retrieved_and_normalized":
                if (
                    interval.snapshot_state != "complete"
                    or not _is_sha256(interval.raw_snapshot_sha256)
                    or not isinstance(interval.json_line_count, int)
                    or isinstance(interval.json_line_count, bool)
                    or interval.json_line_count < 0
                    or interval.terminal_reason is not None
                ):
                    raise NormalizationIntegrityError(
                        "successful terminal interval facts are contradictory"
                    )
            elif interval.outcome == "provider_gap":
                if interval.snapshot_state == "missing":
                    if (
                        interval.raw_snapshot_sha256 is not None
                        or interval.json_line_count is not None
                    ):
                        raise NormalizationIntegrityError(
                            "missing terminal interval has snapshot facts"
                        )
                elif interval.snapshot_state == "invalid":
                    if (
                        not _is_sha256(interval.raw_snapshot_sha256)
                        or not isinstance(interval.json_line_count, int)
                        or isinstance(interval.json_line_count, bool)
                        or interval.json_line_count < 0
                        or interval.terminal_reason not in _SAFE_SNAPSHOT_ERROR_CODES
                    ):
                        raise NormalizationIntegrityError(
                            "invalid terminal interval has malformed snapshot facts"
                        )
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
            previous_end = end
            if position and intervals[position - 1].start_at >= interval.start_at:
                raise NormalizationIntegrityError(
                    "terminal chronology is not in canonical interval order"
                )
        if not intervals:
            if watermark is not None:
                raise NormalizationIntegrityError(
                    "terminal watermark is not justified by interval facts"
                )
            return
        watermark_time = _parse_chronology_minute(watermark, "next_expected_interval_start")
        if previous_end != watermark_time:
            raise NormalizationIntegrityError(
                "terminal watermark is not justified by interval facts"
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
        self._validate_chronology(self._terminal_intervals, self._next_expected_interval_start)
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
        except (ArticleValidationError, TypeError, ValueError) as exc:
            raise NormalizationIntegrityError("normalizer state payload validation failed") from exc

        normalizer._terminal_intervals = tuple(terminal_intervals)
        normalizer._next_expected_interval_start = chronology_payload[
            "next_expected_interval_start"
        ]
        normalizer._validate_chronology(
            normalizer._terminal_intervals,
            normalizer._next_expected_interval_start,
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

        referenced_raw_hashes = {article.raw_snapshot_sha256 for article in articles}
        for observation_id in set(normalizer._links) | set(normalizer._exclusions):
            referenced_raw_hashes.add(_raw_hash_from_observation_id(observation_id))
        referenced_raw_hashes.update(
            interval.raw_snapshot_sha256
            for interval in normalizer._terminal_intervals
            if interval.raw_snapshot_sha256 is not None
        )
        for digest in sorted(referenced_raw_hashes):
            try:
                store.get_bytes(digest)
            except SentimentStorageError as exc:
                raise NormalizationIntegrityError(
                    f"state transitive raw object failed verification: {digest}"
                ) from exc

        normalizer._validate_state_components(
            normalizer._versions_by_id,
            normalizer._versions_by_fingerprint,
            normalizer._links,
            normalizer._exclusions,
            normalizer._groups_by_id,
            normalizer._article_groups,
        )
        normalizer._validate_rights_bound_state(normalizer._versions_by_id, normalizer._links)
        if normalizer.export_state_files() != verified.files:
            raise NormalizationIntegrityError(
                "hydrated state does not reproduce the exact canonical state files"
            )
        return normalizer

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
        for observation_id, link in links.items():
            if observation_id != link.provider_observation_id:
                raise NormalizationIntegrityError("observation-link index key mismatch")
            if link.article_version_id not in versions:
                raise NormalizationIntegrityError("observation link references a missing version")
            if (
                not _is_sha256(link.raw_snapshot_sha256)
                or _raw_hash_from_observation_id(observation_id) != link.raw_snapshot_sha256
                or link.input_class not in {"provider_response", "synthetic_fixture"}
            ):
                raise NormalizationIntegrityError("observation link provenance is invalid")
        for observation_id, exclusion in exclusions.items():
            if observation_id != exclusion.provider_observation_id:
                raise NormalizationIntegrityError("exclusion index key mismatch")
            if exclusion.reason not in _EXCLUSION_RANK:
                raise NormalizationIntegrityError("persisted exclusion reason is unknown")
            if (
                not _is_sha256(exclusion.raw_snapshot_sha256)
                or _raw_hash_from_observation_id(observation_id) != exclusion.raw_snapshot_sha256
                or exclusion.input_class not in {"provider_response", "synthetic_fixture"}
            ):
                raise NormalizationIntegrityError("exclusion provenance is invalid")
        for article in versions.values():
            first_link = links.get(article.provider_observation_id)
            if (
                first_link is None
                or first_link.article_version_id != article.article_version_id
                or first_link.raw_snapshot_sha256 != article.raw_snapshot_sha256
            ):
                raise NormalizationIntegrityError(
                    "article first observation is not bound to its provenance link"
                )
        used_group_ids = set(article_groups.values())
        if set(groups) != used_group_ids:
            raise NormalizationIntegrityError("group anchors do not match used groups")
        for group_id, group in groups.items():
            if group_id != group.duplicate_group_id:
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


def _terminal_interval_facts(
    plan: RetrievalPlan, snapshots: Sequence[SnapshotResult]
) -> tuple[TerminalInterval, ...]:
    """Derive batch-boundary-independent facts for an already terminal plan."""
    snapshot_by_timestamp = {
        snapshot.receipt.filename_timestamp: snapshot for snapshot in snapshots
    }
    facts: list[TerminalInterval] = []
    for interval in plan.intervals:
        start = _parse_minute_timestamp(interval.filename_timestamp, "interval.filename_timestamp")
        end = format_utc_timestamp(start + EXPECTED_INTERVAL)
        snapshot = snapshot_by_timestamp.get(interval.filename_timestamp)
        if snapshot is None:
            facts.append(
                TerminalInterval(
                    start_at=interval.filename_timestamp,
                    end_at_exclusive=end,
                    outcome="provider_gap",
                    snapshot_state="missing",
                    raw_snapshot_sha256=None,
                    json_line_count=None,
                    terminal_reason="missing_after_due_time",
                )
            )
            continue
        _validate_snapshot_terminal_fact(snapshot, interval.filename_timestamp)
        if snapshot.state == "complete":
            facts.append(
                TerminalInterval(
                    start_at=interval.filename_timestamp,
                    end_at_exclusive=end,
                    outcome="retrieved_and_normalized",
                    snapshot_state="complete",
                    raw_snapshot_sha256=snapshot.receipt.raw_snapshot_sha256,
                    json_line_count=snapshot.json_line_count,
                    terminal_reason=None,
                )
            )
        else:
            facts.append(
                TerminalInterval(
                    start_at=interval.filename_timestamp,
                    end_at_exclusive=end,
                    outcome="provider_gap",
                    snapshot_state="invalid",
                    raw_snapshot_sha256=snapshot.receipt.raw_snapshot_sha256,
                    json_line_count=snapshot.json_line_count,
                    terminal_reason=snapshot.error_code,
                )
            )
    return tuple(facts)


def _validate_snapshot_terminal_fact(snapshot: SnapshotResult, interval_start: str) -> None:
    if snapshot.receipt.filename_timestamp != interval_start:
        raise NormalizationIntegrityError(
            "snapshot receipt does not belong to its terminal interval"
        )
    if not _is_sha256(snapshot.receipt.raw_snapshot_sha256):
        raise NormalizationIntegrityError("terminal snapshot raw hash is invalid")
    if (
        not isinstance(snapshot.json_line_count, int)
        or isinstance(snapshot.json_line_count, bool)
        or snapshot.json_line_count < 0
    ):
        raise NormalizationIntegrityError("terminal snapshot line count is invalid")
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
        if snapshot.observations or snapshot.error_code not in _SAFE_SNAPSHOT_ERROR_CODES:
            raise NormalizationIntegrityError("invalid terminal snapshot facts are contradictory")
    else:
        raise NormalizationIntegrityError("snapshot has an unsupported terminal state")
    for observation in snapshot.observations:
        if (
            observation.filename_timestamp != interval_start
            or observation.raw_snapshot_sha256 != snapshot.receipt.raw_snapshot_sha256
        ):
            raise NormalizationIntegrityError(
                "terminal snapshot observations have inconsistent provenance"
            )


def build_coverage_report(
    plan: RetrievalPlan, snapshots: Iterable[SnapshotResult], *, as_of: str
) -> CoverageReport:
    """Reconcile expected intervals and fail closed on missing or corrupt provider files."""
    as_of_time = _parse_required_timestamp(as_of, "as_of")
    snapshot_by_timestamp: dict[str, SnapshotResult] = {}
    for snapshot in snapshots:
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

    due_count = 0
    complete_count = 0
    zero_line_count = 0
    pending_count = 0
    gap_points: list[tuple[datetime, str]] = []
    observed_receipts: list[dict[str, Any]] = []
    for interval in plan.intervals:
        due_at = _parse_required_timestamp(interval.due_at, "interval.due_at")
        snapshot = snapshot_by_timestamp.get(interval.filename_timestamp)
        if snapshot is not None:
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
            pending_count += 1
            continue
        due_count += 1
        interval_time = _parse_minute_timestamp(
            interval.filename_timestamp, "interval.filename_timestamp"
        )
        if snapshot is not None and snapshot.state == "complete":
            complete_count += 1
            if snapshot.json_line_count == 0:
                zero_line_count += 1
        else:
            reason = snapshot.error_code if snapshot is not None else "missing_after_due_time"
            gap_points.append((interval_time, reason or "invalid_snapshot"))
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
        "zero_line_intervals": zero_line_count,
    }
    return CoverageReport(
        plan_id=plan.plan_id,
        as_of=format_utc_timestamp(as_of_time),
        expected_due_intervals=due_count,
        complete_intervals=complete_count,
        zero_line_intervals=zero_line_count,
        pending_intervals=pending_count,
        gap_intervals=len(gap_points),
        retrieval_rate=retrieval_rate,
        maximum_gap_minutes=base_payload["maximum_gap_minutes"],
        gaps=gaps,
        expected_schedule_sha256=base_payload["expected_schedule_sha256"],
        observed_receipts_sha256=base_payload["observed_receipts_sha256"],
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
    try:
        return store.publish_bundle(
            publication_id,
            files,
            metadata={
                "coverage_semantic_sha256": coverage.semantic_sha256,
                "normalization_semantic_sha256": result.semantic_sha256,
                "protocol_config_sha256": result.protocol_config_sha256,
                "provider": PROVIDER_ID,
                "rights_approval_sha256": result.rights_approval_sha256,
                "scope": "normalized_offline_adapter_batch",
            },
        )
    except PublicationCollisionError as exc:
        try:
            verified = store.read_publication(publication_id)
            if verified.files == files:
                return store.publications_root / publication_id
        except SentimentStorageError:
            pass
        raise ProviderIngestionError(
            f"normalization batch ID collision with different bytes: {batch_id}"
        ) from exc


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
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NormalizationIntegrityError(f"{description} is not valid strict JSON") from exc
    if canonicalize(payload) != value:
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
    match = re.fullmatch(r"gdelt-gsg:.*:([0-9a-f]{64}):\d+:(?:from|to)", value)
    if match is None:
        raise NormalizationIntegrityError("persisted provider observation ID is malformed")
    return match.group(1)


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
