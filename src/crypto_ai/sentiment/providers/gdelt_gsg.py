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
    ProviderIngestionError,
    PublicationCollisionError,
    SentimentStorageError,
)
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize
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
NORMALIZER_VERSION = "gdelt-gsg-normalizer-v1"
URL_NORMALIZER_VERSION = "url-canonicalization-v1"
TEXT_NORMALIZER_VERSION = "text-normalization-v1"
LANGUAGE_MAP_VERSION = "language-map-v1"
EXPECTED_INTERVAL = timedelta(minutes=1)
PROVIDER_LAG = timedelta(minutes=30)
MAX_PLAN_INTERVALS = 10_080
MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_JSON_LINES = 1_000_000

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExcludedObservation:
    provider_observation_id: str
    reason: str
    diagnostic: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    articles: tuple[ArticleRecord, ...]
    observation_links: tuple[ObservationLink, ...]
    exclusions: tuple[ExcludedObservation, ...]
    semantic_sha256: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "articles": [article.to_dict() for article in self.articles],
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
            "normalizer_version": NORMALIZER_VERSION,
            "observation_links": [link.to_dict() for link in self.observation_links],
        }
        if include_hash:
            value["semantic_sha256"] = self.semantic_sha256
        return value


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

        digest = self.store.put_bytes(raw_response_bytes)
        snapshot_id = canonical_sha256(
            {
                "collection_mode": collection_mode,
                "filename_timestamp": format_utc_timestamp(filename_time),
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
        ):
            raise ProviderIngestionError("snapshot receipt identity collision")
        return self._parse_snapshot(receipt, raw_response_bytes)

    def _load_existing_receipt(self, publication_id: str) -> RawSnapshotReceipt | None:
        publication = self.store.publications_root / publication_id
        if not publication.exists():
            return None
        try:
            self.store.verify_publication(publication_id)
            raw = (publication / "receipt.json").read_bytes()
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
    """Deterministic version, revision, observation-link, and exact-dedup state."""

    def __init__(self) -> None:
        self._versions_by_fingerprint: dict[tuple[str, str], ArticleRecord] = {}
        self._versions_by_id: dict[str, ArticleRecord] = {}
        self._links: dict[str, ObservationLink] = {}
        self._article_groups: dict[str, str] = {}
        self._dedup_anchors: dict[str, list[tuple[datetime, str, str, str]]] = {}

    def normalize(
        self,
        snapshots: Iterable[SnapshotResult],
        *,
        retrieval_plan: RetrievalPlan,
        terminal_as_of: str,
    ) -> NormalizationResult:
        """Normalize only after every planned interval is terminally complete or a gap."""
        materialized_snapshots = tuple(snapshots)
        coverage = build_coverage_report(
            retrieval_plan, materialized_snapshots, as_of=terminal_as_of
        )
        if coverage.pending_intervals:
            raise ProviderIngestionError(
                "normalization watermark is not terminal; planned intervals remain pending"
            )
        observations = [
            observation
            for snapshot in materialized_snapshots
            if snapshot.state == "complete"
            for observation in snapshot.observations
        ]
        observations.sort(key=_observation_sort_key)
        touched_versions: dict[str, ArticleRecord] = {}
        links: dict[str, ObservationLink] = {}
        exclusions: dict[str, ExcludedObservation] = {}
        for observation in observations:
            existing_link = self._links.get(observation.provider_observation_id)
            if existing_link is not None:
                links[existing_link.provider_observation_id] = existing_link
                touched_versions[existing_link.article_version_id] = self._versions_by_id[
                    existing_link.article_version_id
                ]
                continue
            try:
                normalized = self._normalize_observation(observation)
            except _ObservationExcluded as exc:
                exclusions[observation.provider_observation_id] = ExcludedObservation(
                    observation.provider_observation_id, exc.reason, exc.diagnostic
                )
                continue
            record, reused = normalized
            link = ObservationLink(
                provider_observation_id=observation.provider_observation_id,
                article_version_id=record.article_version_id,
                reused_existing_version=reused,
            )
            self._links[link.provider_observation_id] = link
            links[link.provider_observation_id] = link
            touched_versions[record.article_version_id] = record

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
        }
        return NormalizationResult(
            articles=articles_tuple,
            observation_links=links_tuple,
            exclusions=exclusions_tuple,
            semantic_sha256=canonical_sha256(hash_payload),
        )

    def _normalize_observation(self, observation: GSGObservation) -> tuple[ArticleRecord, bool]:
        if observation.collection_mode == "historical_backfill":
            raise _ObservationExcluded(
                "historical_backfill_without_availability",
                "historical GSG observations are audit-only and never model eligible",
            )
        if not isinstance(observation.raw_url, str) or not observation.raw_url.strip():
            raise _ObservationExcluded("invalid_url_or_identifier", "endpoint URL is missing")
        if not isinstance(observation.raw_title, str):
            raise _ObservationExcluded("missing_title_and_content", "endpoint title is missing")
        if not isinstance(observation.raw_language, str):
            raise _ObservationExcluded("unsupported_language", "endpoint language is missing")
        try:
            canonical_url = canonicalize_url(observation.raw_url)
            title = normalize_title(observation.raw_title)
            language = normalize_gsg_language(observation.raw_language)
            provider_first_seen_at = normalize_provider_timestamp(
                observation.raw_provider_first_seen_at
            )
        except _ObservationExcluded:
            raise
        except (TypeError, ValueError, UnicodeError, idna.IDNAError) as exc:
            raise _ObservationExcluded("malformed_record", _safe_error_code(exc)) from exc
        if not title:
            raise _ObservationExcluded("missing_title_and_content", "normalized title is blank")
        if not selects_direct_btc(title):
            raise _ObservationExcluded("asset_mismatch", "title does not match direct BTC selector")
        source = urlsplit(canonical_url).hostname
        if source is None:
            raise _ObservationExcluded("invalid_url_or_identifier", "canonical URL has no host")
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
        key = (article_id, fingerprint)
        existing = self._versions_by_fingerprint.get(key)
        if existing is not None:
            return existing, True

        first_seen = _parse_required_timestamp(observation.raw_published_at, "raw_published_at")
        same_article_versions = [
            record for record in self._versions_by_id.values() if record.article_id == article_id
        ]
        if any(
            record.first_seen_at == observation.raw_published_at for record in same_article_versions
        ):
            raise _ObservationExcluded(
                "revision_time_unknown",
                "different content for one article has the same availability timestamp",
            )
        duplicate_group_id = self._assign_duplicate_group(
            article_id=article_id,
            canonical_url=canonical_url,
            source=source,
            title=title,
            language=language,
            first_seen=first_seen,
        )
        first_seen_text = format_utc_timestamp(first_seen)
        version_id = derive_article_version_id(
            article_id=article_id,
            first_seen_at=first_seen_text,
            language=language,
            content_hash=content_hash,
        )
        value = {
            "article_id": article_id,
            "article_version_id": version_id,
            "provider": PROVIDER_ID,
            "provider_article_id": None,
            "provider_observation_id": observation.provider_observation_id,
            "source": source,
            "canonical_url": canonical_url,
            "title": title,
            "content": None,
            "language": language,
            "published_at": None,
            "provider_first_seen_at": provider_first_seen_at,
            "first_seen_at": first_seen_text,
            "ingested_at": observation.ingested_at,
            "provider_updated_at": None,
            "asset": "BTC",
            "content_hash": content_hash,
            "raw_snapshot_sha256": observation.raw_snapshot_sha256,
            "point_in_time_eligible": True,
            "exclusion_reason": None,
            "duplicate_group_id": duplicate_group_id,
        }
        try:
            record = validate_article_record(value)
        except ArticleValidationError as exc:
            raise _ObservationExcluded("malformed_record", str(exc)) from exc
        self._versions_by_fingerprint[key] = record
        self._versions_by_id[record.article_version_id] = record
        return record, False

    def _assign_duplicate_group(
        self,
        *,
        article_id: str,
        canonical_url: str,
        source: str,
        title: str,
        language: str,
        first_seen: datetime,
    ) -> str:
        existing_article_group = self._article_groups.get(article_id)
        if existing_article_group is not None:
            return existing_article_group
        dedup_fingerprint = canonical_sha256(
            {
                "content": "",
                "language": language,
                "serialization_version": "dedup-fingerprint-v1",
                "title_casefold": title.casefold(),
            }
        )
        candidates = self._dedup_anchors.setdefault(dedup_fingerprint, [])
        tolerance = timedelta(hours=72)
        for anchor_time, anchor_article_id, anchor_url, anchor_source in candidates:
            if first_seen - anchor_time > tolerance:
                continue
            if canonical_url != anchor_url or source != anchor_source:
                group_id = derive_duplicate_group_id(anchor_article_id)
                self._article_groups[article_id] = group_id
                return group_id
        group_id = derive_duplicate_group_id(article_id)
        candidates.append((first_seen, article_id, canonical_url, source))
        candidates.sort(key=lambda item: (item[0], item[1]))
        self._article_groups[article_id] = group_id
        return group_id


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
                "provider": PROVIDER_ID,
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
                "provider": PROVIDER_ID,
                "scope": "normalized_offline_adapter_batch",
            },
        )
    except PublicationCollisionError as exc:
        directory = store.publications_root / publication_id
        try:
            store.verify_publication(publication_id)
            if all((directory / name).read_bytes() == data for name, data in files.items()):
                return directory
        except (OSError, SentimentStorageError):
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
