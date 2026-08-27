"""Strict, scorer-independent Phase 2 article and score contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from crypto_ai.exceptions import ArticleValidationError, ScoreValidationError
from crypto_ai.sentiment.canonical import canonical_sha256, sha256_bytes

ARTICLE_CONTRACT_VERSION = "article-contract-v1"
SCORE_CONTRACT_VERSION = "scoring-contract-v1-draft"
SENTIMENT_INPUT_VERSION = "sentiment-input-v1"
ARTICLE_FINGERPRINT_VERSION = "article-version-fingerprint-v1"

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
UTC_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")

EXCLUSION_REASONS = frozenset(
    {
        "missing_first_seen",
        "undocumented_first_seen_semantics",
        "historical_backfill_without_availability",
        "missing_identity",
        "missing_title_and_content",
        "unsupported_language",
        "asset_mismatch",
        "invalid_timestamp",
        "invalid_url_or_identifier",
        "hash_mismatch",
        "revision_time_unknown",
        "license_restricted",
        "provider_gap",
        "duplicate_unresolved",
        "malformed_record",
    }
)

SCORE_STATES = frozenset(
    {
        "pending",
        "succeeded",
        "input_too_long",
        "transient_exhausted",
        "invalid_output",
        "permanent_error",
        "hash_mismatch",
        "budget_blocked",
        "license_blocked",
    }
)


@dataclass(frozen=True, slots=True)
class ArticleRecord:
    """One immutable normalized article version."""

    article_id: str
    article_version_id: str
    provider: str
    provider_article_id: str | None
    provider_observation_id: str
    source: str
    canonical_url: str | None
    title: str | None
    content: str | None
    language: str
    published_at: str | None
    provider_first_seen_at: str | None
    first_seen_at: str | None
    ingested_at: str
    provider_updated_at: str | None
    asset: str
    content_hash: str
    raw_snapshot_sha256: str
    point_in_time_eligible: bool
    exclusion_reason: str | None
    duplicate_group_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScorePayload:
    """Strict normalized scorer payload; no scorer is selected or invoked here."""

    sentiment_score: float
    relevance_score: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """Immutable score envelope keyed by the frozen scoring inputs."""

    score_id: str
    article_version_id: str
    content_hash: str
    asset: str
    sentiment_model_id: str
    sentiment_model_version: str
    prompt_version: str
    scoring_config_hash: str
    input_sha256: str
    raw_response_sha256: str | None
    score_payload_sha256: str | None
    scored_at: str | None
    state: str
    payload: ScorePayload | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value


ARTICLE_FIELDS = frozenset(ArticleRecord.__dataclass_fields__)
SCORE_FIELDS = frozenset(ScoreRecord.__dataclass_fields__)
SCORE_PAYLOAD_FIELDS = frozenset(ScorePayload.__dataclass_fields__)


def parse_utc_timestamp(value: object, *, field: str, nullable: bool = False) -> datetime | None:
    """Validate the contract's unambiguous UTC timestamp representation."""
    if value is None and nullable:
        return None
    if not isinstance(value, str) or UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid UTC timestamp") from exc
    if parsed.tzinfo != UTC:
        raise ValueError(f"{field} must use UTC")
    return parsed


def format_utc_timestamp(value: datetime) -> str:
    """Render an aware UTC datetime in the contract form."""
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    utc_value = value.astimezone(UTC)
    timespec = "microseconds" if utc_value.microsecond else "seconds"
    return utc_value.isoformat(timespec=timespec).replace("+00:00", "Z")


def derive_article_id(
    provider: str, provider_article_id: str | None, canonical_url: str | None
) -> str:
    """Derive the frozen stable article identity."""
    if provider_article_id:
        material = f"article-id-v1\n{provider}\n{provider_article_id}"
    elif canonical_url:
        material = f"article-url-v1\n{provider}\n{canonical_url}"
    else:
        raise ArticleValidationError("article identity requires a provider ID or canonical URL")
    return sha256_bytes(material.encode("utf-8"))


def sentiment_input_payload(
    *, asset: str, content: str | None, language: str, source: str, title: str | None
) -> dict[str, Any]:
    return {
        "asset": asset,
        "content": content,
        "language": language,
        "serialization_version": SENTIMENT_INPUT_VERSION,
        "source": source,
        "title": title,
    }


def derive_content_hash(
    *, asset: str, content: str | None, language: str, source: str, title: str | None
) -> str:
    """Hash the exact JCS bytes supplied to a future scorer."""
    return canonical_sha256(
        sentiment_input_payload(
            asset=asset, content=content, language=language, source=source, title=title
        )
    )


def derive_version_fingerprint(
    *,
    article_id: str,
    content_hash: str,
    content: str | None,
    language: str,
    provider: str,
    source: str,
    title: str | None,
) -> str:
    return canonical_sha256(
        {
            "article_id": article_id,
            "content": content,
            "content_hash": content_hash,
            "language": language,
            "provider": provider,
            "serialization_version": ARTICLE_FINGERPRINT_VERSION,
            "source": source,
            "title": title,
        }
    )


def derive_article_version_id(
    *, article_id: str, first_seen_at: str, language: str, content_hash: str
) -> str:
    material = f"article-version-v1\n{article_id}\n{first_seen_at}\n{language}\n{content_hash}"
    return sha256_bytes(material.encode("utf-8"))


def derive_duplicate_group_id(anchor_article_id: str) -> str:
    return sha256_bytes(f"duplicate-group-v1\n{anchor_article_id}".encode())


def derive_score_id(
    *,
    content_hash: str,
    asset: str,
    sentiment_model_id: str,
    sentiment_model_version: str,
    prompt_version: str,
    scoring_config_hash: str,
) -> str:
    return canonical_sha256(
        {
            "asset": asset,
            "content_hash": content_hash,
            "prompt_version": prompt_version,
            "scoring_config_hash": scoring_config_hash,
            "sentiment_model_id": sentiment_model_id,
            "sentiment_model_version": sentiment_model_version,
        }
    )


def validate_article_record(value: Mapping[str, Any]) -> ArticleRecord:
    """Strictly validate and return one immutable article record."""
    _require_exact_fields(value, ARTICLE_FIELDS, ArticleValidationError, "article")
    try:
        record = ArticleRecord(**dict(value))
        _validate_article(record)
    except ArticleValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ArticleValidationError(str(exc)) from exc
    return record


def _validate_article(record: ArticleRecord) -> None:
    _require_nullable_string(record.title, "title", ArticleValidationError)
    _require_nullable_string(record.content, "content", ArticleValidationError)
    _require_hash(record.article_id, "article_id", ArticleValidationError)
    _require_hash(record.article_version_id, "article_version_id", ArticleValidationError)
    _require_hash(record.content_hash, "content_hash", ArticleValidationError)
    _require_hash(record.raw_snapshot_sha256, "raw_snapshot_sha256", ArticleValidationError)
    if record.duplicate_group_id is not None:
        _require_hash(record.duplicate_group_id, "duplicate_group_id", ArticleValidationError)
    _require_nonblank(record.provider, "provider", ArticleValidationError)
    _require_nonblank(
        record.provider_observation_id, "provider_observation_id", ArticleValidationError
    )
    _require_nonblank(record.source, "source", ArticleValidationError)
    if record.provider_article_id is not None:
        _require_nonblank(record.provider_article_id, "provider_article_id", ArticleValidationError)
    _validate_url(record.canonical_url, required=record.provider_article_id is None)
    if (record.title is None or not record.title.strip()) and (
        record.content is None or not record.content.strip()
    ):
        raise ArticleValidationError("title and content cannot both be blank")
    if record.provider == "gdelt_gsg":
        if record.title is None or not record.title.strip() or record.content is not None:
            raise ArticleValidationError("GDELT GSG requires a nonblank title and null content")
    if record.language != "en":
        raise ArticleValidationError("language must be exactly en")
    if record.asset != "BTC":
        raise ArticleValidationError("asset must be exactly BTC")
    ingested_at = parse_utc_timestamp(record.ingested_at, field="ingested_at")
    first_seen_at = parse_utc_timestamp(record.first_seen_at, field="first_seen_at", nullable=True)
    for field in ("published_at", "provider_first_seen_at", "provider_updated_at"):
        parse_utc_timestamp(getattr(record, field), field=field, nullable=True)
    if first_seen_at is not None and ingested_at is not None and first_seen_at < ingested_at:
        raise ArticleValidationError("first_seen_at cannot precede ingested_at")
    if not isinstance(record.point_in_time_eligible, bool):
        raise ArticleValidationError("point_in_time_eligible must be a boolean")
    if record.exclusion_reason is not None and record.exclusion_reason not in EXCLUSION_REASONS:
        raise ArticleValidationError("exclusion_reason is not in the frozen enum")
    if record.point_in_time_eligible:
        if record.exclusion_reason is not None:
            raise ArticleValidationError("eligible articles cannot have an exclusion reason")
        if record.first_seen_at is None or record.duplicate_group_id is None:
            raise ArticleValidationError(
                "eligible articles require first_seen_at and a dedup group"
            )
    elif record.exclusion_reason is None:
        raise ArticleValidationError("ineligible articles require an exclusion reason")

    expected_article_id = derive_article_id(
        record.provider, record.provider_article_id, record.canonical_url
    )
    if record.article_id != expected_article_id:
        raise ArticleValidationError("article_id hash mismatch")
    expected_content_hash = derive_content_hash(
        asset=record.asset,
        content=record.content,
        language=record.language,
        source=record.source,
        title=record.title,
    )
    if record.content_hash != expected_content_hash:
        raise ArticleValidationError("content_hash mismatch")
    if record.first_seen_at is None:
        raise ArticleValidationError("article_version_id cannot be derived without first_seen_at")
    expected_version = derive_article_version_id(
        article_id=record.article_id,
        first_seen_at=record.first_seen_at,
        language=record.language,
        content_hash=record.content_hash,
    )
    if record.article_version_id != expected_version:
        raise ArticleValidationError("article_version_id hash mismatch")


def validate_article_collection(records: Iterable[ArticleRecord]) -> tuple[ArticleRecord, ...]:
    """Reject duplicate or conflicting identities in a materialized article collection."""
    materialized = tuple(records)
    versions: dict[str, ArticleRecord] = {}
    fingerprints: dict[tuple[str, str], ArticleRecord] = {}
    observations: set[str] = set()
    for record in materialized:
        if record.article_version_id in versions:
            raise ArticleValidationError(
                f"duplicate article_version_id: {record.article_version_id}"
            )
        if record.provider_observation_id in observations:
            raise ArticleValidationError(
                f"duplicate provider_observation_id: {record.provider_observation_id}"
            )
        fingerprint = derive_version_fingerprint(
            article_id=record.article_id,
            content_hash=record.content_hash,
            content=record.content,
            language=record.language,
            provider=record.provider,
            source=record.source,
            title=record.title,
        )
        key = (record.article_id, fingerprint)
        existing = fingerprints.get(key)
        if existing is not None and existing.first_seen_at != record.first_seen_at:
            raise ArticleValidationError("first_seen_at is immutable for an article fingerprint")
        versions[record.article_version_id] = record
        fingerprints[key] = record
        observations.add(record.provider_observation_id)
    return materialized


def validate_score_record(value: Mapping[str, Any]) -> ScoreRecord:
    """Strictly validate a score envelope without choosing or invoking a scorer."""
    _require_exact_fields(value, SCORE_FIELDS, ScoreValidationError, "score")
    raw_payload = value.get("payload")
    payload: ScorePayload | None
    if raw_payload is None:
        payload = None
    elif isinstance(raw_payload, Mapping):
        _require_exact_fields(
            raw_payload, SCORE_PAYLOAD_FIELDS, ScoreValidationError, "score payload"
        )
        try:
            payload = ScorePayload(**dict(raw_payload))
        except TypeError as exc:
            raise ScoreValidationError(str(exc)) from exc
    else:
        raise ScoreValidationError("payload must be an object or null")
    try:
        record = ScoreRecord(**{**dict(value), "payload": payload})
        _validate_score(record)
    except ScoreValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ScoreValidationError(str(exc)) from exc
    return record


def _validate_score(record: ScoreRecord) -> None:
    for field in (
        "score_id",
        "article_version_id",
        "content_hash",
        "scoring_config_hash",
        "input_sha256",
    ):
        _require_hash(getattr(record, field), field, ScoreValidationError)
    for field in ("raw_response_sha256", "score_payload_sha256"):
        value = getattr(record, field)
        if value is not None:
            _require_hash(value, field, ScoreValidationError)
    for field in (
        "sentiment_model_id",
        "sentiment_model_version",
        "prompt_version",
    ):
        _require_nonblank(getattr(record, field), field, ScoreValidationError)
    if record.asset != "BTC":
        raise ScoreValidationError("asset must be exactly BTC")
    if record.state not in SCORE_STATES:
        raise ScoreValidationError("state is not in the frozen enum")
    parse_utc_timestamp(record.scored_at, field="scored_at", nullable=True)
    if record.state == "succeeded":
        if record.payload is None or record.raw_response_sha256 is None or record.scored_at is None:
            raise ScoreValidationError(
                "successful scores require payload, response hash, and scored_at"
            )
        sentiment = _finite_number(record.payload.sentiment_score, "sentiment_score")
        relevance = _finite_number(record.payload.relevance_score, "relevance_score")
        if not -1.0 <= sentiment <= 1.0:
            raise ScoreValidationError("sentiment_score must be between -1 and 1")
        if not 0.0 <= relevance <= 1.0:
            raise ScoreValidationError("relevance_score must be between 0 and 1")
        expected_payload_hash = canonical_sha256(record.payload.to_dict())
        if record.score_payload_sha256 != expected_payload_hash:
            raise ScoreValidationError("score_payload_sha256 mismatch")
    elif record.payload is not None or record.score_payload_sha256 is not None:
        raise ScoreValidationError("non-success states cannot contain a normalized payload")
    expected_score_id = derive_score_id(
        content_hash=record.content_hash,
        asset=record.asset,
        sentiment_model_id=record.sentiment_model_id,
        sentiment_model_version=record.sentiment_model_version,
        prompt_version=record.prompt_version,
        scoring_config_hash=record.scoring_config_hash,
    )
    if record.score_id != expected_score_id:
        raise ScoreValidationError("score_id mismatch")


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreValidationError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ScoreValidationError(f"{field} must be finite")
    return number


def _require_hash(value: object, field: str, error: type[Exception]) -> None:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise error(f"{field} must be a lowercase SHA-256 digest")


def _require_nonblank(value: object, field: str, error: type[Exception]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{field} must be a nonblank string")


def _require_nullable_string(value: object, field: str, error: type[Exception]) -> None:
    if value is not None and not isinstance(value, str):
        raise error(f"{field} must be a string or null")


def _require_exact_fields(
    value: Mapping[str, Any], fields: frozenset[str], error: type[Exception], label: str
) -> None:
    if not isinstance(value, Mapping):
        raise error(f"{label} must be an object")
    actual = frozenset(value)
    missing = sorted(fields - actual)
    extra = sorted(actual - fields)
    if missing or extra:
        raise error(f"{label} fields mismatch; missing={missing}, extra={extra}")


def _validate_url(value: str | None, *, required: bool) -> None:
    if value is None:
        if required:
            raise ArticleValidationError("canonical_url is required without provider_article_id")
        return
    if not isinstance(value, str):
        raise ArticleValidationError("canonical_url must be a string or null")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ArticleValidationError("canonical_url is not canonical HTTP(S)")
