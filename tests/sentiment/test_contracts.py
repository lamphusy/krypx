"""Strict article and score contract tests."""

from copy import deepcopy

import pytest

from crypto_ai.exceptions import ArticleValidationError, ScoreValidationError
from crypto_ai.sentiment.canonical import canonical_sha256
from crypto_ai.sentiment.contracts import (
    derive_article_id,
    derive_article_version_id,
    derive_content_hash,
    derive_duplicate_group_id,
    derive_score_id,
    validate_article_collection,
    validate_article_record,
    validate_score_record,
)

RAW_HASH = "1" * 64
CONFIG_HASH = "2" * 64
INPUT_HASH = "3" * 64
RESPONSE_HASH = "4" * 64


def valid_article(**overrides: object) -> dict[str, object]:
    provider = "fixture_provider"
    provider_article_id = "article-1"
    title = "Bitcoin rises after synthetic event"
    source = "fixture.example"
    content_hash = derive_content_hash(
        asset="BTC", content=None, language="en", source=source, title=title
    )
    article_id = derive_article_id(provider, provider_article_id, None)
    first_seen_at = "2026-08-14T01:02:04Z"
    value: dict[str, object] = {
        "article_id": article_id,
        "article_version_id": derive_article_version_id(
            article_id=article_id,
            first_seen_at=first_seen_at,
            language="en",
            content_hash=content_hash,
        ),
        "provider": provider,
        "provider_article_id": provider_article_id,
        "provider_observation_id": "fixture:1",
        "source": source,
        "canonical_url": "https://fixture.example/story",
        "title": title,
        "content": None,
        "language": "en",
        "published_at": "2026-08-14T00:55:00Z",
        "provider_first_seen_at": None,
        "first_seen_at": first_seen_at,
        "ingested_at": "2026-08-14T01:02:03.123456Z",
        "provider_updated_at": None,
        "asset": "BTC",
        "content_hash": content_hash,
        "raw_snapshot_sha256": RAW_HASH,
        "point_in_time_eligible": True,
        "exclusion_reason": None,
        "duplicate_group_id": derive_duplicate_group_id(article_id),
    }
    value.update(overrides)
    return value


def valid_score(**overrides: object) -> dict[str, object]:
    payload = {"sentiment_score": 0.25, "relevance_score": 0.75}
    content_hash = "5" * 64
    identity = {
        "content_hash": content_hash,
        "asset": "BTC",
        "sentiment_model_id": "synthetic-model-not-approved",
        "sentiment_model_version": "fixture-v1",
        "prompt_version": "fixture-prompt-v1",
        "scoring_config_hash": CONFIG_HASH,
    }
    value: dict[str, object] = {
        "score_id": derive_score_id(**identity),
        "article_version_id": "6" * 64,
        **identity,
        "input_sha256": INPUT_HASH,
        "raw_response_sha256": RESPONSE_HASH,
        "score_payload_sha256": canonical_sha256(payload),
        "scored_at": "2026-08-14T01:03:00Z",
        "state": "succeeded",
        "payload": payload,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-14 01:02:03Z",
        "2026-08-14T01:02:03",
        "2026-08-14T01:02:03+00:00",
        "2026-13-14T01:02:03Z",
        "2026-08-14T01:02:03.1234567Z",
    ],
)
def test_article_rejects_malformed_or_noncontract_timestamps(timestamp: str) -> None:
    with pytest.raises(ArticleValidationError):
        validate_article_record(valid_article(ingested_at=timestamp))


def test_article_rejects_hash_mismatch_and_extra_fields() -> None:
    with pytest.raises(ArticleValidationError, match="content_hash mismatch"):
        validate_article_record(valid_article(content_hash="0" * 64))
    with pytest.raises(ArticleValidationError, match="extra"):
        validate_article_record({**valid_article(), "surprise": True})


def test_collection_rejects_duplicate_identities() -> None:
    first = validate_article_record(valid_article())
    second_value = deepcopy(valid_article(provider_observation_id="fixture:2"))
    second = validate_article_record(second_value)
    with pytest.raises(ArticleValidationError, match="duplicate article_version_id"):
        validate_article_collection([first, second])


def test_collection_rejects_changed_first_seen_for_same_fingerprint() -> None:
    first = validate_article_record(valid_article())
    changed = valid_article(
        provider_observation_id="fixture:2", first_seen_at="2026-08-14T01:03:00Z"
    )
    changed["article_version_id"] = derive_article_version_id(
        article_id=str(changed["article_id"]),
        first_seen_at=str(changed["first_seen_at"]),
        language="en",
        content_hash=str(changed["content_hash"]),
    )
    second = validate_article_record(changed)
    with pytest.raises(ArticleValidationError, match="first_seen_at is immutable"):
        validate_article_collection([first, second])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_score_rejects_nonfinite_values(score: float) -> None:
    value = valid_score()
    assert isinstance(value["payload"], dict)
    value["payload"]["sentiment_score"] = score
    with pytest.raises(ScoreValidationError, match="finite"):
        validate_score_record(value)


def test_score_validates_exact_payload_hash_and_identity() -> None:
    record = validate_score_record(valid_score())
    assert record.payload is not None and record.payload.relevance_score == 0.75
    with pytest.raises(ScoreValidationError, match="score_payload_sha256 mismatch"):
        validate_score_record(valid_score(score_payload_sha256="0" * 64))
