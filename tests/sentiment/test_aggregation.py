"""Synthetic-only point-in-time aggregation and independent binary64 reference."""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import socket
import struct
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment import aggregation as aggregation_module
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.aggregation import (
    AggregationArtifact,
    AggregationAuthorizationError,
    AggregationCoverageError,
    AggregationInputError,
    AggregationIntegrityError,
    AggregationStore,
    FeatureValues,
    OfflineFeatureAggregator,
    SyntheticAggregationInput,
)
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import ScorePayload, format_utc_timestamp
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GapAttempt,
    GSGAdapter,
    GSGNormalizer,
    RightsApproval,
    TerminalGapEvidence,
    plan_retrieval,
)
from crypto_ai.sentiment.scoring import (
    MockScorer,
    OfflineScoringEngine,
    ScoreArtifact,
    ScoringStore,
    SyntheticInput,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

START = datetime(2026, 9, 1, tzinfo=UTC)
DECISION = START + timedelta(hours=25)
PROTOCOL_HASH = sha256_bytes(b"synthetic milestone four protocol fixture")
FEATURE_NAMES = (
    "sentiment_mean_6h",
    "sentiment_mean_24h",
    "news_count_1h",
    "news_count_6h",
    "news_count_24h",
    "sentiment_recency_6h",
    "sentiment_recency_24h",
    "positive_share_24h",
    "negative_share_24h",
    "sentiment_dispersion_24h",
    "source_count_24h",
    "hours_since_latest_article",
    "news_missing_24h",
)
INTEGER_FEATURES = {
    "news_count_1h",
    "news_count_6h",
    "news_count_24h",
    "source_count_24h",
    "news_missing_24h",
}


def instant(value: datetime) -> str:
    return format_utc_timestamp(value)


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("aggregation tests must remain entirely offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@dataclass(frozen=True)
class FixtureCorpus:
    store: ContentAddressedStore
    populated: SyntheticAggregationInput
    empty: SyntheticAggregationInput
    gap: SyntheticAggregationInput
    prepared: Any


def synthetic_line(name: str, *, title: str | None = None, source: str = "desk") -> bytes:
    """One real parser input, deliberately not a publisher fetch or article body."""
    return (
        canonicalize(
            {
                "fromDate": "20260901000000",
                "fromUrl": f"https://{source}.invalid/{name}",
                "fromTitle": title or f"Bitcoin synthetic {name} headline",
                "fromLang": "English",
            }
        )
        + b"\n"
    )


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> FixtureCorpus:
    """Pay the full immutable 1-minute receipt verification cost once per module."""
    root = tmp_path_factory.mktemp("synthetic-aggregation")
    store = ContentAddressedStore(root / "provider")
    empty_raw = gzip.compress(b"", mtime=0)
    plan = plan_retrieval(instant(START), instant(START + timedelta(hours=28)))
    publication_clock = [START]
    adapter = GSGAdapter(store, clock=lambda: publication_clock[0])
    empty_snapshots = []
    populated_snapshots = []
    # Availability is exactly source minute + release lag. One event lies at each
    # 24h/6h/1h lower bound, one at t, and one strictly after t.
    event_ages = {
        30: ("expired", "Bitcoin synthetic expired headline", "old"),
        31: ("old", "Bitcoin synthetic retained old headline", "old"),
        1110: ("six", "Bitcoin synthetic six hour boundary", "six"),
        1111: ("anchor", "Bitcoin synthetic permanent duplicate anchor", "anchor"),
        1112: ("syndicated", "Bitcoin synthetic permanent duplicate anchor", "copy"),
        1410: ("one", "Bitcoin synthetic one hour boundary", "one"),
        1411: ("recent", "Bitcoin synthetic recent headline", "recent"),
        1470: ("now", "Bitcoin synthetic exact decision headline", "now"),
        1471: ("anchor", "Bitcoin synthetic revised anchor headline", "anchor"),
        1472: ("future", "Bitcoin synthetic future headline", "future"),
    }
    for index, interval in enumerate(plan.intervals):
        publication_clock[0] = START + timedelta(minutes=index + 30)
        kwargs = {
            "filename_timestamp": interval.filename_timestamp,
            "ingested_at": instant(publication_clock[0]),
            "source_locator": (
                "https://data.gdeltproject.org/gdeltv3/gsg/"
                f"{(START + timedelta(minutes=index)).strftime('%Y%m%d%H%M%S')}.gsg.json.gz"
            ),
            "collection_mode": "prospective",
            "input_class": "synthetic_fixture",
        }
        empty = adapter.ingest_snapshot(empty_raw, **kwargs)
        empty_snapshots.append(empty)
        event = event_ages.get(index)
        if event is None:
            populated_snapshots.append(empty)
        else:
            name, title, source = event
            populated_snapshots.append(
                adapter.ingest_snapshot(
                    gzip.compress(synthetic_line(name, title=title, source=source), mtime=0),
                    **kwargs,
                )
            )
    approval = RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=PROTOCOL_HASH,
        raw_snapshot_sha256={
            snapshot.receipt.raw_snapshot_sha256
            for snapshot in (*empty_snapshots, *populated_snapshots)
        },
    )
    as_of = instant(START + timedelta(hours=29))

    def state(name: str, snapshots: list, gaps: tuple = ()) -> GSGNormalizer:
        normalizer = GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH, rights_approval=approval)
        normalizer.normalize(
            snapshots, retrieval_plan=plan, terminal_as_of=as_of, gap_evidence=gaps
        )
        normalizer.publish_state(store, name)
        return normalizer

    populated = state("populated", populated_snapshots)
    state("empty", empty_snapshots)
    gap_index = 1470  # The source gap starts at t - 30m, not at release/terminal time.
    gap_interval = plan.intervals[gap_index]
    evidence = TerminalGapEvidence.create(
        interval_start=gap_interval.filename_timestamp,
        interval_end_exclusive=instant(START + timedelta(minutes=gap_index + 1)),
        expected_source_locator=empty_snapshots[gap_index].receipt.source_locator,
        attempts=(GapAttempt(1, gap_interval.due_at, 404, None, None, "gap"),),
        terminal_at=gap_interval.due_at,
        protocol_config_sha256=PROTOCOL_HASH,
    )
    state("gap", [s for i, s in enumerate(empty_snapshots) if i != gap_index], (evidence,))
    score_store = ScoringStore(root / "scores")
    score_artifacts = []
    for index, article in enumerate(json.loads(populated.export_state_files()["articles.json"])):
        raw = canonicalize(
            {"sentiment_score": (-0.5 if index % 2 else 0.75), "relevance_score": 0.5}
        )
        scorer = OfflineScoringEngine(
            score_store,
            MockScorer(steps=(raw,)),
            # A far later audit clock must never become a causal article filter.
            clock=lambda: datetime(2099, 1, 1, tzinfo=UTC),
        )
        score_artifacts.append(
            scorer.score(SyntheticInput(source=article["source"], title=article["title"]))
        )

    def request(name: str, scores: tuple[ScoreArtifact, ...]) -> SyntheticAggregationInput:
        return SyntheticAggregationInput(
            state_publication_id=f"gsg-normalizer-state-{name}",
            score_artifacts=scores,
            coverage_as_of=as_of,
            protocol_config_sha256=PROTOCOL_HASH,
        )

    populated_request = request("populated", tuple(score_artifacts))
    return FixtureCorpus(
        store,
        populated_request,
        request("empty", ()),
        request("gap", ()),
        aggregation_module._prepare(store, populated_request),
    )


def reference_features(prepared: Any, decision: datetime) -> dict[str, int | float]:
    """Deliberately slow group/version enumeration; no production reduction helper."""
    selected = []
    scores = {score.content_hash: score for score in prepared.scores}
    for group in prepared.groups:
        arrival = timestamp(group.initial_first_seen_at)
        if arrival > decision:
            continue
        versions = []
        for article in prepared.articles:
            if (
                article.article_id == group.anchor_article_id
                and article.point_in_time_eligible is True
                and article.asset == "BTC"
                and article.first_seen_at is not None
                and timestamp(article.first_seen_at) <= decision
            ):
                versions.append(article)
        if not versions:
            continue
        latest = max(versions, key=lambda article: timestamp(article.first_seen_at))
        score = scores.get(latest.content_hash)
        selected.append((arrival, group.duplicate_group_id, group.source, score))
    selected.sort(key=lambda item: (item[0], item[1]))
    result: dict[str, int | float] = {}
    for hours in (1, 6, 24):
        within = [item for item in selected if item[0] > decision - timedelta(hours=hours)]
        result[f"news_count_{hours}h"] = len(within)
        if hours == 1:
            continue
        relevant = []
        for arrival, _, _, score in within:
            if score is not None and score.state == "succeeded" and score.payload is not None:
                relevance = float(score.payload.relevance_score)
                if relevance >= 0.20:
                    relevant.append(
                        (
                            float((decision - arrival).total_seconds() / 3600),
                            float(score.payload.sentiment_score),
                            relevance,
                        )
                    )
        denominator = numerator = decayed_denominator = decayed_numerator = 0.0
        positive = negative = 0.0
        for age, sentiment, relevance in relevant:
            denominator = float(denominator + relevance)
            numerator = float(numerator + float(relevance * sentiment))
            decayed = float(relevance * math.pow(2.0, float(-age / float(hours))))
            decayed_denominator = float(decayed_denominator + decayed)
            decayed_numerator = float(decayed_numerator + float(decayed * sentiment))
            positive = float(positive + (relevance if sentiment > 0.20 else 0.0))
            negative = float(negative + (relevance if sentiment < -0.20 else 0.0))
        mean = float(numerator / denominator) if denominator else 0.0
        result[f"sentiment_mean_{hours}h"] = mean or 0.0
        result[f"sentiment_recency_{hours}h"] = (
            float(decayed_numerator / decayed_denominator) if decayed_denominator else 0.0
        ) or 0.0
        if hours == 24:
            variance_numerator = 0.0
            for _, sentiment, relevance in relevant:
                delta = float(sentiment - mean)
                square = float(delta * delta)
                variance_numerator = float(variance_numerator + float(relevance * square))
            result["positive_share_24h"] = float(positive / denominator) if denominator else 0.0
            result["negative_share_24h"] = float(negative / denominator) if denominator else 0.0
            result["sentiment_dispersion_24h"] = (
                math.sqrt(float(variance_numerator / denominator)) if denominator else 0.0
            ) or 0.0
            result["source_count_24h"] = len({item[2] for item in within})
            result["news_missing_24h"] = int(not within)
    result["hours_since_latest_article"] = (
        min(24.0, float((decision - max(item[0] for item in selected)).total_seconds() / 3600))
        if selected
        else 24.0
    )
    return {name: result[name] for name in FEATURE_NAMES}


def assert_exact_features(actual: Any, expected: dict) -> None:
    values = actual.to_dict()
    assert set(values) == set(FEATURE_NAMES)
    for name, value in values.items():
        if name in INTEGER_FEATURES:
            assert type(value) is int
            assert value == expected[name]
        else:
            assert type(value) is float
            assert math.isfinite(value)
            assert struct.pack(">d", value) == struct.pack(">d", expected[name]), name
    assert canonicalize(values) == canonicalize(expected)


def pure_rows(prepared: Any, *decisions: datetime) -> tuple:
    return aggregation_module._aggregate_rows(prepared, tuple(instant(t) for t in decisions))


def scene(corpus: FixtureCorpus, points: list[tuple[float, float, float]]) -> Any:
    """Trusted internal arithmetic fixture; public tests retain actual verified parents."""
    seed_article = corpus.prepared.articles[0]
    seed_group = corpus.prepared.groups[0]
    seed_score = corpus.prepared.scores[0]
    articles, groups, scores = [], [], []
    for index, (age_hours, sentiment, relevance) in enumerate(points):
        identity = sha256_bytes(f"synthetic scene {index}".encode())
        arrival = instant(DECISION - timedelta(hours=age_hours))
        article = replace(
            seed_article,
            article_id=identity,
            article_version_id=identity,
            duplicate_group_id=identity,
            content_hash=identity,
            source=f"source-{index % 3}.invalid",
            first_seen_at=arrival,
            asset="BTC",
            point_in_time_eligible=True,
        )
        articles.append(article)
        groups.append(
            replace(
                seed_group,
                duplicate_group_id=identity,
                anchor_article_id=identity,
                initial_first_seen_at=arrival,
                source=article.source,
            )
        )
        scores.append(
            replace(
                seed_score,
                article_version_id=identity,
                content_hash=identity,
                state="succeeded",
                payload=ScorePayload(float(sentiment), float(relevance)),
            )
        )
    return replace(
        corpus.prepared, articles=tuple(articles), groups=tuple(groups), scores=tuple(scores)
    )


def test_public_real_parser_cas_and_mock_scores_integrate(corpus: FixtureCorpus) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(
        corpus.populated, (instant(DECISION), instant(DECISION + timedelta(hours=1)))
    )
    assert len(artifact.rows) == 2
    for row in artifact.rows:
        assert row.exclusion_reason is None
        assert row.features is not None
        assert_exact_features(
            row.features, reference_features(corpus.prepared, timestamp(row.decision_at))
        )
    # A syndicated non-anchor source cannot inflate count or source count.
    assert artifact.rows[0].features.news_count_24h == 6
    assert artifact.rows[0].features.source_count_24h == 6
    assert all(score.scored_at.startswith("2099-") for score in corpus.prepared.scores)
    assert artifact.rows[0].features.sentiment_mean_24h != 0.0


def test_delivered_empty_intervals_are_no_news_not_provider_gap(corpus: FixtureCorpus) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(corpus.empty, (instant(DECISION),))
    row = artifact.rows[0]
    expected = {name: 0 if name in INTEGER_FEATURES else 0.0 for name in FEATURE_NAMES}
    expected.update(hours_since_latest_article=24.0, news_missing_24h=1)
    assert row.exclusion_reason is None
    assert_exact_features(row.features, expected)
    assert row.diagnostics.to_dict()["provider_gap_indicator"] == 0


def test_verified_gap_excludes_without_fabricated_zero_row(corpus: FixtureCorpus) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(corpus.gap, (instant(DECISION),))
    row = artifact.rows[0]
    assert row.exclusion_reason == "provider_gap_window"
    assert row.features is None
    assert row.diagnostics.to_dict()["provider_gap_indicator"] == 1


def test_hand_calculated_all_thirteen_features(corpus: FixtureCorpus) -> None:
    prepared = scene(corpus, [(0.0, 1.0, 1.0), (0.0, -1.0, 1.0)])
    features = pure_rows(prepared, DECISION)[0].features
    expected = {
        "sentiment_mean_6h": 0.0,
        "sentiment_mean_24h": 0.0,
        "news_count_1h": 2,
        "news_count_6h": 2,
        "news_count_24h": 2,
        "sentiment_recency_6h": 0.0,
        "sentiment_recency_24h": 0.0,
        "positive_share_24h": 0.5,
        "negative_share_24h": 0.5,
        "sentiment_dispersion_24h": 1.0,
        "source_count_24h": 2,
        "hours_since_latest_article": 0.0,
        "news_missing_24h": 0,
    }
    assert_exact_features(features, expected)


def test_hand_calculated_relevance_weighted_population_dispersion_and_decay(
    corpus: FixtureCorpus,
) -> None:
    prepared = scene(corpus, [(0.0, 1.0, 0.75), (3.0, -1.0, 0.25)])
    features = pure_rows(prepared, DECISION)[0].features
    assert features.sentiment_mean_6h == 0.5
    assert features.sentiment_mean_24h == 0.5
    assert features.positive_share_24h == 0.75
    assert features.negative_share_24h == 0.25
    assert features.sentiment_dispersion_24h == math.sqrt(0.75)
    for hours in (6, 24):
        older_weight = 0.25 * math.pow(2.0, -3.0 / hours)
        expected = (0.75 - older_weight) / (0.75 + older_weight)
        assert struct.pack(">d", features.to_dict()[f"sentiment_recency_{hours}h"]) == struct.pack(
            ">d", expected
        )


def test_score_failure_diagnostics_reconcile_without_changing_news_counts(
    corpus: FixtureCorpus,
) -> None:
    prepared = scene(corpus, [(1.0, 1.0, 1.0), (10.0, -1.0, 1.0), (2.0, 0.5, 0.1)])
    prepared = replace(
        prepared,
        scores=(
            prepared.scores[0],
            replace(prepared.scores[1], state="permanent_error", payload=None),
            prepared.scores[2],
        ),
    )
    row = pure_rows(prepared, DECISION)[0]
    assert row.features.news_count_24h == 3
    assert row.features.news_count_6h == 2
    assert row.features.news_missing_24h == 0
    assert row.features.sentiment_mean_24h == 1.0
    assert row.diagnostics.to_dict() == {
        "scoring_failure_count_6h": 0,
        "scoring_failure_count_24h": 1,
        "scoring_failure_rate_6h": 0.0,
        "scoring_failure_rate_24h": 1.0 / 3.0,
        "provider_gap_indicator": 0,
        "low_relevance_count": 1,
    }


@pytest.mark.parametrize("window", [1, 6, 24])
@pytest.mark.parametrize("offset_us,expected", [(0, 0), (1, 1), (-1, 0)])
def test_open_left_exact_microsecond_boundaries(
    corpus: FixtureCorpus, window: int, offset_us: int, expected: int
) -> None:
    prepared = scene(corpus, [(float(window), 0.5, 1.0)])
    changed = instant(DECISION - timedelta(hours=window) + timedelta(microseconds=offset_us))
    prepared = replace(
        prepared,
        articles=(replace(prepared.articles[0], first_seen_at=changed),),
        groups=(replace(prepared.groups[0], initial_first_seen_at=changed),),
    )
    row = pure_rows(prepared, DECISION)[0]
    assert row.features.to_dict()[f"news_count_{window}h"] == expected
    assert_exact_features(row.features, reference_features(prepared, DECISION))


@pytest.mark.parametrize("offset_us,expected", [(-1, 1), (0, 1), (1, 0)])
def test_closed_right_exact_microsecond_boundaries(
    corpus: FixtureCorpus, offset_us: int, expected: int
) -> None:
    prepared = scene(corpus, [(0.0, 1.0, 1.0)])
    changed = instant(DECISION + timedelta(microseconds=offset_us))
    prepared = replace(
        prepared,
        articles=(replace(prepared.articles[0], first_seen_at=changed),),
        groups=(replace(prepared.groups[0], initial_first_seen_at=changed),),
    )
    row = pure_rows(prepared, DECISION)[0]
    assert row.features.news_count_24h == expected
    assert_exact_features(row.features, reference_features(prepared, DECISION))


@pytest.mark.parametrize(
    "relevance", [0.0, math.nextafter(0.2, 0.0), 0.2, math.nextafter(0.2, 1.0), 1.0]
)
def test_relevance_floor_keeps_counts_and_missingness(
    corpus: FixtureCorpus, relevance: float
) -> None:
    prepared = scene(corpus, [(1.0, 1.0, relevance)])
    row = pure_rows(prepared, DECISION)[0]
    assert row.features.news_count_24h == 1
    assert row.features.news_missing_24h == 0
    assert row.features.sentiment_mean_24h == (1.0 if relevance >= 0.2 else 0.0)
    assert row.diagnostics.to_dict()["low_relevance_count"] == int(relevance < 0.2)
    assert_exact_features(row.features, reference_features(prepared, DECISION))


@pytest.mark.parametrize(
    "sentiment", [-1.0, math.nextafter(-0.2, -1.0), -0.2, 0.0, 0.2, math.nextafter(0.2, 1.0), 1.0]
)
def test_strict_positive_negative_share_cutoffs(corpus: FixtureCorpus, sentiment: float) -> None:
    prepared = scene(corpus, [(1.0, sentiment, 0.2)])
    features = pure_rows(prepared, DECISION)[0].features
    assert features.positive_share_24h == float(sentiment > 0.2)
    assert features.negative_share_24h == float(sentiment < -0.2)
    assert_exact_features(features, reference_features(prepared, DECISION))


@pytest.mark.parametrize("seed", range(12))
def test_independent_reference_randomized_zero_ulp(corpus: FixtureCorpus, seed: int) -> None:
    rng = random.Random(seed)
    points = [
        (
            rng.choice([0.0, 1.0, 6.0, 24.0, 24.5, -1.0, rng.uniform(0, 24)]),
            rng.choice([-1.0, 1.0, -0.2, 0.2, 5e-324, -5e-324, rng.uniform(-1, 1)]),
            rng.choice([0.0, 0.2, 0.25, 1.0, rng.uniform(0, 1)]),
        )
        for _ in range(30)
    ]
    prepared = scene(corpus, points)
    for decision in (DECISION, DECISION + timedelta(hours=1), DECISION + timedelta(hours=2)):
        assert_exact_features(
            pure_rows(prepared, decision)[0].features, reference_features(prepared, decision)
        )
    shuffled = replace(
        prepared,
        articles=tuple(reversed(prepared.articles)),
        groups=tuple(reversed(prepared.groups)),
        scores=tuple(reversed(prepared.scores)),
    )
    assert canonicalize(pure_rows(shuffled, DECISION)[0].to_dict()) == canonicalize(
        pure_rows(prepared, DECISION)[0].to_dict()
    )


@pytest.mark.parametrize(
    "state",
    [
        "pending",
        "invalid_output",
        "permanent_error",
        "transient_exhausted",
        "missing",
        "low_relevance",
    ],
)
def test_latest_revision_never_resurrects_old_success(corpus: FixtureCorpus, state: str) -> None:
    prepared = scene(corpus, [(2.0, 1.0, 1.0)])
    old = prepared.articles[0]
    revised_hash = sha256_bytes(b"synthetic newer revision")
    revised = replace(
        old,
        article_version_id=revised_hash,
        content_hash=revised_hash,
        first_seen_at=instant(DECISION),
    )
    scores = prepared.scores
    if state != "missing":
        scores += (
            replace(
                scores[0],
                content_hash=revised_hash,
                article_version_id=revised_hash,
                state="succeeded" if state == "low_relevance" else state,
                payload=ScorePayload(-1.0, 0.1) if state == "low_relevance" else None,
            ),
        )
    revised_input = replace(prepared, articles=(old, revised), scores=scores)
    before, at = pure_rows(revised_input, DECISION - timedelta(hours=1), DECISION)
    assert before.features.sentiment_mean_24h == 1.0
    assert at.features.sentiment_mean_24h == 0.0
    assert at.features.news_count_24h == 1
    assert at.features.hours_since_latest_article == 2.0
    assert at.features.news_missing_24h == 0
    assert_exact_features(at.features, reference_features(revised_input, DECISION))


def test_future_add_remove_modify_and_revision_perturbations_are_byte_identical(
    corpus: FixtureCorpus,
) -> None:
    baseline = scene(corpus, [(4.0, -0.5, 0.75), (2.0, 0.75, 0.5)])
    decisions = (DECISION - timedelta(hours=1), DECISION)
    future = scene(corpus, [(4.0, -0.5, 0.75), (2.0, 0.75, 0.5), (-1.0, -1.0, 1.0)])
    # A fixed score corpus may legitimately include unused future input scores.
    baseline = replace(baseline, scores=future.scores)
    expected = tuple(canonicalize(row.to_dict()) for row in pure_rows(baseline, *decisions))
    anchor = baseline.articles[0]
    revised = replace(
        anchor,
        article_version_id=sha256_bytes(b"future revision"),
        content_hash=sha256_bytes(b"future revision content"),
        first_seen_at=instant(DECISION + timedelta(microseconds=1)),
    )
    future = replace(future, articles=(*future.articles, revised))
    for candidate in (
        future,
        replace(
            future,
            articles=(
                *future.articles[:-1],
                replace(revised, title="Bitcoin extreme future title"),
            ),
        ),
    ):
        assert candidate.terminal_intervals == baseline.terminal_intervals
        assert candidate.scores == baseline.scores
        assert (
            tuple(canonicalize(row.to_dict()) for row in pure_rows(candidate, *decisions))
            == expected
        )


def test_non_btc_and_ineligible_records_never_enter_features(corpus: FixtureCorpus) -> None:
    prepared = scene(corpus, [(1.0, 1.0, 1.0), (2.0, -1.0, 1.0)])
    prepared = replace(
        prepared,
        articles=(
            replace(prepared.articles[0], asset="ETH"),
            replace(prepared.articles[1], point_in_time_eligible=False),
        ),
        groups=(),
    )
    features = pure_rows(prepared, DECISION)[0].features
    assert features.news_count_24h == 0
    assert features.news_missing_24h == 1


def test_group_without_eligible_anchor_fails_closed(corpus: FixtureCorpus) -> None:
    prepared = scene(corpus, [(1.0, 1.0, 1.0)])
    prepared = replace(prepared, articles=(replace(prepared.articles[0], asset="ETH"),))
    with pytest.raises(AggregationIntegrityError):
        pure_rows(prepared, DECISION)


@pytest.mark.parametrize(
    "gap_index,excluded", [(59, False), (60, True), (1500, True), (1501, False)]
)
def test_gap_interval_exact_overlap_endpoints(
    corpus: FixtureCorpus, gap_index: int, excluded: bool
) -> None:
    prepared = scene(corpus, [])
    # This isolated overlap test starts with the fully verified contiguous ledger;
    # public evidence binding itself is exercised by the separate gap publication.
    intervals = list(prepared.terminal_intervals)
    intervals[gap_index] = replace(intervals[gap_index], outcome="provider_gap")
    prepared = replace(prepared, terminal_intervals=tuple(intervals))
    row = pure_rows(prepared, DECISION)[0]
    assert (row.exclusion_reason == "provider_gap_window") is excluded
    assert (row.features is None) is excluded
    assert row.diagnostics.provider_gap_indicator == int(excluded)


def test_permanent_source_and_arrival_survive_member_and_revision_changes(
    corpus: FixtureCorpus,
) -> None:
    prepared = scene(corpus, [(2.0, -0.5, 1.0)])
    anchor = prepared.articles[0]
    member = replace(
        anchor,
        article_id=sha256_bytes(b"later syndicated member"),
        article_version_id=sha256_bytes(b"later syndicated version"),
        first_seen_at=instant(DECISION - timedelta(hours=1)),
        source="must-not-count.invalid",
    )
    revision = replace(
        anchor,
        article_version_id=sha256_bytes(b"later anchor version"),
        first_seen_at=instant(DECISION),
        source="must-not-replace-anchor.invalid",
    )
    changed = replace(prepared, articles=(anchor, member, revision))
    row = pure_rows(changed, DECISION)[0]
    assert row.features.news_count_24h == 1
    assert row.features.news_count_1h == 0
    assert row.features.source_count_24h == 1
    assert row.features.hours_since_latest_article == 2.0
    assert_exact_features(row.features, reference_features(changed, DECISION))


@pytest.mark.parametrize("age", [24.0, 24.5, 1000.0])
def test_expired_groups_and_revisions_cannot_reset_capped_age(
    corpus: FixtureCorpus, age: float
) -> None:
    prepared = scene(corpus, [(age, 1.0, 1.0)])
    revised = replace(
        prepared.articles[0],
        first_seen_at=instant(DECISION),
        article_version_id=sha256_bytes(b"expired anchor revision"),
    )
    prepared = replace(prepared, articles=(*prepared.articles, revised))
    features = pure_rows(prepared, DECISION)[0].features
    assert features.news_count_24h == 0
    assert features.hours_since_latest_article == 24.0
    assert features.news_missing_24h == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("sentiment_mean_6h", value)
        for value in (
            True,
            0,
            "0",
            None,
            float("nan"),
            float("inf"),
            float("-inf"),
            -1.01,
            1.01,
            -0.0,
        )
    ]
    + [
        ("positive_share_24h", -0.01),
        ("negative_share_24h", 1.01),
        ("sentiment_dispersion_24h", 1.01),
        ("hours_since_latest_article", 24.01),
        ("hours_since_latest_article", -0.01),
        ("news_count_1h", True),
        ("news_count_24h", 0.0),
        ("news_count_24h", -1),
        ("news_count_24h", 2**63),
        ("news_missing_24h", 2),
    ],
)
def test_feature_schema_rejects_nonfinite_wrong_types_and_bounds(field: str, value: Any) -> None:
    values = {name: 0 if name in INTEGER_FEATURES else 0.0 for name in FEATURE_NAMES}
    values.update(hours_since_latest_article=24.0, news_missing_24h=1)
    values[field] = value
    with pytest.raises(CryptoAIError):
        FeatureValues(**values).to_dict()


@pytest.mark.parametrize(
    "error",
    [
        AggregationAuthorizationError,
        AggregationCoverageError,
        AggregationInputError,
        AggregationIntegrityError,
    ],
)
def test_all_aggregation_exceptions_are_project_specific(error: type[Exception]) -> None:
    assert issubclass(error, CryptoAIError)


@pytest.mark.parametrize("value", [None, {}, "state", 1, True])
def test_public_engine_rejects_non_synthetic_input(corpus: FixtureCorpus, value: Any) -> None:
    with pytest.raises(CryptoAIError):
        OfflineFeatureAggregator(corpus.store).aggregate(value, (instant(DECISION),))


@pytest.mark.parametrize(
    "decisions",
    [
        (),
        [],
        ("2026-09-02",),
        ("2026-09-02T01:01:00Z",),
        ("2026-09-02T01:00:00",),
        ("2026-09-02T01:00:00+01:00",),
        ("not-time",),
        (1,),
        (True,),
        (instant(DECISION), instant(DECISION)),
        (instant(DECISION + timedelta(hours=1)), instant(DECISION)),
    ],
)
def test_malformed_or_non_hourly_decisions_fail_closed(
    corpus: FixtureCorpus, decisions: Any
) -> None:
    with pytest.raises(CryptoAIError):
        OfflineFeatureAggregator(corpus.store).aggregate(corpus.populated, decisions)


@pytest.mark.parametrize(
    "decision",
    [
        START,
        START + timedelta(hours=23),
        START + timedelta(hours=28),
        datetime(2099, 1, 1, tzinfo=UTC),
    ],
)
def test_incomplete_or_out_of_ledger_coverage_cannot_certify_no_news(
    corpus: FixtureCorpus, decision: datetime
) -> None:
    with pytest.raises(CryptoAIError):
        OfflineFeatureAggregator(corpus.store).aggregate(corpus.empty, (instant(decision),))


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_config_sha256", "0" * 64),
        ("protocol_config_sha256", "xyz"),
        ("coverage_as_of", instant(START)),
        ("coverage_as_of", "2099-13-01T00:00:00Z"),
        ("state_publication_id", "../escape"),
        ("state_publication_id", "missing"),
    ],
)
def test_forged_or_malformed_input_lineage_fails_closed(
    corpus: FixtureCorpus, field: str, value: Any
) -> None:
    with pytest.raises(CryptoAIError):
        request = replace(corpus.populated, **{field: value})
        OfflineFeatureAggregator(corpus.store).aggregate(request, (instant(DECISION),))


def test_score_artifact_corruption_and_duplicate_input_fail_closed(corpus: FixtureCorpus) -> None:
    first = corpus.populated.score_artifacts[0]
    files = dict(first.files)
    files["record.json"] += b" "
    bad = ScoreArtifact(tuple(sorted(files.items())))
    for scores in ((bad,), (*corpus.populated.score_artifacts, first)):
        with pytest.raises(CryptoAIError):
            request = replace(corpus.populated, score_artifacts=scores)
            OfflineFeatureAggregator(corpus.store).aggregate(request, (instant(DECISION),))


def test_publication_and_reload_are_immutable_and_deterministic(corpus: FixtureCorpus) -> None:
    engine = OfflineFeatureAggregator(corpus.store)
    first = engine.aggregate(corpus.populated, (instant(DECISION),))
    repeated = engine.aggregate(corpus.populated, (instant(DECISION),))
    assert first.files == repeated.files
    assert first.aggregation_id == repeated.aggregation_id
    repository = AggregationStore(corpus.store)
    repository.publish(first)
    loaded = repository.get(first.aggregation_id)
    assert loaded is not None
    assert loaded.files == first.files
    assert canonicalize(loaded.rows[0].to_dict()) == canonicalize(first.rows[0].to_dict())
    assert_exact_features(loaded.rows[0].features, first.rows[0].features.to_dict())
    repository.publish(repeated)
    assert repository.get("0" * 64) is None


def test_forged_artifact_payload_cannot_publish(corpus: FixtureCorpus) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(corpus.empty, (instant(DECISION),))
    files = dict(artifact.files)
    files["rows.json"] = files["rows.json"].replace(b'"news_count_24h":0', b'"news_count_24h":999')
    assert files["rows.json"] != dict(artifact.files)["rows.json"]
    with pytest.raises(CryptoAIError):
        AggregationStore(corpus.store).publish(AggregationArtifact(tuple(sorted(files.items()))))


def rehash_artifact(files: dict[str, bytes]) -> AggregationArtifact:
    identity = {
        "schema_version": aggregation_module.SCHEMA,
        "files": {
            name: sha256_bytes(raw)
            for name, raw in sorted(files.items())
            if name != "envelope.json"
        },
    }
    files["envelope.json"] = canonicalize(
        {**identity, "aggregation_id": canonical_sha256(identity)}
    )
    return AggregationArtifact(tuple(sorted(files.items())))


@pytest.mark.parametrize("forgery", ["feature", "parent", "extra_field", "duplicate_key"])
def test_recomputed_hashes_do_not_bypass_semantic_or_strict_json_validation(
    corpus: FixtureCorpus, forgery: str
) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(corpus.empty, (instant(DECISION),))
    files = dict(artifact.files)
    if forgery == "feature":
        rows = json.loads(files["rows.json"])
        rows[0]["features"]["sentiment_mean_24h"] = 0.5
        files["rows.json"] = canonicalize(rows)
    elif forgery == "parent":
        lineage = json.loads(files["lineage.json"])
        lineage["state_sha256"] = "0" * 64
        files["lineage.json"] = canonicalize(lineage)
    elif forgery == "extra_field":
        inputs = json.loads(files["input.json"])
        inputs["network_authorized"] = True
        files["input.json"] = canonicalize(inputs)
    else:
        files["rows.json"] = files["rows.json"].replace(
            b'"news_count_24h":0', b'"news_count_24h":0,"news_count_24h":0'
        )
    with pytest.raises(CryptoAIError):
        AggregationStore(corpus.store).publish(rehash_artifact(files))


def test_unused_valid_mock_scores_and_shuffled_score_order_are_safe(
    corpus: FixtureCorpus,
) -> None:
    engine = OfflineFeatureAggregator(corpus.store)
    no_news = engine.aggregate(corpus.empty, (instant(DECISION),))
    with_unused_scores = engine.aggregate(
        replace(corpus.empty, score_artifacts=corpus.populated.score_artifacts),
        (instant(DECISION),),
    )
    assert canonicalize(no_news.rows[0].to_dict()) == canonicalize(
        with_unused_scores.rows[0].to_dict()
    )
    baseline = engine.aggregate(corpus.populated, (instant(DECISION),))
    shuffled = engine.aggregate(
        replace(
            corpus.populated, score_artifacts=tuple(reversed(corpus.populated.score_artifacts))
        ),
        (instant(DECISION),),
    )
    assert baseline.files == shuffled.files


def test_conflicting_valid_scores_for_same_content_fail_closed(
    corpus: FixtureCorpus, tmp_path: Path
) -> None:
    first = corpus.populated.score_artifacts[0]
    original = json.loads(dict(first.files)["input.json"])
    conflicting = OfflineScoringEngine(
        ScoringStore(tmp_path),
        MockScorer((b'{"sentiment_score":0,"relevance_score":0}',)),
        clock=lambda: START,
    ).score(SyntheticInput(source=original["source"], title=original["title"]))
    with pytest.raises(CryptoAIError):
        OfflineFeatureAggregator(corpus.store).aggregate(
            replace(
                corpus.populated, score_artifacts=(*corpus.populated.score_artifacts, conflicting)
            ),
            (instant(DECISION),),
        )


def test_altered_publication_metadata_is_not_an_idempotent_cache_hit(
    corpus: FixtureCorpus,
) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(
        corpus.empty, (instant(DECISION + timedelta(hours=2)),)
    )
    corpus.store.publish_bundle(
        "aggregation-" + artifact.aggregation_id,
        dict(artifact.files),
        metadata={
            "schema_version": aggregation_module.SCHEMA,
            "aggregation_id": artifact.aggregation_id,
            "envelope_sha256": "0" * 64,
        },
    )
    with pytest.raises(CryptoAIError):
        AggregationStore(corpus.store).publish(artifact)


@pytest.mark.parametrize("defect", ["non_string", "non_hex", "unexpected_key"])
def test_invalid_outer_metadata_opens_only_manifest_before_aborting(
    corpus: FixtureCorpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(corpus.empty, (instant(DECISION),))
    isolated = ContentAddressedStore(tmp_path)
    metadata = {
        "schema_version": aggregation_module.SCHEMA,
        "aggregation_id": artifact.aggregation_id,
        "envelope_sha256": sha256_bytes(dict(artifact.files)["envelope.json"]),
    }
    if defect == "non_string":
        metadata["envelope_sha256"] = 7
    elif defect == "non_hex":
        metadata["envelope_sha256"] = "z" * 64
    else:
        metadata["unexpected"] = "reject before payload capture"
    isolated.publish_bundle(
        "aggregation-" + artifact.aggregation_id, dict(artifact.files), metadata=metadata
    )
    real_read = storage_module._read_regular_file_at_once
    opened = []

    def record_open(parent_descriptor: int, name: str, *, description: str) -> tuple:
        opened.append(name)
        assert name == "manifest.json", "malformed metadata must abort before any payload I/O"
        return real_read(parent_descriptor, name, description=description)

    monkeypatch.setattr(storage_module, "_read_regular_file_at_once", record_open)
    with pytest.raises(CryptoAIError):
        AggregationStore(isolated).get(artifact.aggregation_id)
    assert opened == ["manifest.json"]


@pytest.mark.parametrize("kind", ["incomplete", "fifo", "symlink"])
def test_incomplete_and_unmanifested_publication_objects_fail_closed(
    corpus: FixtureCorpus, tmp_path: Path, kind: str
) -> None:
    artifact = OfflineFeatureAggregator(corpus.store).aggregate(corpus.empty, (instant(DECISION),))
    isolated = ContentAddressedStore(tmp_path)
    publication_id = "aggregation-" + artifact.aggregation_id
    if kind == "incomplete":
        for _, raw in artifact.files:
            isolated.put_bytes(raw)
        assert AggregationStore(isolated).get(artifact.aggregation_id) is None
        (isolated.publications_root / publication_id).mkdir()
    else:
        published = isolated.publish_bundle(
            publication_id,
            dict(artifact.files),
            metadata={
                "schema_version": aggregation_module.SCHEMA,
                "aggregation_id": artifact.aggregation_id,
                "envelope_sha256": sha256_bytes(dict(artifact.files)["envelope.json"]),
            },
        )
        nested = published / "unmanifested"
        nested.mkdir()
        if kind == "fifo":
            os.mkfifo(nested / "blocked")
        else:
            (nested / "escape").symlink_to(corpus.store.publications_root, target_is_directory=True)
    with pytest.raises(CryptoAIError):
        AggregationStore(isolated).get(artifact.aggregation_id)


@pytest.mark.parametrize("object_kind", ["symlink", "fifo"])
def test_storage_parent_mutation_fails_closed_without_opening_special_files(
    corpus: FixtureCorpus, tmp_path: Path, object_kind: str
) -> None:
    root = tmp_path / "corrupt"
    root.mkdir()
    if object_kind == "symlink":
        (root / "publications").symlink_to(corpus.store.publications_root, target_is_directory=True)
    else:
        os.mkfifo(root / "publications")
    with pytest.raises(CryptoAIError):
        store = ContentAddressedStore(root)
        OfflineFeatureAggregator(store).aggregate(corpus.populated, (instant(DECISION),))
