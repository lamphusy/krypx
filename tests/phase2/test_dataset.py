"""Offline synthetic integration, exact labels, and transitive prepared provenance."""

from __future__ import annotations

import csv
import gzip
import io
import json
import os
import shutil
import socket
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from crypto_ai.config import settings
from crypto_ai.costs import minimum_gross_return_for_net_edge
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.features.build import compute_features, get_expected_feature_columns
from crypto_ai.features.labels import add_labels
from crypto_ai.phase2 import dataset as dataset_module
from crypto_ai.phase2 import evaluation
from crypto_ai.phase2.dataset import (
    COMBINED_COLUMNS,
    LABEL_COLUMNS,
    SENTIMENT_COLUMNS,
    TECHNICAL_COLUMNS,
    DatasetStore,
    OfflineDatasetBuilder,
    PreparedDatasetArtifact,
    SyntheticDatasetInput,
    SyntheticMarket,
)
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.aggregation import (
    AggregationStore,
    OfflineFeatureAggregator,
    SyntheticAggregationInput,
)
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import (
    format_utc_timestamp,
    validate_article_record,
    validate_score_record,
)
from crypto_ai.sentiment.providers import gdelt_gsg
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
    ScoringStore,
    SyntheticInput,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

START = datetime(2026, 9, 1, tzinfo=UTC)
PROTOCOL_BYTES = canonicalize({"synthetic": True, "fixture": "milestone-five"})
CONTEXT_COLUMNS = (
    "market_ordinal",
    "decision_at",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
CORE_HASHES = (
    "market_snapshot_sha256",
    "article_snapshot_sha256",
    "score_snapshot_sha256",
    "sentiment_feature_sha256",
    "combined_feature_sha256",
    "labeled_dataset_sha256",
)


def instant(value: datetime) -> str:
    return format_utc_timestamp(value)


def evaluation_parent_case(
    corpus: Corpus, tmp_path: Path, *, news: str, development: str
) -> tuple[evaluation.VerifiedNewsParent, dict[str, Any]]:
    """Retain actual immutable parents and the M5 artifact's frozen fit rows."""
    copied_root = tmp_path / "isolated-verified-news-cas"
    shutil.copytree(corpus.store.root, copied_root)
    isolated = ContentAddressedStore(copied_root)
    prepared = DatasetStore(isolated).publish(corpus.artifacts[development])
    source_files = dict(corpus.artifacts[news].files)
    articles = tuple(
        validate_article_record(value)
        for value in json.loads(source_files["parents/articles/articles.json"])
    )
    score_names = sorted(
        name
        for name in source_files
        if name.startswith("parents/scores/") and name.endswith("/record.json")
    )
    scores = tuple(validate_score_record(json.loads(source_files[name])) for name in score_names)
    terminal_intervals = json.loads(source_files["parents/articles/chronology.json"])[
        "terminal_intervals"
    ]
    evidence_raw = canonicalize(
        {
            "coverage_start_at": terminal_intervals[0]["start_at"],
            "coverage_end_at_exclusive": terminal_intervals[-1]["end_at_exclusive"],
            "terminal_gap_evidence": json.loads(source_files["parents/articles/gap-evidence.json"]),
        }
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("market_ordinal", "timestamp", "open", "high", "low", "close", "volume"))
    for ordinal, row in corpus.market.frame().iterrows():
        writer.writerow(
            (ordinal, instant(row.timestamp.to_pydatetime()))
            + tuple(repr(float(row[name])) for name in ("open", "high", "low", "close", "volume"))
        )
    market_raw = buffer.getvalue().encode("utf-8")
    fit_rows = {row["market_ordinal"]: row for row in prepared.labeled.to_dict("records")}
    payload = evaluation._development_rows_payload(market_raw, corpus.market.hours - 6)
    payload["rows"] = [
        {
            **row,
            "features": [fit_rows[row["market_ordinal"]][name] for name in COMBINED_COLUMNS],
        }
        for row in payload["rows"]
        if row["market_ordinal"] in fit_rows
    ]
    rows_raw = canonicalize(payload)
    opened_at = [value.to_pydatetime() for value in corpus.market.frame().timestamp]
    first = len(opened_at) - len(corpus.decisions)
    last = len(opened_at) - 1
    parent = evaluation.VerifiedNewsParent(
        isolated.root, corpus.aggregations[news].aggregation_id, prepared.dataset_id
    )
    arguments = dict(
        evidence_raw=evidence_raw,
        articles=articles,
        scores=scores,
        opened_at=opened_at,
        first_ordinal=first,
        last_decision_ordinal=last,
        development_market_raw=market_raw,
        development_rows_raw=rows_raw,
        protocol_sha256=sha256_bytes(PROTOCOL_BYTES),
    )
    return parent, arguments


@pytest.mark.parametrize("name", ["empty", "populated", "gap"])
def test_evaluation_news_parent_accepts_matching_m4_m5_causal_history(
    corpus: Corpus, tmp_path: Path, name: str
) -> None:
    parent, arguments = evaluation_parent_case(corpus, tmp_path, news=name, development=name)
    expected = evaluation._verified_news_parent_expectations(parent, **arguments)
    first = arguments["first_ordinal"]
    assert evaluation._feature_rows_bits(expected) == evaluation._feature_rows_bits(
        {first + index: row.features for index, row in enumerate(corpus.aggregations[name].rows)}
    )


@pytest.mark.parametrize(
    ("news", "development"),
    [("populated", "empty"), ("empty", "populated"), ("gap", "empty"), ("empty", "gap")],
)
def test_evaluation_news_parent_rejects_contradictory_m4_m5_causal_history(
    corpus: Corpus, tmp_path: Path, news: str, development: str
) -> None:
    """Separate authentic publications cannot certify conflicting news/gap histories."""
    parent, arguments = evaluation_parent_case(corpus, tmp_path, news=news, development=development)
    with pytest.raises(evaluation.EvaluationIntegrityError, match="M4/M5 overlapping"):
        evaluation._verified_news_parent_expectations(parent, **arguments)


def test_evaluation_news_parent_reconciles_coverage_outside_published_m4_grid(
    corpus: Corpus, tmp_path: Path
) -> None:
    """A single published M4 decision cannot hide other contradictory covered windows."""
    parent, arguments = evaluation_parent_case(
        corpus, tmp_path, news="populated", development="empty"
    )
    isolated = ContentAddressedStore(parent.cas_root)
    _, request, _ = dataset_module._parent(isolated, parent.aggregation_id)
    last_only = OfflineFeatureAggregator(isolated).aggregate(request, corpus.decisions[-1:])
    AggregationStore(isolated).publish(last_only)
    assert len(last_only.rows) == 1
    parent = replace(parent, aggregation_id=last_only.aggregation_id)
    arguments["first_ordinal"] = arguments["last_decision_ordinal"]
    with pytest.raises(evaluation.EvaluationIntegrityError, match="M4/M5 overlapping"):
        evaluation._verified_news_parent_expectations(parent, **arguments)


def test_evaluation_news_parent_rejects_changed_score_with_identical_counts(
    corpus: Corpus, tmp_path: Path
) -> None:
    """Matching counts and missingness do not excuse different causal numeric features."""
    parent, arguments = evaluation_parent_case(
        corpus, tmp_path, news="populated", development="populated"
    )
    isolated = ContentAddressedStore(parent.cas_root)
    _, request, _ = dataset_module._parent(isolated, parent.aggregation_id)
    article = arguments["articles"][0]
    score = OfflineScoringEngine(
        ScoringStore(tmp_path / "alternate-scores"),
        MockScorer((canonicalize({"sentiment_score": 0.25, "relevance_score": 0.5}),)),
        clock=lambda: START,
    ).score(SyntheticInput(source=article.source, title=article.title))
    altered = OfflineFeatureAggregator(isolated).aggregate(
        replace(request, score_artifacts=(score,)), corpus.decisions
    )
    AggregationStore(isolated).publish(altered)
    original = corpus.aggregations["populated"].rows[0].features
    assert altered.rows[0].features.news_count_24h == original.news_count_24h
    assert altered.rows[0].features.news_missing_24h == original.news_missing_24h
    assert altered.rows[0].features.sentiment_mean_24h != original.sentiment_mean_24h
    parent = replace(parent, aggregation_id=altered.aggregation_id)
    arguments["scores"] = (score.record,)
    with pytest.raises(evaluation.EvaluationIntegrityError, match="M4/M5 overlapping"):
        evaluation._verified_news_parent_expectations(parent, **arguments)


@pytest.mark.parametrize(
    ("news", "development", "matches"),
    [
        ("populated", "empty", False),
        ("empty", "populated", False),
        ("gap", "empty", False),
        ("empty", "gap", False),
        ("populated", "populated", True),
        ("empty", "empty", True),
        ("gap", "gap", True),
        ("changed_score", "populated", False),
        ("missing_score", "populated", False),
    ],
)
def test_evaluation_news_parent_reconciles_partially_overlapping_chronology(
    corpus: Corpus, tmp_path: Path, news: str, development: str, matches: bool
) -> None:
    """Shared authentic minute facts agree even without a complete common 24h window."""
    source_name = "populated" if news in {"changed_score", "missing_score"} else news
    parent, arguments = evaluation_parent_case(
        corpus, tmp_path, news=source_name, development=development
    )
    isolated = ContentAddressedStore(parent.cas_root)
    _, source, prepared = dataset_module._parent(isolated, parent.aggregation_id)
    original = GSGNormalizer.hydrate(isolated, source.state_publication_id)
    first = datetime.fromisoformat(corpus.decisions[0].replace("Z", "+00:00"))
    partial_start = instant(first - timedelta(hours=3))
    last = first + timedelta(hours=26)
    coverage_end = instant(last + timedelta(minutes=1))
    as_of = instant(last + timedelta(hours=1))
    assert datetime.fromisoformat(corpus.decisions[-1].replace("Z", "+00:00")) < first + timedelta(
        hours=21
    )
    intervals = tuple(
        item for item in prepared.terminal_intervals if item.start_at >= partial_start
    )
    snapshots = []
    raw_cache: dict[str, bytes] = {}
    for interval in intervals:
        if interval.snapshot_id is None:
            continue
        receipt, raw = gdelt_gsg._load_verified_snapshot_receipt(
            isolated, interval.snapshot_id, raw_object_cache=raw_cache
        )
        adapter = GSGAdapter(
            isolated,
            clock=lambda receipt=receipt: datetime.fromisoformat(
                receipt.raw_published_at.replace("Z", "+00:00")
            ),
        )
        snapshots.append(
            adapter.ingest_snapshot(
                raw,
                filename_timestamp=receipt.filename_timestamp,
                ingested_at=receipt.ingested_at,
                source_locator=receipt.source_locator,
                collection_mode=receipt.collection_mode,
                input_class=receipt.input_class,
            )
        )
    # Extend only synthetic terminal coverage after Development; no market
    # outcomes are needed to certify the later, holdout-only decision grid.
    for interval in plan_retrieval(intervals[-1].end_at_exclusive, coverage_end).intervals:
        source_at = datetime.fromisoformat(interval.filename_timestamp.replace("Z", "+00:00"))
        published = source_at + timedelta(minutes=30)
        adapter = GSGAdapter(isolated, clock=lambda published=published: published)
        snapshots.append(
            adapter.ingest_snapshot(
                gzip.compress(b"", mtime=0),
                filename_timestamp=interval.filename_timestamp,
                ingested_at=instant(published),
                source_locator=(
                    "https://data.gdeltproject.org/gdeltv3/gsg/"
                    f"{source_at.strftime('%Y%m%d%H%M%S')}.gsg.json.gz"
                ),
                collection_mode="prospective",
                input_class="synthetic_fixture",
            )
        )
    state = GSGNormalizer(
        protocol_config_sha256=source.protocol_config_sha256,
        rights_approval=original.rights_approval,
    )
    state.normalize(
        snapshots,
        retrieval_plan=plan_retrieval(partial_start, coverage_end),
        terminal_as_of=as_of,
        gap_evidence=tuple(
            item for item in original.terminal_gap_evidence if item.interval_start >= partial_start
        ),
    )
    state.publish_state(isolated, "dataset-partial-" + news)
    partial_source = replace(
        source,
        state_publication_id="gsg-normalizer-state-dataset-partial-" + news,
        coverage_as_of=as_of,
    )
    if news == "changed_score":
        article = arguments["articles"][0]
        altered = OfflineScoringEngine(
            ScoringStore(tmp_path / "partial-alternate-scores"),
            MockScorer((canonicalize({"sentiment_score": 0.25, "relevance_score": 0.5}),)),
            clock=lambda: START,
        ).score(SyntheticInput(source=article.source, title=article.title))
        partial_source = replace(partial_source, score_artifacts=(altered,))
        arguments["scores"] = (altered.record,)
    elif news == "missing_score":
        partial_source = replace(partial_source, score_artifacts=())
        arguments["scores"] = ()
    partial = OfflineFeatureAggregator(isolated).aggregate(partial_source, (instant(last),))
    AggregationStore(isolated).publish(partial)
    parent = replace(parent, aggregation_id=partial.aggregation_id)
    ordinal = int((last - START).total_seconds() // 3600) - 1
    arguments["opened_at"] = [START + timedelta(hours=index) for index in range(ordinal + 1)]
    arguments["first_ordinal"] = arguments["last_decision_ordinal"] = ordinal
    arguments["evidence_raw"] = canonicalize(
        {
            "coverage_start_at": intervals[0].start_at,
            "coverage_end_at_exclusive": coverage_end,
            "terminal_gap_evidence": json.loads(state.export_state_files()["gap-evidence.json"]),
        }
    )
    # The original first Development decision has only three hours of shared
    # source coverage. The final published M4 row does not itself expose news
    # that was already over 24 hours old, so comparing published rows is insufficient.
    assert first - timedelta(hours=24) < datetime.fromisoformat(
        partial_start.replace("Z", "+00:00")
    )
    if news != "gap":
        assert partial.rows[0].features.news_missing_24h == 1
    if matches:
        result = evaluation._verified_news_parent_expectations(parent, **arguments)
        assert evaluation._feature_rows_bits(result) == evaluation._feature_rows_bits(
            {arguments["last_decision_ordinal"]: partial.rows[0].features}
        )
    else:
        with pytest.raises(evaluation.EvaluationIntegrityError, match="M4/M5 overlapping"):
            evaluation._verified_news_parent_expectations(parent, **arguments)


def test_evaluation_news_parent_replays_actual_m4_m5_publications(
    corpus: Corpus, tmp_path: Path
) -> None:
    """The preclaim verifier uses real immutable GSG/score and prepared parents."""
    parent, arguments = evaluation_parent_case(
        corpus, tmp_path, news="populated", development="populated"
    )
    evaluation._verified_news_parent_expectations(parent, **arguments)
    articles = arguments["articles"]
    scores = arguments["scores"]
    with pytest.raises(evaluation.EvaluationPreflightError, match="raw/response parents"):
        evaluation._verified_news_parent_expectations(
            parent,
            **{
                **arguments,
                "articles": (replace(articles[0], raw_snapshot_sha256="a" * 64),),
            },
        )
    with pytest.raises(evaluation.EvaluationPreflightError, match="raw/response parents"):
        evaluation._verified_news_parent_expectations(
            parent,
            **{
                **arguments,
                "scores": (replace(scores[0], raw_response_sha256="b" * 64),),
            },
        )
    forged_coverage = json.loads(arguments["evidence_raw"])
    forged_coverage["coverage_start_at"] = instant(datetime(2099, 1, 1, tzinfo=UTC))
    with pytest.raises(evaluation.EvaluationPreflightError, match="gap/coverage evidence"):
        evaluation._verified_news_parent_expectations(
            parent,
            **{**arguments, "evidence_raw": canonicalize(forged_coverage)},
        )


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dataset integration tests must remain entirely offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@dataclass(frozen=True)
class Corpus:
    store: ContentAddressedStore
    requests: dict[str, SyntheticDatasetInput]
    artifacts: dict[str, PreparedDatasetArtifact]
    aggregations: dict[str, Any]
    market: SyntheticMarket
    decisions: tuple[str, ...]
    gap_at: datetime
    raw_article_hash: str


@pytest.fixture(scope="module")
def synthetic_clean_build():
    """Mock read-only Git evidence, never bypass the clean-commit/lock validator."""
    commit = "1" * 40
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        if args in (("rev-parse", "--verify", commit + "^{commit}"), ("rev-parse", "HEAD")):
            return commit.encode() + b"\n"
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return b""
        if len(args) == 2 and args[0] == "show" and args[1].startswith(commit + ":"):
            name = args[1].split(":", 1)[1]
            if name in (
                "src/crypto_ai/phase2/dataset.py",
                "requirements-lock.txt",
                "requirements-phase2.txt",
            ):
                return (root / name).read_bytes()
        raise AssertionError(f"unexpected synthetic Git request: {args!r}")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dataset_module, "_git", git)
        yield


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory, synthetic_clean_build) -> Corpus:
    """Construct actual accepted parent contracts without accessing external data."""
    root = tmp_path_factory.mktemp("synthetic-prepared-dataset")
    store = ContentAddressedStore(root / "provider")
    market = SyntheticMarket(instant(START), 48, seed=17)
    technical = compute_features(market.frame())
    decisions = tuple(
        instant(value.to_pydatetime() + timedelta(hours=1)) for value in technical.timestamp
    )
    first = datetime.fromisoformat(decisions[0].replace("Z", "+00:00"))
    last = datetime.fromisoformat(decisions[-1].replace("Z", "+00:00"))
    coverage_start = first - timedelta(hours=24)
    plan = plan_retrieval(instant(coverage_start), instant(last + timedelta(minutes=1)))
    clock = [coverage_start]
    adapter = GSGAdapter(store, clock=lambda: clock[0])
    empty_snapshots = []
    populated_snapshots = []
    empty_raw = gzip.compress(b"", mtime=0)
    # One scored title is genuinely available before the first decision.
    article_at = first - timedelta(hours=2)
    raw_article = gzip.compress(
        canonicalize(
            {
                "fromDate": "20260901000000",
                "fromUrl": "https://synthetic.invalid/bitcoin-dataset-contract",
                "fromTitle": "Bitcoin synthetic dataset integration headline",
                "fromLang": "English",
            }
        )
        + b"\n",
        mtime=0,
    )
    for interval in plan.intervals:
        source_at = datetime.fromisoformat(interval.filename_timestamp.replace("Z", "+00:00"))
        clock[0] = source_at + timedelta(minutes=30)
        kwargs = {
            "filename_timestamp": interval.filename_timestamp,
            "ingested_at": instant(clock[0]),
            "source_locator": (
                "https://data.gdeltproject.org/gdeltv3/gsg/"
                f"{source_at.strftime('%Y%m%d%H%M%S')}.gsg.json.gz"
            ),
            "collection_mode": "prospective",
            "input_class": "synthetic_fixture",
        }
        empty = adapter.ingest_snapshot(empty_raw, **kwargs)
        empty_snapshots.append(empty)
        populated_snapshots.append(
            adapter.ingest_snapshot(raw_article, **kwargs) if clock[0] == article_at else empty
        )
    protocol_hash = sha256_bytes(PROTOCOL_BYTES)
    approval = RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=protocol_hash,
        raw_snapshot_sha256={sha256_bytes(empty_raw), sha256_bytes(raw_article)},
    )
    as_of = instant(last + timedelta(hours=1))
    gap_at = first + timedelta(hours=2)
    gap_index = int((gap_at - coverage_start).total_seconds() // 60)
    gap_interval = plan.intervals[gap_index]
    evidence = TerminalGapEvidence.create(
        interval_start=gap_interval.filename_timestamp,
        interval_end_exclusive=instant(gap_at + timedelta(minutes=1)),
        expected_source_locator=empty_snapshots[gap_index].receipt.source_locator,
        attempts=(GapAttempt(1, gap_interval.due_at, 404, None, None, "gap"),),
        terminal_at=gap_interval.due_at,
        protocol_config_sha256=protocol_hash,
    )
    states = {}
    for name, snapshots, gaps in (
        ("empty", empty_snapshots, ()),
        ("populated", populated_snapshots, ()),
        ("gap", [row for i, row in enumerate(empty_snapshots) if i != gap_index], (evidence,)),
    ):
        state = GSGNormalizer(protocol_config_sha256=protocol_hash, rights_approval=approval)
        state.normalize(snapshots, retrieval_plan=plan, terminal_as_of=as_of, gap_evidence=gaps)
        state.publish_state(store, "dataset-" + name)
        states[name] = state
    article = json.loads(states["populated"].export_state_files()["articles.json"])[0]
    score = OfflineScoringEngine(
        ScoringStore(root / "scores"),
        MockScorer((canonicalize({"sentiment_score": 0.75, "relevance_score": 0.5}),)),
        clock=lambda: START,
    ).score(SyntheticInput(source=article["source"], title=article["title"]))
    aggregates = {}
    requests = {}
    artifacts = {}
    for name in states:
        source = SyntheticAggregationInput(
            state_publication_id="gsg-normalizer-state-dataset-" + name,
            score_artifacts=(score,) if name == "populated" else (),
            coverage_as_of=as_of,
            protocol_config_sha256=protocol_hash,
        )
        aggregate = OfflineFeatureAggregator(store).aggregate(source, decisions)
        AggregationStore(store).publish(aggregate)
        aggregates[name] = aggregate
        request = SyntheticDatasetInput(
            market=market,
            aggregation_id=aggregate.aggregation_id,
            protocol_bytes=PROTOCOL_BYTES,
            code_commit="1" * 40,
            dependency_lock_bytes=(
                Path(__file__).resolve().parents[2] / "requirements-lock.txt"
            ).read_bytes(),
            phase2_dependency_lock_bytes=(
                Path(__file__).resolve().parents[2] / "requirements-phase2.txt"
            ).read_bytes(),
        )
        requests[name] = request
        artifacts[name] = OfflineDatasetBuilder(store).prepare(request)
    return Corpus(
        store,
        requests,
        artifacts,
        aggregates,
        market,
        decisions,
        gap_at,
        sha256_bytes(raw_article),
    )


def artifact_file(artifact: PreparedDatasetArtifact, digest_field: str) -> str:
    digest = artifact.manifest[digest_field]
    candidates = [name for name, raw in artifact.files if sha256_bytes(raw) == digest]
    assert len(candidates) == 1, (digest_field, candidates)
    return candidates[0]


def replace_file(
    artifact: PreparedDatasetArtifact, name: str, raw: bytes, *, rehash: bool = False
) -> PreparedDatasetArtifact:
    files = dict(artifact.files)
    old = files[name]
    files[name] = raw
    if rehash:
        manifest = json.loads(files["prepared_dataset_manifest.json"])
        manifest["files"][name] = {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        for key, value in manifest.items():
            if key.endswith("_sha256") and value == sha256_bytes(old):
                manifest[key] = sha256_bytes(raw)
        files["prepared_dataset_manifest.json"] = canonicalize(manifest)
    return PreparedDatasetArtifact(tuple(sorted(files.items())))


def assert_bitwise_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    for column in left:
        if left[column].dtype == np.dtype("float64"):
            assert left[column].to_numpy().tobytes() == right[column].to_numpy().tobytes()


def test_exact_24_13_37_feature_and_six_label_order(corpus: Corpus) -> None:
    artifact = corpus.artifacts["populated"]
    assert tuple(TECHNICAL_COLUMNS) == tuple(get_expected_feature_columns())
    assert len(TECHNICAL_COLUMNS) == 24
    assert len(SENTIMENT_COLUMNS) == 13
    assert tuple(COMBINED_COLUMNS) == tuple(TECHNICAL_COLUMNS) + tuple(SENTIMENT_COLUMNS)
    assert len(set(COMBINED_COLUMNS)) == 37
    assert tuple(LABEL_COLUMNS) == (
        "entry_timestamp",
        "exit_timestamp",
        "entry_open",
        "exit_open",
        "gross_forward_return",
        "label",
    )
    assert tuple(artifact.features.columns) == CONTEXT_COLUMNS + tuple(COMBINED_COLUMNS)
    assert tuple(artifact.labeled.columns) == (
        CONTEXT_COLUMNS + tuple(COMBINED_COLUMNS) + tuple(LABEL_COLUMNS)
    )


@pytest.mark.parametrize("name", ["empty", "populated", "gap"])
@pytest.mark.parametrize("labeled", [False, True])
def test_cells_share_exact_decisions_and_project_only_frozen_features(
    corpus: Corpus, name: str, labeled: bool
) -> None:
    artifact = corpus.artifacts[name]
    views = {cell: artifact.cell(cell, labeled=labeled) for cell in "ABCD"}
    for cell, frame in views.items():
        expected = TECHNICAL_COLUMNS if cell in "AB" else COMBINED_COLUMNS
        assert tuple(frame.columns) == tuple(expected)
        assert frame.index.name == "decision_at"
        assert frame.index.equals(views["A"].index)
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
    assert_bitwise_equal(views["A"], views["B"])
    assert_bitwise_equal(views["C"], views["D"])
    assert_bitwise_equal(views["A"], views["C"].loc[:, list(TECHNICAL_COLUMNS)])


def test_all_dtypes_are_preserved_after_canonical_integer_json_hydration(corpus: Corpus) -> None:
    artifact = corpus.artifacts["empty"]
    frame = artifact.labeled
    assert str(frame.market_ordinal.dtype) == "int64"
    for column in TECHNICAL_COLUMNS:
        assert str(frame[column].dtype) == "float64"
    for column in SENTIMENT_COLUMNS:
        expected = (
            "int8"
            if column == "news_missing_24h"
            else (
                "int64"
                if column.startswith("news_count") or column == "source_count_24h"
                else "float64"
            )
        )
        assert str(frame[column].dtype) == expected
    assert str(frame.label.dtype) == "int8"
    for column in ("timestamp", "decision_at", "entry_timestamp", "exit_timestamp"):
        assert isinstance(frame[column].dtype, pd.DatetimeTZDtype)
        assert str(frame[column].dt.tz) == "UTC"


def test_verified_no_news_preserves_zero_plus_indicator_without_fill(corpus: Corpus) -> None:
    frame = corpus.artifacts["empty"].features
    for column in SENTIMENT_COLUMNS:
        expected = (
            24 if column == "hours_since_latest_article" else int(column == "news_missing_24h")
        )
        assert (frame[column] == expected).all()
        if str(frame[column].dtype) == "float64" and expected == 0:
            assert not np.signbit(frame[column].to_numpy()).any()


def test_sentiment_rows_are_bit_preserved_not_recomputed_at_join(corpus: Corpus) -> None:
    artifact = corpus.artifacts["populated"]
    frame = artifact.features.set_index("decision_at")
    for row in corpus.aggregations["populated"].rows:
        values = row.features.to_dict()
        for column, value in values.items():
            actual = frame.loc[pd.Timestamp(row.decision_at), column]
            assert actual == value
            if type(value) is float:
                assert np.float64(actual).tobytes() == np.float64(value).tobytes()


def test_gap_is_removed_from_every_cell_without_compressing_price_context(corpus: Corpus) -> None:
    artifact = corpus.artifacts["gap"]
    all_frame = corpus.artifacts["empty"].features
    expected = all_frame.loc[all_frame.decision_at < corpus.gap_at].reset_index(drop=True)
    assert len(expected) == 2
    assert_bitwise_equal(artifact.features.reset_index(drop=True), expected)
    market = corpus.market.frame()
    for row in artifact.labeled.itertuples():
        ordinal = row.market_ordinal
        assert row.entry_open == market.open.iloc[ordinal + 1]
        assert row.exit_open == market.open.iloc[ordinal + 5]
        assert row.entry_timestamp == market.timestamp.iloc[ordinal + 1]
        assert row.exit_timestamp == market.timestamp.iloc[ordinal + 5]
        assert row.exit_timestamp > corpus.gap_at
    for cell in "ABCD":
        assert all(artifact.cell(cell).index < corpus.gap_at)


def test_labels_match_unmodified_phase1_math_and_exact_close_mapping(corpus: Corpus) -> None:
    artifact = corpus.artifacts["empty"]
    threshold = minimum_gross_return_for_net_edge(0.001, 2.0, 1.0, 5.0)
    assert threshold == 0.003105688415184993
    technical = compute_features(corpus.market.frame())
    expected = add_labels(technical, horizon=4, minimum_required_return=threshold)
    actual = artifact.labeled
    assert_bitwise_equal(
        actual.loc[:, list(LABEL_COLUMNS)].reset_index(drop=True),
        expected.loc[:, list(LABEL_COLUMNS)].reset_index(drop=True),
    )
    assert (actual.entry_timestamp == actual.decision_at).all()
    assert (actual.exit_timestamp == actual.decision_at + pd.Timedelta(hours=4)).all()
    assert (actual.decision_at == actual.timestamp + pd.Timedelta(hours=1)).all()
    assert len(artifact.features) - len(actual) == 5
    assert artifact.features.market_ordinal.iloc[-1] == corpus.market.hours - 1
    assert actual.market_ordinal.iloc[-1] == corpus.market.hours - 6


def test_synthetic_integration_exercises_both_cost_aware_label_outcomes(corpus: Corpus) -> None:
    assert set(corpus.artifacts["empty"].labeled.label) == {0, 1}


def test_technical_values_are_unchanged_phase1_features_before_gap_filter(corpus: Corpus) -> None:
    technical = compute_features(corpus.market.frame()).reset_index(drop=True)
    actual = corpus.artifacts["empty"].features
    assert_bitwise_equal(
        actual.loc[:, list(TECHNICAL_COLUMNS)].reset_index(drop=True),
        technical.loc[:, list(TECHNICAL_COLUMNS)],
    )


def test_manifest_binds_all_core_bytes_orders_locks_and_identity(corpus: Corpus) -> None:
    artifact = corpus.artifacts["populated"]
    files = dict(artifact.files)
    manifest = artifact.manifest
    assert artifact.dataset_id == sha256_bytes(files["prepared_dataset_manifest.json"])
    assert manifest["synthetic"] is True
    assert manifest["schema_version"] == "phase2-prepared-dataset-v1"
    for field in CORE_HASHES:
        name = artifact_file(artifact, field)
        assert manifest["files"][name]["sha256"] == sha256_bytes(files[name])
        assert manifest["files"][name]["size_bytes"] == len(files[name])
    assert manifest["technical_feature_columns"] == list(TECHNICAL_COLUMNS)
    assert manifest["sentiment_feature_columns"] == list(SENTIMENT_COLUMNS)
    assert manifest["combined_feature_columns"] == list(COMBINED_COLUMNS)
    assert manifest["label_columns"] == list(LABEL_COLUMNS)
    request = corpus.requests["populated"]
    assert manifest["protocol_sha256"] == sha256_bytes(request.protocol_bytes)
    assert manifest["dependency_lock_sha256"] == sha256_bytes(request.dependency_lock_bytes)
    assert manifest["phase2_dependency_lock_sha256"] == sha256_bytes(
        request.phase2_dependency_lock_bytes
    )
    assert manifest["code_commit"] == request.code_commit
    assert set(manifest["files"]) == set(files) - {"prepared_dataset_manifest.json"}


@pytest.mark.parametrize("field", CORE_HASHES)
def test_one_byte_payload_change_breaks_manifest_verification(corpus: Corpus, field: str) -> None:
    artifact = corpus.artifacts["populated"]
    name = artifact_file(artifact, field)
    changed = replace_file(artifact, name, dict(artifact.files)[name] + b" ")
    with pytest.raises(CryptoAIError):
        DatasetStore(corpus.store).publish(changed)


@pytest.mark.parametrize("field", ["combined_feature_sha256", "labeled_dataset_sha256"])
@pytest.mark.parametrize("forgery", ["value", "order", "dtype", "extra_column", "duplicate_key"])
def test_rehashed_table_forgeries_fail_semantic_replay(
    corpus: Corpus, field: str, forgery: str
) -> None:
    artifact = corpus.artifacts["empty"]
    name = artifact_file(artifact, field)
    table = json.loads(dict(artifact.files)[name])
    if forgery == "value":
        index = table["columns"].index("ema_short")
        table["rows"][0][index] += 1.0
    elif forgery == "order":
        table["rows"].reverse()
    elif forgery == "dtype":
        table["dtypes"]["news_missing_24h"] = "float64"
    elif forgery == "extra_column":
        table["columns"].append("leaked_label")
        table["dtypes"]["leaked_label"] = "int8"
        for row in table["rows"]:
            row.append(1)
    raw = canonicalize(table)
    if forgery == "duplicate_key":
        raw = raw.replace(b'{"columns":', b'{"extra":0,"extra":0,"columns":', 1)
    changed = replace_file(artifact, name, raw, rehash=True)
    with pytest.raises(CryptoAIError):
        DatasetStore(corpus.store).publish(changed)


@pytest.mark.parametrize("field", CORE_HASHES)
def test_forged_core_manifest_hashes_cannot_publish(corpus: Corpus, field: str) -> None:
    artifact = corpus.artifacts["empty"]
    manifest = artifact.manifest
    manifest[field] = "0" * 64
    changed = replace_file(artifact, "prepared_dataset_manifest.json", canonicalize(manifest))
    with pytest.raises(CryptoAIError):
        DatasetStore(corpus.store).publish(changed)


@pytest.mark.parametrize("cell", ["", "E", "a", None, 1, True])
def test_unknown_experiment_cells_fail_closed(corpus: Corpus, cell: Any) -> None:
    with pytest.raises(CryptoAIError):
        corpus.artifacts["empty"].cell(cell)


@pytest.mark.parametrize("value", [None, {}, "market.csv", 1, True])
def test_non_synthetic_inputs_cannot_reach_builder(corpus: Corpus, value: Any) -> None:
    with pytest.raises(CryptoAIError):
        OfflineDatasetBuilder(corpus.store).prepare(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("aggregation_id", "0" * 64),
        ("aggregation_id", "../outside"),
        ("aggregation_id", True),
        ("protocol_bytes", canonicalize({"synthetic": True, "fixture": "wrong"})),
        ("protocol_bytes", b'{"synthetic":true,"synthetic":true}'),
        ("protocol_bytes", b'{"synthetic":false}'),
        ("protocol_bytes", "{}"),
        ("code_commit", "xyz"),
        ("code_commit", "A" * 40),
        ("code_commit", True),
        ("dependency_lock_bytes", b""),
        ("phase2_dependency_lock_bytes", "not bytes"),
        ("market", None),
    ],
)
def test_malformed_request_provenance_fails_closed(corpus: Corpus, field: str, value: Any) -> None:
    with pytest.raises(CryptoAIError):
        OfflineDatasetBuilder(corpus.store).prepare(
            replace(corpus.requests["empty"], **{field: value})
        )


@pytest.mark.parametrize(
    "start,hours,seed",
    [
        ("2026-09-01", 48, 0),
        ("2026-09-01T00:00:00", 48, 0),
        ("2026-09-01T00:00:00+01:00", 48, 0),
        ("2026-09-01T00:01:00Z", 48, 0),
        (instant(START), True, 0),
        (instant(START), 0, 0),
        (instant(START), -1, 0),
        (instant(START), 48.0, 0),
        (instant(START), 48, True),
        (instant(START), 48, "seed"),
    ],
)
def test_synthetic_generator_rejects_invalid_grid_and_types(
    start: Any, hours: Any, seed: Any
) -> None:
    with pytest.raises(CryptoAIError):
        SyntheticMarket(start, hours, seed).snapshot_bytes()


def test_deterministic_generator_is_finite_continuous_and_extends_causally() -> None:
    first = SyntheticMarket(instant(START), 48, seed=17)
    repeated = SyntheticMarket(instant(START), 48, seed=17)
    extended = SyntheticMarket(instant(START), 52, seed=17)
    assert first.snapshot_bytes() == repeated.snapshot_bytes()
    assert_bitwise_equal(first.frame(), extended.frame().iloc[:48].reset_index(drop=True))
    frame = first.frame()
    assert np.isfinite(frame.loc[:, list(settings.RAW_COLUMNS[1:])].to_numpy()).all()
    assert (frame.timestamp.diff().dropna() == pd.Timedelta(hours=1)).all()
    assert (frame.low <= frame[["open", "close"]].min(axis=1)).all()
    assert (frame.high >= frame[["open", "close"]].max(axis=1)).all()


def test_preparation_rerun_is_byte_identical(corpus: Corpus) -> None:
    repeated = OfflineDatasetBuilder(corpus.store).prepare(corpus.requests["empty"])
    assert repeated.files == corpus.artifacts["empty"].files
    assert repeated.dataset_id == corpus.artifacts["empty"].dataset_id


def test_publication_and_verified_cache_reload_preserve_exact_buffers(corpus: Corpus) -> None:
    repository = DatasetStore(corpus.store)
    artifact = corpus.artifacts["empty"]
    published = repository.publish(artifact)
    repeated = repository.publish(artifact)
    loaded = repository.get(artifact.dataset_id)
    assert loaded is not None
    assert published.files == repeated.files == loaded.files == artifact.files
    assert_bitwise_equal(loaded.features, artifact.features)
    assert_bitwise_equal(loaded.labeled, artifact.labeled)
    assert repository.get("0" * 64) is None


def test_feature_frames_are_detached_mutable_views_not_artifact_storage(corpus: Corpus) -> None:
    artifact = corpus.artifacts["empty"]
    before = artifact.files
    view = artifact.features
    view.loc[0, "ema_short"] = -123.0
    assert artifact.files == before
    assert artifact.features.ema_short.iloc[0] != -123.0


def test_all_dataset_files_are_captured_once_then_parsed_from_those_buffers(
    corpus: Corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = DatasetStore(corpus.store)
    artifact = corpus.artifacts["empty"]
    repository.publish(artifact)
    directory = corpus.store.publications_root / ("prepared-dataset-" + artifact.dataset_id)
    identities = {}
    for path in (directory, *(entry for entry in directory.rglob("*") if entry.is_dir())):
        stat = path.stat()
        identities[(stat.st_dev, stat.st_ino)] = path.relative_to(directory)
    reads: Counter[str] = Counter()
    original = storage_module._read_regular_file_at_once

    def capture(descriptor: int, name: str, *, description: str):
        stat = os.fstat(descriptor)
        relative = identities.get((stat.st_dev, stat.st_ino))
        if relative is not None:
            reads[(relative / name).as_posix()] += 1
        return original(descriptor, name, description=description)

    monkeypatch.setattr(storage_module, "_read_regular_file_at_once", capture)
    loaded = repository.get(artifact.dataset_id)
    assert loaded is not None
    assert_bitwise_equal(loaded.features, artifact.features)
    assert_bitwise_equal(loaded.labeled, artifact.labeled)
    assert reads == Counter({name: 1 for name in ("manifest.json", *dict(artifact.files))})


@pytest.mark.parametrize("parent", ["score", "article", "aggregation"])
def test_rehashed_embedded_parent_bytes_cannot_bypass_parent_replay(
    corpus: Corpus, parent: str
) -> None:
    artifact = corpus.artifacts["populated"]
    prefix = {
        "score": "parents/scores/",
        "article": "parents/articles/",
        "aggregation": "parents/aggregation/",
    }[parent]
    name = next(name for name, _ in artifact.files if name.startswith(prefix))
    changed = replace_file(artifact, name, dict(artifact.files)[name] + b" ", rehash=True)
    with pytest.raises(CryptoAIError):
        DatasetStore(corpus.store).publish(changed)


@pytest.mark.parametrize("operation", ["prepare", "publish", "reload"])
def test_single_raw_article_cas_byte_corruption_breaks_transitive_verification(
    corpus: Corpus, operation: str
) -> None:
    repository = DatasetStore(corpus.store)
    artifact = corpus.artifacts["populated"]
    if operation == "reload":
        repository.publish(artifact)
    path = corpus.store._object_path(corpus.raw_article_hash)
    original = path.read_bytes()
    path.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
    try:
        with pytest.raises(CryptoAIError):
            if operation == "prepare":
                OfflineDatasetBuilder(corpus.store).prepare(corpus.requests["populated"])
            elif operation == "publish":
                repository.publish(artifact)
            else:
                repository.get(artifact.dataset_id)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate", "nonbytes", "unsafe_path"])
def test_exact_artifact_inventory_rejects_incomplete_or_ambiguous_payloads(
    corpus: Corpus, kind: str
) -> None:
    files = list(corpus.artifacts["empty"].files)
    if kind == "missing":
        files.pop()
    elif kind == "extra":
        files.append(("unmanifested.json", b"{}"))
    elif kind == "duplicate":
        files.append(files[0])
    elif kind == "nonbytes":
        files[0] = (files[0][0], "not bytes")
    else:
        files.append(("../outside.json", b"{}"))
    with pytest.raises(CryptoAIError):
        DatasetStore(corpus.store).publish(PreparedDatasetArtifact(tuple(files)))


@pytest.mark.parametrize(
    "kind", ["noncanonical", "duplicate_key", "legacy", "extra", "boolean_count"]
)
def test_strict_prepared_manifest_schema_fails_closed(corpus: Corpus, kind: str) -> None:
    artifact = corpus.artifacts["empty"]
    manifest = artifact.manifest
    if kind == "legacy":
        manifest["schema_version"] = "phase1-prepared-dataset-v1"
    elif kind == "extra":
        manifest["live_authorized"] = True
    elif kind == "boolean_count":
        first = next(iter(manifest["row_counts"]))
        manifest["row_counts"][first] = True
    raw = canonicalize(manifest)
    if kind == "noncanonical":
        raw += b"\n"
    elif kind == "duplicate_key":
        raw = raw.replace(b"{", b'{"duplicate":0,"duplicate":0,', 1)
    with pytest.raises(CryptoAIError):
        DatasetStore(corpus.store).publish(
            replace_file(artifact, "prepared_dataset_manifest.json", raw)
        )


def test_missing_sentiment_decision_is_not_silently_inner_joined_or_zero_filled(
    corpus: Corpus,
) -> None:
    aggregate = corpus.aggregations["empty"]
    source = json.loads(dict(aggregate.files)["input.json"])
    request = SyntheticAggregationInput(
        source["state_publication_id"],
        (),
        source["coverage_as_of"],
        source["protocol_config_sha256"],
    )
    shortened = OfflineFeatureAggregator(corpus.store).aggregate(request, corpus.decisions[1:])
    AggregationStore(corpus.store).publish(shortened)
    with pytest.raises(CryptoAIError):
        OfflineDatasetBuilder(corpus.store).prepare(
            replace(corpus.requests["empty"], aggregation_id=shortened.aggregation_id)
        )


@pytest.mark.parametrize("malformation", ["non_string", "non_hex", "unexpected_key", "legacy"])
def test_invalid_outer_metadata_reads_only_manifest_before_abort(
    corpus: Corpus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformation: str
) -> None:
    artifact = corpus.artifacts["empty"]
    DatasetStore(corpus.store).publish(artifact)
    publication_id = "prepared-dataset-" + artifact.dataset_id
    original_manifest = json.loads(
        (corpus.store.publications_root / publication_id / "manifest.json").read_bytes()
    )
    metadata = original_manifest["metadata"]
    hash_key = next(key for key in metadata if key.endswith("sha256"))
    if malformation == "non_string":
        metadata[hash_key] = 7
    elif malformation == "non_hex":
        metadata[hash_key] = "z" * 64
    elif malformation == "unexpected_key":
        metadata["unexpected"] = "must abort before any parent or payload"
    else:
        metadata["schema_version"] = "legacy"
    store = ContentAddressedStore(tmp_path / "malformed-metadata")
    store.publish_bundle(publication_id, dict(artifact.files), metadata=metadata)
    opened = []
    original = storage_module._read_regular_file_at_once

    def read_manifest_only(descriptor: int, name: str, *, description: str):
        opened.append(name)
        assert name == "manifest.json", "malformed metadata must abort before payload I/O"
        return original(descriptor, name, description=description)

    monkeypatch.setattr(storage_module, "_read_regular_file_at_once", read_manifest_only)
    with pytest.raises(CryptoAIError):
        DatasetStore(store).get(artifact.dataset_id)
    assert opened == ["manifest.json"]


def test_altered_collision_metadata_is_not_an_idempotent_hit(corpus: Corpus) -> None:
    artifact = corpus.artifacts["empty"]
    repository = DatasetStore(corpus.store)
    repository.publish(artifact)
    path = (
        corpus.store.publications_root
        / ("prepared-dataset-" + artifact.dataset_id)
        / "manifest.json"
    )
    original = path.read_bytes()
    manifest = json.loads(original)
    manifest["metadata"]["unexpected"] = "altered immutable metadata"
    path.write_bytes(canonicalize(manifest))
    try:
        with pytest.raises(CryptoAIError):
            repository.publish(artifact)
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("kind", ["incomplete", "fifo", "symlink"])
def test_incomplete_and_nonregular_publications_cannot_be_cache_hits(
    corpus: Corpus, tmp_path: Path, kind: str
) -> None:
    artifact = corpus.artifacts["empty"]
    DatasetStore(corpus.store).publish(artifact)
    publication_id = "prepared-dataset-" + artifact.dataset_id
    manifest = json.loads(
        (corpus.store.publications_root / publication_id / "manifest.json").read_bytes()
    )
    store = ContentAddressedStore(tmp_path / "bad-storage")
    if kind == "incomplete":
        (store.publications_root / publication_id).mkdir()
    else:
        directory = store.publish_bundle(
            publication_id, dict(artifact.files), metadata=manifest["metadata"]
        )
        nested = directory / "parents" / "articles"
        if kind == "fifo":
            os.mkfifo(nested / "unmanifested-pipe")
        else:
            (nested / "escape").symlink_to(corpus.store.publications_root, target_is_directory=True)
    with pytest.raises(CryptoAIError):
        DatasetStore(store).get(artifact.dataset_id)


def test_staged_byte_failure_aborts_before_completion_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path / "staged-readback")
    original = dataset_module._write_fsynced_at
    writes = []

    def corrupt_payload(descriptor: int, name: str, raw: bytes):
        writes.append(name)
        original(descriptor, name, raw + b" " if name == "payload.json" else raw)

    monkeypatch.setattr(dataset_module, "_write_fsynced_at", corrupt_payload)
    with pytest.raises(CryptoAIError):
        dataset_module._publish_verified(
            store,
            "prepared-test-readback",
            {"payload.json": b"{}"},
            {"synthetic": True},
        )
    assert writes == ["payload.json"]
    assert list(store.publications_root.iterdir()) == []


def test_manifest_is_written_after_all_staged_payload_hash_readbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path / "manifest-last")
    events = []
    write = dataset_module._write_fsynced_at
    read = dataset_module._read_regular_file_at_once

    def record_write(descriptor: int, name: str, raw: bytes):
        events.append(("write", name))
        return write(descriptor, name, raw)

    def record_read(descriptor: int, name: str, *, description: str):
        events.append(("read", name))
        return read(descriptor, name, description=description)

    monkeypatch.setattr(dataset_module, "_write_fsynced_at", record_write)
    monkeypatch.setattr(dataset_module, "_read_regular_file_at_once", record_read)
    payloads = {"a.json": b"{}", "nested/b.json": b"[]"}
    dataset_module._publish_verified(store, "prepared-manifest-last", payloads, {"synthetic": True})
    marker = events.index(("write", "manifest.json"))
    for name in ("a.json", "b.json"):
        assert events.index(("write", name)) < events.index(("read", name)) < marker
    assert store.read_publication("prepared-manifest-last").files == payloads
    assert not any(path.name.startswith(".staging-") for path in store.publications_root.iterdir())


def test_atomic_publisher_cannot_replace_an_existing_directory(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "no-replace")
    dataset_module._publish_verified(store, "prepared-collision", {"payload.json": b"{}"}, {})
    with pytest.raises(CryptoAIError):
        dataset_module._publish_verified(store, "prepared-collision", {"payload.json": b"[]"}, {})
    assert store.read_publication("prepared-collision").files == {"payload.json": b"{}"}


@pytest.mark.parametrize("consumer", [OfflineDatasetBuilder, DatasetStore])
def test_subclass_or_arbitrary_store_cannot_inject_real_execution(consumer, tmp_path: Path) -> None:
    class InjectedStore(ContentAddressedStore):
        pass

    with pytest.raises(CryptoAIError):
        consumer(InjectedStore(tmp_path / "unsupported-store"))


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_cas_storage_ancestors_are_never_followed(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "bad-ancestor"
    root.mkdir()
    if kind == "symlink":
        destination = tmp_path / "elsewhere"
        destination.mkdir()
        (root / "publications").symlink_to(destination, target_is_directory=True)
    else:
        os.mkfifo(root / "publications")
    with pytest.raises(CryptoAIError):
        ContentAddressedStore(root)


def test_no_backtest_training_holdout_or_live_execution_modules_are_used() -> None:
    source = Path(dataset_module.__file__).read_text()
    for forbidden in (
        "crypto_ai.backtesting",
        "crypto_ai.modeling",
        "crypto_ai.workflow",
        "crypto_ai.data.fetch",
        "requests.get",
        "urlopen(",
        "read_csv(",
    ):
        assert forbidden not in source
