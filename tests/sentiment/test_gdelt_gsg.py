"""Fixture-only GDELT GSG adapter and normalization tests."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from crypto_ai.exceptions import ProviderIngestionError
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.providers.gdelt_gsg import (
    MAX_PLAN_INTERVALS,
    CoverageReport,
    ExcludedObservation,
    GapAttempt,
    GSGAdapter,
    GSGNormalizer,
    GSGRetryPolicy,
    NormalizationResult,
    RetrievalPlan,
    RightsApproval,
    SnapshotResult,
    TerminalGapEvidence,
    build_coverage_report,
    canonicalize_url,
    decisions_intersecting_gaps,
    plan_retrieval,
    publish_normalization,
    redact_headers,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

FIXTURES = Path(__file__).parents[1] / "fixtures" / "gdelt_gsg"
PROTOCOL_CONFIG = Path(__file__).parents[2] / "config" / "phase2_protocol.json"
PROTOCOL_CONFIG_SHA256 = sha256_bytes(PROTOCOL_CONFIG.read_bytes())


def gzip_fixture(name: str) -> bytes:
    return gzip.compress((FIXTURES / name).read_bytes(), compresslevel=9, mtime=0)


def adapter_at(tmp_path: Path, *times: datetime) -> GSGAdapter:
    clock_values = iter(times)
    return GSGAdapter(ContentAddressedStore(tmp_path), clock=lambda: next(clock_values))


def ingest(
    adapter: GSGAdapter,
    raw: bytes,
    minute: int,
    *,
    mode: str = "prospective",
    input_class: str = "synthetic_fixture",
    locator: str = "https://data.gdeltproject.org/gdeltv3/gsg/fixture.gz",
):
    return adapter.ingest_snapshot(
        raw,
        filename_timestamp=f"2026-08-14T01:{minute:02d}:00Z",
        ingested_at=f"2026-08-14T01:{minute:02d}:30Z",
        source_locator=locator,
        collection_mode=mode,
        input_class=input_class,
    )


def approved_normalizer(*snapshots) -> GSGNormalizer:
    approval = RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=PROTOCOL_CONFIG_SHA256,
        raw_snapshot_sha256={snapshot.receipt.raw_snapshot_sha256 for snapshot in snapshots},
    )
    return GSGNormalizer(
        protocol_config_sha256=PROTOCOL_CONFIG_SHA256,
        rights_approval=approval,
    )


@dataclass(frozen=True)
class NormalizedPublicationInputs:
    adapter: GSGAdapter
    normalizer: GSGNormalizer
    plan: RetrievalPlan
    snapshots: tuple[SnapshotResult, ...]
    as_of: str
    result: NormalizationResult
    coverage: CoverageReport


def normalized_publication_inputs(
    tmp_path: Path,
) -> NormalizedPublicationInputs:
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    snapshot = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z")
    coverage = build_coverage_report(plan, [snapshot], as_of="2026-08-14T01:30:00Z")
    normalizer = approved_normalizer(snapshot)
    result = normalizer.normalize(
        [snapshot], retrieval_plan=plan, terminal_as_of="2026-08-14T01:30:00Z"
    )
    return NormalizedPublicationInputs(
        adapter=adapter,
        normalizer=normalizer,
        plan=plan,
        snapshots=(snapshot,),
        as_of="2026-08-14T01:30:00Z",
        result=result,
        coverage=coverage,
    )


def publish_inputs(
    inputs: NormalizedPublicationInputs,
    batch_id: str,
    *,
    result: NormalizationResult | None = None,
    plan: RetrievalPlan | None = None,
    snapshots: tuple[SnapshotResult, ...] | None = None,
    as_of: str | None = None,
    gap_evidence: tuple[TerminalGapEvidence, ...] = (),
):
    return publish_normalization(
        inputs.adapter.store,
        batch_id,
        result if result is not None else inputs.result,
        normalizer=inputs.normalizer,
        retrieval_plan=plan if plan is not None else inputs.plan,
        snapshots=snapshots if snapshots is not None else inputs.snapshots,
        as_of=as_of if as_of is not None else inputs.as_of,
        gap_evidence=gap_evidence,
    )


def rehash_result(result: NormalizationResult) -> NormalizationResult:
    return replace(
        result,
        semantic_sha256=canonical_sha256(result.to_dict(include_hash=False)),
    )


def normalize_terminal(
    normalizer: GSGNormalizer,
    snapshots: list,
    *,
    start: str,
    end: str,
    as_of: str,
):
    return normalizer.normalize(
        snapshots,
        retrieval_plan=plan_retrieval(start, end),
        terminal_as_of=as_of,
    )


def test_retrieval_plan_is_minute_aligned_bounded_and_due_after_30_minutes() -> None:
    plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:03:00Z", maximum_intervals=3)

    assert len(plan.intervals) == 3
    assert plan.intervals[0].due_at == "2026-08-14T01:30:00Z"
    assert plan.intervals[-1].relative_path.endswith("20260814010200.gsg.json.gz")
    with pytest.raises(ProviderIngestionError, match="interval cap"):
        plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:04:00Z", maximum_intervals=3)
    with pytest.raises(ProviderIngestionError, match="exact UTC minute"):
        plan_retrieval("2026-08-14T01:00:01Z", "2026-08-14T01:02:00Z")

    forged = replace(plan, plan_id="0" * 64)
    with pytest.raises(ProviderIngestionError, match="plan identity"):
        build_coverage_report(forged, (), as_of="2026-08-14T01:00:00Z")


def test_retry_policy_is_bounded_and_honors_longer_retry_after() -> None:
    policy = GSGRetryPolicy()

    assert policy.decide(attempt=1, http_status=500).delay_seconds == 2
    assert policy.decide(attempt=2, http_status=429, retry_after_seconds=9).delay_seconds == 9
    assert policy.decide(attempt=3, http_status=599).disposition == "gap"
    assert policy.decide(attempt=1, http_status=404).disposition == "gap"
    assert policy.decide(attempt=1, http_status=200).disposition == "complete"
    with pytest.raises(ProviderIngestionError, match="finite"):
        policy.decide(attempt=1, http_status=429, retry_after_seconds=float("inf"))


def test_exact_raw_snapshot_is_content_addressed_redacted_and_idempotent(tmp_path: Path) -> None:
    raw = gzip_fixture("base.jsonl")
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    locator = "https://user:secret@data.example/file.gz?token=top-secret&part=1#private"

    first = ingest(adapter, raw, 0, locator=locator)
    second = ingest(adapter, raw, 0, locator=locator)

    assert first == second
    assert first.state == "complete" and len(first.observations) == 2
    assert first.receipt.raw_snapshot_sha256 == sha256_bytes(raw)
    assert adapter.store.get_bytes(first.receipt.raw_snapshot_sha256) == raw
    stored_receipt = (
        adapter.store.publications_root
        / f"gsg-snapshot-{first.receipt.snapshot_id}"
        / "receipt.json"
    ).read_text()
    assert "secret" not in stored_receipt and "top-secret" not in stored_receipt
    assert "REDACTED" in stored_receipt
    assert redact_headers(
        {"Authorization": "Bearer private", "X-API-Key": "private", "Accept": "gzip"}
    ) == {
        "Authorization": "REDACTED",
        "X-API-Key": "REDACTED",
        "Accept": "gzip",
    }


def test_normalization_preserves_revisions_reuses_repeat_and_deduplicates(
    tmp_path: Path,
) -> None:
    adapter = adapter_at(
        tmp_path,
        datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC),
        datetime(2026, 8, 14, 1, 1, 31, tzinfo=UTC),
    )
    first = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    revision = ingest(adapter, gzip_fixture("revision.jsonl"), 1)
    normalizer = approved_normalizer(first, revision)

    first_result = normalize_terminal(
        normalizer,
        [first],
        start="2026-08-14T01:00:00Z",
        end="2026-08-14T01:01:00Z",
        as_of="2026-08-14T01:30:00Z",
    )
    second_result = normalize_terminal(
        normalizer,
        [revision],
        start="2026-08-14T01:01:00Z",
        end="2026-08-14T01:02:00Z",
        as_of="2026-08-14T01:31:31Z",
    )

    assert len(first_result.articles) == 2
    assert len({article.duplicate_group_id for article in first_result.articles}) == 1
    first_by_host = {article.source: article for article in first_result.articles}
    second_by_title = {article.title: article for article in second_result.articles}
    changed = second_by_title["Bitcoin rises, then steadies in synthetic revision"]
    assert changed.article_id == first_by_host["news.example"].article_id
    assert changed.article_version_id != first_by_host["news.example"].article_version_id
    assert changed.duplicate_group_id == first_by_host["news.example"].duplicate_group_id
    repeated = second_by_title["Bitcoin rises after synthetic approval"]
    assert repeated.article_version_id == first_by_host["wire.example"].article_version_id
    assert any(link.reused_existing_version for link in second_result.observation_links)
    assert first_by_host["news.example"].first_seen_at == "2026-08-14T01:00:31Z"
    assert changed.first_seen_at == "2026-08-14T01:01:31Z"


def test_latest_revision_batch_with_reused_links_remains_publishable(tmp_path: Path) -> None:
    adapter = adapter_at(
        tmp_path,
        datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC),
        datetime(2026, 8, 14, 1, 1, 31, tzinfo=UTC),
    )
    first = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    revision = ingest(adapter, gzip_fixture("revision.jsonl"), 1)
    normalizer = approved_normalizer(first, revision)
    first_plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z")
    second_plan = plan_retrieval("2026-08-14T01:01:00Z", "2026-08-14T01:02:00Z")
    normalizer.normalize(
        (first,),
        retrieval_plan=first_plan,
        terminal_as_of="2026-08-14T01:30:00Z",
    )
    result = normalizer.normalize(
        (revision,),
        retrieval_plan=second_plan,
        terminal_as_of="2026-08-14T01:31:31Z",
    )

    published = publish_normalization(
        adapter.store,
        "revision-and-reuse",
        result,
        normalizer=normalizer,
        retrieval_plan=second_plan,
        snapshots=(revision,),
        as_of="2026-08-14T01:31:31Z",
    )

    assert published.name == "gsg-normalized-revision-and-reuse"
    assert any(link.reused_existing_version for link in result.observation_links)
    representative_links = [
        json.loads(line)
        for line in (published / "representative-observation-links.jsonl").read_bytes().splitlines()
    ]
    batch_observation_ids = {link.provider_observation_id for link in result.observation_links}
    assert len(representative_links) == len(result.articles)
    assert all(not link["reused_existing_version"] for link in representative_links)
    assert any(
        link["provider_observation_id"] not in batch_observation_ids
        for link in representative_links
    )
    representative_snapshots = json.loads(
        (published / "representative-snapshot-references.json").read_bytes()
    )
    assert {item["filename_timestamp"] for item in representative_snapshots} == {
        "2026-08-14T01:00:00Z",
        "2026-08-14T01:01:00Z",
    }


def test_publication_rejects_reused_link_retargeted_in_state_and_result(
    tmp_path: Path,
) -> None:
    adapter = adapter_at(
        tmp_path,
        datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC),
        datetime(2026, 8, 14, 1, 1, 31, tzinfo=UTC),
    )
    first = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    revision = ingest(adapter, gzip_fixture("revision.jsonl"), 1)
    normalizer = approved_normalizer(first, revision)
    normalizer.normalize(
        (first,),
        retrieval_plan=plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z"),
        terminal_as_of="2026-08-14T01:30:00Z",
    )
    plan = plan_retrieval("2026-08-14T01:01:00Z", "2026-08-14T01:02:00Z")
    result = normalizer.normalize(
        (revision,),
        retrieval_plan=plan,
        terminal_as_of="2026-08-14T01:31:31Z",
    )
    reused = next(link for link in result.observation_links if link.reused_existing_version)
    wrong_version = next(
        article.article_version_id
        for article in result.articles
        if article.article_version_id != reused.article_version_id
    )
    forged_link = replace(reused, article_version_id=wrong_version)
    forged_links = tuple(
        sorted(
            (
                (
                    forged_link
                    if link.provider_observation_id == reused.provider_observation_id
                    else link
                )
                for link in result.observation_links
            ),
            key=lambda link: link.provider_observation_id,
        )
    )
    normalizer._links[reused.provider_observation_id] = forged_link
    referenced_versions = {link.article_version_id for link in forged_links}
    forged = rehash_result(
        replace(
            result,
            articles=tuple(
                article
                for article in result.articles
                if article.article_version_id in referenced_versions
            ),
            observation_links=forged_links,
        )
    )

    with pytest.raises(ProviderIngestionError, match="deterministic raw-byte replay"):
        publish_normalization(
            adapter.store,
            "forged-reused-target",
            forged,
            normalizer=normalizer,
            retrieval_plan=plan,
            snapshots=(revision,),
            as_of="2026-08-14T01:31:31Z",
        )


def test_publication_rejects_eligible_reused_observation_forged_as_excluded(
    tmp_path: Path,
) -> None:
    adapter = adapter_at(
        tmp_path,
        datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC),
        datetime(2026, 8, 14, 1, 1, 31, tzinfo=UTC),
    )
    first = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    revision = ingest(adapter, gzip_fixture("revision.jsonl"), 1)
    normalizer = approved_normalizer(first, revision)
    normalizer.normalize(
        (first,),
        retrieval_plan=plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z"),
        terminal_as_of="2026-08-14T01:30:00Z",
    )
    plan = plan_retrieval("2026-08-14T01:01:00Z", "2026-08-14T01:02:00Z")
    result = normalizer.normalize(
        (revision,),
        retrieval_plan=plan,
        terminal_as_of="2026-08-14T01:31:31Z",
    )
    reused = next(link for link in result.observation_links if link.reused_existing_version)
    forged_exclusion = ExcludedObservation(
        provider_observation_id=reused.provider_observation_id,
        reason="asset_mismatch",
        diagnostic="forged eligible-observation exclusion",
        raw_snapshot_sha256=reused.raw_snapshot_sha256,
        input_class=reused.input_class,
    )
    remaining_links = tuple(
        link
        for link in result.observation_links
        if link.provider_observation_id != reused.provider_observation_id
    )
    referenced_versions = {link.article_version_id for link in remaining_links}
    forged = rehash_result(
        replace(
            result,
            articles=tuple(
                article
                for article in result.articles
                if article.article_version_id in referenced_versions
            ),
            observation_links=remaining_links,
            exclusions=tuple(
                sorted(
                    (*result.exclusions, forged_exclusion),
                    key=lambda exclusion: exclusion.provider_observation_id,
                )
            ),
        )
    )
    normalizer._links.pop(reused.provider_observation_id)
    normalizer._exclusions[reused.provider_observation_id] = forged_exclusion

    with pytest.raises(ProviderIngestionError, match="deterministic raw-byte replay"):
        publish_normalization(
            adapter.store,
            "forged-eligible-exclusion",
            forged,
            normalizer=normalizer,
            retrieval_plan=plan,
            snapshots=(revision,),
            as_of="2026-08-14T01:31:31Z",
        )


def test_historical_records_are_reported_but_never_normalized_as_eligible(tmp_path: Path) -> None:
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    historical = ingest(adapter, gzip_fixture("base.jsonl"), 0, mode="historical_backfill")

    result = normalize_terminal(
        approved_normalizer(historical),
        [historical],
        start="2026-08-14T01:00:00Z",
        end="2026-08-14T01:01:00Z",
        as_of="2026-08-14T01:30:00Z",
    )

    assert result.articles == ()
    assert len(result.exclusions) == 2
    assert {item.reason for item in result.exclusions} == {
        "historical_backfill_without_availability"
    }


def test_coverage_treats_zero_line_as_complete_and_omissions_as_unresolved(
    tmp_path: Path,
) -> None:
    adapter = adapter_at(
        tmp_path,
        datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC),
        datetime(2026, 8, 14, 1, 1, 31, tzinfo=UTC),
        datetime(2026, 8, 14, 1, 3, 31, tzinfo=UTC),
    )
    complete = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    empty = ingest(adapter, gzip.compress(b"", mtime=0), 1)
    invalid = ingest(adapter, b"not a gzip", 3)
    plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:04:00Z")

    report = build_coverage_report(plan, [invalid, empty, complete], as_of="2026-08-14T01:34:00Z")

    assert report.expected_due_intervals == 4
    assert report.complete_intervals == 2
    assert report.zero_line_intervals == 1
    assert report.gap_intervals == 0
    assert report.unresolved_intervals == 2
    assert report.retrieval_rate == 0.5
    assert report.maximum_gap_minutes == 0
    assert report.gaps == ()
    affected = decisions_intersecting_gaps(
        report,
        ["2026-08-14T01:01:00Z", "2026-08-14T02:00:00Z", "2026-08-16T02:00:00Z"],
    )
    assert affected == ()


def test_deterministic_rerun_produces_identical_semantic_bytes(tmp_path: Path) -> None:
    raw = gzip_fixture("base.jsonl")
    results = []
    for name in ("one", "two"):
        adapter = adapter_at(tmp_path / name, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
        snapshot = ingest(adapter, raw, 0)
        plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z")
        normalized = approved_normalizer(snapshot).normalize(
            [snapshot], retrieval_plan=plan, terminal_as_of="2026-08-14T01:30:00Z"
        )
        coverage = build_coverage_report(plan, [snapshot], as_of="2026-08-14T01:30:00Z")
        results.append(
            canonicalize(
                {
                    "coverage": coverage.to_dict(),
                    "normalization": normalized.to_dict(),
                    "snapshot": snapshot.to_dict(),
                }
            )
        )
    assert results[0] == results[1]


def test_normalized_batch_publication_is_exact_and_idempotent(tmp_path: Path) -> None:
    inputs = normalized_publication_inputs(tmp_path)

    first = publish_inputs(inputs, "fixture-001")
    second = publish_inputs(inputs, "fixture-001")

    assert first == second
    inputs.adapter.store.verify_publication("gsg-normalized-fixture-001")
    lines = (first / "articles.jsonl").read_bytes().splitlines()
    assert len(lines) == 2
    assert all(canonicalize(json.loads(line)) == line for line in lines)
    assert json.loads((first / "coverage.json").read_bytes()) == inputs.coverage.to_dict()
    assert json.loads((first / "retrieval-plan.json").read_bytes()) == inputs.plan.to_dict()
    snapshot_references = json.loads((first / "snapshot-references.json").read_bytes())
    assert snapshot_references == [
        {
            "error_code": snapshot.error_code,
            "filename_timestamp": snapshot.receipt.filename_timestamp,
            "json_line_count": snapshot.json_line_count,
            "raw_snapshot_sha256": snapshot.receipt.raw_snapshot_sha256,
            "snapshot_id": snapshot.receipt.snapshot_id,
            "snapshot_publication_id": f"gsg-snapshot-{snapshot.receipt.snapshot_id}",
            "snapshot_result_sha256": canonical_sha256(snapshot.to_dict()),
            "state": snapshot.state,
        }
        for snapshot in inputs.snapshots
    ]
    assert json.loads((first / "terminal-gap-evidence.json").read_bytes()) == []
    representative_links = [
        json.loads(line)
        for line in (first / "representative-observation-links.jsonl").read_bytes().splitlines()
    ]
    expected_representative_links = [
        link.to_dict()
        for link in inputs.result.observation_links
        if not link.reused_existing_version
    ]
    assert representative_links == expected_representative_links
    representative_snapshots = json.loads(
        (first / "representative-snapshot-references.json").read_bytes()
    )
    assert representative_snapshots == snapshot_references
    summary = json.loads((first / "summary.json").read_bytes())
    assert summary["representative_observation_link_count"] == len(inputs.result.articles)
    assert summary["representative_snapshot_count"] == 1
    assert summary["representative_provenance_sha256"] == canonical_sha256(
        {
            "observation_links": representative_links,
            "snapshot_references": representative_snapshots,
        }
    )


def test_normalized_publication_reconstructs_coverage_and_rejects_bad_as_of(
    tmp_path: Path,
) -> None:
    inputs = normalized_publication_inputs(tmp_path)

    forged = replace(
        inputs.coverage,
        complete_intervals=999,
        expected_due_intervals=999,
        retrieval_rate=1.0,
    )
    forged = replace(
        forged,
        semantic_sha256=canonical_sha256(forged.to_dict(include_hash=False)),
    )
    with pytest.raises(ProviderIngestionError, match="caller-supplied coverage"):
        publish_normalization(
            inputs.adapter.store,
            "counterfeit-coverage",
            inputs.result,
            forged,
            normalizer=inputs.normalizer,
            retrieval_plan=inputs.plan,
            snapshots=inputs.snapshots,
            as_of=inputs.as_of,
        )

    with pytest.raises(
        ProviderIngestionError,
        match="availability lies after|future receipt|latest committed",
    ):
        publish_inputs(inputs, "future-coverage", as_of="2026-08-14T01:00:30Z")
    with pytest.raises(ProviderIngestionError, match="timestamp"):
        publish_inputs(inputs, "malformed-as-of", as_of="not-a-timestamp")
    with pytest.raises(ProviderIngestionError, match="canonical UTC"):
        publish_inputs(
            inputs,
            "noncanonical-as-of",
            as_of="2026-08-14T01:30:00.000000Z",
        )


def test_normalized_publication_rejects_missing_committed_representative_link(
    tmp_path: Path,
) -> None:
    inputs = normalized_publication_inputs(tmp_path)
    representative_id = inputs.result.articles[0].provider_observation_id
    inputs.normalizer._links.pop(representative_id)

    with pytest.raises(
        ProviderIngestionError,
        match="first observation|representative link",
    ):
        publish_inputs(inputs, "missing-representative")


def test_normalized_publication_rejects_out_of_plan_and_over_cap_evidence(
    tmp_path: Path,
) -> None:
    inputs = normalized_publication_inputs(tmp_path)
    out_of_plan = TerminalGapEvidence.create(
        interval_start="2099-01-01T00:00:00Z",
        interval_end_exclusive="2099-01-01T00:01:00Z",
        expected_source_locator=(
            "https://data.gdeltproject.org/gdeltv3/gsg/20990101000000.gsg.json.gz"
        ),
        attempts=(
            GapAttempt(
                attempt_number=1,
                attempted_at="2099-01-01T00:30:00Z",
                http_status=404,
                error_kind=None,
                retry_after_seconds=None,
                retry_disposition="gap",
            ),
        ),
        terminal_at="2099-01-01T00:30:00Z",
        protocol_config_sha256=PROTOCOL_CONFIG_SHA256,
    )
    with pytest.raises(ProviderIngestionError, match="outside the plan"):
        publish_inputs(
            inputs,
            "future-out-of-plan",
            as_of="2099-01-01T01:00:00Z",
            gap_evidence=(out_of_plan,),
        )

    forged_over_cap = replace(
        inputs.plan,
        intervals=inputs.plan.intervals * (MAX_PLAN_INTERVALS + 1),
    )
    with pytest.raises(
        ProviderIngestionError,
        match="retrieval plan is malformed|plan identity",
    ):
        publish_inputs(inputs, "over-cap", plan=forged_over_cap)


def test_normalized_publication_persists_exact_committed_gap_evidence(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z")
    evidence = TerminalGapEvidence.create(
        interval_start="2026-08-14T01:00:00Z",
        interval_end_exclusive="2026-08-14T01:01:00Z",
        expected_source_locator=(
            "https://data.gdeltproject.org/gdeltv3/gsg/20260814010000.gsg.json.gz"
        ),
        attempts=(
            GapAttempt(
                attempt_number=1,
                attempted_at="2026-08-14T01:30:00Z",
                http_status=404,
                error_kind=None,
                retry_after_seconds=None,
                retry_disposition="gap",
            ),
        ),
        terminal_at="2026-08-14T01:30:00Z",
        protocol_config_sha256=PROTOCOL_CONFIG_SHA256,
    )
    normalizer = GSGNormalizer(protocol_config_sha256=PROTOCOL_CONFIG_SHA256)
    result = normalizer.normalize(
        (),
        retrieval_plan=plan,
        terminal_as_of="2026-08-14T01:30:00Z",
        gap_evidence=(evidence,),
    )

    published = publish_normalization(
        store,
        "committed-gap",
        result,
        normalizer=normalizer,
        retrieval_plan=plan,
        snapshots=(),
        as_of="2026-08-14T01:30:00Z",
        gap_evidence=(evidence,),
    )

    assert json.loads((published / "terminal-gap-evidence.json").read_bytes()) == [
        evidence.to_dict()
    ]
    coverage = json.loads((published / "coverage.json").read_bytes())
    assert coverage["gap_intervals"] == 1
    assert coverage["terminal_gap_evidence_sha256"] == canonical_sha256(
        [{"evidence_id": evidence.evidence_id, "evidence_sha256": evidence.evidence_sha256}]
    )


def test_normalized_publication_rejects_snapshot_not_equal_to_raw_replay(
    tmp_path: Path,
) -> None:
    inputs = normalized_publication_inputs(tmp_path)
    observation = inputs.snapshots[0].observations[0]
    forged_observation = replace(observation, raw_title="Bitcoin forged after parsing")
    forged_snapshot = replace(
        inputs.snapshots[0],
        observations=(forged_observation, *inputs.snapshots[0].observations[1:]),
    )

    with pytest.raises(ProviderIngestionError, match="immutable raw-byte replay"):
        publish_inputs(inputs, "forged-snapshot", snapshots=(forged_snapshot,))


@pytest.mark.parametrize("damage", ["linkless", "missing", "swapped", "forged"])
def test_normalized_publication_rejects_incomplete_or_forged_provenance(
    tmp_path: Path, damage: str
) -> None:
    inputs = normalized_publication_inputs(tmp_path)
    links = list(inputs.result.observation_links)
    if damage == "linkless":
        forged = replace(inputs.result, observation_links=())
    elif damage == "missing":
        forged = replace(inputs.result, observation_links=tuple(links[:-1]))
    elif damage == "swapped":
        first, second = links
        links[0] = replace(first, article_version_id=second.article_version_id)
        links[1] = replace(second, article_version_id=first.article_version_id)
        forged = replace(inputs.result, observation_links=tuple(links))
    else:
        links[0] = replace(links[0], raw_snapshot_sha256="0" * 64)
        forged = replace(inputs.result, observation_links=tuple(links))
    forged = rehash_result(forged)

    with pytest.raises(ProviderIngestionError, match="closed batch|exactly account|provenance"):
        publish_inputs(inputs, f"forged-{damage}", result=forged)


def test_normalized_publication_rejects_forged_normalization_hash(tmp_path: Path) -> None:
    inputs = normalized_publication_inputs(tmp_path)
    forged = replace(inputs.result, semantic_sha256="0" * 64)

    with pytest.raises(ProviderIngestionError, match="normalization semantic hash mismatch"):
        publish_inputs(inputs, "forged-normalization", result=forged)


def test_normalized_publication_collision_rejects_altered_manifest_metadata(
    tmp_path: Path,
) -> None:
    inputs = normalized_publication_inputs(tmp_path)
    published = publish_inputs(inputs, "metadata-collision")
    manifest_path = published / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["metadata"]["provider"] = "forged-provider"
    manifest["metadata"]["scope"] = "forged-scope"
    manifest_path.write_bytes(canonicalize(manifest))

    with pytest.raises(ProviderIngestionError, match="collision with different bytes"):
        publish_inputs(inputs, "metadata-collision")


def test_normalizer_refuses_a_nonterminal_expected_interval(tmp_path: Path) -> None:
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    snapshot = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:02:00Z")

    with pytest.raises(ProviderIngestionError, match="remain pending"):
        approved_normalizer(snapshot).normalize(
            [snapshot], retrieval_plan=plan, terminal_as_of="2026-08-14T01:30:00Z"
        )


def test_url_normalization_removes_tracking_default_port_fragment_and_dot_segments() -> None:
    assert (
        canonicalize_url("HTTPS://Example.COM:443/a/../bitcoin?utm_source=x&b=2&a=1#fragment")
        == "https://example.com/bitcoin?a=1&b=2"
    )


def test_corrupt_snapshot_error_never_contains_fixture_or_credentials(tmp_path: Path) -> None:
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    result = ingest(
        adapter,
        b"private corrupt bytes",
        0,
        locator="https://example.test/file?api_key=private-key",
    )

    assert result.state == "invalid"
    serialized = json.dumps(result.to_dict())
    assert "private-key" not in serialized and "private corrupt bytes" not in serialized
