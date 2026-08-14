"""Fixture-only GDELT GSG adapter and normalization tests."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from crypto_ai.exceptions import ProviderIngestionError
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GSGAdapter,
    GSGNormalizer,
    GSGRetryPolicy,
    build_coverage_report,
    canonicalize_url,
    decisions_intersecting_gaps,
    plan_retrieval,
    publish_normalization,
    redact_headers,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

FIXTURES = Path(__file__).parents[1] / "fixtures" / "gdelt_gsg"


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
    locator: str = "https://data.gdeltproject.org/gdeltv3/gsg/fixture.gz",
):
    return adapter.ingest_snapshot(
        raw,
        filename_timestamp=f"2026-08-14T01:{minute:02d}:00Z",
        ingested_at=f"2026-08-14T01:{minute:02d}:30Z",
        source_locator=locator,
        collection_mode=mode,
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
    normalizer = GSGNormalizer()

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


def test_historical_records_are_reported_but_never_normalized_as_eligible(tmp_path: Path) -> None:
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    historical = ingest(adapter, gzip_fixture("base.jsonl"), 0, mode="historical_backfill")

    result = normalize_terminal(
        GSGNormalizer(),
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


def test_coverage_treats_zero_line_as_complete_and_missing_or_corrupt_as_gaps(
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
    assert report.gap_intervals == 2
    assert report.retrieval_rate == 0.5
    assert report.maximum_gap_minutes == 2
    assert len(report.gaps) == 1
    affected = decisions_intersecting_gaps(
        report,
        ["2026-08-14T01:01:00Z", "2026-08-14T02:00:00Z", "2026-08-16T02:00:00Z"],
    )
    assert affected == ("2026-08-14T02:00:00Z",)


def test_deterministic_rerun_produces_identical_semantic_bytes(tmp_path: Path) -> None:
    raw = gzip_fixture("base.jsonl")
    results = []
    for name in ("one", "two"):
        adapter = adapter_at(tmp_path / name, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
        snapshot = ingest(adapter, raw, 0)
        plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z")
        normalized = GSGNormalizer().normalize(
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
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    snapshot = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:01:00Z")
    coverage = build_coverage_report(plan, [snapshot], as_of="2026-08-14T01:30:00Z")
    result = GSGNormalizer().normalize(
        [snapshot], retrieval_plan=plan, terminal_as_of="2026-08-14T01:30:00Z"
    )

    first = publish_normalization(adapter.store, "fixture-001", result, coverage)
    second = publish_normalization(adapter.store, "fixture-001", result, coverage)

    assert first == second
    adapter.store.verify_publication("gsg-normalized-fixture-001")
    lines = (first / "articles.jsonl").read_bytes().splitlines()
    assert len(lines) == 2
    assert all(canonicalize(json.loads(line)) == line for line in lines)


def test_normalizer_refuses_a_nonterminal_expected_interval(tmp_path: Path) -> None:
    adapter = adapter_at(tmp_path, datetime(2026, 8, 14, 1, 0, 31, tzinfo=UTC))
    snapshot = ingest(adapter, gzip_fixture("base.jsonl"), 0)
    plan = plan_retrieval("2026-08-14T01:00:00Z", "2026-08-14T01:02:00Z")

    with pytest.raises(ProviderIngestionError, match="remain pending"):
        GSGNormalizer().normalize(
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
