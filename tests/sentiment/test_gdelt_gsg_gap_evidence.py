"""Offline regressions for immutable terminal GSG gap evidence."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.exceptions import (
    NormalizationIntegrityError,
    ProviderIngestionError,
    SentimentStorageError,
)
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GapAttempt,
    GSGAdapter,
    GSGNormalizer,
    RightsApproval,
    TerminalGapEvidence,
    expected_gsg_source_locator,
    plan_retrieval,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

PROJECT_ROOT = Path(__file__).parents[2]
PROTOCOL_HASH = sha256_bytes((PROJECT_ROOT / "config" / "phase2_protocol.json").read_bytes())
INTEGRITY_FAILURES = (NormalizationIntegrityError, ProviderIngestionError)


def plan_for(minute: int):
    start = datetime(2026, 8, 14, 1, minute, tzinfo=UTC)
    return plan_retrieval(
        format_utc_timestamp(start),
        format_utc_timestamp(start + timedelta(minutes=1)),
    )


def attempt_at(
    minute: int,
    attempt_number: int,
    seconds_after_due: int,
    *,
    http_status: int | None = 404,
    error_kind: str | None = None,
    retry_after_seconds: float | None = None,
    disposition: str = "gap",
) -> GapAttempt:
    due = datetime(2026, 8, 14, 1, minute, tzinfo=UTC) + timedelta(minutes=30)
    return GapAttempt(
        attempt_number=attempt_number,
        attempted_at=format_utc_timestamp(due + timedelta(seconds=seconds_after_due)),
        http_status=http_status,
        error_kind=error_kind,
        retry_after_seconds=retry_after_seconds,
        retry_disposition=disposition,
    )


def nonretryable_attempt(minute: int = 0) -> GapAttempt:
    return attempt_at(minute, 1, 0)


def exhausted_attempts(minute: int = 0) -> tuple[GapAttempt, ...]:
    return (
        attempt_at(minute, 1, 0, http_status=500, disposition="retry"),
        attempt_at(minute, 2, 2, http_status=500, disposition="retry"),
        attempt_at(minute, 3, 6, http_status=500, disposition="gap"),
    )


def gap_evidence(
    minute: int = 0,
    *,
    attempts: tuple[GapAttempt, ...] | None = None,
    terminal_at: str | None = None,
    **overrides: Any,
) -> TerminalGapEvidence:
    plan = plan_for(minute)
    materialized_attempts = attempts or (nonretryable_attempt(minute),)
    values: dict[str, Any] = {
        "interval_start": plan.start_at,
        "interval_end_exclusive": plan.end_at_exclusive,
        "expected_source_locator": expected_gsg_source_locator(plan.intervals[0]),
        "attempts": materialized_attempts,
        "terminal_at": terminal_at or materialized_attempts[-1].attempted_at,
        "protocol_config_sha256": PROTOCOL_HASH,
    }
    values.update(overrides)
    return TerminalGapEvidence.create(**values)


def empty_normalizer() -> GSGNormalizer:
    return GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH)


def normalize_gap(
    normalizer: GSGNormalizer,
    minute: int,
    evidence: tuple[TerminalGapEvidence, ...] = (),
    *,
    snapshots: tuple[Any, ...] = (),
    as_of: str = "2026-08-14T03:00:00Z",
):
    return normalizer.normalize(
        snapshots,
        retrieval_plan=plan_for(minute),
        terminal_as_of=as_of,
        gap_evidence=evidence,
    )


def relation(minute: int) -> dict[str, object]:
    return {
        "fromDate": "20260814003000",
        "fromLang": "English",
        "fromTitle": f"Bitcoin synthetic gap-evidence article {minute}",
        "fromUrl": f"https://news.example/gap-evidence/{minute}",
        "similarity": 0.9,
        "toDate": "20260814003100",
        "toLang": "English",
        "toTitle": f"Bitcoin synthetic gap-evidence companion {minute}",
        "toUrl": f"https://wire.example/gap-evidence/{minute}",
    }


def gzip_records(records: list[dict[str, object]]) -> bytes:
    payload = b"\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() for record in records
    )
    return gzip.compress(payload, compresslevel=9, mtime=0)


def complete_snapshot(store: ContentAddressedStore, minute: int = 0):
    plan = plan_for(minute)
    published_at = datetime(2026, 8, 14, 1, minute, tzinfo=UTC) + timedelta(minutes=30, seconds=30)
    adapter = GSGAdapter(store, clock=lambda: published_at)
    return adapter.ingest_snapshot(
        gzip_records([relation(minute)]),
        filename_timestamp=plan.start_at,
        ingested_at=format_utc_timestamp(datetime(2026, 8, 14, 1, minute, 30, tzinfo=UTC)),
        source_locator=expected_gsg_source_locator(plan.intervals[0]),
        collection_mode="prospective",
        input_class="synthetic_fixture",
    )


def invalid_snapshot(store: ContentAddressedStore, minute: int = 0):
    plan = plan_for(minute)
    published_at = datetime(2026, 8, 14, 1, minute, tzinfo=UTC) + timedelta(minutes=30, seconds=1)
    adapter = GSGAdapter(store, clock=lambda: published_at)
    return adapter.ingest_snapshot(
        b"synthetic bytes that are not gzip",
        filename_timestamp=plan.start_at,
        ingested_at=format_utc_timestamp(datetime(2026, 8, 14, 1, minute, 30, tzinfo=UTC)),
        source_locator=expected_gsg_source_locator(plan.intervals[0]),
        collection_mode="prospective",
        input_class="synthetic_fixture",
    )


def approved_normalizer(*snapshots: Any) -> GSGNormalizer:
    approval = RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=PROTOCOL_HASH,
        raw_snapshot_sha256={item.receipt.raw_snapshot_sha256 for item in snapshots},
    )
    return GSGNormalizer(
        protocol_config_sha256=PROTOCOL_HASH,
        rights_approval=approval,
    )


def publish_modified_state(
    store: ContentAddressedStore,
    publication_id: str,
    files: dict[str, bytes],
) -> Path:
    state_index = json.loads(files["state.json"])
    state_index["files"] = {
        name: {"sha256": sha256_bytes(files[name]), "size_bytes": len(files[name])}
        for name in sorted(state_index["files"])
    }
    identity = dict(state_index)
    identity.pop("state_sha256")
    state_index["state_sha256"] = canonical_sha256(identity)
    files["state.json"] = canonicalize(state_index)
    return store.publish_bundle(
        publication_id,
        files,
        metadata={
            "protocol_config_sha256": state_index["protocol_config_sha256"],
            "provider": state_index["provider"],
            "rights_approval_sha256": state_index["rights_approval_sha256"],
            "state_sha256": state_index["state_sha256"],
        },
    )


def test_missing_snapshot_without_evidence_does_not_advance_or_mutate_state() -> None:
    normalizer = empty_normalizer()
    before = normalizer.export_state_files()

    with pytest.raises(ProviderIngestionError, match="lack verified gap evidence"):
        normalize_gap(normalizer, 0)

    assert normalizer.export_state_files() == before
    assert normalizer.next_expected_interval_start is None
    assert normalizer.closed_availability_through is None
    assert normalizer.terminal_intervals == ()
    assert normalizer.terminal_gap_evidence == ()


def test_valid_synthetic_nonretryable_404_advances_verified_gap_chronology() -> None:
    normalizer = empty_normalizer()
    evidence = gap_evidence()

    normalize_gap(normalizer, 0, (evidence,))

    assert normalizer.next_expected_interval_start == "2026-08-14T01:01:00Z"
    assert normalizer.closed_availability_through == "2026-08-14T01:30:00.000001Z"
    assert normalizer.terminal_gap_evidence == (evidence,)
    interval = normalizer.terminal_intervals[0]
    assert interval.outcome == "provider_gap"
    assert interval.snapshot_state == "missing"
    assert interval.gap_evidence_id == evidence.evidence_id
    assert interval.gap_evidence_sha256 == evidence.evidence_sha256
    assert interval.terminal_reason == "verified_terminal_gap_evidence"


def test_retryable_failure_before_exhaustion_is_rejected_transactionally() -> None:
    normalizer = empty_normalizer()
    before = normalizer.export_state_files()
    early = gap_evidence(
        attempts=(attempt_at(0, 1, 0, http_status=500, disposition="retry"),),
        final_terminal_disposition="retry_exhausted",
    )

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (early,))

    assert normalizer.export_state_files() == before


def test_retryable_failure_is_accepted_only_after_exact_three_attempt_exhaustion() -> None:
    normalizer = empty_normalizer()
    evidence = gap_evidence(
        attempts=exhausted_attempts(),
        final_terminal_disposition="retry_exhausted",
    )

    normalize_gap(normalizer, 0, (evidence,))

    assert evidence.attempt_count == 3
    assert [item.retry_disposition for item in evidence.attempts] == [
        "retry",
        "retry",
        "gap",
    ]
    assert normalizer.terminal_gap_evidence == (evidence,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider": "different_provider"},
        {"scope": "different_scope"},
        {"expected_source_locator": "https://example.invalid/different.gz"},
        {"collection_mode": "historical_backfill"},
        {"input_class": "provider_response"},
        {"network_access_authorized": True},
        {"retry_policy_version": "different-retry-policy"},
        {"protocol_config_sha256": "f" * 64},
    ],
)
def test_mismatched_provider_protocol_scope_endpoint_or_policy_is_rejected(
    overrides: dict[str, Any],
) -> None:
    normalizer = empty_normalizer()
    before = normalizer.export_state_files()
    evidence = gap_evidence(**overrides)

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (evidence,))

    assert normalizer.export_state_files() == before


def test_evidence_for_another_interval_and_reuse_are_rejected_without_mutation() -> None:
    wrong_interval_normalizer = empty_normalizer()
    minute_one_evidence = gap_evidence(1)
    before = wrong_interval_normalizer.export_state_files()
    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(wrong_interval_normalizer, 0, (minute_one_evidence,))
    assert wrong_interval_normalizer.export_state_files() == before

    normalizer = empty_normalizer()
    minute_zero_evidence = gap_evidence(0)
    normalize_gap(normalizer, 0, (minute_zero_evidence,))
    committed = normalizer.export_state_files()
    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 1, (minute_zero_evidence,))
    assert normalizer.export_state_files() == committed


def test_duplicate_evidence_identity_is_rejected_without_mutation() -> None:
    normalizer = empty_normalizer()
    evidence = gap_evidence()
    before = normalizer.export_state_files()

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (evidence, evidence))

    assert normalizer.export_state_files() == before


@pytest.mark.parametrize(
    "attempts",
    [
        (attempt_at(0, 2, 0),),
        (
            attempt_at(0, 1, 0, http_status=500, disposition="retry"),
            attempt_at(0, 2, 0, http_status=404, disposition="gap"),
        ),
        (
            attempt_at(0, 1, 0, http_status=500, disposition="retry"),
            attempt_at(0, 2, 1, http_status=404, disposition="gap"),
        ),
        (
            attempt_at(
                0,
                1,
                0,
                http_status=None,
                error_kind="unsupported_transport_error",
            ),
        ),
        (
            attempt_at(
                0,
                1,
                0,
                http_status=500,
                error_kind="network_transport_error",
                disposition="retry",
            ),
        ),
        (attempt_at(0, 1, 0, http_status=999),),
        (attempt_at(0, 1, 0, retry_after_seconds=1.0),),
        (attempt_at(0, 1, 0, http_status=404, disposition="retry"),),
    ],
)
def test_reordered_malformed_or_policy_contradictory_attempts_are_rejected(
    attempts: tuple[GapAttempt, ...],
) -> None:
    normalizer = empty_normalizer()
    before = normalizer.export_state_files()
    evidence = gap_evidence(attempts=attempts)

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (evidence,))

    assert normalizer.export_state_files() == before


@pytest.mark.parametrize(
    ("attempts", "terminal_at", "as_of"),
    [
        (
            (
                GapAttempt(
                    attempt_number=1,
                    attempted_at="2026-08-14T01:29:59Z",
                    http_status=404,
                    error_kind=None,
                    retry_after_seconds=None,
                    retry_disposition="gap",
                ),
            ),
            "2026-08-14T01:30:00Z",
            "2026-08-14T03:00:00Z",
        ),
        (
            (attempt_at(0, 1, 2),),
            "2026-08-14T01:30:01Z",
            "2026-08-14T03:00:00Z",
        ),
        (
            (attempt_at(0, 1, 10),),
            "2026-08-14T01:30:10Z",
            "2026-08-14T01:30:09Z",
        ),
        (
            (
                GapAttempt(
                    attempt_number=1,
                    attempted_at="2026-08-14T01:30:00.000000Z",
                    http_status=404,
                    error_kind=None,
                    retry_after_seconds=None,
                    retry_disposition="gap",
                ),
            ),
            "2026-08-14T01:30:00Z",
            "2026-08-14T03:00:00Z",
        ),
    ],
)
def test_future_regressing_contradictory_or_noncanonical_attempt_times_are_rejected(
    attempts: tuple[GapAttempt, ...], terminal_at: str, as_of: str
) -> None:
    normalizer = empty_normalizer()
    before = normalizer.export_state_files()
    evidence = gap_evidence(attempts=attempts, terminal_at=terminal_at)

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (evidence,), as_of=as_of)

    assert normalizer.export_state_files() == before


def test_hash_mismatched_evidence_is_rejected_before_state_mutation() -> None:
    normalizer = empty_normalizer()
    evidence = replace(gap_evidence(), evidence_sha256="0" * 64)
    before = normalizer.export_state_files()

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (evidence,))

    assert normalizer.export_state_files() == before


def test_unhashable_gap_error_kind_raises_project_error_without_mutation() -> None:
    normalizer = empty_normalizer()
    malformed_attempt = replace(
        nonretryable_attempt(),
        http_status=None,
        error_kind=[],  # type: ignore[arg-type]
    )
    evidence = gap_evidence(attempts=(malformed_attempt,))
    before = normalizer.export_state_files()

    with pytest.raises(INTEGRITY_FAILURES, match="verification|retry policy"):
        normalize_gap(normalizer, 0, (evidence,))

    assert normalizer.export_state_files() == before


def test_persist_hydrate_preserves_gap_evidence_and_all_state_bytes(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    normalizer = empty_normalizer()
    evidence = gap_evidence(
        attempts=exhausted_attempts(),
        final_terminal_disposition="retry_exhausted",
    )
    normalize_gap(normalizer, 0, (evidence,))
    before = normalizer.export_state_files()

    normalizer.publish_state(store, "gap-roundtrip")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-gap-roundtrip")

    assert hydrated.export_state_files() == before
    assert hydrated.terminal_gap_evidence == (evidence,)
    assert hydrated.terminal_gap_evidence[0].canonical_bytes() == evidence.canonical_bytes()
    assert hydrated.closed_availability_through == normalizer.closed_availability_through
    assert hydrated.next_expected_interval_start == normalizer.next_expected_interval_start


@pytest.mark.parametrize("damage", ["noncanonical", "hash_mismatch"])
def test_noncanonical_or_hash_mismatched_gap_evidence_fails_hydration(
    tmp_path: Path,
    damage: str,
) -> None:
    store = ContentAddressedStore(tmp_path)
    normalizer = empty_normalizer()
    normalize_gap(normalizer, 0, (gap_evidence(),))
    files = normalizer.export_state_files()
    payload = json.loads(files["gap-evidence.json"])
    if damage == "noncanonical":
        files["gap-evidence.json"] = json.dumps(payload, indent=2).encode()
    else:
        payload[0]["evidence_sha256"] = "0" * 64
        files["gap-evidence.json"] = canonicalize(payload)
    publication_id = f"gsg-normalizer-state-damaged-gap-{damage}"
    publish_modified_state(store, publication_id, files)

    with pytest.raises(NormalizationIntegrityError):
        GSGNormalizer.hydrate(store, publication_id)


def test_hydration_rejects_legacy_state_without_migration(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    files = empty_normalizer().export_state_files()
    state_index = json.loads(files["state.json"])
    files.pop("gap-evidence.json")
    state_index["files"].pop("gap-evidence.json")
    state_index["normalizer_version"] = "gdelt-gsg-normalizer-v2"
    state_index["schema_version"] = "gdelt-gsg-normalizer-state-v2"
    identity = dict(state_index)
    identity.pop("state_sha256")
    state_index["state_sha256"] = canonical_sha256(identity)
    files["state.json"] = canonicalize(state_index)
    publication_id = "gsg-normalizer-state-legacy-v2"
    store.publish_bundle(
        publication_id,
        files,
        metadata={
            "protocol_config_sha256": state_index["protocol_config_sha256"],
            "provider": state_index["provider"],
            "rights_approval_sha256": state_index["rights_approval_sha256"],
            "state_sha256": state_index["state_sha256"],
        },
    )

    with pytest.raises(NormalizationIntegrityError, match="inventory|version"):
        GSGNormalizer.hydrate(store, publication_id)


def test_gap_evidence_is_canonical_and_immutable_in_content_addressed_storage(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    evidence = gap_evidence(
        attempts=exhausted_attempts(),
        final_terminal_disposition="retry_exhausted",
    )
    raw = evidence.canonical_bytes()
    identity = evidence.identity_payload()

    assert raw == canonicalize(json.loads(raw))
    assert evidence.evidence_id == canonical_sha256(
        {
            "identity": identity,
            "identity_version": "gdelt-gsg-terminal-gap-identity-v1",
        }
    )
    assert evidence.evidence_sha256 == canonical_sha256(
        {**identity, "evidence_id": evidence.evidence_id}
    )
    object_sha256 = store.put_bytes(raw)
    assert object_sha256 == sha256_bytes(raw)
    assert store.get_bytes(object_sha256) == raw

    normalizer = empty_normalizer()
    normalize_gap(normalizer, 0, (evidence,))
    files = normalizer.export_state_files()
    state_index = json.loads(files["state.json"])
    assert state_index["files"]["gap-evidence.json"] == {
        "sha256": sha256_bytes(files["gap-evidence.json"]),
        "size_bytes": len(files["gap-evidence.json"]),
    }


def test_gap_validation_failure_preserves_all_state_facets_and_both_watermarks(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    snapshot = complete_snapshot(store)
    normalizer = approved_normalizer(snapshot)
    normalizer.normalize(
        [snapshot],
        retrieval_plan=plan_for(0),
        terminal_as_of="2026-08-14T03:00:00Z",
    )
    before = normalizer.export_state_files()
    before_intervals = normalizer.terminal_intervals
    before_gap_evidence = normalizer.terminal_gap_evidence
    before_filename_watermark = normalizer.next_expected_interval_start
    before_availability_watermark = normalizer.closed_availability_through
    early = gap_evidence(
        1,
        attempts=(attempt_at(1, 1, 0, http_status=500, disposition="retry"),),
        final_terminal_disposition="retry_exhausted",
    )

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 1, (early,))

    assert normalizer.export_state_files() == before
    assert normalizer.terminal_intervals == before_intervals
    assert normalizer.terminal_gap_evidence == before_gap_evidence
    assert normalizer.next_expected_interval_start == before_filename_watermark
    assert normalizer.closed_availability_through == before_availability_watermark
    for filename in (
        "articles.json",
        "groups.json",
        "exclusions.json",
        "observation-links.json",
        "chronology.json",
        "gap-evidence.json",
    ):
        assert normalizer.export_state_files()[filename] == before[filename]


@pytest.mark.parametrize("failure_point", ["gap-evidence", "state", "manifest", "rename"])
def test_gap_state_publication_failure_leaves_no_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    store = ContentAddressedStore(tmp_path)
    normalizer = empty_normalizer()
    normalize_gap(normalizer, 0, (gap_evidence(),))
    before = normalizer.export_state_files()
    original_write = storage_module._write_fsynced_at
    original_rename = storage_module._atomic_rename_directory_no_replace

    def fail_write(parent_descriptor: int, name: str, data: bytes) -> None:
        target = {
            "gap-evidence": "gap-evidence.json",
            "state": "state.json",
            "manifest": "manifest.json",
        }.get(failure_point)
        if name == target:
            raise SentimentStorageError(f"simulated {failure_point} failure")
        original_write(parent_descriptor, name, data)

    def fail_rename(parent_descriptor: int, source_name: str, destination_name: str) -> None:
        if failure_point == "rename":
            raise SentimentStorageError("simulated rename failure")
        original_rename(parent_descriptor, source_name, destination_name)

    monkeypatch.setattr(storage_module, "_write_fsynced_at", fail_write)
    monkeypatch.setattr(storage_module, "_atomic_rename_directory_no_replace", fail_rename)
    state_name = f"gap-failure-{failure_point}"

    with pytest.raises(SentimentStorageError, match="simulated"):
        normalizer.publish_state(store, state_name)

    publication_id = f"gsg-normalizer-state-{state_name}"
    assert normalizer.export_state_files() == before
    assert not (store.publications_root / publication_id).exists()
    assert not list(store.publications_root.glob(f".staging-{publication_id}-*"))


def test_verified_invalid_snapshot_evidence_is_preserved_as_a_terminal_gap(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    snapshot = invalid_snapshot(store)
    assert snapshot.state == "invalid"
    assert snapshot.error_code == "invalid_gzip"
    evidence = gap_evidence(
        attempts=(
            attempt_at(
                0,
                1,
                2,
                http_status=None,
                error_kind="invalid_gzip",
                disposition="gap",
            ),
        ),
        terminal_at="2026-08-14T01:30:02Z",
        observed_snapshot_id=snapshot.receipt.snapshot_id,
        observed_raw_snapshot_sha256=snapshot.receipt.raw_snapshot_sha256,
    )
    normalizer = empty_normalizer()

    normalize_gap(normalizer, 0, (evidence,), snapshots=(snapshot,))

    interval = normalizer.terminal_intervals[0]
    assert interval.outcome == "provider_gap"
    assert interval.snapshot_state == "invalid"
    assert interval.snapshot_id == snapshot.receipt.snapshot_id
    assert interval.raw_snapshot_sha256 == snapshot.receipt.raw_snapshot_sha256
    assert interval.terminal_reason == "invalid_gzip"
    assert normalizer.terminal_gap_evidence == (evidence,)


def test_custom_frozen_parser_bounds_round_trip_deterministically(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    plan = plan_for(0)
    adapter = GSGAdapter(
        store,
        clock=lambda: datetime(2026, 8, 14, 1, 30, 1, tzinfo=UTC),
        max_json_lines=1,
    )
    snapshot = adapter.ingest_snapshot(
        gzip_records([relation(0), relation(1)]),
        filename_timestamp=plan.start_at,
        ingested_at="2026-08-14T01:00:30Z",
        source_locator=expected_gsg_source_locator(plan.intervals[0]),
        collection_mode="prospective",
        input_class="synthetic_fixture",
    )
    assert snapshot.state == "invalid"
    assert snapshot.error_code == "resource_limit_exceeded"
    evidence = gap_evidence(
        attempts=(
            attempt_at(
                0,
                1,
                2,
                http_status=None,
                error_kind="resource_limit_exceeded",
                disposition="gap",
            ),
        ),
        terminal_at="2026-08-14T01:30:02Z",
        observed_snapshot_id=snapshot.receipt.snapshot_id,
        observed_raw_snapshot_sha256=snapshot.receipt.raw_snapshot_sha256,
    )
    normalizer = empty_normalizer()
    normalize_gap(normalizer, 0, (evidence,), snapshots=(snapshot,))
    before = normalizer.export_state_files()
    normalizer.publish_state(store, "custom-parser-policy")

    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-custom-parser-policy")

    assert hydrated.export_state_files() == before
    assert hydrated.terminal_intervals[0].terminal_reason == "resource_limit_exceeded"


@pytest.mark.parametrize("binding", ["missing", "wrong_snapshot", "wrong_raw_hash"])
def test_invalid_snapshot_evidence_must_exactly_bind_the_observed_raw_receipt(
    tmp_path: Path,
    binding: str,
) -> None:
    store = ContentAddressedStore(tmp_path)
    snapshot = invalid_snapshot(store)
    observed_snapshot_id: str | None = snapshot.receipt.snapshot_id
    observed_raw_hash: str | None = snapshot.receipt.raw_snapshot_sha256
    if binding == "missing":
        observed_snapshot_id = None
        observed_raw_hash = None
    elif binding == "wrong_snapshot":
        observed_snapshot_id = "1" * 64
    else:
        observed_raw_hash = "2" * 64
    evidence = gap_evidence(
        attempts=(
            attempt_at(
                0,
                1,
                2,
                http_status=None,
                error_kind="invalid_gzip",
                disposition="gap",
            ),
        ),
        terminal_at="2026-08-14T01:30:02Z",
        observed_snapshot_id=observed_snapshot_id,
        observed_raw_snapshot_sha256=observed_raw_hash,
    )
    normalizer = empty_normalizer()
    before = normalizer.export_state_files()

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (evidence,), snapshots=(snapshot,))

    assert normalizer.export_state_files() == before


def test_complete_snapshot_cannot_consume_gap_evidence(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    snapshot = complete_snapshot(store)
    normalizer = approved_normalizer(snapshot)
    before = normalizer.export_state_files()

    with pytest.raises(INTEGRITY_FAILURES):
        normalize_gap(normalizer, 0, (gap_evidence(),), snapshots=(snapshot,))

    assert normalizer.export_state_files() == before
