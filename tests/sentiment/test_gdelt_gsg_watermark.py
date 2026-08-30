"""Persistent global chronology regressions for the offline GSG normalizer."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.exceptions import (
    NormalizationIntegrityError,
    ProviderIngestionError,
    SentimentStorageError,
)
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
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


def relation(url: str, title: str, companion: str) -> dict[str, object]:
    return {
        "fromDate": "20260814003000",
        "fromLang": "English",
        "fromTitle": title,
        "fromUrl": url,
        "similarity": 0.9,
        "toDate": "20260814003100",
        "toLang": "English",
        "toTitle": f"Bitcoin synthetic companion {companion}",
        "toUrl": f"https://companion.example/{companion}",
    }


def gzip_records(records: list[dict[str, object]]) -> bytes:
    payload = b"\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() for record in records
    )
    return gzip.compress(payload, compresslevel=9, mtime=0)


def snapshot(
    store: ContentAddressedStore,
    minute: int,
    *,
    url: str | None = None,
    title: str | None = None,
    raw: bytes | None = None,
):
    if raw is None:
        raw = gzip_records(
            [
                relation(
                    url or f"https://news{minute}.example/story",
                    title or f"Bitcoin synthetic minute {minute}",
                    str(minute),
                )
            ]
        )
    adapter = GSGAdapter(
        store,
        clock=lambda: datetime(2026, 8, 14, 1, minute, 31, tzinfo=UTC),
    )
    return adapter.ingest_snapshot(
        raw,
        filename_timestamp=f"2026-08-14T01:{minute:02d}:00Z",
        ingested_at=f"2026-08-14T01:{minute:02d}:30Z",
        source_locator="https://data.gdeltproject.org/gdeltv3/gsg/synthetic.gz",
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


def normalize(
    instance: GSGNormalizer,
    snapshots: list[Any],
    start_minute: int,
    end_minute: int,
    *,
    as_of: str = "2026-08-14T03:00:00Z",
    gap_evidence: list[TerminalGapEvidence] | None = None,
):
    plan = plan_retrieval(
        f"2026-08-14T01:{start_minute:02d}:00Z",
        f"2026-08-14T01:{end_minute:02d}:00Z",
    )
    return instance.normalize(
        snapshots,
        retrieval_plan=plan,
        terminal_as_of=as_of,
        gap_evidence=gap_evidence or (),
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
    state_index["state_sha256"] = sha256_bytes(canonicalize(identity))
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


def test_late_older_batch_is_rejected_before_mutation_without_restart(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    older = snapshot(store, 0)
    newer = snapshot(store, 1)
    instance = approved_normalizer(older, newer)
    normalize(instance, [newer], 1, 2)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="regresses or overlaps"):
        normalize(instance, [older], 0, 1)

    assert instance.export_state_files() == before
    assert instance.next_expected_interval_start == "2026-08-14T01:02:00Z"


def test_late_older_batch_is_rejected_after_restart_without_publication(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    older = snapshot(store, 0)
    newer = snapshot(store, 1)
    instance = approved_normalizer(older, newer)
    normalize(instance, [newer], 1, 2)
    instance.publish_state(store, "newer-first")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-newer-first")
    before = hydrated.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="regresses or overlaps"):
        normalize(hydrated, [older], 0, 1)

    assert hydrated.export_state_files() == before
    assert not (store.publications_root / "gsg-normalizer-state-late").exists()


def test_contiguous_forward_progress_survives_restart_and_advances_watermark(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    minute_0 = snapshot(store, 0)
    minute_1 = snapshot(store, 1)
    instance = approved_normalizer(minute_0, minute_1)
    normalize(instance, [minute_0], 0, 1)
    instance.publish_state(store, "minute-0")

    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-minute-0")
    normalize(hydrated, [minute_1], 1, 2)

    assert hydrated.next_expected_interval_start == "2026-08-14T01:02:00Z"
    assert [item.start_at for item in hydrated.terminal_intervals] == [
        "2026-08-14T01:00:00Z",
        "2026-08-14T01:01:00Z",
    ]


def test_forward_gap_is_rejected_without_mutation(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    minute_0 = snapshot(store, 0)
    minute_2 = snapshot(store, 2)
    instance = approved_normalizer(minute_0, minute_2)
    normalize(instance, [minute_0], 0, 1)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="unrecorded interval"):
        normalize(instance, [minute_2], 2, 3)

    assert instance.export_state_files() == before


def test_explicit_provider_gap_advances_chronology(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    minute_0 = snapshot(store, 0)
    instance = approved_normalizer(minute_0)
    normalize(instance, [minute_0], 0, 1)

    plan = plan_retrieval("2026-08-14T01:01:00Z", "2026-08-14T01:02:00Z")
    evidence = TerminalGapEvidence.create(
        interval_start=plan.start_at,
        interval_end_exclusive=plan.end_at_exclusive,
        expected_source_locator=expected_gsg_source_locator(plan.intervals[0]),
        attempts=(
            GapAttempt(
                attempt_number=1,
                attempted_at="2026-08-14T01:31:00Z",
                http_status=404,
                error_kind=None,
                retry_after_seconds=None,
                retry_disposition="gap",
            ),
        ),
        terminal_at="2026-08-14T01:31:00Z",
        protocol_config_sha256=PROTOCOL_HASH,
    )
    normalize(instance, [], 1, 2, gap_evidence=[evidence])

    assert instance.next_expected_interval_start == "2026-08-14T01:02:00Z"
    gap = instance.terminal_intervals[-1]
    assert gap.outcome == "provider_gap"
    assert gap.snapshot_state == "missing"
    assert gap.terminal_reason == "verified_terminal_gap_evidence"
    assert gap.gap_evidence_id == evidence.evidence_id


def test_pending_interval_cannot_advance_watermark(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    minute_0 = snapshot(store, 0)
    minute_1 = snapshot(store, 1)
    instance = approved_normalizer(minute_0, minute_1)
    normalize(instance, [minute_0], 0, 1)
    before = instance.export_state_files()

    with pytest.raises(ProviderIngestionError, match="remain pending"):
        normalize(
            instance,
            [minute_1],
            1,
            2,
            as_of="2026-08-14T01:30:30Z",
        )

    assert instance.export_state_files() == before
    assert instance.next_expected_interval_start == "2026-08-14T01:01:00Z"


def test_overlap_and_exact_replay_are_rejected_as_unsupported(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    minute_0 = snapshot(store, 0)
    instance = approved_normalizer(minute_0)
    normalize(instance, [minute_0], 0, 1)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="regresses or overlaps"):
        normalize(instance, [minute_0], 0, 1)

    assert instance.export_state_files() == before


def test_conflicting_replay_is_rejected_without_mutation(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    original = snapshot(store, 0)
    conflict = snapshot(
        store,
        0,
        url="https://conflict.example/story",
        title="Bitcoin synthetic conflicting replay",
    )
    instance = approved_normalizer(original, conflict)
    normalize(instance, [original], 0, 1)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="regresses or overlaps"):
        normalize(instance, [conflict], 0, 1)

    assert instance.export_state_files() == before


def test_partition_independence_produces_byte_identical_state(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    title = "Bitcoin synthetic partition independence"
    minute_0 = snapshot(store, 0, url="https://early.example/story", title=title)
    minute_1 = snapshot(store, 1, url="https://later.example/story", title=title)
    whole = approved_normalizer(minute_0, minute_1)
    split = approved_normalizer(minute_0, minute_1)

    normalize(whole, [minute_1, minute_0], 0, 2)
    normalize(split, [minute_0], 0, 1)
    normalize(split, [minute_1], 1, 2)

    assert split.export_state_files() == whole.export_state_files()


def test_restart_independence_produces_byte_identical_state(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    minute_0 = snapshot(store, 0)
    minute_1 = snapshot(store, 1)
    continuous = approved_normalizer(minute_0, minute_1)
    restarted = approved_normalizer(minute_0, minute_1)
    normalize(continuous, [minute_0], 0, 1)
    normalize(continuous, [minute_1], 1, 2)
    normalize(restarted, [minute_0], 0, 1)
    restarted.publish_state(store, "restart-boundary")

    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-restart-boundary")
    normalize(hydrated, [minute_1], 1, 2)

    assert hydrated.export_state_files() == continuous.export_state_files()


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "noncanonical",
        "hash_mismatch",
        "malformed",
        "duplicate",
        "overlap",
        "discontinuous",
        "contradictory",
        "unjustified_watermark",
    ],
)
def test_hydration_rejects_corrupt_chronology(tmp_path: Path, damage: str) -> None:
    store = ContentAddressedStore(tmp_path)
    minute_0 = snapshot(store, 0)
    minute_1 = snapshot(store, 1)
    instance = approved_normalizer(minute_0, minute_1)
    normalize(instance, [minute_0, minute_1], 0, 2)
    files = instance.export_state_files()

    if damage == "missing":
        files.pop("chronology.json")
        state_index = json.loads(files["state.json"])
        state_index["files"].pop("chronology.json")
        identity = dict(state_index)
        identity.pop("state_sha256")
        state_index["state_sha256"] = sha256_bytes(canonicalize(identity))
        files["state.json"] = canonicalize(state_index)
    elif damage == "noncanonical":
        files["chronology.json"] = json.dumps(
            json.loads(files["chronology.json"]), indent=2
        ).encode()
    elif damage == "hash_mismatch":
        publication = publish_modified_state(store, "gsg-normalizer-state-damage-hash", dict(files))
        (publication / "chronology.json").write_bytes(b"{}")
        with pytest.raises(NormalizationIntegrityError):
            GSGNormalizer.hydrate(store, "gsg-normalizer-state-damage-hash")
        return
    else:
        chronology = json.loads(files["chronology.json"])
        intervals = chronology["terminal_intervals"]
        if damage == "malformed":
            intervals[0]["start_at"] = "not-a-timestamp"
        elif damage == "duplicate":
            intervals.insert(1, dict(intervals[0]))
        elif damage == "overlap":
            intervals[1]["start_at"] = intervals[0]["start_at"]
        elif damage == "discontinuous":
            intervals[1]["start_at"] = "2026-08-14T01:02:00Z"
            intervals[1]["end_at_exclusive"] = "2026-08-14T01:03:00Z"
            chronology["next_expected_interval_start"] = "2026-08-14T01:03:00Z"
        elif damage == "contradictory":
            intervals[0]["snapshot_state"] = "missing"
        elif damage == "unjustified_watermark":
            chronology["next_expected_interval_start"] = "2026-08-14T01:05:00Z"
        files["chronology.json"] = canonicalize(chronology)

    publication_id = f"gsg-normalizer-state-damage-{damage}"
    publish_modified_state(store, publication_id, files)
    with pytest.raises(NormalizationIntegrityError):
        GSGNormalizer.hydrate(store, publication_id)


@pytest.mark.parametrize("failure_point", ["chronology", "state", "manifest", "rename"])
def test_publication_failure_does_not_mutate_state_or_leave_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(store, 0)
    instance = approved_normalizer(item)
    normalize(instance, [item], 0, 1)
    before = instance.export_state_files()
    unrelated = store.publications_root / ".staging-unrelated-preserve"
    unrelated.mkdir()
    (unrelated / "sentinel").write_bytes(b"preserve")
    original_write = storage_module._write_fsynced_at
    original_rename = storage_module._atomic_rename_directory_no_replace

    def fail_write(parent_descriptor: int, name: str, data: bytes) -> None:
        target = {
            "chronology": "chronology.json",
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
    state_name = f"failure-{failure_point}"
    with pytest.raises(SentimentStorageError, match="simulated"):
        instance.publish_state(store, state_name)

    assert instance.export_state_files() == before
    assert not (store.publications_root / f"gsg-normalizer-state-{state_name}").exists()
    assert not list(store.publications_root.glob(f".staging-gsg-normalizer-state-{state_name}-*"))
    assert (unrelated / "sentinel").read_bytes() == b"preserve"


def test_reverse_interval_reproduction_is_rejected_before_anchor_mutation(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    title = "Bitcoin synthetic causal watermark anchor"
    older = snapshot(store, 0, url="https://older.example/story", title=title)
    newer = snapshot(store, 1, url="https://newer.example/story", title=title)
    approval = RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=PROTOCOL_HASH,
        raw_snapshot_sha256={
            older.receipt.raw_snapshot_sha256,
            newer.receipt.raw_snapshot_sha256,
        },
    )
    reverse = GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH, rights_approval=approval)
    chronological = GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH, rights_approval=approval)
    normalize(reverse, [newer], 1, 2)
    reverse_before = reverse.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="regresses or overlaps"):
        normalize(reverse, [older], 0, 1)

    normalize(chronological, [older], 0, 1)
    normalize(chronological, [newer], 1, 2)
    assert reverse.export_state_files() == reverse_before
    reverse_groups = json.loads(reverse_before["groups.json"])
    chronological_groups = json.loads(chronological.export_state_files()["groups.json"])
    assert any(item["canonical_url"] == "https://newer.example/story" for item in reverse_groups)
    assert any(
        item["canonical_url"] == "https://older.example/story" for item in chronological_groups
    )
    assert not any(
        item["canonical_url"] == "https://newer.example/story" for item in chronological_groups
    )
