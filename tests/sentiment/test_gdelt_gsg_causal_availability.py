"""Offline regressions for the GSG normalizer's causal availability boundary."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.exceptions import NormalizationIntegrityError, ProviderIngestionError
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GSGAdapter,
    GSGNormalizer,
    RightsApproval,
    SnapshotResult,
    plan_retrieval,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

PROJECT_ROOT = Path(__file__).parents[2]
PROTOCOL_HASH = sha256_bytes((PROJECT_ROOT / "config" / "phase2_protocol.json").read_bytes())
SOURCE_LOCATOR = "https://data.gdeltproject.org/gdeltv3/gsg/offline-synthetic.gz"


def _relation(
    url: str,
    title: str,
    companion: str,
    *,
    companion_url: str | None = None,
    companion_title: str | None = None,
) -> dict[str, object]:
    return {
        "fromDate": "20260814003000",
        "fromLang": "English",
        "fromTitle": title,
        "fromUrl": url,
        "similarity": 0.9,
        "toDate": "20260814003100",
        "toLang": "English",
        "toTitle": companion_title or f"Bitcoin synthetic companion {companion}",
        "toUrl": companion_url or f"https://companion.example/{companion}",
    }


def _gzip_records(records: Iterable[dict[str, object]]) -> bytes:
    payload = b"\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        for record in records
    )
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _snapshot(
    store: ContentAddressedStore,
    filename_minute: int,
    published_at: datetime,
    *,
    records: Iterable[dict[str, object]] | None = None,
) -> SnapshotResult:
    if records is None:
        records = (
            _relation(
                f"https://news{filename_minute}.example/story",
                f"Bitcoin synthetic minute {filename_minute}",
                str(filename_minute),
            ),
        )
    adapter = GSGAdapter(store, clock=lambda: published_at)
    return adapter.ingest_snapshot(
        _gzip_records(records),
        filename_timestamp=f"2026-08-14T01:{filename_minute:02d}:00Z",
        ingested_at=f"2026-08-14T01:{filename_minute:02d}:30Z",
        source_locator=SOURCE_LOCATOR,
        collection_mode="prospective",
        input_class="synthetic_fixture",
    )


def _approval(*snapshots: SnapshotResult) -> RightsApproval:
    return RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=PROTOCOL_HASH,
        raw_snapshot_sha256={item.receipt.raw_snapshot_sha256 for item in snapshots},
    )


def _normalizer(
    *snapshots: SnapshotResult, approval: RightsApproval | None = None
) -> GSGNormalizer:
    return GSGNormalizer(
        protocol_config_sha256=PROTOCOL_HASH,
        rights_approval=approval if approval is not None else _approval(*snapshots),
    )


def _normalize(
    instance: GSGNormalizer,
    snapshots: Iterable[SnapshotResult],
    start_minute: int,
    end_minute: int,
) -> Any:
    return instance.normalize(
        snapshots,
        retrieval_plan=plan_retrieval(
            f"2026-08-14T01:{start_minute:02d}:00Z",
            f"2026-08-14T01:{end_minute:02d}:00Z",
        ),
        terminal_as_of="2026-08-14T04:00:00Z",
    )


def _replace_receipt_times(
    snapshot: SnapshotResult,
    *,
    ingested_at: str | None = None,
    raw_published_at: str | None = None,
) -> SnapshotResult:
    receipt_changes: dict[str, str] = {}
    observation_changes: dict[str, str] = {}
    if ingested_at is not None:
        receipt_changes["ingested_at"] = ingested_at
        observation_changes["ingested_at"] = ingested_at
    if raw_published_at is not None:
        receipt_changes["raw_published_at"] = raw_published_at
        observation_changes["raw_published_at"] = raw_published_at
    return replace(
        snapshot,
        receipt=replace(snapshot.receipt, **receipt_changes),
        observations=tuple(
            replace(observation, **observation_changes) for observation in snapshot.observations
        ),
    )


def _publish_modified_state(
    store: ContentAddressedStore,
    publication_id: str,
    files: dict[str, bytes],
) -> None:
    state_index = json.loads(files["state.json"])
    state_index["files"] = {
        name: {"sha256": sha256_bytes(files[name]), "size_bytes": len(files[name])}
        for name in sorted(state_index["files"])
    }
    identity = dict(state_index)
    identity.pop("state_sha256")
    state_index["state_sha256"] = sha256_bytes(canonicalize(identity))
    files["state.json"] = canonicalize(state_index)
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


def test_raw_publication_regression_in_one_call_is_transactional(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    later_availability = _snapshot(store, 0, datetime(2026, 8, 14, 3, 0, tzinfo=UTC))
    earlier_availability = _snapshot(store, 1, datetime(2026, 8, 14, 2, 0, tzinfo=UTC))
    instance = _normalizer(later_availability, earlier_availability)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="causal availability"):
        _normalize(instance, [later_availability, earlier_availability], 0, 2)

    assert instance.export_state_files() == before
    assert instance.closed_availability_through is None
    assert instance.next_expected_interval_start is None
    assert instance.terminal_intervals == ()


def test_raw_publication_regression_across_calls_preserves_exact_state(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    later_availability = _snapshot(store, 0, datetime(2026, 8, 14, 3, 0, tzinfo=UTC))
    earlier_availability = _snapshot(store, 1, datetime(2026, 8, 14, 2, 0, tzinfo=UTC))
    instance = _normalizer(later_availability, earlier_availability)
    _normalize(instance, [later_availability], 0, 1)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="causal availability"):
        _normalize(instance, [earlier_availability], 1, 2)

    assert instance.export_state_files() == before
    assert instance.closed_availability_through == "2026-08-14T03:00:00.000001Z"
    assert instance.next_expected_interval_start == "2026-08-14T01:01:00Z"


def test_raw_publication_regression_after_restart_preserves_exact_state(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    later_availability = _snapshot(store, 0, datetime(2026, 8, 14, 3, 0, tzinfo=UTC))
    earlier_availability = _snapshot(store, 1, datetime(2026, 8, 14, 2, 0, tzinfo=UTC))
    approval = _approval(later_availability, earlier_availability)
    instance = _normalizer(approval=approval)
    _normalize(instance, [later_availability], 0, 1)
    instance.publish_state(store, "causal-regression-restart")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-causal-regression-restart")
    before = hydrated.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="causal availability"):
        _normalize(hydrated, [earlier_availability], 1, 2)

    assert hydrated.export_state_files() == before
    assert hydrated.closed_availability_through == "2026-08-14T03:00:00.000001Z"


def test_valid_stream_is_identical_whole_split_and_after_restart(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    snapshots = (
        _snapshot(store, 0, datetime(2026, 8, 14, 2, 0, 0, 0, tzinfo=UTC)),
        _snapshot(store, 1, datetime(2026, 8, 14, 2, 0, 0, 1, tzinfo=UTC)),
        _snapshot(store, 2, datetime(2026, 8, 14, 2, 0, 0, 2, tzinfo=UTC)),
    )
    approval = _approval(*snapshots)
    whole = _normalizer(approval=approval)
    split = _normalizer(approval=approval)
    restarted = _normalizer(approval=approval)

    _normalize(whole, reversed(snapshots), 0, 3)
    for minute, item in enumerate(snapshots):
        _normalize(split, [item], minute, minute + 1)
    _normalize(restarted, [snapshots[0]], 0, 1)
    restarted.publish_state(store, "valid-causal-prefix")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-valid-causal-prefix")
    _normalize(hydrated, snapshots[1:], 1, 3)

    expected = whole.export_state_files()
    assert split.export_state_files() == expected
    assert hydrated.export_state_files() == expected
    assert whole.closed_availability_through == "2026-08-14T02:00:00.000003Z"


def test_equal_publication_timestamps_from_distinct_snapshots_are_rejected(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    published_at = datetime(2026, 8, 14, 2, 0, tzinfo=UTC)
    first = _snapshot(store, 0, published_at)
    second = _snapshot(store, 1, published_at)
    instance = _normalizer(first, second)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="causal availability"):
        _normalize(instance, [second, first], 0, 2)

    assert instance.export_state_files() == before


def test_same_time_conflicts_within_one_snapshot_are_all_excluded(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    item = _snapshot(
        store,
        0,
        datetime(2026, 8, 14, 2, 0, tzinfo=UTC),
        records=(
            _relation(
                "https://conflict.example/same-article",
                "Bitcoin synthetic conflict version one",
                "ignored",
                companion_url="https://conflict.example/same-article",
                companion_title="Bitcoin synthetic conflict version two",
            ),
        ),
    )
    instance = _normalizer(item)

    result = _normalize(instance, [item], 0, 1)

    assert result.articles == ()
    assert result.observation_links == ()
    assert len(result.exclusions) == 2
    assert {item.reason for item in result.exclusions} == {"revision_time_unknown"}
    assert len(json.loads(instance.export_state_files()["exclusions.json"])) == 2
    assert instance.closed_availability_through == "2026-08-14T02:00:00.000001Z"


def test_equal_time_conflict_against_prior_immutable_state_fails_without_mutation(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    published_at = datetime(2026, 8, 14, 2, 0, tzinfo=UTC)
    first = _snapshot(
        store,
        0,
        published_at,
        records=(
            _relation(
                "https://immutable.example/article",
                "Bitcoin synthetic immutable version one",
                "first",
            ),
        ),
    )
    conflicting = _snapshot(
        store,
        1,
        published_at,
        records=(
            _relation(
                "https://immutable.example/article",
                "Bitcoin synthetic immutable version two",
                "second",
            ),
        ),
    )
    instance = _normalizer(first, conflicting)
    _normalize(instance, [first], 0, 1)
    before = instance.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="causal availability"):
        _normalize(instance, [conflicting], 1, 2)

    assert instance.export_state_files() == before
    assert len(json.loads(before["articles.json"])) == 2


@pytest.mark.parametrize(
    ("ingested_at", "raw_published_at"),
    [
        (None, "2026-08-14T02:00:00.0Z"),
        (None, "2026-08-14T02:00:00+00:00"),
        ("2026-08-14T01:00:30.0Z", None),
        ("2026-08-14T01:00:30Z", "2026-08-14T01:00:29Z"),
    ],
)
def test_complete_snapshot_receipt_timestamps_are_strict_and_transactional(
    tmp_path: Path,
    ingested_at: str | None,
    raw_published_at: str | None,
) -> None:
    store = ContentAddressedStore(tmp_path)
    original = _snapshot(store, 0, datetime(2026, 8, 14, 2, 0, tzinfo=UTC))
    damaged = _replace_receipt_times(
        original,
        ingested_at=ingested_at,
        raw_published_at=raw_published_at,
    )
    instance = _normalizer(original)
    before = instance.export_state_files()

    with pytest.raises(
        (NormalizationIntegrityError, ProviderIngestionError),
        match="coverage snapshot terminal facts are invalid",
    ):
        _normalize(instance, [damaged], 0, 1)

    assert instance.export_state_files() == before


def test_zero_line_snapshot_advances_the_exclusive_availability_boundary(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    empty = _snapshot(
        store,
        0,
        datetime(2026, 8, 14, 2, 0, tzinfo=UTC),
        records=(),
    )
    immediately_next = _snapshot(
        store,
        1,
        datetime(2026, 8, 14, 2, 0, 0, 1, tzinfo=UTC),
    )
    instance = _normalizer(empty, immediately_next)

    empty_result = _normalize(instance, [empty], 0, 1)
    assert empty.json_line_count == 0
    assert empty_result.articles == ()
    assert instance.closed_availability_through == "2026-08-14T02:00:00.000001Z"

    _normalize(instance, [immediately_next], 1, 2)
    assert instance.closed_availability_through == "2026-08-14T02:00:00.000002Z"
    assert [item.json_line_count for item in instance.terminal_intervals] == [0, 1]


@pytest.mark.parametrize(
    "damage",
    ["missing", "regressed", "noncanonical_timestamp", "noncanonical_encoding"],
)
def test_hydration_rejects_corrupt_causal_availability_boundary(
    tmp_path: Path, damage: str
) -> None:
    store = ContentAddressedStore(tmp_path)
    item = _snapshot(store, 0, datetime(2026, 8, 14, 2, 0, tzinfo=UTC))
    instance = _normalizer(item)
    _normalize(instance, [item], 0, 1)
    files = dict(instance.export_state_files())
    chronology = json.loads(files["chronology.json"])

    if damage == "missing":
        chronology.pop("closed_availability_through")
        files["chronology.json"] = canonicalize(chronology)
    elif damage == "regressed":
        chronology["closed_availability_through"] = "2026-08-14T01:59:59Z"
        files["chronology.json"] = canonicalize(chronology)
    elif damage == "noncanonical_timestamp":
        chronology["closed_availability_through"] = "2026-08-14T02:00:00.0000010Z"
        files["chronology.json"] = canonicalize(chronology)
    else:
        files["chronology.json"] = json.dumps(chronology, indent=2).encode("utf-8")

    publication_id = f"gsg-normalizer-state-corrupt-causal-{damage}"
    _publish_modified_state(store, publication_id, files)
    with pytest.raises(NormalizationIntegrityError):
        GSGNormalizer.hydrate(store, publication_id)


def test_late_failure_in_multi_interval_batch_rolls_back_every_component(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    prefix = _snapshot(store, 0, datetime(2026, 8, 14, 2, 0, tzinfo=UTC))
    first_new = _snapshot(store, 1, datetime(2026, 8, 14, 2, 0, 2, tzinfo=UTC))
    regressing_new = _snapshot(store, 2, datetime(2026, 8, 14, 2, 0, 1, tzinfo=UTC))
    instance = _normalizer(prefix, first_new, regressing_new)
    _normalize(instance, [prefix], 0, 1)
    before_files = instance.export_state_files()
    before_intervals = instance.terminal_intervals
    before_boundary = instance.closed_availability_through
    before_watermark = instance.next_expected_interval_start

    with pytest.raises(NormalizationIntegrityError, match="causal availability"):
        _normalize(instance, [first_new, regressing_new], 1, 3)

    assert instance.export_state_files() == before_files
    assert instance.terminal_intervals == before_intervals
    assert instance.closed_availability_through == before_boundary
    assert instance.next_expected_interval_start == before_watermark
