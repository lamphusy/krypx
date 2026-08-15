"""Regression tests for restart, causal ordering, conflict, and rights integrity."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.exceptions import (
    NormalizationIntegrityError,
    PublicationCollisionError,
    SentimentStorageError,
)
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import (
    derive_article_version_id,
    derive_content_hash,
    derive_duplicate_group_id,
)
from crypto_ai.sentiment.providers.gdelt_gsg import (
    PROVIDER_ID,
    RIGHTS_SCOPE,
    GSGAdapter,
    GSGNormalizer,
    RightsApproval,
    plan_retrieval,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

PROJECT_ROOT = Path(__file__).parents[2]
PROTOCOL_HASH = sha256_bytes((PROJECT_ROOT / "config" / "phase2_protocol.json").read_bytes())


def relation(
    from_url: object,
    from_title: object,
    to_url: object,
    to_title: object,
    *,
    from_language: object = "English",
    to_language: object = "English",
    from_date: object = "20260814003000",
    to_date: object = "20260814003100",
) -> dict[str, object]:
    return {
        "fromDate": from_date,
        "fromLang": from_language,
        "fromTitle": from_title,
        "fromUrl": from_url,
        "similarity": 0.9,
        "toDate": to_date,
        "toLang": to_language,
        "toTitle": to_title,
        "toUrl": to_url,
    }


def gzip_records(records: list[dict[str, object]]) -> bytes:
    payload = b"\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() for record in records
    )
    return gzip.compress(payload, compresslevel=9, mtime=0)


def snapshot(
    store: ContentAddressedStore,
    raw: bytes,
    minute: int,
    *,
    published_minute: int = 0,
    input_class: str = "synthetic_fixture",
    mode: str = "prospective",
):
    published = datetime(2026, 8, 14, 2, published_minute, tzinfo=UTC)
    adapter = GSGAdapter(store, clock=lambda: published)
    return adapter.ingest_snapshot(
        raw,
        filename_timestamp=f"2026-08-14T01:{minute:02d}:00Z",
        ingested_at=f"2026-08-14T01:{minute:02d}:30Z",
        source_locator="https://data.gdeltproject.org/gdeltv3/gsg/synthetic.gz",
        collection_mode=mode,
        input_class=input_class,
    )


def synthetic_approval(*snapshots: Any, protocol_hash: str = PROTOCOL_HASH) -> RightsApproval:
    return RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=protocol_hash,
        raw_snapshot_sha256={item.receipt.raw_snapshot_sha256 for item in snapshots},
    )


def normalizer(*snapshots: Any, approval: RightsApproval | None = None) -> GSGNormalizer:
    return GSGNormalizer(
        protocol_config_sha256=PROTOCOL_HASH,
        rights_approval=approval if approval is not None else synthetic_approval(*snapshots),
    )


def normalize(
    instance: GSGNormalizer,
    snapshots: list[Any],
    *,
    start_minute: int,
    end_minute: int,
):
    return instance.normalize(
        snapshots,
        retrieval_plan=plan_retrieval(
            f"2026-08-14T01:{start_minute:02d}:00Z",
            f"2026-08-14T01:{end_minute:02d}:00Z",
        ),
        terminal_as_of="2026-08-14T03:00:00Z",
    )


def group_map(result: Any) -> dict[str, str | None]:
    return {article.article_id: article.duplicate_group_id for article in result.articles}


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
    state_identity = dict(state_index)
    state_identity.pop("state_sha256")
    state_index["state_sha256"] = sha256_bytes(canonicalize(state_identity))
    files["state.json"] = canonicalize(state_index)
    return store.publish_bundle(publication_id, files)


def test_restart_repeat_preserves_first_seen_version_group_links_and_semantics(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    raw = gzip_records(
        [
            relation(
                "https://a.example/bitcoin",
                "Bitcoin synthetic repeat",
                "https://b.example/bitcoin",
                "Bitcoin synthetic repeat",
            )
        ]
    )
    first = snapshot(store, raw, 0, published_minute=0)
    repeated = snapshot(store, raw, 1, published_minute=1)
    approval = synthetic_approval(first, repeated)

    before_restart = normalizer(first, repeated, approval=approval)
    first_result = normalize(before_restart, [first], start_minute=0, end_minute=1)
    before_restart.publish_state(store, "restart-repeat")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-restart-repeat")
    restarted_result = normalize(hydrated, [repeated], start_minute=1, end_minute=2)

    uninterrupted = normalizer(first, repeated, approval=approval)
    normalize(uninterrupted, [first], start_minute=0, end_minute=1)
    uninterrupted_result = normalize(uninterrupted, [repeated], start_minute=1, end_minute=2)

    assert restarted_result.to_dict() == uninterrupted_result.to_dict()
    assert hydrated.export_state_files() == uninterrupted.export_state_files()
    originals = {item.article_id: item for item in first_result.articles}
    for article in restarted_result.articles:
        assert article.first_seen_at == originals[article.article_id].first_seen_at
        assert article.article_version_id == originals[article.article_id].article_version_id
        assert article.duplicate_group_id == originals[article.article_id].duplicate_group_id
    assert all(item.reused_existing_version for item in restarted_result.observation_links)


def test_restart_before_revision_preserves_history_and_group(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    first = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/story",
                    "Bitcoin synthetic first title",
                    "https://b.example/story",
                    "Bitcoin synthetic companion",
                )
            ]
        ),
        0,
    )
    revision = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/story",
                    "Bitcoin synthetic revised title",
                    "https://b.example/story",
                    "Bitcoin synthetic companion",
                )
            ]
        ),
        1,
        published_minute=1,
    )
    approval = synthetic_approval(first, revision)
    original = normalizer(first, revision, approval=approval)
    first_result = normalize(original, [first], start_minute=0, end_minute=1)
    original.publish_state(store, "before-revision")

    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-before-revision")
    revision_result = normalize(hydrated, [revision], start_minute=1, end_minute=2)
    old = next(item for item in first_result.articles if item.source == "a.example")
    new = next(item for item in revision_result.articles if item.source == "a.example")

    assert new.article_id == old.article_id
    assert new.article_version_id != old.article_version_id
    assert new.duplicate_group_id == old.duplicate_group_id
    assert len(hydrated.export_state_files()["articles.json"]) > 0


def test_restart_before_syndicated_duplicate_preserves_permanent_anchor(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    first = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/syndicated",
                    "Bitcoin synthetic syndicated title",
                    "https://b.example/unique",
                    "Bitcoin synthetic unique first",
                )
            ]
        ),
        0,
    )
    later = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://c.example/syndicated",
                    "Bitcoin synthetic syndicated title",
                    "https://d.example/unique",
                    "Bitcoin synthetic unique later",
                )
            ]
        ),
        1,
        published_minute=1,
    )
    approval = synthetic_approval(first, later)
    initial = normalizer(first, later, approval=approval)
    first_result = normalize(initial, [first], start_minute=0, end_minute=1)
    initial.publish_state(store, "before-syndication")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-before-syndication")
    later_result = normalize(hydrated, [later], start_minute=1, end_minute=2)

    anchor = next(item for item in first_result.articles if item.source == "a.example")
    duplicate = next(item for item in later_result.articles if item.source == "c.example")
    assert duplicate.duplicate_group_id == anchor.duplicate_group_id


def test_state_round_trip_is_deterministic_and_collision_is_no_overwrite(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/state",
                    "Bitcoin synthetic state",
                    "https://b.example/state",
                    "Bitcoin synthetic state duplicate",
                )
            ]
        ),
        0,
    )
    instance = normalizer(item)
    normalize(instance, [item], start_minute=0, end_minute=1)
    exported = instance.export_state_files()
    instance.publish_state(store, "roundtrip")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-roundtrip")

    assert hydrated.export_state_files() == exported
    with pytest.raises(PublicationCollisionError):
        instance.publish_state(store, "roundtrip")


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_hydration_rejects_corrupt_or_missing_referenced_state_file(
    tmp_path: Path, damage: str
) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/damage",
                    "Bitcoin synthetic damage",
                    "https://b.example/damage",
                    "Bitcoin synthetic damage two",
                )
            ]
        ),
        0,
    )
    instance = normalizer(item)
    normalize(instance, [item], start_minute=0, end_minute=1)
    publication = instance.publish_state(store, f"damage-{damage}")
    articles_path = publication / "articles.json"
    if damage == "corrupt":
        articles_path.write_bytes(b"[]")
    else:
        articles_path.unlink()

    with pytest.raises(NormalizationIntegrityError):
        GSGNormalizer.hydrate(store, f"gsg-normalizer-state-damage-{damage}")


def test_hydration_rejects_noncanonical_conflicting_partial_and_missing_transitive_state(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/deep-state",
                    "Bitcoin synthetic deep state",
                    "https://b.example/deep-state",
                    "Bitcoin synthetic deep state two",
                )
            ]
        ),
        0,
    )
    instance = normalizer(item)
    normalize(instance, [item], start_minute=0, end_minute=1)
    original_files = instance.export_state_files()

    noncanonical_files = dict(original_files)
    articles = json.loads(noncanonical_files["articles.json"])
    noncanonical_files["articles.json"] = json.dumps(articles, indent=2).encode()
    publish_modified_state(store, "gsg-normalizer-state-noncanonical", noncanonical_files)
    with pytest.raises(NormalizationIntegrityError, match="canonical"):
        GSGNormalizer.hydrate(store, "gsg-normalizer-state-noncanonical")

    conflicting_files = dict(original_files)
    conflicting_articles = json.loads(conflicting_files["articles.json"])
    original = dict(conflicting_articles[0])
    original["title"] = f"{original['title']} conflicting revision"
    original["content_hash"] = derive_content_hash(
        asset=original["asset"],
        content=original["content"],
        language=original["language"],
        source=original["source"],
        title=original["title"],
    )
    original["article_version_id"] = derive_article_version_id(
        article_id=original["article_id"],
        first_seen_at=original["first_seen_at"],
        language=original["language"],
        content_hash=original["content_hash"],
    )
    conflicting_articles.append(original)
    conflicting_files["articles.json"] = canonicalize(conflicting_articles)
    publish_modified_state(store, "gsg-normalizer-state-conflicting", conflicting_files)
    with pytest.raises(NormalizationIntegrityError, match="same-time"):
        GSGNormalizer.hydrate(store, "gsg-normalizer-state-conflicting")

    partial = store.publications_root / ".staging-gsg-normalizer-state-partial-fixture"
    partial.mkdir()
    (partial / "state.json").write_bytes(original_files["state.json"])
    with pytest.raises(NormalizationIntegrityError):
        GSGNormalizer.hydrate(store, "gsg-normalizer-state-partial")

    instance.publish_state(store, "missing-raw")
    raw_path = (
        store.objects_root / item.receipt.raw_snapshot_sha256[:2] / item.receipt.raw_snapshot_sha256
    )
    raw_path.unlink()
    with pytest.raises(NormalizationIntegrityError, match="transitive raw object"):
        GSGNormalizer.hydrate(store, "gsg-normalizer-state-missing-raw")


def test_equal_time_endpoint_and_line_order_use_lexicographically_smallest_article_anchor(
    tmp_path: Path,
) -> None:
    records = [
        relation(
            "https://a.example/order",
            "Bitcoin synthetic equal anchor",
            "https://b.example/order",
            "Bitcoin synthetic equal anchor",
        ),
        relation(
            "https://c.example/order",
            "Bitcoin synthetic equal anchor",
            "https://d.example/order",
            "Bitcoin synthetic equal anchor",
        ),
    ]
    maps = []
    for index, variant in enumerate(
        [
            records,
            list(reversed(records)),
            [
                relation(
                    records[0]["toUrl"],
                    records[0]["toTitle"],
                    records[0]["fromUrl"],
                    records[0]["fromTitle"],
                ),
                records[1],
            ],
        ]
    ):
        store = ContentAddressedStore(tmp_path / str(index))
        item = snapshot(store, gzip_records(variant), 0)
        result = normalize(normalizer(item), [item], start_minute=0, end_minute=1)
        maps.append(group_map(result))

    assert maps[0] == maps[1] == maps[2]
    smallest_article_id = min(maps[0])
    assert set(maps[0].values()) == {derive_duplicate_group_id(smallest_article_id)}


def test_snapshot_input_order_and_future_append_do_not_change_prior_groups(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    first = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/input",
                    "Bitcoin synthetic snapshot order",
                    "https://b.example/input",
                    "Bitcoin synthetic first unique",
                )
            ]
        ),
        0,
    )
    second = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://c.example/input",
                    "Bitcoin synthetic snapshot order",
                    "https://d.example/input",
                    "Bitcoin synthetic second unique",
                )
            ]
        ),
        1,
        published_minute=1,
    )
    approval = synthetic_approval(first, second)
    forward = normalizer(first, second, approval=approval)
    reverse = normalizer(first, second, approval=approval)
    forward_result = normalize(forward, [first, second], start_minute=0, end_minute=2)
    reverse_result = normalize(reverse, [second, first], start_minute=0, end_minute=2)
    assert forward_result.to_dict() == reverse_result.to_dict()

    incremental = normalizer(first, second, approval=approval)
    initial = normalize(incremental, [first], start_minute=0, end_minute=1)
    prior_groups = group_map(initial)
    normalize(incremental, [second], start_minute=1, end_minute=2)
    state_articles = json.loads(incremental.export_state_files()["articles.json"])
    assert {
        item["article_id"]: item["duplicate_group_id"]
        for item in state_articles
        if item["article_id"] in prior_groups
    } == prior_groups


def test_same_time_conflicts_exclude_every_fingerprint_and_are_order_independent(
    tmp_path: Path,
) -> None:
    conflict = relation(
        "https://same.example/conflict",
        "Bitcoin synthetic conflict one",
        "https://same.example/conflict",
        "Bitcoin synthetic conflict two",
    )
    unrelated = relation(
        "https://u.example/one",
        "Bitcoin synthetic unrelated one",
        "https://v.example/two",
        "Bitcoin synthetic unrelated two",
    )
    results = []
    for index, records in enumerate(([conflict, unrelated], [unrelated, conflict])):
        store = ContentAddressedStore(tmp_path / str(index))
        item = snapshot(store, gzip_records(list(records)), 0)
        result = normalize(normalizer(item), [item], start_minute=0, end_minute=1)
        results.append(result)
    for result in results:
        assert len(result.articles) == 2
        assert len(result.exclusions) == 2
        assert {item.reason for item in result.exclusions} == {"revision_time_unknown"}
        assert all(item.source in {"u.example", "v.example"} for item in result.articles)
    assert {item.article_id for item in results[0].articles} == {
        item.article_id for item in results[1].articles
    }


def test_revision_conflict_precedence_is_stable_without_rights_approval(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://same.example/precedence",
                    "Bitcoin synthetic precedence one",
                    "https://same.example/precedence",
                    "Bitcoin synthetic precedence two",
                )
            ]
        ),
        0,
    )
    instance = GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH)
    result = normalize(instance, [item], start_minute=0, end_minute=1)
    repeated = normalize(
        GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH),
        [item],
        start_minute=0,
        end_minute=1,
    )

    assert result.articles == ()
    assert [item.reason for item in result.exclusions] == [
        "revision_time_unknown",
        "revision_time_unknown",
    ]
    assert repeated.to_dict() == result.to_dict()


def test_conflicts_across_snapshots_are_order_independent(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    first = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://same.example/across",
                    "Bitcoin synthetic across one",
                    "https://u.example/across",
                    "Bitcoin synthetic unrelated across one",
                )
            ]
        ),
        0,
        published_minute=0,
    )
    second = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://same.example/across",
                    "Bitcoin synthetic across two",
                    "https://v.example/across",
                    "Bitcoin synthetic unrelated across two",
                )
            ]
        ),
        1,
        published_minute=0,
    )
    approval = synthetic_approval(first, second)
    results = [
        normalize(
            normalizer(first, second, approval=approval),
            order,
            start_minute=0,
            end_minute=2,
        )
        for order in ([first, second], [second, first])
    ]
    assert results[0].to_dict() == results[1].to_dict()
    assert len(results[0].articles) == 2
    assert len(results[0].exclusions) == 2


def test_conflict_after_hydration_fails_closed_without_partial_publication(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path)
    first = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://same.example/restart-conflict",
                    "Bitcoin synthetic restart one",
                    "https://u.example/restart-conflict",
                    "Bitcoin synthetic unrelated restart",
                )
            ]
        ),
        0,
        published_minute=0,
    )
    conflicting = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://same.example/restart-conflict",
                    "Bitcoin synthetic restart two",
                    "https://v.example/restart-conflict",
                    "Bitcoin synthetic unrelated later",
                )
            ]
        ),
        1,
        published_minute=0,
    )
    approval = synthetic_approval(first, conflicting)
    initial = normalizer(first, conflicting, approval=approval)
    normalize(initial, [first], start_minute=0, end_minute=1)
    initial.publish_state(store, "before-conflict")
    hydrated = GSGNormalizer.hydrate(store, "gsg-normalizer-state-before-conflict")
    before = hydrated.export_state_files()

    with pytest.raises(NormalizationIntegrityError, match="same-time revision"):
        normalize(hydrated, [conflicting], start_minute=1, end_minute=2)
    assert hydrated.export_state_files() == before
    assert not (store.publications_root / "gsg-normalizer-state-after-conflict").exists()


def test_conflict_state_publication_failure_leaves_no_partial_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://same.example/publication-conflict",
                    "Bitcoin synthetic publication one",
                    "https://same.example/publication-conflict",
                    "Bitcoin synthetic publication two",
                )
            ]
        ),
        0,
    )
    instance = normalizer(item)
    normalize(instance, [item], start_minute=0, end_minute=1)
    original_write = storage_module._write_fsynced

    def fail_state_index(path: Path, data: bytes) -> None:
        if path.name == "state.json":
            raise SentimentStorageError("simulated state publication failure")
        original_write(path, data)

    monkeypatch.setattr(storage_module, "_write_fsynced", fail_state_index)
    with pytest.raises(SentimentStorageError, match="state publication failure"):
        instance.publish_state(store, "failed-conflict")
    assert not (store.publications_root / "gsg-normalizer-state-failed-conflict").exists()
    assert not list(store.publications_root.glob(".staging-gsg-normalizer-state-failed-conflict-*"))


def test_missing_or_mismatched_rights_approval_fails_closed(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/rights",
                    "Bitcoin synthetic rights",
                    "https://b.example/rights",
                    "Bitcoin synthetic rights two",
                )
            ]
        ),
        0,
    )
    approvals = [
        None,
        RightsApproval.create(
            approval_kind="synthetic_fixture_only",
            approved=False,
            provider=PROVIDER_ID,
            scope=RIGHTS_SCOPE,
            protocol_config_sha256=PROTOCOL_HASH,
            authorized_fixture_sha256=[item.receipt.raw_snapshot_sha256],
        ),
        RightsApproval.create(
            approval_kind="synthetic_fixture_only",
            approved=True,
            provider=PROVIDER_ID,
            scope="wrong_scope",
            protocol_config_sha256=PROTOCOL_HASH,
            authorized_fixture_sha256=[item.receipt.raw_snapshot_sha256],
        ),
        RightsApproval.synthetic_fixture_only(
            protocol_config_sha256="f" * 64,
            raw_snapshot_sha256=[item.receipt.raw_snapshot_sha256],
        ),
    ]
    for approval in approvals:
        instance = GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH, rights_approval=approval)
        result = normalize(instance, [item], start_minute=0, end_minute=1)
        assert result.articles == ()
        assert {exclusion.reason for exclusion in result.exclusions} == {"license_restricted"}


def test_synthetic_approval_cannot_authorize_provider_response_or_network(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    raw = gzip_records(
        [
            relation(
                "https://a.example/provider",
                "Bitcoin synthetic provider class",
                "https://b.example/provider",
                "Bitcoin synthetic provider class two",
            )
        ]
    )
    item = snapshot(store, raw, 0, input_class="provider_response")
    approval = RightsApproval.synthetic_fixture_only(
        protocol_config_sha256=PROTOCOL_HASH,
        raw_snapshot_sha256=[item.receipt.raw_snapshot_sha256],
    )
    result = normalize(
        GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH, rights_approval=approval),
        [item],
        start_minute=0,
        end_minute=1,
    )
    assert result.articles == ()
    assert {item.reason for item in result.exclusions} == {"license_restricted"}
    assert approval.network_access_authorized is False
    assert not hasattr(GSGAdapter, "fetch")

    synthetic_item = snapshot(store, raw, 1, input_class="synthetic_fixture")
    provider_item = snapshot(store, raw, 2, input_class="provider_response")
    synthetic_state = normalizer(synthetic_item)
    normalize(synthetic_state, [synthetic_item], start_minute=1, end_minute=2)
    provider_result = normalize(synthetic_state, [provider_item], start_minute=2, end_minute=3)
    assert provider_result.articles == ()
    assert {item.reason for item in provider_result.exclusions} == {"license_restricted"}


def test_rights_and_protocol_hashes_change_semantic_identity(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/config",
                    "Bitcoin synthetic config",
                    "https://b.example/config",
                    "Bitcoin synthetic config two",
                )
            ]
        ),
        0,
    )
    without_approval = GSGNormalizer(protocol_config_sha256=PROTOCOL_HASH)
    first = normalize(without_approval, [item], start_minute=0, end_minute=1)
    different_protocol = GSGNormalizer(protocol_config_sha256="e" * 64)
    second = normalize(different_protocol, [item], start_minute=0, end_minute=1)
    approved = normalize(normalizer(item), [item], start_minute=0, end_minute=1)

    assert first.semantic_sha256 != second.semantic_sha256
    assert first.protocol_config_sha256 != second.protocol_config_sha256
    assert approved.rights_approval_sha256 != first.rights_approval_sha256
    assert approved.semantic_sha256 != first.semantic_sha256


def test_historical_and_multi_failure_precedence_are_stable(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    raw = gzip_records(
        [
            relation(
                None,
                None,
                None,
                None,
                from_language=None,
                to_language=None,
                from_date="invalid",
                to_date="invalid",
            )
        ]
    )
    historical = snapshot(store, raw, 0, mode="historical_backfill")
    approval = synthetic_approval(historical)
    instance = normalizer(historical, approval=approval)
    result = normalize(instance, [historical], start_minute=0, end_minute=1)
    reverse_result = normalize(
        normalizer(historical, approval=approval),
        [historical],
        start_minute=0,
        end_minute=1,
    )

    assert [item.reason for item in result.exclusions] == [
        "invalid_timestamp",
        "invalid_timestamp",
    ]
    assert result.to_dict() == reverse_result.to_dict()


def test_historical_remains_ineligible_with_valid_rights(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    item = snapshot(
        store,
        gzip_records(
            [
                relation(
                    "https://a.example/history",
                    "Bitcoin synthetic history",
                    "https://b.example/history",
                    "Bitcoin synthetic history two",
                )
            ]
        ),
        0,
        mode="historical_backfill",
    )
    result = normalize(normalizer(item), [item], start_minute=0, end_minute=1)
    assert result.articles == ()
    assert {item.reason for item in result.exclusions} == {
        "historical_backfill_without_availability"
    }
