"""Immutable exact-byte storage and publication tests."""

import os
from pathlib import Path

import pytest

from crypto_ai.exceptions import PublicationCollisionError, SentimentStorageError
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.canonical import sha256_bytes
from crypto_ai.sentiment.storage import ContentAddressedStore


def test_content_addressed_storage_uses_exact_bytes_and_is_idempotent(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    data = b"exact\x00fixture\r\nbytes"
    digest = store.put_bytes(data)

    assert digest == sha256_bytes(data)
    assert store.put_bytes(data, expected_sha256=digest) == digest
    assert store.get_bytes(digest) == data


def test_content_object_hash_mismatch_is_detected(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    digest = store.put_bytes(b"original")
    object_path = store.objects_root / digest[:2] / digest
    object_path.write_bytes(b"tampered")

    with pytest.raises(SentimentStorageError, match="hash mismatch"):
        store.get_bytes(digest)


def test_replacement_race_accepts_identical_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    data = b"racing writer"
    digest = sha256_bytes(data)
    original_link = os.link

    def competing_link(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        raise FileExistsError

    monkeypatch.setattr(storage_module.os, "link", competing_link)
    assert store.put_bytes(data) == digest
    monkeypatch.setattr(storage_module.os, "link", original_link)
    assert store.get_bytes(digest) == data


def test_replacement_race_rejects_conflicting_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)

    def competing_link(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"conflict")
        raise FileExistsError

    monkeypatch.setattr(storage_module.os, "link", competing_link)
    with pytest.raises(SentimentStorageError, match="hash mismatch"):
        store.put_bytes(b"intended")


def test_incomplete_object_write_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)

    def fail_write(path: Path, data: bytes) -> None:
        raise SentimentStorageError("simulated incomplete write")

    monkeypatch.setattr(storage_module, "_write_fsynced", fail_write)
    digest = sha256_bytes(b"never visible")
    with pytest.raises(SentimentStorageError, match="incomplete"):
        store.put_bytes(b"never visible")
    assert not (store.objects_root / digest[:2] / digest).exists()
    assert not list(store.staging_root.iterdir())


def test_publication_is_manifest_last_verified_and_no_overwrite(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle(
        "fixture-run-001",
        {"articles.jsonl": b'{"fixture":true}\n', "nested/raw.bin": b"raw"},
        metadata={"scope": "offline-fixture"},
    )

    assert published.is_dir()
    manifest = store.verify_publication("fixture-run-001")
    assert manifest["metadata"] == {"scope": "offline-fixture"}
    with pytest.raises(PublicationCollisionError):
        store.publish_bundle("fixture-run-001", {"articles.jsonl": b"replacement"})
    assert (published / "articles.jsonl").read_bytes() == b'{"fixture":true}\n'


def test_incomplete_bundle_is_cleaned_and_never_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    original_write = storage_module._write_fsynced

    def fail_manifest(path: Path, data: bytes) -> None:
        if path.name == "manifest.json":
            raise SentimentStorageError("simulated manifest failure")
        original_write(path, data)

    monkeypatch.setattr(storage_module, "_write_fsynced", fail_manifest)
    with pytest.raises(SentimentStorageError, match="manifest failure"):
        store.publish_bundle("incomplete", {"payload.bin": b"payload"})
    assert not (store.publications_root / "incomplete").exists()
    assert not list(store.publications_root.glob(".staging-*"))


def test_publication_collision_keeps_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    destination = store.publications_root / "collision"

    def race(source: Path, target: Path) -> None:
        destination.mkdir()
        (destination / "winner.txt").write_bytes(b"winner")
        raise PublicationCollisionError("simulated publication collision")

    monkeypatch.setattr(storage_module, "_atomic_rename_directory_no_replace", race)
    with pytest.raises(PublicationCollisionError):
        store.publish_bundle("collision", {"loser.txt": b"loser"})
    assert (destination / "winner.txt").read_bytes() == b"winner"
    assert not (destination / "loser.txt").exists()
