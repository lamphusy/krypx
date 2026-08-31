"""Immutable exact-byte storage and publication tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from crypto_ai.exceptions import (
    PublicationCollisionError,
    SentimentStorageError,
    StorageIntegrityError,
)
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.storage import ContentAddressedStore


def assert_publication_rejected_without_blocking(root: Path, publication_id: str) -> str:
    probe = """
import sys
from pathlib import Path

from crypto_ai.exceptions import SentimentStorageError
from crypto_ai.sentiment.storage import ContentAddressedStore

try:
    ContentAddressedStore(Path(sys.argv[1])).verify_publication(sys.argv[2])
except SentimentStorageError as exc:
    print(f"{type(exc).__name__}: {exc}")
else:
    raise SystemExit("invalid publication storage object was accepted")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(root), publication_id],
        capture_output=True,
        check=False,
        text=True,
        timeout=1.0,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("SentimentStorageError:")
    return completed.stdout


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


def test_get_bytes_returns_the_verified_open_file_during_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    original = b"verified original"
    digest = store.put_bytes(original)
    object_path = store.objects_root / digest[:2] / digest
    replacement_path = object_path.with_name("replacement")
    replacement_path.write_bytes(b"unverified replacement")
    real_open = os.open
    replaced = False

    def racing_open(
        path: str | os.PathLike[str],
        flags: int,
        *args: object,
        **kwargs: int,
    ) -> int:
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == digest and not replaced:
            replaced = True
            os.replace(replacement_path, object_path)
        return descriptor

    monkeypatch.setattr(storage_module.os, "open", racing_open)

    assert store.get_bytes(digest) == original
    assert object_path.read_bytes() == b"unverified replacement"


def test_get_bytes_never_performs_a_second_path_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    original = b"single captured buffer"
    digest = store.put_bytes(original)

    def reject_second_read(path: Path) -> bytes:
        raise AssertionError("an unverified second path read was attempted")

    monkeypatch.setattr(Path, "read_bytes", reject_second_read)
    assert store.get_bytes(digest) == original


def test_get_bytes_rejects_symlink_disappearance_and_read_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    digest = store.put_bytes(b"protected")
    object_path = store.objects_root / digest[:2] / digest
    target = tmp_path / "target"
    target.write_bytes(b"protected")
    object_path.unlink()
    object_path.symlink_to(target)
    with pytest.raises(SentimentStorageError):
        store.get_bytes(digest)

    object_path.unlink()
    with pytest.raises(SentimentStorageError, match="missing"):
        store.get_bytes(digest)

    object_path.write_bytes(b"protected")

    def fail_fdopen(*args: object, **kwargs: object) -> object:
        raise OSError("simulated read failure")

    monkeypatch.setattr(storage_module.os, "fdopen", fail_fdopen)
    with pytest.raises(SentimentStorageError, match="simulated read failure"):
        store.get_bytes(digest)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_get_bytes_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    digest = store.put_bytes(b"replace with named pipe")
    object_path = store.objects_root / digest[:2] / digest
    object_path.unlink()
    os.mkfifo(object_path)

    probe = """
import sys
from pathlib import Path

from crypto_ai.exceptions import SentimentStorageError
from crypto_ai.sentiment.storage import ContentAddressedStore

try:
    ContentAddressedStore(Path(sys.argv[1])).get_bytes(sys.argv[2])
except SentimentStorageError as exc:
    print(f"{type(exc).__name__}: {exc}")
else:
    raise SystemExit("FIFO was accepted as a content object")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(tmp_path), digest],
        capture_output=True,
        check=False,
        text=True,
        timeout=1.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("SentimentStorageError:")
    assert "not a regular file" in completed.stdout


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
@pytest.mark.parametrize("ancestor", ["objects", "sha256", "bucket"])
def test_get_bytes_rejects_symlinked_object_ancestor(tmp_path: Path, ancestor: str) -> None:
    store = ContentAddressedStore(tmp_path)
    digest = store.put_bytes(b"protected by anchored traversal")
    path = {
        "objects": store.root / "objects",
        "sha256": store.objects_root,
        "bucket": store.objects_root / digest[:2],
    }[ancestor]
    relocated = tmp_path / f"relocated-{ancestor}"
    path.rename(relocated)
    path.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.get_bytes(digest)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
@pytest.mark.parametrize("ancestor", ["objects", "sha256", "bucket", "staging"])
def test_put_bytes_rejects_symlinked_ancestor_without_external_write(
    tmp_path: Path, ancestor: str
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    data = b"must never cross the trusted store boundary"
    digest = sha256_bytes(data)
    external = tmp_path / f"external-{ancestor}"
    external.mkdir()
    path = {
        "objects": store.root / "objects",
        "sha256": store.objects_root,
        "bucket": store.objects_root / digest[:2],
        "staging": store.staging_root,
    }[ancestor]
    if path.exists():
        path.rename(tmp_path / f"relocated-write-{ancestor}")
    path.symlink_to(external, target_is_directory=True)

    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.put_bytes(data)

    assert list(external.iterdir()) == []


def test_replacement_race_accepts_identical_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    data = b"racing writer"
    digest = sha256_bytes(data)
    original_link = os.link

    def competing_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        assert source
        assert src_dir_fd >= 0
        assert follow_symlinks is False
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise FileExistsError

    monkeypatch.setattr(storage_module.os, "link", competing_link)
    assert store.put_bytes(data) == digest
    monkeypatch.setattr(storage_module.os, "link", original_link)
    assert store.get_bytes(digest) == data


def test_replacement_race_rejects_conflicting_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)

    def competing_link(
        source: str,
        destination: str,
        *,
        src_dir_fd: int,
        dst_dir_fd: int,
        follow_symlinks: bool,
    ) -> None:
        assert source
        assert src_dir_fd >= 0
        assert follow_symlinks is False
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(descriptor, b"conflict")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        raise FileExistsError

    monkeypatch.setattr(storage_module.os, "link", competing_link)
    with pytest.raises(SentimentStorageError, match="hash mismatch"):
        store.put_bytes(b"intended")


def test_incomplete_object_write_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)

    def fail_write(parent_descriptor: int, name: str, data: bytes) -> None:
        assert parent_descriptor >= 0
        assert name
        assert data
        raise SentimentStorageError("simulated incomplete write")

    monkeypatch.setattr(storage_module, "_write_fsynced_at", fail_write)
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
    captured = store.read_publication("fixture-run-001")
    assert captured.files == {
        "articles.jsonl": b'{"fixture":true}\n',
        "nested/raw.bin": b"raw",
    }
    with pytest.raises(PublicationCollisionError):
        store.publish_bundle("fixture-run-001", {"articles.jsonl": b"replacement"})
    assert (published / "articles.jsonl").read_bytes() == b'{"fixture":true}\n'


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
@pytest.mark.parametrize("ancestor", ["publications", "publication"])
def test_publication_rejects_symlinked_publication_ancestor(tmp_path: Path, ancestor: str) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("symlinked-publication", {"payload.bin": b"payload"})
    path = store.publications_root if ancestor == "publications" else published
    relocated = tmp_path / f"relocated-{ancestor}"
    path.rename(relocated)
    path.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.verify_publication("symlinked-publication")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_publish_bundle_rejects_symlinked_parent_without_external_write(
    tmp_path: Path,
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    external = tmp_path / "external-publications"
    external.mkdir()
    store.publications_root.rename(tmp_path / "relocated-publications-write")
    store.publications_root.symlink_to(external, target_is_directory=True)

    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.publish_bundle("must-not-escape", {"payload.bin": b"payload"})

    assert list(external.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_replaced_store_root_rejects_reads_and_writes_without_external_effects(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = ContentAddressedStore(root)
    digest = store.put_bytes(b"trusted-root-object")
    relocated = tmp_path / "relocated-store"
    external = tmp_path / "external-root"
    external.mkdir()
    root.rename(relocated)
    root.symlink_to(external, target_is_directory=True)

    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.get_bytes(digest)
    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.put_bytes(b"must-not-escape")
    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.publish_bundle("must-not-escape", {"payload.bin": b"payload"})

    assert list(external.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_constructor_rejects_symlinked_store_root_without_external_effects(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-constructor-root"
    external.mkdir()
    root = tmp_path / "symlinked-store"
    root.symlink_to(external, target_is_directory=True)

    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        ContentAddressedStore(root)

    assert list(external.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_constructor_rejects_immediate_and_higher_configured_root_ancestor_symlinks(
    tmp_path: Path,
) -> None:
    for depth in ("immediate", "higher"):
        case = tmp_path / f"constructor-{depth}"
        case.mkdir()
        external = case / "external"
        external.mkdir()
        configured_parent = case / "configured-parent"
        configured_parent.symlink_to(external, target_is_directory=True)
        suffix = ("store",) if depth == "immediate" else ("nested", "store")
        root = configured_parent.joinpath(*suffix)

        with pytest.raises(SentimentStorageError, match="must not be a symlink"):
            ContentAddressedStore(root)

        assert list(external.iterdir()) == []


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_replaced_configured_root_ancestors_reject_all_operations(
    tmp_path: Path,
) -> None:
    for depth in ("immediate", "higher"):
        case = tmp_path / f"replacement-{depth}"
        case.mkdir()
        configured_parent = case / "configured-parent"
        suffix = ("store",) if depth == "immediate" else ("nested", "store")
        root = configured_parent.joinpath(*suffix)
        store = ContentAddressedStore(root)
        digest = store.put_bytes(b"trusted-before-ancestor-replacement")
        external = case / "external"
        external.mkdir()
        relocated_parent = external / "relocated-parent"
        configured_parent.rename(relocated_parent)
        configured_parent.symlink_to(relocated_parent, target_is_directory=True)
        escaped_digest = sha256_bytes(b"must-not-cross-ancestor-symlink")

        with pytest.raises(SentimentStorageError, match="must not be a symlink"):
            store.get_bytes(digest)
        with pytest.raises(SentimentStorageError, match="must not be a symlink"):
            store.put_bytes(b"must-not-cross-ancestor-symlink")
        with pytest.raises(SentimentStorageError, match="must not be a symlink"):
            store.publish_bundle("must-not-cross-ancestor-symlink", {"payload": b"payload"})

        relocated_store = relocated_parent.joinpath(*suffix)
        assert not (
            relocated_store / "objects" / "sha256" / escaped_digest[:2] / escaped_digest
        ).exists()
        assert not (relocated_store / "publications" / "must-not-cross-ancestor-symlink").exists()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_publication_rejects_symlinked_nested_member_parent(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle(
        "symlinked-member-parent",
        {"nested/payload.bin": b"payload"},
    )
    nested = published / "nested"
    relocated = tmp_path / "relocated-nested-directory"
    nested.rename(relocated)
    nested.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(SentimentStorageError, match="must not be a symlink"):
        store.verify_publication("symlinked-member-parent")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_publication_rejects_unmanifested_fifo_without_blocking(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("unmanifested-fifo", {"payload.bin": b"payload"})
    os.mkfifo(published / "unexpected.fifo")

    output = assert_publication_rejected_without_blocking(tmp_path, "unmanifested-fifo")
    assert "not a regular file or directory" in output


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_publication_rejects_manifested_fifo_without_blocking(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("manifested-fifo", {"payload.bin": b"payload"})
    payload = published / "payload.bin"
    payload.unlink()
    os.mkfifo(payload)

    output = assert_publication_rejected_without_blocking(tmp_path, "manifested-fifo")
    assert "not a regular file or directory" in output


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes are unavailable")
def test_publication_rejects_fifo_injected_during_payload_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ContentAddressedStore(tmp_path)
    publication = store.publish_bundle("late-fifo", {"payload.bin": b"payload"})
    read_regular_file = storage_module._read_regular_file_at_once
    injected = False

    def inject_fifo_before_payload_read(
        parent_descriptor: int,
        name: str,
        *,
        description: str,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal injected
        if name == "payload.bin" and not injected:
            os.mkfifo(publication / "late.fifo")
            injected = True
        return read_regular_file(parent_descriptor, name, description=description)

    monkeypatch.setattr(
        storage_module,
        "_read_regular_file_at_once",
        inject_fifo_before_payload_read,
    )

    with pytest.raises(StorageIntegrityError, match="changed during verification"):
        store.verify_publication("late-fifo")
    assert injected is True
    assert (publication / "late.fifo").is_fifo()


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks are unavailable")
def test_publication_rejects_directory_replaced_during_payload_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    publication = store.publish_bundle(
        "late-directory-symlink",
        {"nested/payload.bin": b"payload"},
    )
    nested = publication / "nested"
    relocated = tmp_path / "relocated-nested"
    external = tmp_path / "external"
    external.mkdir()
    read_regular_file = storage_module._read_regular_file_at_once
    injected = False

    def replace_directory_before_payload_read(
        parent_descriptor: int,
        name: str,
        *,
        description: str,
    ) -> tuple[bytes, os.stat_result]:
        nonlocal injected
        if name == "payload.bin" and not injected:
            nested.rename(relocated)
            nested.symlink_to(external, target_is_directory=True)
            injected = True
        return read_regular_file(parent_descriptor, name, description=description)

    monkeypatch.setattr(
        storage_module,
        "_read_regular_file_at_once",
        replace_directory_before_payload_read,
    )

    with pytest.raises(StorageIntegrityError, match="changed during verification"):
        store.verify_publication("late-directory-symlink")
    assert injected is True
    assert nested.is_symlink()
    assert list(external.iterdir()) == []


def test_publication_rejects_unmanifested_empty_directory(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("unexpected-directory", {"payload.bin": b"payload"})
    (published / "unexpected-empty-directory").mkdir()

    with pytest.raises(SentimentStorageError, match="unmanifested or missing"):
        store.verify_publication("unexpected-directory")


def test_publication_rejects_unmanifested_regular_file(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("unexpected-file", {"payload.bin": b"payload"})
    (published / "unexpected.bin").write_bytes(b"not in manifest")

    with pytest.raises(SentimentStorageError, match="unmanifested or missing"):
        store.verify_publication("unexpected-file")


def test_publication_rejects_unknown_outer_schema(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("unknown-schema", {"payload.bin": b"payload"})
    manifest_path = published / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["schema_version"] = "unknown-publication-v999"
    manifest_path.write_bytes(canonicalize(manifest))

    with pytest.raises(SentimentStorageError, match="unsupported.*schema"):
        store.verify_publication("unknown-schema")


def test_publication_rejects_unexpected_outer_fields(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("extra-field", {"payload.bin": b"payload"})
    manifest_path = published / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["unexpected_field"] = "must fail closed"
    manifest_path.write_bytes(canonicalize(manifest))

    with pytest.raises(SentimentStorageError, match="unexpected fields"):
        store.verify_publication("extra-field")


def test_publication_rejects_non_object_metadata(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path)
    published = store.publish_bundle("bad-metadata", {"payload.bin": b"payload"})
    manifest_path = published / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["metadata"] = ["not", "an", "object"]
    manifest_path.write_bytes(canonicalize(manifest))

    with pytest.raises(SentimentStorageError, match="metadata must be an object"):
        store.verify_publication("bad-metadata")


def test_incomplete_bundle_is_cleaned_and_never_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    original_write = storage_module._write_fsynced_at

    def fail_manifest(parent_descriptor: int, name: str, data: bytes) -> None:
        if name == "manifest.json":
            raise SentimentStorageError("simulated manifest failure")
        original_write(parent_descriptor, name, data)

    monkeypatch.setattr(storage_module, "_write_fsynced_at", fail_manifest)
    with pytest.raises(SentimentStorageError, match="manifest failure"):
        store.publish_bundle("incomplete", {"payload.bin": b"payload"})
    assert not (store.publications_root / "incomplete").exists()
    assert not list(store.publications_root.glob(".staging-*"))


def test_publication_collision_keeps_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ContentAddressedStore(tmp_path)
    destination = store.publications_root / "collision"

    def race(parent_descriptor: int, source_name: str, destination_name: str) -> None:
        assert source_name.startswith(".staging-")
        assert destination_name == "collision"
        os.mkdir(destination_name, dir_fd=parent_descriptor)
        destination_descriptor = os.open(
            destination_name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            dir_fd=parent_descriptor,
        )
        try:
            winner_descriptor = os.open(
                "winner.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=destination_descriptor,
            )
            try:
                os.write(winner_descriptor, b"winner")
                os.fsync(winner_descriptor)
            finally:
                os.close(winner_descriptor)
        finally:
            os.close(destination_descriptor)
        raise PublicationCollisionError("simulated publication collision")

    monkeypatch.setattr(storage_module, "_atomic_rename_directory_no_replace", race)
    with pytest.raises(PublicationCollisionError):
        store.publish_bundle("collision", {"loser.txt": b"loser"})
    assert (destination / "winner.txt").read_bytes() == b"winner"
    assert not (destination / "loser.txt").exists()
