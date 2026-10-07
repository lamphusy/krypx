"""Offline adversarial filesystem tests for the production registry primitive."""

from __future__ import annotations

import fcntl
import json
import os
import socket
import tempfile
from pathlib import Path

import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2 import production_store as module
from crypto_ai.phase2.production_store import (
    ProductionCollisionError,
    ProductionError,
    ProductionIntegrityError,
    ProductionStore,
)
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("production store regression tests must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "versions"
    files = {
        "model.json": b'{"synthetic_model":true}',
        "feature_columns.json": b'["synthetic_feature"]',
        "prepared_dataset_manifest.json": b'{"synthetic":true}',
        "evidence/nested/authorization.json": b'{"authorized":true}',
    }
    metadata = {"synthetic": True, "nested": {"marker": "fixed"}}
    store = ProductionStore(root)

    def verify(manifest, payloads):
        assert manifest["metadata"] == metadata
        assert payloads == files

    return store, files, metadata, verify


def test_nested_publication_roundtrip_and_defensive_immutable_views(case):
    store, files, metadata, verify = case
    artifact = store.publish("version-1", files, metadata, verify=verify)
    assert artifact == store.get("version-1", verify=verify)
    assert artifact.manifest["production_artifact_hashes"] == {
        name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)} for name, raw in files.items()
    }
    artifact.manifest["metadata"]["nested"]["marker"] = "changed"
    artifact.files["model.json"] = b"changed"
    assert artifact.manifest["metadata"] == metadata
    assert artifact.files == files
    assert not (store.root / "active_model.json").exists()


@pytest.mark.parametrize("version", ["", ".", "..", ".hidden", "a/b", "a\\b", "a\x00b", 1, True])
def test_unsafe_version_rejected_before_writes(case, version):
    store, files, metadata, verify = case
    with pytest.raises(ProductionError):
        store.publish(version, files, metadata, verify=verify)
    assert not list(store.root.iterdir())


@pytest.mark.parametrize("existing", ["empty", "file", "fifo", "symlink", "completed"])
def test_all_occupied_version_names_preserved_without_semantic_work(case, existing):
    store, files, metadata, verify = case
    target = store.root / "v1"
    if existing == "empty":
        target.mkdir()
    elif existing == "file":
        target.write_bytes(b"unrelated")
    elif existing == "fifo":
        os.mkfifo(target)
    elif existing == "symlink":
        target.symlink_to(store.root.parent)
    else:
        store.publish("v1", files, metadata, verify=verify)
    original = target.lstat()

    def forbidden(*args):
        raise AssertionError("occupied versions must reject before verification")

    with pytest.raises(ProductionCollisionError):
        store.publish("v1", files, metadata, verify=forbidden)
    with pytest.raises(ProductionCollisionError):
        store.assert_available("v1")
    assert target.lstat().st_ino == original.st_ino
    if existing == "completed":
        assert store.get("v1", verify=verify).files == files


def test_missing_semantic_verifier_never_silently_skips_checks(case):
    store, files, metadata, verify = case
    with pytest.raises(ProductionError, match="verifier"):
        store.publish("v1", files, metadata)
    store.publish("v1", files, metadata, verify=verify)
    with pytest.raises(ProductionError, match="verifier"):
        store.get("v1")


def test_version_flock_is_nonblocking_exclusive_and_not_deleted(case):
    store, files, metadata, verify = case
    lock = store.root / ".lock-v1"
    with lock.open("wb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ProductionCollisionError, match="locked"):
            store.publish("v1", files, metadata, verify=verify)
    assert lock.is_file()
    assert not (store.root / "v1").exists()
    store.publish("v1", files, metadata, verify=verify)


@pytest.mark.parametrize("kind", ["fifo", "symlink", "directory", "hardlink"])
def test_unsafe_version_lock_rejected_without_blocking(case, kind):
    store, files, metadata, verify = case
    lock = store.root / ".lock-v1"
    if kind == "fifo":
        os.mkfifo(lock)
    elif kind == "directory":
        lock.mkdir()
    elif kind == "symlink":
        lock.symlink_to(store.root.parent / "unrelated")
    else:
        other = store.root.parent / "other"
        other.write_bytes(b"")
        os.link(other, lock)
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert not (store.root / "v1").exists()


def test_root_and_parent_symlinks_rejected(case):
    store, files, metadata, verify = case
    alias = store.root.parent / "alias"
    alias.symlink_to(store.root, target_is_directory=True)
    with pytest.raises(ProductionError):
        ProductionStore(alias)
    with pytest.raises(ProductionError):
        ProductionStore(alias / "subdirectory")
    displaced = store.root.parent / "displaced"
    store.root.rename(displaced)
    store.root.mkdir()
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)


@pytest.mark.parametrize(
    "name", ["../escape", "/escape", ".hidden", "manifest.json", "x/manifest.json", "a/../b"]
)
def test_invalid_payload_paths_rejected(case, name):
    store, files, metadata, verify = case
    with pytest.raises(ProductionError):
        store.publish("v1", {**files, name: b"x"}, metadata, verify=verify)


def test_manifest_completion_is_atomic_after_payload_and_directory_fsync(case, monkeypatch):
    store, files, metadata, verify = case
    events = []
    write = module._write_fsynced_at
    sync = module._fsync_tree_directories_at
    sync_one = module._fsync_directory_descriptor
    rename = module._atomic_rename_directory_no_replace

    def writing(descriptor, name, raw):
        events.append(("write", name))
        write(descriptor, name, raw)

    def syncing(descriptor, **kwargs):
        events.append(("directory-fsync", None))
        sync(descriptor, **kwargs)

    def syncing_one(descriptor, *, description):
        if description == "production version reservation":
            events.append(("reservation-fsync", None))
        sync_one(descriptor, description=description)

    def renaming(descriptor, source, destination):
        assert source.startswith(".pending-manifest-")
        assert destination == "manifest.json"
        assert (store.root / "v1").is_dir()
        assert not (store.root / "v1" / "manifest.json").exists()
        events.append(("rename", None))
        rename(descriptor, source, destination)

    monkeypatch.setattr(module, "_write_fsynced_at", writing)
    monkeypatch.setattr(module, "_fsync_tree_directories_at", syncing)
    monkeypatch.setattr(module, "_fsync_directory_descriptor", syncing_one)
    monkeypatch.setattr(module, "_atomic_rename_directory_no_replace", renaming)
    store.publish("v1", files, metadata, verify=verify)
    assert [name for event, name in events if event == "write"][-1].startswith(".pending-manifest-")
    assert events.index(("reservation-fsync", None)) < next(
        index for index, event in enumerate(events) if event[0] == "write"
    )
    assert events.index(("directory-fsync", None)) < events.index(("rename", None))


def test_caller_metadata_mutation_during_writes_fails_closed(case, monkeypatch):
    store, files, metadata, verify = case
    original = module._write_fsynced_at

    def mutate(descriptor, name, raw):
        original(descriptor, name, raw)
        metadata["nested"]["marker"] = "mutated"

    monkeypatch.setattr(module, "_write_fsynced_at", mutate)
    with pytest.raises(ProductionIntegrityError, match="metadata mutated"):
        store.publish("v1", files, metadata, verify=verify)
    assert not (store.root / "v1" / "manifest.json").exists()
    with pytest.raises(ProductionError):
        store.get("v1", verify=verify)


def test_forged_pending_manifest_exact_bytes_rejected(case, monkeypatch):
    store, files, metadata, verify = case
    original = module._write_fsynced_at

    def forge(descriptor, name, raw):
        if name.startswith(".pending-manifest-"):
            parsed = json.loads(raw)
            parsed["metadata"]["nested"]["marker"] = "forged"
            raw = canonicalize(parsed)
        original(descriptor, name, raw)

    monkeypatch.setattr(module, "_write_fsynced_at", forge)
    with pytest.raises(ProductionIntegrityError, match="pending production manifest bytes changed"):
        store.publish("v1", files, metadata, verify=verify)
    assert not (store.root / "v1" / "manifest.json").exists()
    with pytest.raises(ProductionError):
        store.get("v1", verify=verify)


def test_post_rename_corruption_invalidates_owned_completion_marker(case, monkeypatch):
    store, files, metadata, verify = case
    original = module._atomic_rename_directory_no_replace

    def corrupt(descriptor, source, destination):
        original(descriptor, source, destination)
        assert destination == "manifest.json"
        (store.root / "v1" / "model.json").write_bytes(b"corrupted")

    monkeypatch.setattr(module, "_atomic_rename_directory_no_replace", corrupt)
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert not (store.root / "v1" / "manifest.json").exists()
    with pytest.raises(ProductionError):
        store.get("v1", verify=verify)


def test_rollback_unlink_failure_retires_completion_through_owned_descriptor(case, monkeypatch):
    store, files, metadata, verify = case
    original = module._atomic_rename_directory_no_replace
    unlink = os.unlink
    attempts = []

    def corrupt(descriptor, source, destination):
        original(descriptor, source, destination)
        if destination == "manifest.json":
            (store.root / "v1" / "model.json").write_bytes(b"corrupted")

    def deny_manifest(name, *args, **kwargs):
        if name == "manifest.json":
            attempts.append(name)
            raise OSError("injected manifest unlink failure")
        return unlink(name, *args, **kwargs)

    monkeypatch.setattr(module, "_atomic_rename_directory_no_replace", corrupt)
    monkeypatch.setattr(os, "unlink", deny_manifest)
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert attempts
    assert not (store.root / "v1" / "manifest.json").exists()


def test_rollback_preserves_concurrent_replacement_directory(case, monkeypatch):
    store, files, metadata, verify = case
    original = module._atomic_rename_directory_no_replace
    displaced = store.root / "owned-displaced"
    replacement = store.root / "v1"

    def swap(descriptor, source, destination):
        original(descriptor, source, destination)
        replacement.rename(displaced)
        replacement.mkdir()
        (replacement / "manifest.json").write_bytes(b"unrelated manifest")
        (replacement / "unrelated").write_bytes(b"preserve")
        raise OSError("concurrent directory replacement")

    monkeypatch.setattr(module, "_atomic_rename_directory_no_replace", swap)
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert (replacement / "manifest.json").read_bytes() == b"unrelated manifest"
    assert (replacement / "unrelated").read_bytes() == b"preserve"
    assert not (displaced / "manifest.json").exists()


def test_forged_staging_name_cannot_become_a_published_version(case, monkeypatch):
    store, files, metadata, verify = case
    forged = store.root / ".staging-v1-attacker-controlled"
    forged.mkdir()
    (forged / "manifest.json").write_bytes(b"forged staging manifest")
    rename = module._atomic_rename_directory_no_replace

    def no_stage_publication(descriptor, source, destination):
        assert destination != "v1"
        rename(descriptor, source, destination)

    monkeypatch.setattr(module, "_atomic_rename_directory_no_replace", no_stage_publication)
    artifact = store.publish("v1", files, metadata, verify=verify)
    assert store.get("v1", verify=verify) == artifact
    assert (forged / "manifest.json").read_bytes() == b"forged staging manifest"


def test_post_reservation_public_swap_preserves_unrelated_replacement(case, monkeypatch):
    store, files, metadata, verify = case
    displaced = store.root / ".displaced-owned-version"
    replacement = store.root / "v1"
    write = module._write_fsynced_at
    injected = False

    def swap_after_first_owned_payload(descriptor, name, raw):
        nonlocal injected
        write(descriptor, name, raw)
        if name == "model.json" and replacement.is_dir() and not injected:
            injected = True
            replacement.rename(displaced)
            replacement.mkdir()
            (replacement / "manifest.json").write_bytes(b"unrelated manifest")
            (replacement / "unrelated").write_bytes(b"preserve")

    monkeypatch.setattr(module, "_write_fsynced_at", swap_after_first_owned_payload)
    with pytest.raises(ProductionError, match="directory was replaced"):
        store.publish("v1", files, metadata, verify=verify)
    assert injected
    assert (replacement / "manifest.json").read_bytes() == b"unrelated manifest"
    assert (replacement / "unrelated").read_bytes() == b"preserve"
    assert not (displaced / "manifest.json").exists()


def test_public_swap_between_payloads_aborts_before_next_write(case, monkeypatch):
    store, files, metadata, verify = case
    displaced = store.root / ".displaced-owned-version"
    replacement = store.root / "v1"
    read = module._read_regular_file_at_once
    write = module._write_fsynced_at
    payload_writes = []
    swapped = False

    def counting_write(descriptor, name, raw):
        if not name.startswith(".pending-manifest-"):
            payload_writes.append(name)
        write(descriptor, name, raw)

    def swap_after_readback(descriptor, name, **kwargs):
        nonlocal swapped
        result = read(descriptor, name, **kwargs)
        if name == "authorization.json" and not swapped:
            swapped = True
            replacement.rename(displaced)
            replacement.mkdir()
            (replacement / "manifest.json").write_bytes(b"unrelated manifest")
            (replacement / "unrelated").write_bytes(b"preserve")
        return result

    monkeypatch.setattr(module, "_write_fsynced_at", counting_write)
    monkeypatch.setattr(module, "_read_regular_file_at_once", swap_after_readback)
    with pytest.raises(ProductionError, match="directory was replaced"):
        store.publish("v1", files, metadata, verify=verify)
    assert swapped
    assert payload_writes == ["authorization.json"]
    assert (replacement / "manifest.json").read_bytes() == b"unrelated manifest"
    assert (replacement / "unrelated").read_bytes() == b"preserve"
    assert not (displaced / "manifest.json").exists()


def test_reader_cannot_observe_completion_before_manifest_publish(case, monkeypatch):
    store, files, metadata, verify = case
    write = module._write_fsynced_at
    checked = False

    def inspect_before_completion(descriptor, name, raw):
        nonlocal checked
        if name.startswith(".pending-manifest-"):
            with pytest.raises(ProductionError):
                store.get("v1", verify=verify)
            checked = True
        write(descriptor, name, raw)

    monkeypatch.setattr(module, "_write_fsynced_at", inspect_before_completion)
    artifact = store.publish("v1", files, metadata, verify=verify)
    assert checked
    assert store.get("v1", verify=verify) == artifact


def test_failing_semantic_replay_after_rename_invalidates_publication(case):
    store, files, metadata, verify = case
    calls = 0

    def replay(manifest, payloads):
        nonlocal calls
        calls += 1
        verify(manifest, payloads)
        if calls == 2:
            raise ProductionIntegrityError("changed source evidence")

    with pytest.raises(ProductionError, match="source evidence"):
        store.publish("v1", files, metadata, verify=replay)
    assert calls == 2
    assert not (store.root / "v1" / "manifest.json").exists()


@pytest.mark.parametrize("kind", ["extra", "fifo", "symlink", "changed", "missing"])
def test_retrieval_rejects_inventory_and_byte_corruption(case, kind):
    store, files, metadata, verify = case
    store.publish("v1", files, metadata, verify=verify)
    target = store.root / "v1"
    if kind == "extra":
        (target / "extra").write_bytes(b"unexpected")
    elif kind == "fifo":
        os.mkfifo(target / "evidence" / "nested" / "injected")
    elif kind == "symlink":
        (target / "model.json").unlink()
        (target / "model.json").symlink_to(store.root.parent / "unrelated")
    elif kind == "changed":
        (target / "model.json").write_bytes(b"changed")
    else:
        (target / "model.json").unlink()
    with pytest.raises(ProductionError):
        store.get("v1", verify=verify)


def test_invalid_outer_metadata_stops_before_any_payload_read(case, monkeypatch):
    store, files, metadata, verify = case
    store.publish("v1", files, metadata, verify=verify)
    path = store.root / "v1" / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["metadata"] = "invalid"
    path.write_bytes(canonicalize(manifest))
    read = module._read_regular_file_at_once
    opened = []

    def sentinel(descriptor, name, **kwargs):
        opened.append(name)
        return read(descriptor, name, **kwargs)

    monkeypatch.setattr(module, "_read_regular_file_at_once", sentinel)
    with pytest.raises(ProductionError):
        store.get("v1", verify=verify)
    assert opened == ["manifest.json"]


def test_semantic_retrieval_mutation_detected_after_verifier_returns(case):
    store, files, metadata, verify = case
    store.publish("v1", files, metadata, verify=verify)

    def mutate(manifest, payloads):
        verify(manifest, payloads)
        (store.root / "v1" / "model.json").write_bytes(b"mutated")

    with pytest.raises(ProductionError):
        store.get("v1", verify=mutate)


def test_assertion_and_integrity_error_hierarchy(case):
    assert issubclass(ProductionError, CryptoAIError)
    assert issubclass(ProductionIntegrityError, ProductionError)
    assert issubclass(ProductionCollisionError, ProductionIntegrityError)


def test_active_registry_sentinel_remains_byte_identical(case):
    store, files, metadata, verify = case
    active = store.root.parent / "active_model.json"
    active.write_bytes(b"human-managed activation")
    store.publish("v1", files, metadata, verify=verify)
    store.get("v1", verify=verify)
    assert active.read_bytes() == b"human-managed activation"


def test_payload_write_failure_leaves_only_incomplete_reserved_tombstone(case, monkeypatch):
    store, files, metadata, verify = case

    def failed(*args, **kwargs):
        raise OSError("injected disk write failure")

    monkeypatch.setattr(module, "_write_fsynced_at", failed)
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert (store.root / "v1").is_dir()
    assert not (store.root / "v1" / "manifest.json").exists()
    assert not list(store.root.glob(".staging-*"))
    with pytest.raises(ProductionError):
        store.get("v1", verify=verify)


def test_lock_inode_replacement_before_rename_fails_closed(case, monkeypatch):
    store, files, metadata, verify = case
    write = module._write_fsynced_at

    def replaced(descriptor, name, raw):
        write(descriptor, name, raw)
        if name.startswith(".pending-manifest-"):
            (store.root / ".lock-v1").rename(store.root / ".displaced-lock")
            (store.root / ".lock-v1").write_bytes(b"replacement")

    monkeypatch.setattr(module, "_write_fsynced_at", replaced)
    with pytest.raises(ProductionError, match="lock was replaced"):
        store.publish("v1", files, metadata, verify=verify)
    assert not (store.root / "v1" / "manifest.json").exists()


def test_atomic_rename_collision_race_preserves_concurrent_empty_directory(case, monkeypatch):
    store, files, metadata, verify = case
    mkdir = os.mkdir
    injected = False

    def race(name, *args, **kwargs):
        nonlocal injected
        if name == "v1" and kwargs.get("dir_fd") is not None and not injected:
            injected = True
            mkdir(name, *args, **kwargs)
        return mkdir(name, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", race)
    with pytest.raises(ProductionCollisionError):
        store.publish("v1", files, metadata, verify=verify)
    assert list((store.root / "v1").iterdir()) == []


def test_nested_fifo_injected_during_capture_rejected(case, monkeypatch):
    store, files, metadata, verify = case
    store.publish("v1", files, metadata, verify=verify)
    from crypto_ai.sentiment import storage

    read = storage._read_regular_file_at_once
    injected = False

    def mutate(descriptor, name, **kwargs):
        nonlocal injected
        result = read(descriptor, name, **kwargs)
        if not injected and name == "model.json":
            injected = True
            os.mkfifo(store.root / "v1" / "evidence" / "nested" / "late-fifo")
        return result

    monkeypatch.setattr(storage, "_read_regular_file_at_once", mutate)
    with pytest.raises(ProductionError):
        store.get("v1", verify=verify)
    assert injected


@pytest.mark.parametrize("root", [Path("/"), Path.cwd(), Path("relative"), "not-a-path"])
def test_non_temporary_or_relative_roots_rejected_without_creation(root):
    with pytest.raises(ProductionError):
        ProductionStore(root)


def test_shared_temporary_root_cannot_be_used_as_registry():
    with pytest.raises(ProductionError):
        ProductionStore(Path(tempfile.gettempdir()))


def test_post_rename_parent_fsync_failure_invalidates_completion(case, monkeypatch):
    store, files, metadata, verify = case
    sync = module._fsync_directory_descriptor

    def fail_after_publication(descriptor, *, description):
        if description == "production versions" and (store.root / "v1").exists():
            raise OSError("injected post-rename durability failure")
        return sync(descriptor, description=description)

    monkeypatch.setattr(module, "_fsync_directory_descriptor", fail_after_publication)
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert not (store.root / "v1" / "manifest.json").exists()


def test_rollback_rename_failure_retries_owned_manifest_unlink(case, monkeypatch):
    store, files, metadata, verify = case
    rename = module._atomic_rename_directory_no_replace
    unlink = os.unlink
    attempts = 0

    def fail_retirement(descriptor, source, destination):
        if source == "manifest.json":
            raise OSError("injected retirement rename failure")
        rename(descriptor, source, destination)
        (store.root / destination / "model.json").write_bytes(b"corrupted")

    def fail_once(name, *args, **kwargs):
        nonlocal attempts
        if name == "manifest.json":
            attempts += 1
            if attempts == 1:
                raise OSError("injected initial unlink failure")
        return unlink(name, *args, **kwargs)

    monkeypatch.setattr(module, "_atomic_rename_directory_no_replace", fail_retirement)
    monkeypatch.setattr(os, "unlink", fail_once)
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert attempts == 2
    assert not (store.root / "v1" / "manifest.json").exists()


@pytest.mark.parametrize(
    "metadata",
    [{}, {"synthetic": False}, {"synthetic": 1}, None, {"synthetic": True, "value": float("inf")}],
)
def test_invalid_synthetic_metadata_rejected(case, metadata):
    store, files, _, verify = case
    with pytest.raises(ProductionError):
        store.publish("v1", files, metadata, verify=verify)
    assert not list(store.root.iterdir())


def test_semantic_verifier_called_for_every_retrieval(case):
    store, files, metadata, verify = case
    store.publish("v1", files, metadata, verify=verify)

    def rejected(*args):
        raise ProductionIntegrityError("trusted parent no longer verifies")

    with pytest.raises(ProductionError, match="parent no longer verifies"):
        store.get("v1", verify=rejected)


def test_post_rename_manifest_only_mutation_rejected_exactly(case, monkeypatch):
    store, files, metadata, verify = case
    rename = module._atomic_rename_directory_no_replace

    def forge(descriptor, source, destination):
        rename(descriptor, source, destination)
        if destination == "manifest.json":
            path = store.root / "v1" / "manifest.json"
            manifest = json.loads(path.read_bytes())
            manifest["metadata"]["nested"]["marker"] = "forged"
            path.write_bytes(canonicalize(manifest))

    monkeypatch.setattr(module, "_atomic_rename_directory_no_replace", forge)
    with pytest.raises(ProductionError, match="manifest bytes changed"):
        store.publish("v1", files, metadata, verify=verify)
    assert not (store.root / "v1" / "manifest.json").exists()
