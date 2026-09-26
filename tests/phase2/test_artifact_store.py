"""Adversarial offline tests for immutable Milestone 7 report publication."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

from crypto_ai.phase2 import artifacts as report_module
from crypto_ai.phase2.experiments import ExperimentStore, generate_run_id
from crypto_ai.sentiment import storage as storage_module
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.storage import ContentAddressedStore


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Milestone 7 publication tests must stay offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def report_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> report_module.ArtifactStore:
    parents = ContentAddressedStore(tmp_path / "parents")
    experiments = ExperimentStore(tmp_path / "runs", parents)
    monkeypatch.setattr(report_module, "_semantic_replay", lambda *_: None)
    return report_module.ArtifactStore(tmp_path / "runs", experiments)


@pytest.fixture
def files() -> dict[str, bytes]:
    documents = {
        name: canonicalize({"schema_version": "synthetic-report-test-v1", "synthetic": True})
        for name in report_module.JSON_PAYLOAD_NAMES
    }
    documents["development_report.md"] = (
        "# Synthetic development report\n"
        + "\n".join(
            f"{name}: {sha256_bytes(documents[name])}"
            for name in sorted(report_module.JSON_PAYLOAD_NAMES)
        )
        + "\n"
    ).encode()
    return documents


@pytest.fixture
def metadata() -> dict[str, object]:
    result: dict[str, object] = {name: "a" * 64 for name in report_module.HASH_FIELDS}
    result.update(
        {
            "specification_id": report_module.SPECIFICATION_ID,
            "synthetic": True,
            "source_run_id": generate_run_id(),
            "source_prediction_sha256": {cell: "b" * 64 for cell in "ABCD"},
        }
    )
    return result


def test_distinct_exact_inventory_and_manifest_last(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = []
    write = report_module._write_fsynced_at

    def observe_write(descriptor, name, raw):
        writes.append(name)
        return write(descriptor, name, raw)

    monkeypatch.setattr(report_module, "_write_fsynced_at", observe_write)
    run_id = report_store.publish(files, metadata)
    assert writes[-1] == "manifest.json"
    assert set(writes) == report_module.PAYLOAD_NAMES | {"manifest.json"}
    report = report_store.get(run_id)
    assert report.files == files
    assert report.manifest["metadata"] == metadata
    assert report.manifest["schema_version"] == report_module.SCHEMA
    assert run_id != metadata["source_run_id"]
    assert set((report_store.root / run_id).iterdir()) == {
        report_store.root / run_id / name for name in writes
    }


@pytest.mark.parametrize("replacement", [None, b"not-json", b'{"x":NaN}'])
def test_exact_payloads_and_canonical_json_required(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    replacement: bytes | None,
) -> None:
    candidate = dict(files)
    if replacement is None:
        del candidate["strategy_metrics.json"]
    else:
        candidate["strategy_metrics.json"] = replacement
    with pytest.raises(report_module.ArtifactError):
        report_store.publish(candidate, metadata)


def test_report_markdown_must_bind_json_bytes(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
) -> None:
    files["development_report.md"] = b"# Unbound synthetic report\n"
    with pytest.raises(report_module.ArtifactInputError):
        report_store.publish(files, metadata)


@pytest.mark.parametrize("corruption", ["non_string", "non_hex", "unexpected_key"])
def test_metadata_checked_before_any_non_manifest_payload_read(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    run_id = report_store.publish(files, metadata)
    path = report_store.root / run_id / "manifest.json"
    manifest = report_module._json(path.read_bytes())
    if corruption == "non_string":
        manifest["metadata"]["source_folds_sha256"] = 123
    elif corruption == "non_hex":
        manifest["metadata"]["source_folds_sha256"] = "Z" * 64
    else:
        manifest["metadata"]["unexpected"] = "forbidden"
    path.write_bytes(canonicalize(manifest))
    opened = []
    read = report_module._read_regular_file_at_once

    def observe_read(descriptor, name, **kwargs):
        opened.append(name)
        return read(descriptor, name, **kwargs)

    monkeypatch.setattr(report_module, "_read_regular_file_at_once", observe_read)
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)
    assert opened == ["manifest.json"]


def test_mutation_during_semantic_replay_rejected(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = report_store.publish(files, metadata)

    def mutate_during_replay(*args):
        target = report_store.root / run_id / "cost_sensitivity.json"
        target.write_bytes(target.read_bytes() + b" ")

    monkeypatch.setattr(report_module, "_semantic_replay", mutate_during_replay)
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)


def test_tampered_payload_and_unmanifested_entry_rejected(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
) -> None:
    run_id = report_store.publish(files, metadata)
    target = report_store.root / run_id / "strategy_metrics.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)
    target.write_bytes(files["strategy_metrics.json"])
    os.mkfifo(report_store.root / run_id / "unexpected.fifo")
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)


def test_late_fifo_injection_during_payload_capture_rejected(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = report_store.publish(files, metadata)
    injected = []
    read = storage_module._read_regular_file_at_once

    def inject_after_payload(descriptor, name, **kwargs):
        result = read(descriptor, name, **kwargs)
        if name == "strategy_metrics.json" and not injected:
            os.mkfifo(report_store.root / run_id / "late.fifo")
            injected.append(True)
        return result

    monkeypatch.setattr(storage_module, "_read_regular_file_at_once", inject_after_payload)
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)
    assert injected


def test_collision_preserves_existing_run_and_metadata(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
) -> None:
    run_id = report_store.publish(files, metadata)
    original = (report_store.root / run_id / "manifest.json").read_bytes()
    altered = dict(metadata, cost_config_sha256="f" * 64)
    with pytest.raises(report_module.ArtifactCollisionError):
        report_store.publish(files, altered, run_id=run_id)
    assert (report_store.root / run_id / "manifest.json").read_bytes() == original
    assert report_store.get(run_id).files == files


def test_existing_experiment_namespace_cannot_be_overwritten(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
) -> None:
    run_id = generate_run_id()
    existing = report_store.root / run_id
    existing.mkdir()
    sentinel = b"existing milestone-six manifest bytes"
    (existing / "manifest.json").write_bytes(sentinel)
    with pytest.raises(report_module.ArtifactCollisionError):
        report_store.publish(files, metadata, run_id=run_id)
    assert (existing / "manifest.json").read_bytes() == sentinel


def test_post_rename_corruption_invalidates_owned_manifest(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = generate_run_id()
    rename = report_module._atomic_rename_directory_no_replace

    def corrupt_then_rename(descriptor, source, target):
        (report_store.root / source / "strategy_metrics.json").write_bytes(b"corrupt")
        return rename(descriptor, source, target)

    monkeypatch.setattr(report_module, "_atomic_rename_directory_no_replace", corrupt_then_rename)
    with pytest.raises(report_module.ArtifactIntegrityError):
        report_store.publish(files, metadata, run_id=run_id)
    assert not (report_store.root / run_id / "manifest.json").exists()
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)


def test_post_rename_semantic_replay_failure_invalidates_owned_manifest(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = generate_run_id()
    calls = []

    def reject_final_replay(*args):
        calls.append(True)
        if len(calls) == 2:
            raise report_module.ArtifactIntegrityError("synthetic final replay mismatch")

    monkeypatch.setattr(report_module, "_semantic_replay", reject_final_replay)
    with pytest.raises(report_module.ArtifactIntegrityError, match="final replay mismatch"):
        report_store.publish(files, metadata, run_id=run_id)
    assert len(calls) == 2
    assert not (report_store.root / run_id / "manifest.json").exists()
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)


def test_failed_rename_after_success_invalidates_owned_manifest(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = generate_run_id()
    rename = report_module._atomic_rename_directory_no_replace

    def succeed_then_raise(descriptor, source, target):
        rename(descriptor, source, target)
        raise OSError("synthetic failure after successful rename")

    monkeypatch.setattr(report_module, "_atomic_rename_directory_no_replace", succeed_then_raise)
    with pytest.raises(report_module.ArtifactIntegrityError):
        report_store.publish(files, metadata, run_id=run_id)
    assert not (report_store.root / run_id / "manifest.json").exists()


def test_incomplete_staged_write_never_publishes_manifest(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write = report_module._write_fsynced_at

    def interrupt_payload(descriptor, name, raw):
        if name == "development_report.md":
            raise OSError("synthetic interrupted payload write")
        return write(descriptor, name, raw)

    monkeypatch.setattr(report_module, "_write_fsynced_at", interrupt_payload)
    run_id = generate_run_id()
    with pytest.raises(report_module.ArtifactIntegrityError):
        report_store.publish(files, metadata, run_id=run_id)
    assert not (report_store.root / run_id).exists()
    assert not any(
        path.name == "manifest.json" for path in report_store.root.rglob("manifest.json")
    )


def test_rollback_pinned_inode_preserves_concurrent_replacement(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = generate_run_id()
    unrelated = report_store.root / ".unrelated"
    unrelated.mkdir()
    (unrelated / "sentinel.txt").write_text("unrelated")
    unrelated_identity = (unrelated.stat().st_dev, unrelated.stat().st_ino)
    detached = report_store.root / ".detached-owned"
    rollback = report_module._rollback_owned_directory

    def fail_post_rename(*args, **kwargs):
        raise report_module.ArtifactIntegrityError("synthetic post-rename failure")

    def swap_then_rollback(descriptor):
        os.rename(report_store.root / run_id, detached)
        os.rename(unrelated, report_store.root / run_id)
        return rollback(descriptor)

    monkeypatch.setattr(report_store, "_check_attachment", fail_post_rename)
    monkeypatch.setattr(report_module, "_rollback_owned_directory", swap_then_rollback)
    with pytest.raises(report_module.ArtifactIntegrityError):
        report_store.publish(files, metadata, run_id=run_id)
    assert (report_store.root / run_id / "sentinel.txt").read_text() == "unrelated"
    assert (
        (report_store.root / run_id).stat().st_dev,
        (report_store.root / run_id).stat().st_ino,
    ) == unrelated_identity
    assert not (detached / "manifest.json").exists()


def test_symlinked_runs_root_rejected_after_initialization(
    report_store: report_module.ArtifactStore,
    files: dict[str, bytes],
    metadata: dict[str, object],
) -> None:
    run_id = report_store.publish(files, metadata)
    moved = report_store.root.with_name("runs-moved")
    report_store.root.rename(moved)
    report_store.root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(report_module.ArtifactError):
        report_store.get(run_id)
