"""Focused descriptor-ownership checks for synthetic evaluation unpublication."""

import os
from pathlib import Path

import pytest

from crypto_ai.phase2.evaluation_store import (
    MANIFEST_NAME,
    EvaluationArtifact,
    EvaluationIntegrityError,
    EvaluationStore,
)
from crypto_ai.sentiment.canonical import canonicalize


def _owned_store(tmp_path: Path) -> tuple[EvaluationStore, EvaluationArtifact, Path]:
    store = EvaluationStore(tmp_path / "evaluations")
    destination = store.root / "synthetic-run"
    destination.mkdir()
    (destination / MANIFEST_NAME).write_bytes(b"owned manifest")
    (destination / "owned.json").write_bytes(b"owned payload")
    artifact = EvaluationArtifact(manifest={"owner": "synthetic"}, files={})
    info = destination.stat()
    store._published_owners["synthetic-run"] = (
        (info.st_dev, info.st_ino),
        canonicalize(artifact.manifest),
        frozenset({"owned.json"}),
    )
    return store, artifact, destination


def test_completion_rollback_invalidates_only_pinned_inode(tmp_path: Path) -> None:
    store, artifact, destination = _owned_store(tmp_path)
    store.unpublish_owned("synthetic-run", artifact)
    assert destination.exists()
    assert list(destination.iterdir()) == []
    assert not (destination / MANIFEST_NAME).exists()


def test_completion_rollback_never_mutates_swapped_unrelated_run(tmp_path: Path) -> None:
    store, artifact, destination = _owned_store(tmp_path)
    detached = store.root / "detached-owned-run"
    destination.rename(detached)
    destination.mkdir()
    (destination / MANIFEST_NAME).write_bytes(b"unrelated manifest")
    (destination / "owned.json").write_bytes(b"unrelated payload")

    with pytest.raises(EvaluationIntegrityError, match="replaced"):
        store.unpublish_owned("synthetic-run", artifact)

    assert (destination / MANIFEST_NAME).read_bytes() == b"unrelated manifest"
    assert (destination / "owned.json").read_bytes() == b"unrelated payload"
    assert (detached / MANIFEST_NAME).read_bytes() == b"owned manifest"
    assert (detached / "owned.json").read_bytes() == b"owned payload"


def test_completion_rollback_retires_manifest_after_unlink_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, artifact, destination = _owned_store(tmp_path)
    original_unlink = os.unlink
    injected = False

    def fail_manifest_unlink(path: str, *, dir_fd: int | None = None) -> None:
        nonlocal injected
        if path == MANIFEST_NAME and not injected:
            injected = True
            raise OSError("injected manifest unlink failure")
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", fail_manifest_unlink)
    store.unpublish_owned("synthetic-run", artifact)
    assert injected
    assert list(destination.iterdir()) == []
