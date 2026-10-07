"""Isolated cache/bounded-read regressions; genuine M9 integration lives elsewhere.

Only the initial expensive M9 semantic replay is mocked here. Every subsequent
capture exercises real descriptor traversal, byte hashing, two-pass inventory,
completion reads, and immutable cache comparisons. These fixtures deliberately
are not complete M9 research publications and never substitute for end-to-end
source-binding tests in test_production.py.
"""

from __future__ import annotations

import json
import os
import socket
from types import SimpleNamespace

import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2 import evaluation_store, holdout, production
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("cache regression must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def source_case(tmp_path, monkeypatch):
    root = tmp_path / "evaluations"
    path = root / "run"
    path.mkdir(parents=True)
    development = tmp_path / "development"
    development.mkdir()
    (development / "retained-marker").write_bytes(b"verified development marker")
    files = {
        holdout.CLAIM_NAME: b'{"synthetic":true}',
        "metrics.json": b'{"research_verdict":"FAIL"}',
        "evaluation_models/augmented.json": b'{"model":"synthetic"}',
    }
    manifest = {
        "run_id": "run",
        "files": {
            name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
            for name, raw in files.items()
        },
    }
    manifest_raw = canonicalize(manifest)

    def write(artifact):
        for name, raw in artifact.files.items():
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_bytes(raw)
        (path / evaluation_store.MANIFEST_NAME).write_bytes(canonicalize(artifact.manifest))
        completion = {
            "evaluation_manifest_sha256": sha256_bytes(canonicalize(artifact.manifest)),
            "evaluation_run_id": "run",
            "claim_sha256": sha256_bytes(artifact.files[holdout.CLAIM_NAME]),
        }
        (development / "holdout_evaluation_completed.json").write_bytes(canonicalize(completion))

    write(evaluation_store.EvaluationArtifact(json.loads(manifest_raw), dict(files)))
    calls = {"full": 0, "claim": 0}

    def initial_replay(store, run_id, *, development_run_dir):
        assert store.root == root
        assert run_id == "run"
        assert development_run_dir == development
        calls["full"] += 1
        return evaluation_store.EvaluationArtifact(json.loads(manifest_raw), dict(files))

    def source_binding(development_run_dir, artifact):
        calls["claim"] += 1
        assert development_run_dir == development
        if (development / "retained-marker").read_bytes() != b"verified development marker":
            raise production.ProductionIntegrityError("retained development evidence changed")

    monkeypatch.setattr(evaluation_store.EvaluationStore, "get", initial_replay)
    monkeypatch.setattr(evaluation_store, "_verify_source_claim", source_binding)
    return SimpleNamespace(
        source=production._Source(root, development, "run"),
        root=root,
        path=path,
        development=development,
        files=files,
        manifest_raw=manifest_raw,
        calls=calls,
        write=write,
    )


def test_unchanged_cache_rechecks_bytes_and_source_without_repeating_fit(source_case):
    first = source_case.source.capture()
    assert source_case.source.capture() == first
    assert source_case.source.capture() == first
    assert source_case.calls == {"full": 1, "claim": 2}


@pytest.mark.parametrize("capture_number", [1, 2])
def test_exposed_mutable_objects_cannot_mutate_private_cache_baseline(source_case, capture_number):
    for _ in range(capture_number):
        artifact, _ = source_case.source.capture()
    artifact.files["metrics.json"] = b'{"research_verdict":"PASS"}'
    artifact.manifest["files"]["metrics.json"]["sha256"] = "0" * 64
    fresh, _ = source_case.source.capture()
    assert fresh.files == source_case.files
    assert canonicalize(fresh.manifest) == source_case.manifest_raw
    assert source_case.calls["full"] == 1


def test_exposed_object_plus_consistently_rehashed_disk_cannot_poison_baseline(source_case):
    artifact, _ = source_case.source.capture()
    changed = b'{"research_verdict":"PASS"}'
    artifact.files["metrics.json"] = changed
    artifact.manifest["files"]["metrics.json"] = {
        "sha256": sha256_bytes(changed),
        "size_bytes": len(changed),
    }
    source_case.write(artifact)
    with pytest.raises(production.ProductionIntegrityError, match="manifest changed"):
        source_case.source.capture()
    assert source_case.calls["full"] == 1


def test_consistently_rehashed_disk_only_is_not_a_new_verified_parent(source_case):
    source_case.source.capture()
    manifest = json.loads(source_case.manifest_raw)
    files = dict(source_case.files)
    files["metrics.json"] += b" "
    manifest["files"]["metrics.json"] = {
        "sha256": sha256_bytes(files["metrics.json"]),
        "size_bytes": len(files["metrics.json"]),
    }
    source_case.write(evaluation_store.EvaluationArtifact(manifest, files))
    with pytest.raises(production.ProductionIntegrityError, match="manifest changed"):
        source_case.source.capture()


def test_single_byte_payload_mutation_rejected_on_cached_capture(source_case):
    source_case.source.capture()
    path = source_case.path / "metrics.json"
    path.write_bytes(path.read_bytes().replace(b"FAIL", b"PASS"))
    with pytest.raises(CryptoAIError):
        source_case.source.capture()


@pytest.mark.parametrize("change", ["missing", "wrong_hash", "wrong_run"])
def test_cached_capture_rechecks_current_durable_completion(source_case, change):
    source_case.source.capture()
    path = source_case.development / "holdout_evaluation_completed.json"
    if change == "missing":
        path.unlink()
    else:
        value = json.loads(path.read_bytes())
        if change == "wrong_hash":
            value["evaluation_manifest_sha256"] = "0" * 64
        else:
            value["evaluation_run_id"] = "another-run"
        path.write_bytes(canonicalize(value))
    with pytest.raises(CryptoAIError):
        source_case.source.capture()
    assert source_case.calls["claim"] == 1


def test_cached_capture_must_recheck_retained_development_source(source_case):
    source_case.source.capture()
    (source_case.development / "retained-marker").write_bytes(b"changed")
    with pytest.raises(production.ProductionIntegrityError, match="development evidence"):
        source_case.source.capture()
    assert source_case.calls["claim"] == 1


@pytest.mark.parametrize("target", ["path", "development"])
def test_cached_capture_rejects_directory_inode_replacement(source_case, target):
    source_case.source.capture()
    path = getattr(source_case, target)
    path.rename(path.with_name(path.name + "-displaced"))
    path.mkdir()
    with pytest.raises(CryptoAIError):
        source_case.source.capture()


def test_new_source_instance_never_inherits_prior_semantic_verification(source_case):
    source_case.source.capture()
    production._Source(source_case.root, source_case.development, "run").capture()
    assert source_case.calls == {"full": 2, "claim": 0}


def test_cached_capture_rejects_injected_deep_fifo_without_blocking(source_case):
    source_case.source.capture()
    os.mkfifo(source_case.path / "evaluation_models" / "unexpected")
    with pytest.raises(CryptoAIError):
        source_case.source.capture()


def test_oversized_authorization_is_rejected_before_any_file_read(tmp_path, monkeypatch):
    path = tmp_path / "authorization.json"
    path.write_bytes(b"x" * 16_385)

    def forbidden_read(*args):
        raise AssertionError("oversized authorization was read before its size check")

    monkeypatch.setattr(production.os, "read", forbidden_read)
    with pytest.raises(production.ProductionAuthorizationError, match="exceeds its bound"):
        production.load_synthetic_authorization(path)


@pytest.mark.parametrize("kind", ["fifo", "symlink", "hardlink"])
def test_authorization_loader_rejects_nonunique_or_nonregular_files(tmp_path, kind):
    path = tmp_path / "authorization.json"
    if kind == "fifo":
        os.mkfifo(path)
    else:
        other = tmp_path / "other"
        other.write_bytes(b"{}")
        if kind == "symlink":
            path.symlink_to(other)
        else:
            os.link(other, path)
    with pytest.raises(production.ProductionError):
        production.load_synthetic_authorization(path)
