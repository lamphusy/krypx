"""Tests for immutable artifact and holdout-claim safeguards."""

import json
from pathlib import Path

import pytest

from crypto_ai.artifacts.registry import (
    claim_holdout_evaluation,
    copy_verified_snapshot,
    create_run_directory,
    generate_run_id,
    save_xgboost_model,
    update_holdout_claim,
)
from crypto_ai.data.storage import sha256_file
from crypto_ai.exceptions import ArtifactError


def test_holdout_evaluation_claim_prevents_second_access(tmp_path: Path) -> None:
    claim = claim_holdout_evaluation(tmp_path)
    with pytest.raises(ArtifactError, match="already exists"):
        claim_holdout_evaluation(tmp_path)
    assert claim.exists()


def test_failed_holdout_claim_is_not_silently_removed(tmp_path: Path) -> None:
    claim = claim_holdout_evaluation(tmp_path)
    update_holdout_claim(claim, "failed", error="simulated")
    payload = json.loads(claim.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    with pytest.raises(ArtifactError):
        claim_holdout_evaluation(tmp_path)


def test_failed_holdout_claim_cannot_be_rewritten_as_completed(tmp_path: Path) -> None:
    claim = claim_holdout_evaluation(tmp_path)
    update_holdout_claim(claim, "failed", error="simulated")

    with pytest.raises(ArtifactError, match="transition"):
        update_holdout_claim(claim, "completed")

    payload = json.loads(claim.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "simulated"


def test_run_ids_are_unique_and_include_research_identity() -> None:
    first = generate_run_id("BTC/USDT", "1h", "abcdef123456")
    second = generate_run_id("BTC/USDT", "1h", "abcdef123456")
    assert first != second
    assert first.endswith("_btc_usdt_1h_abcdef1")


def test_snapshot_copy_is_byte_exact_and_cannot_be_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    destination = tmp_path / "evaluation" / "input_data_snapshot.csv"
    source.write_bytes(b"timestamp,open\n2026-01-01T00:00:00Z,100\n")
    digest = sha256_file(source)
    copy_verified_snapshot(source, destination, digest)
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(ArtifactError, match="already exists"):
        copy_verified_snapshot(source, destination, digest)


def test_create_run_directory_normalizes_parent_failure(tmp_path: Path) -> None:
    """A blocked run root is exposed as a project-specific artifact failure."""
    blocked_root = tmp_path / "blocked"
    blocked_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ArtifactError, match="Unable to create run directory"):
        create_run_directory(blocked_root, "run-1")


def test_create_run_directory_never_reuses_a_version(tmp_path: Path) -> None:
    """An existing version directory and its evidence are never overwritten."""
    version = create_run_directory(tmp_path, "version-1")
    sentinel = version / "manifest.json"
    sentinel.write_bytes(b"existing production evidence\n")

    with pytest.raises(ArtifactError, match="already exists"):
        create_run_directory(tmp_path, "version-1")

    assert sentinel.read_bytes() == b"existing production evidence\n"


@pytest.mark.parametrize("identifier", ["../escape", "nested/run", "..\\escape"])
def test_create_run_directory_rejects_registry_escape(tmp_path: Path, identifier: str) -> None:
    with pytest.raises(ArtifactError, match="one path component"):
        create_run_directory(tmp_path / "runs", identifier)

    assert not (tmp_path / "escape").exists()


def test_create_run_directory_rejects_absolute_identifier(tmp_path: Path) -> None:
    outside = tmp_path / "outside"

    with pytest.raises(ArtifactError, match="one path component"):
        create_run_directory(tmp_path / "runs", str(outside))

    assert not outside.exists()


def test_create_run_directory_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked-run").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactError, match="escapes its registry root"):
        create_run_directory(root, "linked-run")

    assert list(outside.iterdir()) == []


def test_save_model_normalizes_parent_failure(tmp_path: Path) -> None:
    """Model parent creation errors are normalized before the model is called."""
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    class Model:
        def save_model(self, path: Path) -> None:
            raise AssertionError("save_model must not be reached")

    with pytest.raises(ArtifactError, match="Unable to save XGBoost model artifact"):
        save_xgboost_model(Model(), blocked_parent / "model.json")


def test_save_model_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    path.write_bytes(b"existing immutable model\n")

    class Model:
        def save_model(self, destination: Path) -> None:
            raise AssertionError("save_model must not be called for an existing artifact")

    with pytest.raises(ArtifactError, match="already exists"):
        save_xgboost_model(Model(), path)

    assert path.read_bytes() == b"existing immutable model\n"


def test_invalid_utf8_holdout_claim_is_normalized_and_preserved(tmp_path: Path) -> None:
    """A corrupt irreversible claim stays in place and yields an artifact error."""
    claim = tmp_path / "holdout_evaluation_claim.json"
    corrupt_bytes = b"\xff\xfe\x00"
    claim.write_bytes(corrupt_bytes)

    with pytest.raises(ArtifactError, match="Unable to load holdout claim"):
        update_holdout_claim(claim, "failed", error="simulated")

    assert claim.read_bytes() == corrupt_bytes


def test_unhashable_holdout_claim_status_is_normalized_and_preserved(tmp_path: Path) -> None:
    claim = tmp_path / "holdout_evaluation_claim.json"
    corrupt_bytes = b'{"status": []}\n'
    claim.write_bytes(corrupt_bytes)

    with pytest.raises(ArtifactError, match="unsupported current status"):
        update_holdout_claim(claim, "failed", error="simulated")

    assert claim.read_bytes() == corrupt_bytes
