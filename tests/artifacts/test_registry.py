"""Tests for immutable artifact and holdout-claim safeguards."""

import json
from pathlib import Path

import pytest

from crypto_ai.artifacts.registry import (
    claim_holdout_evaluation,
    copy_verified_snapshot,
    generate_run_id,
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
