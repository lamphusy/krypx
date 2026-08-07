"""Unique run IDs, immutable model storage, and holdout access claims."""

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crypto_ai.artifacts.manifest import atomic_write_json, utc_now_iso
from crypto_ai.data.storage import sha256_file, symbol_to_slug
from crypto_ai.exceptions import ArtifactError

if TYPE_CHECKING:
    from xgboost import XGBClassifier


def generate_run_id(symbol: str, timeframe: str, commit: str = "unknown") -> str:
    """Generate a UTC, symbol, timeframe, and short-commit run identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    short_commit = re.sub(r"[^a-zA-Z0-9]", "", commit)[:7] or "unknown"
    return f"{timestamp}_{symbol_to_slug(symbol)}_{timeframe}_{short_commit}"


def create_run_directory(root: Path, run_id: str) -> Path:
    """Create a new run directory and refuse accidental reuse."""
    destination = root / run_id
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ArtifactError(f"Run directory already exists: {destination}") from exc
    return destination


def save_xgboost_model(model: "XGBClassifier", path: Path) -> None:
    """Save an XGBoost model without overwriting an existing immutable artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ArtifactError(f"Model artifact already exists: {path}")
    model.save_model(path)


def load_xgboost_model(path: Path) -> "XGBClassifier":
    """Load an XGBoost classifier artifact."""
    if not path.exists():
        raise ArtifactError(f"Model artifact does not exist: {path}")
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ArtifactError("XGBoost and its native OpenMP runtime are required") from exc
    model = XGBClassifier()
    model.load_model(path)
    return model


def copy_verified_snapshot(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy immutable input bytes and verify the resulting hash."""
    if sha256_file(source) != expected_sha256:
        raise ArtifactError(f"Source snapshot hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ArtifactError(f"Evaluation snapshot already exists: {destination}")
    shutil.copyfile(source, destination)
    if sha256_file(destination) != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ArtifactError(f"Copied snapshot hash mismatch: {destination}")


def claim_holdout_evaluation(run_directory: Path) -> Path:
    """Atomically claim one and only one holdout evaluation attempt for a run."""
    claim_path = run_directory / "holdout_evaluation_claim.json"
    atomic_write_json(
        claim_path,
        {"status": "claimed", "claimed_at_utc": utc_now_iso()},
        exclusive=True,
    )
    return claim_path


def update_holdout_claim(claim_path: Path, status: str, **details: Any) -> None:
    """Update an existing claim while preserving evidence of failed attempts."""
    if status not in {"completed", "failed", "invalidated"}:
        raise ArtifactError(f"Unsupported holdout claim status: {status}")
    if not claim_path.exists():
        raise ArtifactError(f"Holdout claim does not exist: {claim_path}")
    existing = json.loads(claim_path.read_text(encoding="utf-8"))
    existing.update({"status": status, "updated_at_utc": utc_now_iso(), **details})
    atomic_write_json(claim_path, existing)
