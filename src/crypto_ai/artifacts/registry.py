"""Unique run IDs, immutable model storage, and holdout access claims."""

import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crypto_ai.artifacts.manifest import atomic_write_json, utc_now_iso
from crypto_ai.data.storage import sha256_file, symbol_to_slug
from crypto_ai.exceptions import ArtifactError

if TYPE_CHECKING:
    from xgboost import XGBClassifier


def resolve_artifact_subdirectory(root: Path, identifier: str) -> Path:
    """Resolve one safe artifact-registry child without permitting path escape."""
    if (
        not isinstance(identifier, str)
        or not identifier
        or identifier in {".", ".."}
        or "/" in identifier
        or "\\" in identifier
    ):
        raise ArtifactError(f"Artifact identifier must be one path component: {identifier!r}")
    try:
        if Path(identifier).is_absolute():
            raise ArtifactError(f"Artifact identifier must be one path component: {identifier!r}")
        resolved_root = root.resolve(strict=False)
        destination = (resolved_root / identifier).resolve(strict=False)
    except ArtifactError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactError(f"Unable to resolve artifact identifier {identifier!r}: {exc}") from exc
    if destination.parent != resolved_root:
        raise ArtifactError(f"Artifact identifier escapes its registry root: {identifier!r}")
    return destination


def generate_run_id(symbol: str, timeframe: str, commit: str = "unknown") -> str:
    """Generate a UTC, symbol, timeframe, and short-commit run identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    short_commit = re.sub(r"[^a-zA-Z0-9]", "", commit)[:7] or "unknown"
    return f"{timestamp}_{symbol_to_slug(symbol)}_{timeframe}_{short_commit}"


def create_run_directory(root: Path, run_id: str) -> Path:
    """Create a new run directory and refuse accidental reuse."""
    destination = resolve_artifact_subdirectory(root, run_id)
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ArtifactError(f"Run directory already exists: {destination}") from exc
    except OSError as exc:
        raise ArtifactError(f"Unable to create run directory {destination}: {exc}") from exc
    return destination


def save_xgboost_model(model: "XGBClassifier", path: Path) -> None:
    """Save an XGBoost model without overwriting an existing immutable artifact."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError(f"Unable to save XGBoost model artifact {path}: {exc}") from exc
    temporary_path: Path | None = None
    try:
        if path.exists():
            raise ArtifactError(f"Model artifact already exists: {path}")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f"{path.stem}.",
            suffix=path.suffix or ".model",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        model.save_model(temporary_path)
        with temporary_path.open("rb") as file_handle:
            os.fsync(file_handle.fileno())
        os.link(temporary_path, path)
    except ArtifactError:
        raise
    except FileExistsError as exc:
        raise ArtifactError(f"Model artifact already exists: {path}") from exc
    except Exception as exc:
        raise ArtifactError(f"Unable to save XGBoost model artifact {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def load_xgboost_model(path: Path) -> "XGBClassifier":
    """Load an XGBoost classifier artifact."""
    if not path.exists():
        raise ArtifactError(f"Model artifact does not exist: {path}")
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise ArtifactError("XGBoost and its native OpenMP runtime are required") from exc
    try:
        model = XGBClassifier()
        model.load_model(path)
    except Exception as exc:
        raise ArtifactError(f"Unable to load XGBoost model artifact {path}: {exc}") from exc
    return model


def copy_verified_snapshot(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy immutable input bytes and verify the resulting hash."""
    try:
        source_sha256 = sha256_file(source)
    except OSError as exc:
        raise ArtifactError(
            f"Unable to copy verified snapshot {source} to {destination}: {exc}"
        ) from exc
    if source_sha256 != expected_sha256:
        raise ArtifactError(f"Source snapshot hash mismatch: {source}")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError(
            f"Unable to copy verified snapshot {source} to {destination}: {exc}"
        ) from exc

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f"{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        shutil.copyfile(source, temporary_path)
        if sha256_file(temporary_path) != expected_sha256:
            raise ArtifactError(f"Copied snapshot hash mismatch: {destination}")
        os.link(temporary_path, destination)
    except ArtifactError:
        raise
    except FileExistsError as exc:
        raise ArtifactError(f"Evaluation snapshot already exists: {destination}") from exc
    except OSError as exc:
        raise ArtifactError(
            f"Unable to copy verified snapshot {source} to {destination}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


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

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        existing = json.loads(
            claim_path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"Unable to load holdout claim {claim_path}: {exc}") from exc
    if not isinstance(existing, dict):
        raise ArtifactError(f"Holdout claim must contain a JSON object: {claim_path}")
    current_status = existing.get("status")
    allowed_transitions = {
        "claimed": {"completed", "failed", "invalidated"},
        "completed": {"invalidated"},
        "failed": {"invalidated"},
        "invalidated": set(),
    }
    if not isinstance(current_status, str) or current_status not in allowed_transitions:
        raise ArtifactError(
            f"Holdout claim has unsupported current status {current_status!r}: {claim_path}"
        )
    if status not in allowed_transitions[current_status]:
        raise ArtifactError(
            f"Holdout claim transition {current_status!r} -> {status!r} is not allowed"
        )
    existing.update(details)
    existing.update({"status": status, "updated_at_utc": utc_now_iso()})
    atomic_write_json(claim_path, existing)
