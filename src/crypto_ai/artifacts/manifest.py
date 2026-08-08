"""JSON-safe run, model, and evaluation manifest helpers."""

import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai import __version__
from crypto_ai.exceptions import ArtifactError


def utc_now_iso() -> str:
    """Return a second-resolution UTC creation timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    """Convert common scientific Python values to strict JSON-compatible values."""
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path: Path, payload: dict[str, Any], *, exclusive: bool = False) -> None:
    """Write formatted JSON atomically, optionally refusing an existing destination."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ArtifactError(f"Unable to create artifact directory {path.parent}: {exc}") from exc
    if exclusive:
        try:
            with path.open("x", encoding="utf-8") as file_handle:
                json.dump(
                    json_safe(payload),
                    file_handle,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                file_handle.write("\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())
        except FileExistsError as exc:
            raise ArtifactError(
                f"Artifact already exists and cannot be overwritten: {path}"
            ) from exc
        except (OSError, TypeError, ValueError) as exc:
            raise ArtifactError(f"Unable to write JSON artifact {path}: {exc}") from exc
        return

    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=path.name, suffix=".tmp"
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(
                json_safe(payload),
                file_handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_name, path)
    except (OSError, TypeError, ValueError) as exc:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise ArtifactError(f"Unable to write JSON artifact {path}: {exc}") from exc


def git_metadata(project_dir: Path) -> dict[str, Any]:
    """Return commit, branch, and dirty-worktree provenance without mutating Git."""

    def run(*arguments: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(project_dir), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "dirty_worktree": bool(status and status != "unknown"),
    }


def environment_metadata() -> dict[str, Any]:
    """Return reproducibility metadata for Python and primary dependencies."""
    dependencies: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "xgboost", "ta", "ccxt"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "project_version": __version__,
        "python_version": platform.python_version(),
        "dependency_versions": dependencies,
    }
