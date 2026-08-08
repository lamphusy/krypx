"""Tests for normalized, atomic, strict manifest writing."""

import json
from pathlib import Path

import numpy as np
import pytest

from crypto_ai.artifacts.manifest import atomic_write_json
from crypto_ai.exceptions import ArtifactError


@pytest.mark.parametrize("exclusive", [False, True])
def test_atomic_json_write_normalizes_parent_directory_failure(
    tmp_path: Path,
    exclusive: bool,
) -> None:
    """A non-directory parent is reported as an artifact error in either mode."""
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ArtifactError, match="Unable to create artifact directory"):
        atomic_write_json(blocked_parent / "manifest.json", {"ok": True}, exclusive=exclusive)


def test_atomic_json_write_normalizes_temporary_file_creation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Temporary-file allocation cannot escape as a raw filesystem exception."""

    def fail_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError("simulated temporary-file failure")

    monkeypatch.setattr("crypto_ai.artifacts.manifest.tempfile.mkstemp", fail_mkstemp)

    with pytest.raises(ArtifactError, match="simulated temporary-file failure"):
        atomic_write_json(tmp_path / "manifest.json", {"ok": True})


@pytest.mark.parametrize("exclusive", [False, True])
def test_atomic_json_write_normalizes_numpy_nonfinite_values_to_null(
    tmp_path: Path,
    exclusive: bool,
) -> None:
    destination = tmp_path / f"manifest-{exclusive}.json"

    atomic_write_json(
        destination,
        {"nan": np.float64(np.nan), "positive_infinity": np.float64(np.inf)},
        exclusive=exclusive,
    )

    text = destination.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {"nan": None, "positive_infinity": None}
