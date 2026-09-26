"""Immutable, synthetic-only Milestone 7 development-report publications.

This store does not accept a report merely because its hashes are self-consistent.
The offline backtest verifier replays the accepted Milestone 5/6 parents and compares
the complete report buffers before publication and after every read.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from crypto_ai.exceptions import CryptoAIError, PublicationCollisionError
from crypto_ai.phase2.dataset import _json
from crypto_ai.phase2.experiments import ExperimentStore, _run_id, generate_run_id
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.storage import (
    _atomic_rename_directory_no_replace,
    _capture_publication_tree,
    _create_directory_path_without_symlinks,
    _fsync_directory_descriptor,
    _fsync_tree_directories_at,
    _open_directory_at,
    _open_directory_path,
    _read_regular_file_at_once,
    _require_atomic_rename_directory_no_replace_at,
    _require_descriptor_relative_mutations,
    _stat_identity,
    _write_fsynced_at,
)

SCHEMA = "phase2-backtest-publication-v1"
SPECIFICATION_ID = "phase2-milestone7-offline-development-backtests-v1"
PAYLOAD_NAMES = frozenset(
    {
        "strategy_metrics.json",
        "cost_sensitivity.json",
        "baseline_metrics.json",
        "ablation_report.json",
        "development_report.md",
    }
)
JSON_PAYLOAD_NAMES = PAYLOAD_NAMES - {"development_report.md"}
HASH_FIELDS = frozenset(
    {
        "source_manifest_sha256",
        "source_experiment_id",
        "source_folds_sha256",
        "prepared_dataset_id",
        "market_price_context_sha256",
        "decision_window_sha256",
        "cost_config_sha256",
        "baseline_config_sha256",
        "metric_config_sha256",
        "implementation_source_sha256",
        "dependency_lock_sha256",
    }
)
METADATA_FIELDS = HASH_FIELDS | {
    "specification_id",
    "synthetic",
    "source_run_id",
    "source_prediction_sha256",
}


class ArtifactError(CryptoAIError):
    """A Milestone 7 artifact contract failed."""


class ArtifactInputError(ArtifactError):
    """A candidate report has an invalid structure or provenance declaration."""


class ArtifactIntegrityError(ArtifactError):
    """Report bytes, filesystem state, or semantic replay disagree."""


class ArtifactCollisionError(ArtifactIntegrityError):
    """The requested report ID is already occupied and cannot be replaced."""


@dataclass(frozen=True, slots=True)
class VerifiedBacktestReport:
    """Exact captured report bytes and their verified outer manifest."""

    manifest: dict
    files: dict[str, bytes]


@contextmanager
def _errors():
    try:
        yield
    except ArtifactError:
        raise
    except PublicationCollisionError as exc:
        raise ArtifactCollisionError("report run already exists") from exc
    except (
        CryptoAIError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        IndexError,
        RuntimeError,
        RecursionError,
        UnicodeError,
    ) as exc:
        raise ArtifactIntegrityError("invalid offline backtest report evidence") from exc


def _hash(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_metadata(value: object, *, run_id: str) -> dict:
    if type(value) is not dict or set(value) != METADATA_FIELDS:
        raise ArtifactInputError("report metadata fields do not match the frozen schema")
    if value["specification_id"] != SPECIFICATION_ID or value["synthetic"] is not True:
        raise ArtifactInputError("only the frozen synthetic report specification is supported")
    if type(value["source_run_id"]) is not str:
        raise ArtifactInputError("source run ID must be a string")
    _run_id(value["source_run_id"])
    if value["source_run_id"] == run_id:
        raise ArtifactInputError("the Milestone 7 report cannot replace its Milestone 6 parent")
    if any(not _hash(value[name]) for name in HASH_FIELDS):
        raise ArtifactInputError("report metadata contains an invalid SHA-256 value")
    predictions = value["source_prediction_sha256"]
    if (
        type(predictions) is not dict
        or set(predictions) != {"A", "B", "C", "D"}
        or any(not _hash(digest) for digest in predictions.values())
    ):
        raise ArtifactInputError("source prediction hashes must bind all four cells")
    return value


def _validate_files(value: object) -> dict[str, bytes]:
    if not isinstance(value, Mapping) or set(value) != PAYLOAD_NAMES:
        raise ArtifactInputError("a report requires exactly five frozen payload files")
    files = dict(value)
    if any(type(data) is not bytes for data in files.values()):
        raise ArtifactInputError("report payloads must be exact bytes")
    for name in JSON_PAYLOAD_NAMES:
        if type(_json(files[name])) is not dict:
            raise ArtifactInputError(f"{name} must be a canonical JSON object")
    try:
        markdown = files["development_report.md"].decode("utf-8")
    except UnicodeError as exc:
        raise ArtifactInputError("development report must be UTF-8") from exc
    if "\x00" in markdown:
        raise ArtifactInputError("development report contains a NUL byte")
    for name in JSON_PAYLOAD_NAMES:
        if sha256_bytes(files[name]) not in markdown:
            raise ArtifactInputError(f"development report does not bind {name} hash")
    return files


def _inventory(files: Mapping[str, bytes]) -> dict[str, dict[str, object]]:
    return {
        name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        for name, raw in sorted(files.items())
    }


def _candidate(files: object, metadata: object, *, run_id: str) -> tuple[dict[str, bytes], dict]:
    _run_id(run_id)
    _validate_metadata(metadata, run_id=run_id)
    checked_metadata = _validate_metadata(deepcopy(metadata), run_id=run_id)
    checked_files = _validate_files(files)
    return checked_files, checked_metadata


def _validate_manifest(raw: bytes, run_id: str) -> dict:
    value = _json(raw)
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "run_id", "metadata", "files"}
        or value["schema_version"] != SCHEMA
        or value["run_id"] != run_id
    ):
        raise ArtifactIntegrityError("invalid report outer manifest")
    _validate_metadata(value["metadata"], run_id=run_id)
    inventory = value["files"]
    if type(inventory) is not dict or set(inventory) != PAYLOAD_NAMES:
        raise ArtifactIntegrityError("invalid report payload inventory")
    for entry in inventory.values():
        if (
            type(entry) is not dict
            or set(entry) != {"sha256", "size_bytes"}
            or not _hash(entry["sha256"])
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise ArtifactIntegrityError("invalid report payload hash or byte length")
    return value


def _capture(
    descriptor: int, run_id: str, *, expected_manifest_bytes: bytes | None = None
) -> VerifiedBacktestReport:
    raw, info = _read_regular_file_at_once(
        descriptor, "manifest.json", description="report manifest"
    )
    if expected_manifest_bytes is not None and raw != expected_manifest_bytes:
        raise ArtifactIntegrityError("report manifest bytes changed during publication")
    manifest = _validate_manifest(raw, run_id)
    captured, _ = _capture_publication_tree(
        descriptor,
        manifest_data=raw,
        manifest_stat=info,
        publication_id=run_id,
        manifest_files=manifest["files"],
        expected_paths=PAYLOAD_NAMES | {"manifest.json"},
        expected_directories=set(),
    )
    files = {name: captured[name] for name in PAYLOAD_NAMES}
    _validate_files(files)
    return VerifiedBacktestReport(manifest, files)


def _rollback_owned_directory(descriptor: int) -> None:
    """Invalidate only the open inode owned by this attempt, never a swapped name."""
    try:
        os.unlink("manifest.json", dir_fd=descriptor)
    except FileNotFoundError:
        pass
    _fsync_directory_descriptor(descriptor, description="invalid report publication")


def _semantic_replay(experiments: ExperimentStore, files: dict[str, bytes], metadata: dict):
    # Import at call time: the engine imports this publication module for its store.
    from crypto_ai.phase2.backtests import verify_report

    verify_report(experiments, files, metadata)


def _require_metadata_unchanged(metadata: object, frozen_bytes: bytes) -> None:
    """Reject caller mutation while the immutable metadata snapshot is being staged."""
    if type(metadata) is not dict or canonicalize(metadata) != frozen_bytes:
        raise ArtifactIntegrityError("caller report metadata changed during publication")


class ArtifactStore:
    """Descriptor-pinned, no-overwrite Milestone 7 report store."""

    def __init__(self, root: Path, experiments: ExperimentStore):
        if type(experiments) is not ExperimentStore:
            raise ArtifactInputError("a verified Milestone 6 experiment store is required")
        with _errors():
            self.experiments = experiments
            self.root = Path(os.path.abspath(os.fspath(root)))
            if self.root != experiments.root:
                raise ArtifactInputError("reports must share the immutable Milestone 6 runs root")
            descriptor = _create_directory_path_without_symlinks(
                self.root, description="phase2 runs"
            )
            try:
                self._identity = _stat_identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)

    def _check_attachment(self, run_id: str, identity: tuple[int, int]) -> None:
        parent = _open_directory_path(
            self.root, description="phase2 runs", expected_identity=self._identity
        )
        try:
            child = _open_directory_at(parent, run_id, description="published report")
            try:
                if _stat_identity(os.fstat(child)) != identity:
                    raise ArtifactIntegrityError("published report directory was replaced")
                if (
                    _stat_identity(os.stat(run_id, dir_fd=parent, follow_symlinks=False))
                    != identity
                ):
                    raise ArtifactIntegrityError("published report was detached")
                current = _open_directory_path(
                    self.root, description="phase2 runs", expected_identity=self._identity
                )
                os.close(current)
            finally:
                os.close(child)
        finally:
            os.close(parent)

    def get(self, run_id: str) -> VerifiedBacktestReport:
        with _errors():
            _run_id(run_id)
            parent = _open_directory_path(
                self.root, description="phase2 runs", expected_identity=self._identity
            )
            try:
                descriptor = _open_directory_at(parent, run_id, description="published report")
                try:
                    report = _capture(descriptor, run_id)
                    self._check_attachment(run_id, _stat_identity(os.fstat(descriptor)))
                    _semantic_replay(self.experiments, report.files, report.manifest["metadata"])
                    confirmed = _capture(descriptor, run_id)
                    if confirmed != report:
                        raise ArtifactIntegrityError("report changed during semantic replay")
                    self._check_attachment(run_id, _stat_identity(os.fstat(descriptor)))
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent)
            return report

    def publish(
        self,
        files: Mapping[str, bytes],
        metadata: Mapping[str, object],
        *,
        run_id: str | None = None,
    ) -> str:
        with _errors():
            if run_id is None:
                run_id = generate_run_id()
            checked_files, checked_metadata = _candidate(files, metadata, run_id=run_id)
            metadata_bytes = canonicalize(checked_metadata)
            _semantic_replay(self.experiments, dict(checked_files), deepcopy(checked_metadata))
            _require_metadata_unchanged(metadata, metadata_bytes)
            _require_descriptor_relative_mutations()
            parent = _open_directory_path(
                self.root, description="phase2 runs", expected_identity=self._identity
            )
            stage_name = ".staging-" + run_id + "-" + uuid.uuid4().hex
            stage = None
            try:
                _require_atomic_rename_directory_no_replace_at(parent)
                os.mkdir(stage_name, mode=0o700, dir_fd=parent)
                stage = _open_directory_at(parent, stage_name, description="report staging")
                for name, raw in sorted(checked_files.items()):
                    _write_fsynced_at(stage, name, raw)
                    captured, _ = _read_regular_file_at_once(stage, name, description=name)
                    if captured != raw or sha256_bytes(captured) != sha256_bytes(raw):
                        raise ArtifactIntegrityError("staged report payload readback mismatch")
                _require_metadata_unchanged(metadata, metadata_bytes)
                manifest_bytes = canonicalize(
                    {
                        "schema_version": SCHEMA,
                        "run_id": run_id,
                        "metadata": checked_metadata,
                        "files": _inventory(checked_files),
                    }
                )
                _write_fsynced_at(stage, "manifest.json", manifest_bytes)
                if (
                    _capture(stage, run_id, expected_manifest_bytes=manifest_bytes).files
                    != checked_files
                ):
                    raise ArtifactIntegrityError("staged report bytes changed before publication")
                _fsync_tree_directories_at(stage, description="report staging")
                if (
                    _capture(stage, run_id, expected_manifest_bytes=manifest_bytes).files
                    != checked_files
                ):
                    raise ArtifactIntegrityError("staged report changed during directory fsync")
                _require_metadata_unchanged(metadata, metadata_bytes)
                _atomic_rename_directory_no_replace(parent, stage_name, run_id)
                _fsync_directory_descriptor(parent, description="phase2 runs")
                report = _capture(stage, run_id, expected_manifest_bytes=manifest_bytes)
                if report.files != checked_files:
                    raise ArtifactIntegrityError("published report changed after rename")
                self._check_attachment(run_id, _stat_identity(os.fstat(stage)))
                _semantic_replay(self.experiments, report.files, report.manifest["metadata"])
                confirmed = _capture(stage, run_id, expected_manifest_bytes=manifest_bytes)
                if confirmed.files != checked_files:
                    raise ArtifactIntegrityError("published report changed during semantic replay")
                self._check_attachment(run_id, _stat_identity(os.fstat(stage)))
                _require_metadata_unchanged(metadata, metadata_bytes)
            except Exception as exc:
                # The no-replace rename may have completed even if a wrapper raises.
                # The still-open staging descriptor pins only the inode we own.
                if stage is not None:
                    try:
                        _rollback_owned_directory(stage)
                    except Exception as cleanup_exc:
                        raise ArtifactIntegrityError(
                            "failed to invalidate owned report publication"
                        ) from cleanup_exc
                if isinstance(exc, ArtifactError):
                    raise
                if isinstance(exc, PublicationCollisionError):
                    raise ArtifactCollisionError("report run already exists") from exc
                raise ArtifactIntegrityError("report publication failed") from exc
            finally:
                if stage is not None:
                    os.close(stage)
                os.close(parent)
            return run_id
