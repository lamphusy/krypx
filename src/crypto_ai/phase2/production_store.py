"""Descriptor-pinned storage for synthetic-only production registry versions.

This low-level store verifies bytes and filesystem invariants. Its mandatory
semantic verifier is supplied by the production engine and must independently
verify evaluation, authorization, training, and model provenance on every read.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from crypto_ai.exceptions import CryptoAIError, PublicationCollisionError
from crypto_ai.phase2.dataset import _json
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.storage import (
    _atomic_rename_directory_no_replace,
    _capture_publication_tree,
    _create_directory_path_without_symlinks,
    _ensure_directory_chain_at,
    _fsync_directory_descriptor,
    _fsync_tree_directories_at,
    _open_directory_at,
    _open_directory_path,
    _read_regular_file_at_once,
    _require_atomic_rename_directory_no_replace_at,
    _require_descriptor_relative_mutations,
    _stat_identity,
    _validate_relative_path,
    _write_fsynced_at,
)

SCHEMA = "phase2-synthetic-production-publication-v1"
MANIFEST_NAME = "manifest.json"
REQUIRED_PAYLOADS = frozenset(
    {"model.json", "feature_columns.json", "prepared_dataset_manifest.json"}
)
SemanticVerifier = Callable[[dict, dict[str, bytes]], None]


class ProductionError(CryptoAIError):
    """A production foundation contract failed; no production authority is implied."""


class ProductionInputError(ProductionError):
    """An input does not satisfy the explicit synthetic production contract."""


class ProductionIntegrityError(ProductionError):
    """Production bytes, provenance, or filesystem identities disagree."""


class ProductionCollisionError(ProductionIntegrityError):
    """An occupied or concurrently locked version must never be replaced."""


@dataclass(frozen=True, slots=True)
class ProductionArtifact:
    """Immutable captured buffers; mutable views are defensive copies."""

    manifest_bytes: bytes
    _payloads: tuple[tuple[str, bytes], ...]

    @property
    def manifest(self) -> dict:
        return _json(self.manifest_bytes)

    @property
    def files(self) -> dict[str, bytes]:
        return dict(self._payloads)


@contextmanager
def _errors():
    try:
        yield
    except ProductionError:
        raise
    except PublicationCollisionError as exc:
        raise ProductionCollisionError("production version already exists") from exc
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
        raise ProductionIntegrityError("invalid synthetic production publication") from exc


def _version(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value) is None:
        raise ProductionInputError("model version must be one safe nonhidden path component")
    return value


def _hash(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _paths(names: object) -> set[str]:
    if not isinstance(names, (set, frozenset, dict)) or not REQUIRED_PAYLOADS <= set(names):
        raise ProductionInputError("production publication is missing a required payload")
    paths = set(names)
    for name in paths:
        if type(name) is not str:
            raise ProductionInputError("production payload paths must be strings")
        _validate_relative_path(name)
        parts = PurePosixPath(name).parts
        if any(part.startswith(".") for part in parts) or MANIFEST_NAME in parts:
            raise ProductionInputError("reserved production payload name")
        if any(PurePosixPath(*parts[:index]).as_posix() in paths for index in range(1, len(parts))):
            raise ProductionInputError("production payload paths collide with directories")
    return paths


def _directories(names: set[str]) -> set[str]:
    return {
        PurePosixPath(*PurePosixPath(name).parts[:index]).as_posix()
        for name in names
        for index in range(1, len(PurePosixPath(name).parts))
    }


def _metadata(value: object) -> dict:
    if type(value) is not dict or value.get("synthetic") is not True:
        raise ProductionInputError("production store only accepts synthetic metadata")
    # Canonical serialization rejects non-JSON types and nonfinite numerics;
    # the engine's semantic verifier enforces the concrete metadata schema.
    return _json(canonicalize(value))


def _inventory(files: Mapping[str, bytes]) -> dict:
    return {
        name: {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        for name, raw in sorted(files.items())
    }


def _manifest(raw: bytes, model_version: str) -> dict:
    value = _json(raw)
    if (
        type(value) is not dict
        or set(value)
        != {"schema_version", "model_version", "metadata", "production_artifact_hashes"}
        or value["schema_version"] != SCHEMA
        or value["model_version"] != model_version
        or canonicalize(value) != raw
    ):
        raise ProductionIntegrityError("invalid production publication manifest")
    _metadata(value["metadata"])
    inventory = value["production_artifact_hashes"]
    if type(inventory) is not dict:
        raise ProductionIntegrityError("production artifact inventory must be an object")
    _paths(inventory)
    for entry in inventory.values():
        if (
            type(entry) is not dict
            or set(entry) != {"sha256", "size_bytes"}
            or not _hash(entry["sha256"])
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise ProductionIntegrityError("invalid production payload hash or length")
    return value


def _capture(
    descriptor: int, model_version: str, expected: bytes | None = None
) -> ProductionArtifact:
    raw, info = _read_regular_file_at_once(
        descriptor, MANIFEST_NAME, description="production manifest"
    )
    if expected is not None and raw != expected:
        raise ProductionIntegrityError("production manifest bytes changed during publication")
    manifest = _manifest(raw, model_version)
    inventory = manifest["production_artifact_hashes"]
    names = set(inventory)
    captured, _ = _capture_publication_tree(
        descriptor,
        manifest_data=raw,
        manifest_stat=info,
        publication_id=model_version,
        manifest_files=inventory,
        expected_paths=names | {MANIFEST_NAME},
        expected_directories=_directories(names),
    )
    return ProductionArtifact(raw, tuple((name, captured[name]) for name in sorted(names)))


def _capture_prepared_payloads(
    descriptor: int, model_version: str, files: Mapping[str, bytes]
) -> None:
    """Inventory one reserved version before its completion marker exists."""
    names = set(files)
    captured, _ = _capture_publication_tree(
        descriptor,
        manifest_data=b"",
        # There is intentionally no manifest entry at this stage. Inventory
        # rejects one before the storage verifier could use this placeholder.
        manifest_stat=os.fstat(descriptor),
        publication_id=model_version,
        manifest_files=_inventory(files),
        expected_paths=names,
        expected_directories=_directories(names),
    )
    if {name: raw for name, raw in captured.items() if name != MANIFEST_NAME} != files:
        raise ProductionIntegrityError("prepared production payloads changed")


def _verify(verify: SemanticVerifier, artifact: ProductionArtifact) -> None:
    if not callable(verify):
        raise ProductionInputError("mandatory production semantic verifier is missing")
    verify(artifact.manifest, artifact.files)


def _unchanged(metadata: object, frozen: bytes) -> None:
    if type(metadata) is not dict or canonicalize(metadata) != frozen:
        raise ProductionIntegrityError("caller production metadata mutated during publication")


def _reserve_version_directory(parent: int, model_version: str) -> int:
    """Reserve under the pinned parent and verify the descriptor before writing.

    Python's descriptor-relative ``os.stat`` with ``follow_symlinks=False`` is
    the ``fstatat(..., AT_SYMLINK_NOFOLLOW)`` lookup. A writer cooperating with
    the version lock cannot replace this name; the identity checks also reject
    a replacement between the lookup and descriptor open.
    """
    try:
        os.mkdir(model_version, mode=0o700, dir_fd=parent)
    except FileExistsError as exc:
        raise ProductionCollisionError("production version already exists") from exc
    looked_up = os.stat(model_version, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISDIR(looked_up.st_mode):
        raise ProductionIntegrityError("reserved production version is not a directory")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(model_version, flags, dir_fd=parent)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(model_version, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or _stat_identity(looked_up) != _stat_identity(opened)
            or _stat_identity(current) != _stat_identity(opened)
        ):
            raise ProductionIntegrityError("reserved production version directory was replaced")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _rollback_owned_directory(descriptor: int) -> None:
    """Remove completion only through our pinned inode, never a swappable name.

    Keeping incomplete payload residue avoids recursively deleting an unrelated
    replacement child. Incomplete directories cannot be reused or read as models.
    """
    try:
        os.unlink(MANIFEST_NAME, dir_fd=descriptor)
    except FileNotFoundError:
        pass
    except OSError:
        retired = f".invalid-production-manifest-{uuid.uuid4().hex}"
        try:
            _atomic_rename_directory_no_replace(descriptor, MANIFEST_NAME, retired)
        except FileNotFoundError:
            pass
        except (CryptoAIError, OSError):
            os.fchmod(descriptor, 0o700)
            try:
                os.unlink(MANIFEST_NAME, dir_fd=descriptor)
            except FileNotFoundError:
                pass
        else:
            try:
                os.unlink(retired, dir_fd=descriptor)
            except OSError:
                pass
    try:
        os.stat(MANIFEST_NAME, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise ProductionIntegrityError("owned production completion marker survived rollback")
    _fsync_directory_descriptor(descriptor, description="invalid production publication")


class ProductionStore:
    """Locked immutable versions; this never reads or changes active_model.json."""

    def __init__(self, root: Path):
        if not isinstance(root, Path) or not root.is_absolute():
            raise ProductionInputError("production root must be an absolute Path")
        with _errors():
            normalized = Path(os.path.abspath(os.fspath(root)))
            temporary_alias = Path(os.path.abspath(tempfile.gettempdir()))
            temporary_root = temporary_alias.resolve(strict=True)
            # Permit only the OS temporary-root alias (macOS /var -> /private/var).
            # Never resolve caller-controlled descendant symlinks before walking.
            if normalized.is_relative_to(temporary_alias):
                self.root = temporary_root / normalized.relative_to(temporary_alias)
            elif normalized.is_relative_to(temporary_root):
                self.root = normalized
            else:
                raise ProductionInputError("production stores require temporary synthetic roots")
            if self.root == temporary_root:
                raise ProductionInputError("the shared temporary directory cannot be a registry")
            descriptor = _create_directory_path_without_symlinks(
                self.root, description="production versions"
            )
            try:
                self._identity = _stat_identity(os.fstat(descriptor))
            finally:
                os.close(descriptor)

    def _parent(self) -> int:
        return _open_directory_path(
            self.root, description="production versions", expected_identity=self._identity
        )

    @staticmethod
    def _absent(parent: int, model_version: str) -> None:
        try:
            os.stat(model_version, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise ProductionCollisionError(
            "production version is already occupied and cannot be replaced"
        )

    def assert_available(self, model_version: str) -> None:
        """Early collision guard; publish repeats this under its exclusive lock."""
        with _errors():
            _version(model_version)
            parent = self._parent()
            try:
                self._absent(parent, model_version)
            finally:
                os.close(parent)

    @contextmanager
    def _locked(self, parent: int, model_version: str):
        name = f".lock-{model_version}"
        flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=parent)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ProductionIntegrityError("version lock is not a unique regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ProductionCollisionError("production version is currently locked") from exc

            def check() -> None:
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (
                    _stat_identity(current) != _stat_identity(os.fstat(descriptor))
                    or not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                ):
                    raise ProductionIntegrityError("production version lock was replaced")
                attached = self._parent()
                os.close(attached)

            check()
            _fsync_directory_descriptor(parent, description="production version lock")
            yield check
            check()
        finally:
            os.close(descriptor)

    def _attached(self, model_version: str, descriptor: int) -> None:
        parent = self._parent()
        try:
            child = _open_directory_at(
                parent, model_version, description="published production version"
            )
            try:
                if _stat_identity(os.fstat(child)) != _stat_identity(os.fstat(descriptor)):
                    raise ProductionIntegrityError("production version directory was replaced")
                if _stat_identity(
                    os.stat(model_version, dir_fd=parent, follow_symlinks=False)
                ) != _stat_identity(os.fstat(descriptor)):
                    raise ProductionIntegrityError("production version directory was detached")
            finally:
                os.close(child)
        finally:
            os.close(parent)

    def get(
        self, model_version: str, *, verify: SemanticVerifier | None = None
    ) -> ProductionArtifact:
        with _errors():
            _version(model_version)
            if not callable(verify):
                raise ProductionInputError("mandatory production semantic verifier is missing")
            parent = self._parent()
            try:
                descriptor = _open_directory_at(
                    parent, model_version, description="production version"
                )
                try:
                    artifact = _capture(descriptor, model_version)
                    self._attached(model_version, descriptor)
                    _verify(verify, artifact)
                    if _capture(descriptor, model_version, artifact.manifest_bytes) != artifact:
                        raise ProductionIntegrityError(
                            "production bytes changed during semantic verification"
                        )
                    self._attached(model_version, descriptor)
                    return artifact
                finally:
                    os.close(descriptor)
            finally:
                os.close(parent)

    def publish(
        self,
        model_version: str,
        files: Mapping[str, bytes],
        metadata: dict,
        *,
        verify: SemanticVerifier | None = None,
    ) -> ProductionArtifact:
        with _errors():
            _version(model_version)
            if not callable(verify):
                raise ProductionInputError("mandatory production semantic verifier is missing")
            if not isinstance(files, Mapping):
                raise ProductionInputError("production payloads must be a mapping")
            checked_files = dict(files)
            _paths(checked_files)
            if any(type(raw) is not bytes for raw in checked_files.values()):
                raise ProductionInputError("production payloads must be exact byte sequences")
            frozen_metadata = _metadata(deepcopy(metadata))
            metadata_bytes = canonicalize(frozen_metadata)
            manifest_bytes = canonicalize(
                {
                    "schema_version": SCHEMA,
                    "model_version": model_version,
                    "metadata": frozen_metadata,
                    "production_artifact_hashes": _inventory(checked_files),
                }
            )
            _manifest(manifest_bytes, model_version)
            candidate = ProductionArtifact(manifest_bytes, tuple(sorted(checked_files.items())))
            _require_descriptor_relative_mutations()
            parent = self._parent()
            owned_version = None
            try:
                with self._locked(parent, model_version) as check_lock:
                    self._absent(parent, model_version)
                    _verify(verify, candidate)
                    _unchanged(metadata, metadata_bytes)
                    _require_atomic_rename_directory_no_replace_at(parent)
                    check_lock()
                    self._absent(parent, model_version)
                    owned_version = _reserve_version_directory(parent, model_version)
                    self._attached(model_version, owned_version)
                    _fsync_directory_descriptor(
                        parent, description="production version reservation"
                    )
                    for name, raw in sorted(checked_files.items()):
                        self._attached(model_version, owned_version)
                        parts = PurePosixPath(name).parts
                        destination = _ensure_directory_chain_at(
                            owned_version, parts[:-1], description="production version payload"
                        )
                        try:
                            self._attached(model_version, owned_version)
                            _write_fsynced_at(destination, parts[-1], raw)
                            self._attached(model_version, owned_version)
                            captured, _ = _read_regular_file_at_once(
                                destination, parts[-1], description=name
                            )
                            if captured != raw:
                                raise ProductionIntegrityError(
                                    "production version payload readback mismatch"
                                )
                        finally:
                            os.close(destination)
                    self._attached(model_version, owned_version)
                    _capture_prepared_payloads(owned_version, model_version, checked_files)
                    self._attached(model_version, owned_version)
                    _fsync_tree_directories_at(owned_version, description="production version")
                    self._attached(model_version, owned_version)
                    _capture_prepared_payloads(owned_version, model_version, checked_files)
                    self._attached(model_version, owned_version)
                    _unchanged(metadata, metadata_bytes)
                    check_lock()
                    pending_name = f".pending-manifest-{uuid.uuid4().hex}"
                    self._attached(model_version, owned_version)
                    _write_fsynced_at(owned_version, pending_name, manifest_bytes)
                    self._attached(model_version, owned_version)
                    pending_bytes, _ = _read_regular_file_at_once(
                        owned_version, pending_name, description="pending production manifest"
                    )
                    if pending_bytes != manifest_bytes:
                        raise ProductionIntegrityError("pending production manifest bytes changed")
                    self._attached(model_version, owned_version)
                    _atomic_rename_directory_no_replace(owned_version, pending_name, MANIFEST_NAME)
                    _fsync_directory_descriptor(
                        owned_version, description="production version completion"
                    )
                    _fsync_directory_descriptor(parent, description="production versions")
                    self._attached(model_version, owned_version)
                    artifact = _capture(owned_version, model_version, manifest_bytes)
                    if artifact != candidate:
                        raise ProductionIntegrityError("production bytes changed after completion")
                    self._attached(model_version, owned_version)
                    _verify(verify, artifact)
                    self._attached(model_version, owned_version)
                    if _capture(owned_version, model_version, manifest_bytes) != candidate:
                        raise ProductionIntegrityError(
                            "production bytes changed during semantic replay"
                        )
                    self._attached(model_version, owned_version)
                    _unchanged(metadata, metadata_bytes)
                    check_lock()
                return artifact
            except BaseException as exc:
                if owned_version is not None:
                    try:
                        _rollback_owned_directory(owned_version)
                    except Exception as cleanup_exc:
                        raise ProductionIntegrityError(
                            "failed to invalidate owned production publication"
                        ) from cleanup_exc
                if isinstance(exc, (KeyboardInterrupt, SystemExit, ProductionError)):
                    raise
                if isinstance(exc, PublicationCollisionError):
                    raise ProductionCollisionError("production version already exists") from exc
                raise ProductionIntegrityError("production publication failed") from exc
            finally:
                if owned_version is not None:
                    os.close(owned_version)
                os.close(parent)
