"""Content-addressed and immutable manifest-last Phase 2 storage."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import stat
import sys
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from crypto_ai.exceptions import PublicationCollisionError, SentimentStorageError
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.exceptions import StorageIntegrityError

SHA256_LENGTH = 64
PUBLICATION_SCHEMA_VERSION = "immutable-publication-v1"
PUBLICATION_MANIFEST_FIELDS = frozenset({"files", "metadata", "publication_id", "schema_version"})
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_STAT_SUPPORTS_DIR_FD = os.stat in os.supports_dir_fd
_STAT_SUPPORTS_NOFOLLOW = os.stat in os.supports_follow_symlinks
_LINK_SUPPORTS_DIR_FD = os.link in os.supports_dir_fd
_LINK_SUPPORTS_NOFOLLOW = os.link in os.supports_follow_symlinks
_MKDIR_SUPPORTS_DIR_FD = os.mkdir in os.supports_dir_fd
_RMDIR_SUPPORTS_DIR_FD = os.rmdir in os.supports_dir_fd
_UNLINK_SUPPORTS_DIR_FD = os.unlink in os.supports_dir_fd


@dataclass(frozen=True, slots=True)
class VerifiedPublication:
    """One manifest and its exact, single-read, hash-verified file buffers."""

    manifest: dict[str, Any]
    files: dict[str, bytes]


_TreeStatFingerprint = tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _PublicationEntrySnapshot:
    """One descriptor-relative lstat captured before payload reads."""

    fingerprint: _TreeStatFingerprint
    is_directory: bool


@dataclass(frozen=True, slots=True)
class _PublicationDirectorySnapshot:
    """One stable directory inventory held open through payload verification."""

    descriptor: int
    fingerprint: _TreeStatFingerprint
    entries: dict[str, _PublicationEntrySnapshot]


class ContentAddressedStore:
    """Append-only exact-byte objects and atomic immutable publications."""

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(os.fspath(Path(root))))
        self.objects_root = self.root / "objects" / "sha256"
        self.publications_root = self.root / "publications"
        self.staging_root = self.root / ".staging"
        root_descriptor = _create_directory_path_without_symlinks(
            self.root,
            description="content store root",
        )
        self._root_identity = _stat_identity(os.fstat(root_descriptor))
        try:
            objects_descriptor = _ensure_directory_at(
                root_descriptor,
                "objects",
                description="content store objects directory",
            )
            try:
                sha256_descriptor = _ensure_directory_at(
                    objects_descriptor,
                    "sha256",
                    description="content store SHA-256 directory",
                )
                os.close(sha256_descriptor)
            finally:
                os.close(objects_descriptor)
            publications_descriptor = _ensure_directory_at(
                root_descriptor,
                "publications",
                description="content store publications directory",
            )
            os.close(publications_descriptor)
            staging_descriptor = _ensure_directory_at(
                root_descriptor,
                ".staging",
                description="content store staging directory",
            )
            os.close(staging_descriptor)
        finally:
            os.close(root_descriptor)

    def put_bytes(self, data: bytes, *, expected_sha256: str | None = None) -> str:
        """Store exact bytes by digest without ever replacing an existing object."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        digest = sha256_bytes(data)
        if expected_sha256 is not None and expected_sha256 != digest:
            raise SentimentStorageError("provided bytes do not match expected SHA-256")
        _require_descriptor_relative_mutations(require_link=True)
        temporary_name = f"object-{digest}-{uuid.uuid4().hex}.tmp"
        with ExitStack() as descriptors:
            root_descriptor = self._open_root_descriptor()
            descriptors.callback(os.close, root_descriptor)
            objects_descriptor = _open_directory_at(
                root_descriptor,
                "objects",
                description="content store objects directory",
            )
            descriptors.callback(os.close, objects_descriptor)
            sha256_descriptor = _open_directory_at(
                objects_descriptor,
                "sha256",
                description="content store SHA-256 directory",
            )
            descriptors.callback(os.close, sha256_descriptor)
            staging_descriptor = _open_directory_at(
                root_descriptor,
                ".staging",
                description="content store staging directory",
            )
            descriptors.callback(os.close, staging_descriptor)
            bucket_descriptor = _ensure_directory_at(
                sha256_descriptor,
                digest[:2],
                description=f"content object bucket {digest[:2]}",
            )
            descriptors.callback(os.close, bucket_descriptor)
            try:
                _write_fsynced_at(staging_descriptor, temporary_name, data)
                try:
                    os.link(
                        temporary_name,
                        digest,
                        src_dir_fd=staging_descriptor,
                        dst_dir_fd=bucket_descriptor,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    self._verify_object_at(bucket_descriptor, digest)
                except OSError as exc:
                    raise SentimentStorageError(
                        f"unable to publish object {digest}: {exc}"
                    ) from exc
                else:
                    _fsync_directory_descriptor(
                        bucket_descriptor,
                        description=f"content object bucket {digest[:2]}",
                    )
                    self._verify_object_at(bucket_descriptor, digest)
            finally:
                try:
                    os.unlink(temporary_name, dir_fd=staging_descriptor)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    raise SentimentStorageError(
                        f"unable to clean object staging file: {exc}"
                    ) from exc
        return digest

    def put_canonical_json(self, value: Any, *, expected_sha256: str | None = None) -> str:
        return self.put_bytes(canonicalize(value), expected_sha256=expected_sha256)

    def get_bytes(self, digest: str) -> bytes:
        _validate_digest(digest)
        return self._read_verified_object(digest)

    def publish_bundle(
        self,
        publication_id: str,
        files: Mapping[str, bytes],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Stage, sync, and atomically publish a complete immutable bundle.

        ``manifest.json`` is written after every payload file. The staging directory then
        becomes visible in one atomic no-replace rename.
        """
        _validate_publication_id(publication_id)
        if not files:
            raise SentimentStorageError("a publication must contain at least one payload file")
        normalized_files: dict[str, bytes] = {}
        for raw_name, data in files.items():
            name = _validate_relative_path(raw_name)
            if name == "manifest.json":
                raise SentimentStorageError("manifest.json is reserved")
            if not isinstance(data, bytes):
                raise SentimentStorageError(f"publication file {name} must contain bytes")
            normalized_files[name] = data
        if len(normalized_files) != len(files):
            raise SentimentStorageError("publication paths are not unique after normalization")

        _require_descriptor_relative_mutations()
        _require_atomic_rename_directory_no_replace_at()
        final_directory = self.publications_root / publication_id
        staging_name = f".staging-{publication_id}-{uuid.uuid4().hex}"
        with ExitStack() as descriptors:
            root_descriptor = self._open_root_descriptor()
            descriptors.callback(os.close, root_descriptor)
            publications_descriptor = _open_directory_at(
                root_descriptor,
                "publications",
                description="content store publications directory",
            )
            descriptors.callback(os.close, publications_descriptor)
            staging_descriptor: int | None = None
            staging_exists = False
            try:
                try:
                    os.mkdir(
                        staging_name,
                        mode=0o700,
                        dir_fd=publications_descriptor,
                    )
                except FileExistsError as exc:
                    raise SentimentStorageError(
                        f"publication staging directory already exists: {staging_name}"
                    ) from exc
                except OSError as exc:
                    raise SentimentStorageError(
                        f"unable to create publication staging directory: {exc}"
                    ) from exc
                staging_exists = True
                staging_descriptor = _open_directory_at(
                    publications_descriptor,
                    staging_name,
                    description=f"publication staging directory {staging_name}",
                )
                file_manifest: dict[str, dict[str, int | str]] = {}
                for name in sorted(normalized_files):
                    data = normalized_files[name]
                    name_parts = PurePosixPath(name).parts
                    parent_descriptor = _ensure_directory_chain_at(
                        staging_descriptor,
                        name_parts[:-1],
                        description=f"parent directory for publication member {name}",
                    )
                    try:
                        _write_fsynced_at(parent_descriptor, name_parts[-1], data)
                    finally:
                        os.close(parent_descriptor)
                    file_manifest[name] = {
                        "sha256": sha256_bytes(data),
                        "size_bytes": len(data),
                    }
                manifest = {
                    "files": file_manifest,
                    "metadata": dict(metadata or {}),
                    "publication_id": publication_id,
                    "schema_version": PUBLICATION_SCHEMA_VERSION,
                }
                _write_fsynced_at(
                    staging_descriptor,
                    "manifest.json",
                    canonicalize(manifest),
                )
                _fsync_tree_directories_at(
                    staging_descriptor,
                    description=f"publication staging directory {staging_name}",
                )
                _atomic_rename_directory_no_replace(
                    publications_descriptor,
                    staging_name,
                    publication_id,
                )
                staging_exists = False
                _fsync_directory_descriptor(
                    publications_descriptor,
                    description="content store publications directory",
                )
            except PublicationCollisionError:
                if staging_descriptor is not None:
                    os.close(staging_descriptor)
                    staging_descriptor = None
                if staging_exists:
                    _cleanup_staging_at(publications_descriptor, staging_name)
                raise
            except Exception as exc:
                if staging_descriptor is not None:
                    os.close(staging_descriptor)
                    staging_descriptor = None
                if staging_exists:
                    _cleanup_staging_at(publications_descriptor, staging_name)
                if isinstance(exc, SentimentStorageError):
                    raise
                raise SentimentStorageError(f"unable to publish {publication_id}: {exc}") from exc
            finally:
                if staging_descriptor is not None:
                    os.close(staging_descriptor)
        self.verify_publication(publication_id)
        return final_directory

    def verify_publication(self, publication_id: str) -> dict[str, Any]:
        """Load and verify every byte referenced by a publication manifest."""
        return self.read_publication(publication_id).manifest

    def read_publication(
        self,
        publication_id: str,
        *,
        metadata_prevalidator: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> VerifiedPublication:
        """Validate metadata, then capture each payload once from one anchored directory."""
        _validate_publication_id(publication_id)
        publication_descriptor = _open_store_directory(
            self.root,
            ("publications", publication_id),
            description=f"publication directory {publication_id}",
            expected_root_identity=self._root_identity,
        )
        try:
            try:
                raw_manifest, manifest_stat = _read_regular_file_at_once(
                    publication_descriptor,
                    "manifest.json",
                    description=f"publication manifest {publication_id}",
                )
                manifest = json.loads(
                    raw_manifest.decode("utf-8"),
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"invalid JSON constant {value}")
                    ),
                )
            except SentimentStorageError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise SentimentStorageError(
                    f"invalid publication manifest: {publication_id}"
                ) from exc
            if not isinstance(manifest, dict) or canonicalize(manifest) != raw_manifest:
                raise SentimentStorageError("publication manifest is not canonical JCS")
            if set(manifest) != PUBLICATION_MANIFEST_FIELDS:
                raise SentimentStorageError("publication manifest has unexpected fields")
            if manifest["schema_version"] != PUBLICATION_SCHEMA_VERSION:
                raise SentimentStorageError("unsupported publication manifest schema")
            if manifest.get("publication_id") != publication_id:
                raise SentimentStorageError("publication ID does not match its manifest")
            if not isinstance(manifest["metadata"], dict):
                raise SentimentStorageError("publication manifest metadata must be an object")
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                raise SentimentStorageError("publication manifest has no file inventory")
            expected_paths = {"manifest.json"}
            expected_directories: set[str] = set()
            for raw_name, descriptor in files.items():
                name = _validate_relative_path(raw_name)
                expected_paths.add(name)
                name_parts = PurePosixPath(name).parts
                expected_directories.update(
                    PurePosixPath(*name_parts[:index]).as_posix()
                    for index in range(1, len(name_parts))
                )
                if not isinstance(descriptor, dict) or set(descriptor) != {
                    "sha256",
                    "size_bytes",
                }:
                    raise SentimentStorageError(f"invalid manifest entry for {name}")
                size_bytes = descriptor["size_bytes"]
                if (
                    isinstance(size_bytes, bool)
                    or not isinstance(size_bytes, int)
                    or size_bytes < 0
                ):
                    raise SentimentStorageError(f"invalid manifest size for {name}")

            if metadata_prevalidator is not None:
                metadata_prevalidator(deepcopy(manifest["metadata"]))

            captured_files, actual_directories = _capture_publication_tree(
                publication_descriptor,
                manifest_data=raw_manifest,
                manifest_stat=manifest_stat,
                publication_id=publication_id,
                manifest_files=files,
                expected_paths=expected_paths,
                expected_directories=expected_directories,
            )
            if actual_directories != expected_directories:  # pragma: no cover - checked inside
                raise StorageIntegrityError(
                    f"publication {publication_id} changed during verification"
                )
            verified_files = {name: captured_files[name] for name in files}
            return VerifiedPublication(manifest=manifest, files=verified_files)
        finally:
            os.close(publication_descriptor)

    def _object_path(self, digest: str) -> Path:
        _validate_digest(digest)
        return self.objects_root / digest[:2] / digest

    def _open_root_descriptor(self) -> int:
        return _open_directory_path(
            self.root,
            description="content store root",
            expected_identity=self._root_identity,
        )

    def _read_verified_object(self, digest: str) -> bytes:
        data = _read_store_relative_regular_file_once(
            self.root,
            ("objects", "sha256", digest[:2], digest),
            description=f"content object {digest}",
            expected_root_identity=self._root_identity,
        )
        if sha256_bytes(data) != digest:
            raise SentimentStorageError(f"content object hash mismatch: {digest}")
        return data

    def _verify_object(self, digest: str) -> None:
        self._read_verified_object(digest)

    @staticmethod
    def _verify_object_at(bucket_descriptor: int, digest: str) -> None:
        data, _ = _read_regular_file_at_once(
            bucket_descriptor,
            digest,
            description=f"content object {digest}",
        )
        if sha256_bytes(data) != digest:
            raise SentimentStorageError(f"content object hash mismatch: {digest}")


def _validate_digest(digest: str) -> None:
    if (
        not isinstance(digest, str)
        or len(digest) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise SentimentStorageError("digest must be lowercase SHA-256 hex")


def _validate_publication_id(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in value
        )
        or value in {".", ".."}
        or value.startswith(".staging-")
    ):
        raise SentimentStorageError("unsafe publication ID")


def _validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise SentimentStorageError("publication paths must use nonempty POSIX-relative names")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise SentimentStorageError(f"unsafe publication path: {value}")
    return path.as_posix()


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _regular_file_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _stat_content_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stat_tree_fingerprint(value: os.stat_result) -> _TreeStatFingerprint:
    """Fingerprint identity, kind, ownership, links, and mutation-relevant metadata."""
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_descriptor_relative_operations() -> None:
    if not _OPEN_SUPPORTS_DIR_FD or not _STAT_SUPPORTS_DIR_FD:
        raise SentimentStorageError("platform lacks descriptor-relative storage traversal support")
    if not _STAT_SUPPORTS_NOFOLLOW:
        raise SentimentStorageError("platform lacks no-follow storage stat support")


def _require_descriptor_relative_mutations(*, require_link: bool = False) -> None:
    _require_descriptor_relative_operations()
    if not all(
        (
            _MKDIR_SUPPORTS_DIR_FD,
            _RMDIR_SUPPORTS_DIR_FD,
            _UNLINK_SUPPORTS_DIR_FD,
        )
    ):
        raise SentimentStorageError("platform lacks descriptor-relative storage mutation support")
    if require_link and (not _LINK_SUPPORTS_DIR_FD or not _LINK_SUPPORTS_NOFOLLOW):
        raise SentimentStorageError(
            "platform lacks descriptor-relative no-follow hard-link support"
        )


def _ensure_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    description: str,
    mode: int = 0o755,
) -> int:
    """Create or open one child directory without following a preexisting link."""
    _require_descriptor_relative_mutations()
    try:
        os.mkdir(name, mode=mode, dir_fd=parent_descriptor)
    except FileExistsError:
        pass
    except OSError as exc:
        raise SentimentStorageError(f"unable to create {description}: {exc}") from exc
    return _open_directory_at(parent_descriptor, name, description=description)


def _ensure_directory_chain_at(
    parent_descriptor: int,
    relative_parts: tuple[str, ...],
    *,
    description: str,
) -> int:
    """Create publication-owned nested directories below an anchored descriptor."""
    descriptor = os.dup(parent_descriptor)
    try:
        for index, part in enumerate(relative_parts):
            child_descriptor = _ensure_directory_at(
                descriptor,
                part,
                mode=0o700,
                description=(
                    description
                    if index == len(relative_parts) - 1
                    else f"publication staging directory {'/'.join(relative_parts[: index + 1])}"
                ),
            )
            os.close(descriptor)
            descriptor = child_descriptor
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_directory_path(
    path: Path,
    *,
    description: str,
    expected_identity: tuple[int, int] | None = None,
) -> int:
    """Descriptor-walk one absolute path without following any component symlink."""
    parts = _absolute_directory_parts(path)
    descriptor = _open_filesystem_root_descriptor()
    try:
        for index, part in enumerate(parts):
            child_descriptor = _open_directory_at(
                descriptor,
                part,
                description=(
                    description
                    if index == len(parts) - 1
                    else f"content store path component {'/'.join(parts[: index + 1])}"
                ),
            )
            os.close(descriptor)
            descriptor = child_descriptor
        opened_stat = os.fstat(descriptor)
        if expected_identity is not None and _stat_identity(opened_stat) != expected_identity:
            raise SentimentStorageError(f"{description} was replaced")
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _create_directory_path_without_symlinks(path: Path, *, description: str) -> int:
    """Descriptor-walk and mkdir an absolute path without following any symlink."""
    parts = _absolute_directory_parts(path)
    if not parts:
        raise SentimentStorageError("content store root must not be the filesystem root")
    descriptor = _open_filesystem_root_descriptor()
    try:
        for index, part in enumerate(parts):
            child_descriptor = _ensure_directory_at(
                descriptor,
                part,
                description=(
                    description
                    if index == len(parts) - 1
                    else f"content store path component {'/'.join(parts[: index + 1])}"
                ),
            )
            os.close(descriptor)
            descriptor = child_descriptor
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _absolute_directory_parts(path: Path) -> tuple[str, ...]:
    if not path.is_absolute() or path.anchor != os.sep:
        raise SentimentStorageError("content store paths must be absolute POSIX paths")
    return tuple(part for part in path.parts if part != os.sep)


def _open_filesystem_root_descriptor() -> int:
    descriptor: int | None = None
    try:
        descriptor = os.open(os.sep, _directory_open_flags())
        opened_stat = os.fstat(descriptor)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise SentimentStorageError(f"unable to open filesystem root: {exc}") from exc
    if not stat.S_ISDIR(opened_stat.st_mode):  # pragma: no cover - POSIX invariant
        os.close(descriptor)
        raise SentimentStorageError("filesystem root is not a directory")
    return descriptor


def _open_directory_at(parent_descriptor: int, name: str, *, description: str) -> int:
    """Open a direct child directory relative to an already trusted descriptor."""
    _require_descriptor_relative_operations()
    descriptor: int | None = None
    try:
        pre_open_stat = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(pre_open_stat.st_mode):
            raise SentimentStorageError(f"{description} must not be a symlink")
        if not stat.S_ISDIR(pre_open_stat.st_mode):
            raise SentimentStorageError(f"{description} is not a directory")
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise SentimentStorageError(f"{description} is not a directory")
        if _stat_identity(pre_open_stat) != _stat_identity(opened_stat):
            raise SentimentStorageError(f"{description} changed while it was opened")
        result = descriptor
        descriptor = None
        return result
    except FileNotFoundError as exc:
        raise SentimentStorageError(f"{description} is missing") from exc
    except SentimentStorageError:
        raise
    except OSError as exc:
        raise SentimentStorageError(f"unable to open {description}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_store_directory(
    store_root: Path,
    relative_parts: tuple[str, ...],
    *,
    description: str,
    expected_root_identity: tuple[int, int] | None = None,
) -> int:
    """Traverse store-relative directory components from one trusted root descriptor."""
    descriptor = _open_directory_path(
        store_root,
        description="content store root",
        expected_identity=expected_root_identity,
    )
    try:
        for index, part in enumerate(relative_parts):
            child_description = (
                description
                if index == len(relative_parts) - 1
                else f"content store directory {'/'.join(relative_parts[: index + 1])}"
            )
            child_descriptor = _open_directory_at(
                descriptor,
                part,
                description=child_description,
            )
            os.close(descriptor)
            descriptor = child_descriptor
        result = descriptor
        descriptor = -1
        return result
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular_file_at_once(
    parent_descriptor: int,
    name: str,
    *,
    description: str,
) -> tuple[bytes, os.stat_result]:
    """Read one direct child regular file once from an anchored directory descriptor."""
    _require_descriptor_relative_operations()
    descriptor: int | None = None
    try:
        pre_open_stat = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(pre_open_stat.st_mode):
            raise SentimentStorageError(f"{description} must not be a symlink")
        if not stat.S_ISREG(pre_open_stat.st_mode):
            raise SentimentStorageError(f"{description} is not a regular file")
        descriptor = os.open(
            name,
            _regular_file_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise SentimentStorageError(f"{description} is not a regular file")
        if _stat_identity(pre_open_stat) != _stat_identity(opened_stat):
            raise SentimentStorageError(f"{description} changed while it was opened")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            data = handle.read()
            final_stat = os.fstat(handle.fileno())
        if _stat_content_fingerprint(opened_stat) != _stat_content_fingerprint(final_stat):
            raise SentimentStorageError(f"{description} changed while it was read")
    except FileNotFoundError as exc:
        raise SentimentStorageError(f"{description} is missing") from exc
    except SentimentStorageError:
        raise
    except OSError as exc:
        raise SentimentStorageError(f"unable to read {description}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(data, bytes):  # pragma: no cover - binary file contract
        raise SentimentStorageError(f"{description} did not yield bytes")
    return data, opened_stat


def _read_store_relative_regular_file_once(
    store_root: Path,
    relative_parts: tuple[str, ...],
    *,
    description: str,
    expected_root_identity: tuple[int, int] | None = None,
) -> bytes:
    if not relative_parts:
        raise SentimentStorageError("storage read requires a relative file path")
    parent_descriptor = _open_store_directory(
        store_root,
        relative_parts[:-1],
        description=f"parent directory for {description}",
        expected_root_identity=expected_root_identity,
    )
    try:
        data, _ = _read_regular_file_at_once(
            parent_descriptor,
            relative_parts[-1],
            description=description,
        )
        return data
    finally:
        os.close(parent_descriptor)


def _capture_publication_tree(
    publication_descriptor: int,
    *,
    manifest_data: bytes,
    manifest_stat: os.stat_result,
    publication_id: str,
    manifest_files: Mapping[str, Mapping[str, Any]],
    expected_paths: set[str],
    expected_directories: set[str],
) -> tuple[dict[str, bytes], set[str]]:
    """Verify one anchored tree with baseline, capture, and bounded stability sweeps.

    POSIX traversal cannot create an atomic recursive snapshot against an uncooperative
    writer that already has filesystem authority. Returned buffers are descriptor-anchored,
    manifest-verified, and survive two opposite-order confirmation sweeps without another
    content callback after the final sweep.
    """
    captured_files = {"manifest.json": manifest_data}
    directories: set[str] = set()
    snapshots: dict[PurePosixPath, _PublicationDirectorySnapshot] = {}

    def changed_during_verification(
        relative_name: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        suffix = f": {relative_name}" if relative_name else ""
        error = StorageIntegrityError(
            f"publication {publication_id} changed during verification{suffix}"
        )
        if cause is None:
            raise error
        raise error from cause

    def directory_names(
        directory_descriptor: int,
        *,
        changed_is_integrity_failure: bool,
    ) -> tuple[str, ...]:
        try:
            names = sorted(os.listdir(directory_descriptor))
        except OSError as exc:
            if changed_is_integrity_failure:
                changed_during_verification()
            raise SentimentStorageError(
                f"unable to inventory publication {publication_id}: {exc}"
            ) from exc
        for name in names:
            if not isinstance(name, str) or name in {"", ".", ".."} or "/" in name:
                raise SentimentStorageError("publication contains an unsafe directory entry")
        return tuple(names)

    def entry_stat(
        directory_descriptor: int,
        name: str,
        relative_name: str,
        *,
        changed_is_integrity_failure: bool,
    ) -> os.stat_result:
        try:
            return os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            if changed_is_integrity_failure:
                changed_during_verification(relative_name)
            raise SentimentStorageError(
                f"unable to inspect publication entry {relative_name}: {exc}"
            ) from exc

    def assert_entry_kind(
        value: os.stat_result,
        relative_name: str,
        *,
        changed_is_integrity_failure: bool,
    ) -> bool:
        if stat.S_ISLNK(value.st_mode):
            if changed_is_integrity_failure:
                changed_during_verification(relative_name)
            raise SentimentStorageError(f"publication entry {relative_name} must not be a symlink")
        if stat.S_ISDIR(value.st_mode):
            return True
        if stat.S_ISREG(value.st_mode):
            return False
        if changed_is_integrity_failure:
            changed_during_verification(relative_name)
        raise SentimentStorageError(
            f"publication entry {relative_name} is not a regular file or directory"
        )

    with ExitStack() as directory_descriptors:

        def inventory_directory(
            directory_descriptor: int,
            prefix: PurePosixPath,
        ) -> None:
            initial_directory_fingerprint = _stat_tree_fingerprint(os.fstat(directory_descriptor))
            names = directory_names(
                directory_descriptor,
                changed_is_integrity_failure=False,
            )
            entries: dict[str, _PublicationEntrySnapshot] = {}
            for name in names:
                relative_path = prefix / name
                relative_name = relative_path.as_posix()
                initial_stat = entry_stat(
                    directory_descriptor,
                    name,
                    relative_name,
                    changed_is_integrity_failure=False,
                )
                is_directory = assert_entry_kind(
                    initial_stat,
                    relative_name,
                    changed_is_integrity_failure=False,
                )
                initial_fingerprint = _stat_tree_fingerprint(initial_stat)
                entries[name] = _PublicationEntrySnapshot(
                    fingerprint=initial_fingerprint,
                    is_directory=is_directory,
                )
                if not is_directory:
                    continue
                directories.add(relative_name)
                child_descriptor = _open_directory_at(
                    directory_descriptor,
                    name,
                    description=f"publication directory {relative_name}",
                )
                directory_descriptors.callback(os.close, child_descriptor)
                if _stat_tree_fingerprint(os.fstat(child_descriptor)) != initial_fingerprint:
                    changed_during_verification(relative_name)
                inventory_directory(child_descriptor, relative_path)
                final_entry_stat = entry_stat(
                    directory_descriptor,
                    name,
                    relative_name,
                    changed_is_integrity_failure=True,
                )
                if _stat_tree_fingerprint(final_entry_stat) != initial_fingerprint:
                    changed_during_verification(relative_name)

            if (
                directory_names(
                    directory_descriptor,
                    changed_is_integrity_failure=True,
                )
                != names
                or _stat_tree_fingerprint(os.fstat(directory_descriptor))
                != initial_directory_fingerprint
            ):
                changed_during_verification(prefix.as_posix() if prefix.parts else None)
            snapshots[prefix] = _PublicationDirectorySnapshot(
                descriptor=directory_descriptor,
                fingerprint=initial_directory_fingerprint,
                entries=entries,
            )

        def capture_directory(prefix: PurePosixPath) -> None:
            snapshot = snapshots[prefix]
            for name, entry in snapshot.entries.items():
                relative_path = prefix / name
                relative_name = relative_path.as_posix()
                if entry.is_directory:
                    capture_directory(relative_path)
                    continue
                if relative_name == "manifest.json":
                    if entry.fingerprint != _stat_tree_fingerprint(manifest_stat):
                        changed_during_verification(relative_name)
                    continue
                try:
                    data, opened_stat = _read_regular_file_at_once(
                        snapshot.descriptor,
                        name,
                        description=f"publication member {relative_name}",
                    )
                except SentimentStorageError as exc:
                    changed_during_verification(relative_name, cause=exc)
                if _stat_tree_fingerprint(opened_stat) != entry.fingerprint:
                    changed_during_verification(relative_name)
                final_entry_stat = entry_stat(
                    snapshot.descriptor,
                    name,
                    relative_name,
                    changed_is_integrity_failure=True,
                )
                if _stat_tree_fingerprint(final_entry_stat) != entry.fingerprint:
                    changed_during_verification(relative_name)
                captured_files[relative_name] = data

        def verify_directory(prefix: PurePosixPath) -> None:
            snapshot = snapshots[prefix]
            names = directory_names(
                snapshot.descriptor,
                changed_is_integrity_failure=True,
            )
            current_entries: dict[str, _PublicationEntrySnapshot] = {}
            for name in names:
                relative_path = prefix / name
                relative_name = relative_path.as_posix()
                current_stat = entry_stat(
                    snapshot.descriptor,
                    name,
                    relative_name,
                    changed_is_integrity_failure=True,
                )
                current_entries[name] = _PublicationEntrySnapshot(
                    fingerprint=_stat_tree_fingerprint(current_stat),
                    is_directory=assert_entry_kind(
                        current_stat,
                        relative_name,
                        changed_is_integrity_failure=True,
                    ),
                )
            if (
                current_entries != snapshot.entries
                or _stat_tree_fingerprint(os.fstat(snapshot.descriptor)) != snapshot.fingerprint
            ):
                changed_during_verification(prefix.as_posix() if prefix.parts else None)

        inventory_directory(publication_descriptor, PurePosixPath())

        inventoried_paths = {
            (prefix / name).as_posix()
            for prefix, snapshot in snapshots.items()
            for name, entry in snapshot.entries.items()
            if not entry.is_directory
        }
        if inventoried_paths != expected_paths or directories != expected_directories:
            raise SentimentStorageError(
                "publication contains unmanifested or missing files or directories"
            )

        capture_directory(PurePosixPath())

        for name, descriptor in manifest_files.items():
            data = captured_files[name]
            if descriptor["size_bytes"] != len(data) or descriptor["sha256"] != sha256_bytes(data):
                raise SentimentStorageError(f"publication member hash mismatch: {name}")

        verification_order = tuple(sorted(snapshots, key=lambda path: path.as_posix()))
        for prefix in verification_order:
            verify_directory(prefix)
        # Reverse order ensures a mutation injected into an already-confirmed subtree
        # while a later sibling is checked is visited again before any buffers escape.
        for prefix in reversed(verification_order):
            verify_directory(prefix)
    return captured_files, directories


def _write_fsynced_at(parent_descriptor: int, name: str, data: bytes) -> None:
    """Create one immutable file relative to an anchored directory and fsync it."""
    _require_descriptor_relative_mutations()
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise SentimentStorageError(f"unable to write immutable file {name}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory_descriptor(descriptor: int, *, description: str) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise SentimentStorageError(f"unable to fsync {description}: {exc}") from exc


def _fsync_tree_directories_at(descriptor: int, *, description: str) -> None:
    """Fsync an anchored tree without following injected filesystem objects."""
    try:
        names = sorted(os.listdir(descriptor))
    except OSError as exc:
        raise SentimentStorageError(f"unable to inventory {description}: {exc}") from exc
    for name in names:
        try:
            entry_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as exc:
            raise SentimentStorageError(
                f"unable to inspect staged publication entry {name}: {exc}"
            ) from exc
        if stat.S_ISLNK(entry_stat.st_mode):
            raise SentimentStorageError(f"staged publication entry {name} must not be a symlink")
        if stat.S_ISDIR(entry_stat.st_mode):
            child_descriptor = _open_directory_at(
                descriptor,
                name,
                description=f"staged publication directory {name}",
            )
            try:
                _fsync_tree_directories_at(
                    child_descriptor,
                    description=f"staged publication directory {name}",
                )
            finally:
                os.close(child_descriptor)
            continue
        if not stat.S_ISREG(entry_stat.st_mode):
            raise SentimentStorageError(
                f"staged publication entry {name} is not a regular file or directory"
            )
    _fsync_directory_descriptor(descriptor, description=description)


def _remove_entry_at(parent_descriptor: int, name: str) -> None:
    entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode):
        child_descriptor = _open_directory_at(
            parent_descriptor,
            name,
            description=f"publication staging directory {name}",
        )
        try:
            for child_name in os.listdir(child_descriptor):
                _remove_entry_at(child_descriptor, child_name)
        finally:
            os.close(child_descriptor)
        os.rmdir(name, dir_fd=parent_descriptor)
        return
    os.unlink(name, dir_fd=parent_descriptor)


def _cleanup_staging_at(publications_descriptor: int, staging_name: str) -> None:
    if not staging_name.startswith(".staging-"):
        return
    try:
        _remove_entry_at(publications_descriptor, staging_name)
    except (OSError, SentimentStorageError):
        pass


def _require_atomic_rename_directory_no_replace_at() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    symbol = "renameatx_np" if sys.platform == "darwin" else "renameat2"
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise SentimentStorageError(
            "platform lacks descriptor-relative atomic no-replace directory rename"
        )
    if not hasattr(libc, symbol):
        raise SentimentStorageError("atomic no-replace rename is unavailable")


def _atomic_rename_directory_no_replace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically rename below one anchored parent without replacing a destination."""
    source_bytes = os.fsencode(source_name)
    destination_bytes = os.fsencode(destination_name)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_exclusive = libc.renameatx_np
        except AttributeError as exc:
            raise SentimentStorageError("atomic no-replace rename is unavailable") from exc
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = rename_exclusive(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_no_replace = libc.renameat2
        except AttributeError as exc:
            raise SentimentStorageError("atomic no-replace rename is unavailable") from exc
        rename_no_replace.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_no_replace.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = rename_no_replace(
            parent_descriptor,
            source_bytes,
            parent_descriptor,
            destination_bytes,
            0x00000001,
        )
    else:
        raise SentimentStorageError(
            "platform lacks descriptor-relative atomic no-replace directory rename"
        )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PublicationCollisionError(f"publication already exists: {destination_name}")
    raise SentimentStorageError(
        f"unable to publish {destination_name}: {os.strerror(error_number)}"
    )
