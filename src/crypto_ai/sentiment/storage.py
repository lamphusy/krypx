"""Content-addressed and immutable manifest-last Phase 2 storage."""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import stat
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from crypto_ai.exceptions import PublicationCollisionError, SentimentStorageError
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes

SHA256_LENGTH = 64
PUBLICATION_SCHEMA_VERSION = "immutable-publication-v1"
PUBLICATION_MANIFEST_FIELDS = frozenset({"files", "metadata", "publication_id", "schema_version"})


@dataclass(frozen=True, slots=True)
class VerifiedPublication:
    """One manifest and its exact, single-read, hash-verified file buffers."""

    manifest: dict[str, Any]
    files: dict[str, bytes]


class ContentAddressedStore:
    """Append-only exact-byte objects and atomic immutable publications."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.objects_root = self.root / "objects" / "sha256"
        self.publications_root = self.root / "publications"
        self.staging_root = self.root / ".staging"
        for directory in (self.objects_root, self.publications_root, self.staging_root):
            directory.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes, *, expected_sha256: str | None = None) -> str:
        """Store exact bytes by digest without ever replacing an existing object."""
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        digest = sha256_bytes(data)
        if expected_sha256 is not None and expected_sha256 != digest:
            raise SentimentStorageError("provided bytes do not match expected SHA-256")
        destination = self._object_path(digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.staging_root / f"object-{digest}-{uuid.uuid4().hex}.tmp"
        try:
            _write_fsynced(temporary, data)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                self._verify_object(destination, digest)
            except OSError as exc:
                raise SentimentStorageError(f"unable to publish object {digest}: {exc}") from exc
            else:
                _fsync_directory(destination.parent)
                self._verify_object(destination, digest)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise SentimentStorageError(f"unable to clean object staging file: {exc}") from exc
        return digest

    def put_canonical_json(self, value: Any, *, expected_sha256: str | None = None) -> str:
        return self.put_bytes(canonicalize(value), expected_sha256=expected_sha256)

    def get_bytes(self, digest: str) -> bytes:
        path = self._object_path(digest)
        return self._read_verified_object(path, digest)

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

        final_directory = self.publications_root / publication_id
        staging_directory = self.publications_root / (
            f".staging-{publication_id}-{uuid.uuid4().hex}"
        )
        try:
            staging_directory.mkdir(mode=0o700)
            file_manifest: dict[str, dict[str, int | str]] = {}
            for name in sorted(normalized_files):
                data = normalized_files[name]
                target = staging_directory / PurePosixPath(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                _write_fsynced(target, data)
                file_manifest[name] = {"sha256": sha256_bytes(data), "size_bytes": len(data)}
            manifest = {
                "files": file_manifest,
                "metadata": dict(metadata or {}),
                "publication_id": publication_id,
                "schema_version": PUBLICATION_SCHEMA_VERSION,
            }
            _write_fsynced(staging_directory / "manifest.json", canonicalize(manifest))
            _fsync_tree_directories(staging_directory)
            _atomic_rename_directory_no_replace(staging_directory, final_directory)
            _fsync_directory(self.publications_root)
        except PublicationCollisionError:
            _cleanup_staging(staging_directory, self.publications_root)
            raise
        except Exception as exc:
            _cleanup_staging(staging_directory, self.publications_root)
            if isinstance(exc, SentimentStorageError):
                raise
            raise SentimentStorageError(f"unable to publish {publication_id}: {exc}") from exc
        self.verify_publication(publication_id)
        return final_directory

    def verify_publication(self, publication_id: str) -> dict[str, Any]:
        """Load and verify every byte referenced by a publication manifest."""
        return self.read_publication(publication_id).manifest

    def read_publication(self, publication_id: str) -> VerifiedPublication:
        """Capture each publication file once and return those exact verified buffers."""
        _validate_publication_id(publication_id)
        directory = self.publications_root / publication_id
        manifest_path = directory / "manifest.json"
        try:
            raw_manifest = _read_regular_file_once(
                manifest_path, description=f"publication manifest {publication_id}"
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
            raise SentimentStorageError(f"invalid publication manifest: {publication_id}") from exc
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
        verified_files: dict[str, bytes] = {}
        for raw_name, descriptor in files.items():
            name = _validate_relative_path(raw_name)
            expected_paths.add(name)
            if not isinstance(descriptor, dict) or set(descriptor) != {"sha256", "size_bytes"}:
                raise SentimentStorageError(f"invalid manifest entry for {name}")
            size_bytes = descriptor["size_bytes"]
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
                raise SentimentStorageError(f"invalid manifest size for {name}")
            path = directory / PurePosixPath(name)
            data = _read_regular_file_once(path, description=f"publication member {name}")
            if size_bytes != len(data) or descriptor["sha256"] != sha256_bytes(data):
                raise SentimentStorageError(f"publication member hash mismatch: {name}")
            verified_files[name] = data
        actual_paths = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        if actual_paths != expected_paths:
            raise SentimentStorageError("publication contains unmanifested or missing files")
        return VerifiedPublication(manifest=manifest, files=verified_files)

    def _object_path(self, digest: str) -> Path:
        _validate_digest(digest)
        return self.objects_root / digest[:2] / digest

    @staticmethod
    def _read_verified_object(path: Path, digest: str) -> bytes:
        data = _read_regular_file_once(path, description=f"content object {digest}")
        if sha256_bytes(data) != digest:
            raise SentimentStorageError(f"content object hash mismatch: {digest}")
        return data

    @staticmethod
    def _verify_object(path: Path, digest: str) -> None:
        ContentAddressedStore._read_verified_object(path, digest)


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
    if not isinstance(value, str) or not value or "\\" in value:
        raise SentimentStorageError("publication paths must use nonempty POSIX-relative names")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SentimentStorageError(f"unsafe publication path: {value}")
    return path.as_posix()


def _read_regular_file_once(path: Path, *, description: str) -> bytes:
    """Open without following symlinks, read once, and return that captured buffer."""
    descriptor: int | None = None
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    flags |= getattr(os, "O_NONBLOCK", 0)
    nofollow_available = hasattr(os, "O_NOFOLLOW")
    if nofollow_available:
        flags |= os.O_NOFOLLOW
    pre_open_stat: os.stat_result | None = None
    try:
        if not nofollow_available:
            pre_open_stat = path.lstat()
            if stat.S_ISLNK(pre_open_stat.st_mode):
                raise SentimentStorageError(f"{description} must not be a symlink")
        descriptor = os.open(path, flags)
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise SentimentStorageError(f"{description} is not a regular file")
        if pre_open_stat is not None and (
            pre_open_stat.st_dev,
            pre_open_stat.st_ino,
        ) != (opened_stat.st_dev, opened_stat.st_ino):
            raise SentimentStorageError(f"{description} changed while it was opened")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            data = handle.read()
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
    return data


def _write_fsynced(path: Path, data: bytes) -> None:
    """Create one file exclusively, write exact bytes, flush, and fsync."""
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise SentimentStorageError(f"unable to write immutable file {path.name}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise SentimentStorageError(f"unable to fsync directory {path}: {exc}") from exc


def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)


def _cleanup_staging(staging: Path, publications_root: Path) -> None:
    if not staging.exists():
        return
    try:
        resolved = staging.resolve(strict=False)
        root = publications_root.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return
    if resolved.parent == root and staging.name.startswith(".staging-"):
        shutil.rmtree(staging, ignore_errors=True)


def _atomic_rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Use the host's atomic no-replace directory rename primitive."""
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            rename_exclusive = libc.renamex_np
        except AttributeError as exc:
            raise SentimentStorageError("atomic no-replace rename is unavailable") from exc
        rename_exclusive.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename_exclusive.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
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
        result = rename_no_replace(-100, source_bytes, -100, destination_bytes, 0x00000001)
    elif sys.platform == "win32":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise PublicationCollisionError(
                f"publication already exists: {destination.name}"
            ) from exc
        except OSError as exc:
            raise SentimentStorageError(f"unable to publish {destination.name}: {exc}") from exc
        return
    else:
        raise SentimentStorageError("platform lacks an atomic no-replace directory rename")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise PublicationCollisionError(f"publication already exists: {destination.name}")
    raise SentimentStorageError(
        f"unable to publish {destination.name}: {os.strerror(error_number)}"
    )
