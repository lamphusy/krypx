"""Durable, pilot-wide circuit breakers for the offline-tested GSG client.

The caller explicitly creates a pilot once; ordinary resumes never initialize or
repair missing state. Immutable event files are anchored by a fsynced checkpoint
on the locked inode. A crash between event and checkpoint writes fails closed.
This detects accidental truncation, not adversarial rollback of the entire store
and checkpoint together; an independently pinned signed closeout is still needed.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime
from typing import Any

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import (
    SHA256_PATTERN,
    format_utc_timestamp,
    parse_utc_timestamp,
)
from crypto_ai.sentiment.exceptions import NetworkSafetyError
from crypto_ai.sentiment.storage import (
    ContentAddressedStore,
    _ensure_directory_at,
    _open_directory_at,
    _read_regular_file_at_once,
    _stat_tree_fingerprint,
    _write_fsynced_at,
)

MAXIMUM_DOWNLOAD_BYTES = 500_000_000
MAXIMUM_STORAGE_BYTES = 2_000_000_000
_MAXIMUM_EVENTS = 100_000
_JOURNAL_RESERVE = 16_384


def _integer(value: object, label: str, maximum: int, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise NetworkSafetyError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def _timestamp(value: object) -> datetime:
    try:
        parsed = parse_utc_timestamp(value, field="request timestamp")
        if parsed is None or format_utc_timestamp(parsed) != value:
            raise ValueError("timestamp is not canonical UTC")
        return parsed
    except (ValueError, TypeError) as exc:
        raise NetworkSafetyError("invalid request UTC timestamp") from exc


def _strict_json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or canonicalize(value) != raw:
            raise NetworkSafetyError("budget journal is not canonical JSON")
        return value
    except (ValueError, TypeError, CryptoAIError) as exc:
        raise NetworkSafetyError("invalid budget journal JSON") from exc


class PilotBudget:
    """One durable download/rate ledger, with an exclusive whole-session lock."""

    def __init__(
        self,
        store: ContentAddressedStore,
        pilot_id: str,
        protocol_sha256: str,
        *,
        maximum_download_bytes: int = MAXIMUM_DOWNLOAD_BYTES,
        maximum_storage_bytes: int = MAXIMUM_STORAGE_BYTES,
        create: bool = False,
    ) -> None:
        if not isinstance(store, ContentAddressedStore):
            raise NetworkSafetyError("budget requires a content-addressed store")
        if not isinstance(pilot_id, str) or not pilot_id or len(pilot_id) > 128:
            raise NetworkSafetyError("invalid pilot identity")
        if not isinstance(protocol_sha256, str) or not SHA256_PATTERN.fullmatch(protocol_sha256):
            raise NetworkSafetyError("invalid protocol SHA-256")
        if type(create) is not bool:
            raise NetworkSafetyError("create must be boolean")
        self.store = store
        self.pilot_id = pilot_id
        self.protocol_sha256 = protocol_sha256
        self.maximum_download_bytes = _integer(
            maximum_download_bytes, "download cap", MAXIMUM_DOWNLOAD_BYTES, minimum=1
        )
        self.maximum_storage_bytes = _integer(
            maximum_storage_bytes, "storage cap", MAXIMUM_STORAGE_BYTES, minimum=1
        )
        try:
            self.directory_name = sha256_bytes(pilot_id.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise NetworkSafetyError("pilot identity must be valid Unicode") from exc
        self._create = create
        self._directory: int | None = None
        self._checkpoint: int | None = None
        self._total = 0
        self._last: str | None = None
        self._attempts: dict[str, int] = {}
        self._pending = False
        self._halted: str | None = None
        self._head: str | None = None
        self._count = 0

    @property
    def total_download_bytes(self) -> int:
        return self._total

    @property
    def last_request_at_utc(self) -> str | None:
        return self._last

    @property
    def request_attempts(self) -> dict[str, int]:
        return dict(self._attempts)

    @property
    def halted(self) -> bool:
        return self._halted is not None

    @contextmanager
    def locked(self) -> Iterator[PilotBudget]:
        """Load an existing ledger or explicitly create it, without blocking."""
        if self._directory is not None:
            raise NetworkSafetyError("pilot budget is already locked")
        root = parent = directory = checkpoint = None
        try:
            root = self.store._open_root_descriptor()
            if self._create:
                self.assert_storage_capacity(_JOURNAL_RESERVE)
            opener = _ensure_directory_at if self._create else _open_directory_at
            parent = opener(root, ".network-budgets", description="pilot budget root")
            if self._create:
                os.mkdir(self.directory_name, 0o700, dir_fd=parent)
                os.fsync(parent)
            directory = _open_directory_at(
                parent, self.directory_name, description="pilot budget directory"
            )
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if self._create:
                flags |= os.O_CREAT | os.O_EXCL
            checkpoint = os.open("checkpoint.json", flags, 0o600, dir_fd=directory)
            checkpoint_stat = os.fstat(checkpoint)
            if not stat.S_ISREG(checkpoint_stat.st_mode) or checkpoint_stat.st_nlink != 1:
                raise NetworkSafetyError("budget checkpoint is not a unique regular file")
            fcntl.flock(checkpoint, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._directory, self._checkpoint = directory, checkpoint
            if self._create:
                self.assert_storage_capacity(_JOURNAL_RESERVE)
                self._write_checkpoint()
                self._create = False
            else:
                self._load()
            if self._pending:
                raise NetworkSafetyError("unresolved prior request intent; no automatic resume")
            if self._halted or self._total > self.maximum_download_bytes:
                raise NetworkSafetyError("pilot is halted; budgets cannot reset")
            yield self
        except NetworkSafetyError:
            raise
        except (OSError, CryptoAIError, ValueError, TypeError) as exc:
            raise NetworkSafetyError(f"unsafe or unavailable pilot budget: {exc}") from exc
        finally:
            self._directory = self._checkpoint = None
            for descriptor in (checkpoint, directory, parent, root):
                if descriptor is not None:
                    os.close(descriptor)

    def _require_lock(self) -> None:
        if self._directory is None or self._checkpoint is None:
            raise NetworkSafetyError("pilot budget operation requires session lock")
        root = parent = current = None
        try:
            root = self.store._open_root_descriptor()
            parent = _open_directory_at(root, ".network-budgets", description="budget root")
            current = _open_directory_at(parent, self.directory_name, description="budget")
            pinned = os.fstat(self._directory)
            actual = os.fstat(current)
            checkpoint = os.stat("checkpoint.json", dir_fd=current, follow_symlinks=False)
            opened = os.fstat(self._checkpoint)
            if (pinned.st_dev, pinned.st_ino) != (actual.st_dev, actual.st_ino) or (
                checkpoint.st_dev,
                checkpoint.st_ino,
            ) != (opened.st_dev, opened.st_ino):
                raise NetworkSafetyError("budget directory or locked checkpoint was replaced")
        except NetworkSafetyError:
            raise
        except (OSError, CryptoAIError) as exc:
            raise NetworkSafetyError(f"unsafe budget lock identity: {exc}") from exc
        finally:
            for descriptor in (current, parent, root):
                if descriptor is not None:
                    os.close(descriptor)

    def record_request(
        self,
        requested_at_utc: str,
        *,
        filename_timestamp: str | None = None,
        attempt_number: int | None = None,
    ) -> None:
        """Durably record intent before the transport can send a request."""
        self._require_lock()
        if filename_timestamp is None or attempt_number is None:
            raise NetworkSafetyError("a request must identify its minute and attempt")
        event = {
            "kind": "request",
            "requested_at_utc": requested_at_utc,
            "filename_timestamp": filename_timestamp,
            "attempt_number": attempt_number,
        }
        self._validate_event(event)
        self._append(event)

    def record_received(self, count: int) -> None:
        """Persist every returned byte, including the chunk crossing the cap."""
        self._require_lock()
        event = {"kind": "received", "count": count}
        self._validate_event(event)
        self._append(event)
        if self._total > self.maximum_download_bytes:
            raise NetworkSafetyError("cumulative download cap exceeded")

    def finish_request(self) -> None:
        """Call only after the caller durably publishes its signed result evidence."""
        self._require_lock()
        event = {"kind": "finish"}
        self._validate_event(event)
        self._append(event)

    def stop(self, reason: str) -> None:
        """Persist a fatal stop; it cannot be cleared by reopening the budget."""
        self._require_lock()
        event = {"kind": "stop", "reason": reason}
        self._validate_event(event)
        self._append(event)

    def _validate_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        if kind == "request" and set(event) == {
            "kind",
            "requested_at_utc",
            "filename_timestamp",
            "attempt_number",
        }:
            if self._pending or self._halted or self._total > self.maximum_download_bytes:
                raise NetworkSafetyError("request is blocked by prior intent or stop")
            timestamp = _timestamp(event["requested_at_utc"])
            minute = _timestamp(event["filename_timestamp"])
            if minute.second or minute.microsecond:
                raise NetworkSafetyError("filename timestamp must be minute-aligned")
            if format_utc_timestamp(minute) != event["filename_timestamp"]:
                raise NetworkSafetyError("filename timestamp must use canonical UTC form")
            if (
                self._last is not None
                and (timestamp - _timestamp(self._last)).total_seconds() < 5.0
            ):
                raise NetworkSafetyError("request starts must be at least five seconds apart")
            attempt = _integer(event["attempt_number"], "attempt", 4, minimum=1)
            if attempt != self._attempts.get(event["filename_timestamp"], 0) + 1:
                raise NetworkSafetyError("request attempt sequence is not consecutive")
        elif kind == "received" and set(event) == {"kind", "count"}:
            _integer(event["count"], "received bytes", (1 << 53) - 1)
            if not self._pending:
                raise NetworkSafetyError("received bytes require a pending request")
            _integer(self._total + event["count"], "cumulative bytes", (1 << 53) - 1)
        elif kind == "finish" and set(event) == {"kind"}:
            if not self._pending:
                raise NetworkSafetyError("no request is pending")
        elif kind == "stop" and set(event) == {"kind", "reason"}:
            if not isinstance(event["reason"], str) or not 1 <= len(event["reason"]) <= 128:
                raise NetworkSafetyError("stop reason must be a bounded nonempty string")
        else:
            raise NetworkSafetyError("unknown or malformed budget event")

    def _apply(self, event: dict[str, Any]) -> None:
        self._validate_event(event)
        if event["kind"] == "request":
            self._last = event["requested_at_utc"]
            self._attempts[event["filename_timestamp"]] = event["attempt_number"]
            self._pending = True
        elif event["kind"] == "received":
            self._total += event["count"]
        elif event["kind"] == "finish":
            self._pending = False
        else:
            self._halted = event["reason"]

    def _checkpoint_value(self) -> dict[str, Any]:
        return {
            "schema_version": "gsg-pilot-budget-v1",
            "pilot_id": self.pilot_id,
            "protocol_sha256": self.protocol_sha256,
            "maximum_download_bytes": self.maximum_download_bytes,
            "maximum_storage_bytes": self.maximum_storage_bytes,
            "event_count": self._count,
            "head_sha256": self._head,
        }

    def _write_checkpoint(self) -> None:
        self._require_lock()
        assert self._checkpoint is not None and self._directory is not None
        raw = canonicalize(self._checkpoint_value())
        os.lseek(self._checkpoint, 0, os.SEEK_SET)
        offset = 0
        while offset < len(raw):
            written = os.write(self._checkpoint, raw[offset:])
            if written <= 0:
                raise NetworkSafetyError("unable to advance budget checkpoint write")
            offset += written
        os.ftruncate(self._checkpoint, len(raw))
        os.fsync(self._checkpoint)
        os.fsync(self._directory)

    def _append(self, event: dict[str, Any]) -> None:
        self._require_lock()
        if self._count >= _MAXIMUM_EVENTS:
            raise NetworkSafetyError("pilot budget event limit exceeded")
        raw = canonicalize({"event": event, "previous_sha256": self._head})
        self.assert_storage_capacity(len(raw) + _JOURNAL_RESERVE)
        assert self._directory is not None
        _write_fsynced_at(self._directory, f"event-{self._count:06d}.json", raw)
        self._apply(event)
        self._head = sha256_bytes(raw)
        self._count += 1
        self._write_checkpoint()

    def _load(self) -> None:
        self._require_lock()
        assert self._directory is not None
        checkpoint_raw = self._read_journal("checkpoint.json")
        checkpoint = _strict_json(checkpoint_raw)
        count = _integer(checkpoint.get("event_count"), "event count", _MAXIMUM_EVENTS)
        names = {"checkpoint.json"} | {f"event-{index:06d}.json" for index in range(count)}
        if set(os.listdir(self._directory)) != names:
            raise NetworkSafetyError("budget journal inventory disagrees with checkpoint")
        self._total, self._count = 0, 0
        self._last = self._head = self._halted = None
        self._pending = False
        self._attempts = {}
        for index in range(count):
            raw = self._read_journal(f"event-{index:06d}.json")
            envelope = _strict_json(raw)
            if (
                set(envelope) != {"event", "previous_sha256"}
                or envelope["previous_sha256"] != self._head
            ):
                raise NetworkSafetyError("budget journal hash chain is invalid")
            if not isinstance(envelope["event"], dict):
                raise NetworkSafetyError("budget event must be an object")
            self._apply(envelope["event"])
            self._head, self._count = sha256_bytes(raw), index + 1
        if checkpoint != self._checkpoint_value():
            raise NetworkSafetyError("budget checkpoint identity, caps or head mismatch")
        self.assert_storage_capacity(0)

    def _read_journal(self, name: str) -> bytes:
        assert self._directory is not None
        info = os.stat(name, dir_fd=self._directory, follow_symlinks=False)
        if info.st_size > 16_384 or info.st_nlink != 1:
            raise NetworkSafetyError("invalid budget journal size or link count")
        return _read_regular_file_at_once(self._directory, name, description="budget journal")[0]

    def assert_storage_capacity(self, additional_bytes: int) -> None:
        """Inventory the complete store, using lstat and a second, bottom-up pass."""
        _integer(additional_bytes, "additional storage bytes", (1 << 53) - 1)
        try:
            root = self.store._open_root_descriptor()
            try:
                logical, allocated = _inventory(root)
            finally:
                os.close(root)
            reserve = ((additional_bytes + 4095) // 4096) * 4096
            if max(logical + additional_bytes, allocated + reserve) > self.maximum_storage_bytes:
                raise NetworkSafetyError("retained pilot storage cap would be exceeded")
        except NetworkSafetyError:
            raise
        except (OSError, CryptoAIError) as exc:
            raise NetworkSafetyError(f"unsafe retained storage tree: {exc}") from exc

    def payload_inventory(self) -> dict[str, str]:
        """Hash every regular store file from one stable, pinned directory tree.

        The caller must hold the pilot lock and freeze budget writes until the
        closeout and this inventory have been published. The store must be the
        pilot's exclusive namespace. The not-yet-created closeout is the only
        intended exclusion; hidden files, staging, logs and journals are included.
        """
        self._require_lock()
        self.assert_storage_capacity(0)
        try:
            with ExitStack() as descriptors:
                root = self.store._open_root_descriptor()
                descriptors.callback(os.close, root)
                directories: list[tuple[int, os.stat_result, dict[str, os.stat_result]]] = []
                files: list[tuple[int, str, str, os.stat_result]] = []

                def capture(directory: int, relative: str) -> None:
                    before = os.fstat(directory)
                    entries = {
                        name: os.stat(name, dir_fd=directory, follow_symlinks=False)
                        for name in sorted(os.listdir(directory))
                    }
                    directories.append((directory, before, entries))
                    for name, info in entries.items():
                        path = f"{relative}/{name}" if relative else name
                        if stat.S_ISDIR(info.st_mode):
                            child = _open_directory_at(
                                directory, name, description="inventory directory"
                            )
                            descriptors.callback(os.close, child)
                            if _stat_tree_fingerprint(info) != _stat_tree_fingerprint(
                                os.fstat(child)
                            ):
                                raise NetworkSafetyError("inventory directory changed")
                            capture(child, path)
                        elif stat.S_ISREG(info.st_mode):
                            files.append((directory, name, path, info))
                        else:
                            raise NetworkSafetyError("inventory contains a non-regular entry")

                capture(root, "")
                result = {}
                for directory, name, path, info in files:
                    raw, opened = _read_regular_file_at_once(
                        directory, name, description="pilot inventory payload"
                    )
                    if _stat_tree_fingerprint(info) != _stat_tree_fingerprint(opened):
                        raise NetworkSafetyError("inventory payload changed before reading")
                    result[path] = sha256_bytes(raw)
                for directory, before, entries in reversed(directories):
                    if set(os.listdir(directory)) != set(entries):
                        raise NetworkSafetyError("inventory directory entries changed")
                    for name, info in entries.items():
                        after = os.stat(name, dir_fd=directory, follow_symlinks=False)
                        if _stat_tree_fingerprint(info) != _stat_tree_fingerprint(after):
                            raise NetworkSafetyError("inventory entry changed after reading")
                    if _stat_tree_fingerprint(before) != _stat_tree_fingerprint(
                        os.fstat(directory)
                    ):
                        raise NetworkSafetyError("inventory directory changed after reading")
                self._require_lock()
                return result
        except NetworkSafetyError:
            raise
        except (OSError, CryptoAIError) as exc:
            raise NetworkSafetyError(f"unsafe pilot payload inventory: {exc}") from exc


def _inventory(descriptor: int) -> tuple[int, int]:
    """Approve totals only after a whole-tree, pinned bottom-up confirmation.

    Keep every directory descriptor open until the global post-pass completes.
    A recursive local post-check is insufficient: a writer can change an already
    visited grandchild without changing any of its ancestors' entry lists.
    Like other POSIX checks, this is a bounded stability check, not an atomic
    filesystem snapshot against a writer acting after the final observation.
    """
    with ExitStack() as descriptors:
        directories: list[tuple[int, os.stat_result, dict[str, os.stat_result]]] = []
        totals = _capture_capacity_inventory(descriptor, descriptors, directories)
        for directory, before, entries in reversed(directories):
            if set(os.listdir(directory)) != set(entries):
                raise NetworkSafetyError("retained directory entries changed during inventory")
            for name, info in entries.items():
                after = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if _capacity_fingerprint(info) != _capacity_fingerprint(after):
                    raise NetworkSafetyError("retained storage entry changed during inventory")
            if _capacity_fingerprint(before) != _capacity_fingerprint(os.fstat(directory)):
                raise NetworkSafetyError("retained directory changed during inventory")
        return totals


def _capacity_fingerprint(value: os.stat_result) -> tuple[int, ...]:
    # Allocation can change without logical length changing (e.g. a sparse file).
    return (*_stat_tree_fingerprint(value), value.st_blocks)


def _capture_capacity_inventory(
    descriptor: int,
    descriptors: ExitStack,
    directories: list[tuple[int, os.stat_result, dict[str, os.stat_result]]],
) -> tuple[int, int]:
    before = os.fstat(descriptor)
    entries = {
        name: os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        for name in sorted(os.listdir(descriptor))
    }
    directories.append((descriptor, before, entries))
    logical, allocated = before.st_size, before.st_blocks * 512
    for name, info in entries.items():
        if stat.S_ISDIR(info.st_mode):
            child = _open_directory_at(descriptor, name, description="retained directory")
            descriptors.callback(os.close, child)
            if _capacity_fingerprint(info) != _capacity_fingerprint(os.fstat(child)):
                raise NetworkSafetyError("retained directory changed before inventory")
            child_logical, child_allocated = _capture_capacity_inventory(
                child, descriptors, directories
            )
            logical += child_logical
            allocated += child_allocated
        elif stat.S_ISREG(info.st_mode):
            logical += info.st_size
            allocated += info.st_blocks * 512
        else:
            raise NetworkSafetyError("retained storage contains a symlink or non-regular entry")
    return logical, allocated
