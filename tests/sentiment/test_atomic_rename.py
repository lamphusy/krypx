"""Host-filesystem regressions for immutable, no-replace directory publication."""

import ctypes
import errno
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crypto_ai.exceptions import PublicationCollisionError
from crypto_ai.sentiment import storage as storage_module


class _RecordingCFunction:
    def __init__(self, result: int = 0) -> None:
        self.result = result
        self.calls: list[tuple[object, ...]] = []
        self.argtypes: list[object] | None = None
        self.restype: object = None

    def __call__(self, *args: object) -> int:
        self.calls.append(args)
        return self.result


def test_linux_rename_prefers_libc_renameat2() -> None:
    direct = _RecordingCFunction()
    syscall = _RecordingCFunction()
    libc = SimpleNamespace(renameat2=direct, syscall=syscall)

    result = storage_module._linux_rename_noreplace(libc, 17, b"staging", b"final")

    assert result == 0
    assert direct.calls == [(17, b"staging", 17, b"final", 1)]
    assert syscall.calls == []


def test_linux_rename_falls_back_when_libc_returns_enosys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    class UnavailableCFunction(_RecordingCFunction):
        def __call__(self, *args: object) -> int:
            ctypes.set_errno(errno.ENOSYS)
            return super().__call__(*args)

    direct = UnavailableCFunction(result=-1)
    syscall = _RecordingCFunction()
    libc = SimpleNamespace(renameat2=direct, syscall=syscall)

    result = storage_module._linux_rename_noreplace(libc, 23, b"staging", b"final")

    assert result == 0
    assert direct.calls == [(23, b"staging", 23, b"final", 1)]
    assert syscall.calls == [(316, 23, b"staging", 23, b"final", 1)]


@pytest.mark.parametrize(
    ("architecture", "syscall_number"),
    [("x86_64", 316), ("aarch64", 276)],
)
def test_linux_rename_uses_architecture_syscall_when_libc_symbol_is_missing(
    monkeypatch: pytest.MonkeyPatch, architecture: str, syscall_number: int
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: architecture)
    syscall = _RecordingCFunction()
    libc = SimpleNamespace(syscall=syscall)

    result = storage_module._linux_rename_noreplace(libc, 19, b"staging", b"final")

    assert result == 0
    assert syscall.calls == [(syscall_number, 19, b"staging", 19, b"final", 1)]


def test_capability_probe_accepts_eexist_as_no_replace_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "target-filesystem"
    parent.mkdir()
    (parent / "sentinel.txt").write_bytes(b"unmodified")
    calls: list[tuple[bytes, bytes]] = []

    def simulated_rename(probe: int, source: bytes, destination: bytes) -> int:
        probe_stat = os.fstat(probe)
        assert any(
            (entry_stat := os.stat(entry, follow_symlinks=False)).st_dev == probe_stat.st_dev
            and entry_stat.st_ino == probe_stat.st_ino
            for entry in parent.rglob("*")
        )
        calls.append((source, destination))
        if source == b"source":
            os.rename(source, destination, src_dir_fd=probe, dst_dir_fd=probe)
            return 0
        assert source == b"collision"
        ctypes.set_errno(errno.EEXIST)
        return -1

    monkeypatch.setattr(
        storage_module, "_atomic_rename_directory_no_replace_status", simulated_rename
    )

    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        storage_module._require_atomic_rename_directory_no_replace_at(descriptor)
    finally:
        os.close(descriptor)

    assert calls == [(b"source", b"destination"), (b"collision", b"destination")]
    assert [entry.name for entry in parent.iterdir()] == ["sentinel.txt"]
    assert (parent / "sentinel.txt").read_bytes() == b"unmodified"


@pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="atomic no-replace directory rename is supported on Darwin and Linux",
)
def test_atomic_directory_rename_works_on_host_filesystem(tmp_path: Path) -> None:
    """The capability check and rename must agree on an ordinary temp directory."""
    parent = tmp_path / "publications"
    parent.mkdir()
    source = parent / "staging"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"published")

    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        storage_module._require_atomic_rename_directory_no_replace_at(descriptor)
        storage_module._atomic_rename_directory_no_replace(descriptor, "staging", "final")
    finally:
        os.close(descriptor)

    assert not source.exists()
    assert (parent / "final" / "payload.bin").read_bytes() == b"published"


@pytest.mark.skipif(
    sys.platform != "darwin" and not sys.platform.startswith("linux"),
    reason="atomic no-replace directory rename is supported on Darwin and Linux",
)
def test_atomic_directory_rename_preserves_existing_destination(tmp_path: Path) -> None:
    """A collision may not replace either the winner or the unpublished source."""
    parent = tmp_path / "publications"
    parent.mkdir()
    source = parent / "staging"
    source.mkdir()
    (source / "loser.bin").write_bytes(b"loser")
    destination = parent / "final"
    destination.mkdir()
    (destination / "winner.bin").write_bytes(b"winner")

    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(PublicationCollisionError, match="already exists"):
            storage_module._atomic_rename_directory_no_replace(descriptor, "staging", "final")
    finally:
        os.close(descriptor)

    assert (source / "loser.bin").read_bytes() == b"loser"
    assert (destination / "winner.bin").read_bytes() == b"winner"
    assert not (destination / "loser.bin").exists()
