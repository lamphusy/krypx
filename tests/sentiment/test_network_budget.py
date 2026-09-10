"""Offline adversarial tests of persistent, pilot-wide network circuit breakers."""

import json
import os
import stat
from pathlib import Path

import pytest

from crypto_ai.sentiment import network_budget as budget_module
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.exceptions import NetworkSafetyError
from crypto_ai.sentiment.network_budget import PilotBudget
from crypto_ai.sentiment.storage import ContentAddressedStore

PROTOCOL = "a" * 64
MINUTE = "2026-09-09T00:00:00Z"
START = "2026-09-09T00:30:00Z"


def make_budget(tmp_path: Path, **options: object) -> PilotBudget:
    return PilotBudget(ContentAddressedStore(tmp_path), "test-pilot", PROTOCOL, **options)


def request(budget: PilotBudget, *, timestamp: str = START, attempt: int = 1) -> None:
    budget.record_request(timestamp, filename_timestamp=MINUTE, attempt_number=attempt)


def journal(tmp_path: Path, budget: PilotBudget) -> Path:
    return tmp_path / ".network-budgets" / budget.directory_name


def test_budget_resume_preserves_all_bytes_rate_and_attempts(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        request(budget)
        budget.record_received(10)
        budget.record_received(3)
        budget.finish_request()
    resumed = make_budget(tmp_path)
    with resumed.locked():
        assert resumed.total_download_bytes == 13
        assert resumed.last_request_at_utc == START
        assert resumed.request_attempts == {MINUTE: 1}
        with pytest.raises(NetworkSafetyError, match="five seconds"):
            request(resumed, timestamp="2026-09-09T00:30:04.999999Z", attempt=2)
        request(resumed, timestamp="2026-09-09T00:30:05Z", attempt=2)
        resumed.record_received(2)
        resumed.finish_request()
        assert resumed.total_download_bytes == 15


def test_byte_cap_crossing_is_durable_and_blocks_resume(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True, maximum_download_bytes=10)
    with budget.locked():
        request(budget)
        budget.record_received(10)
        with pytest.raises(NetworkSafetyError, match="download cap"):
            budget.record_received(1)
        assert budget.total_download_bytes == 11
    with pytest.raises(NetworkSafetyError, match="unresolved prior request"):
        with make_budget(tmp_path, maximum_download_bytes=10).locked():
            pytest.fail("exhausted budget resumed")
    raw = (journal(tmp_path, budget) / "event-000002.json").read_bytes()
    assert json.loads(raw)["event"] == {"kind": "received", "count": 1}


def test_resume_missing_or_reset_budget_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(NetworkSafetyError):
        with make_budget(tmp_path).locked():
            pytest.fail("missing budget resumed")
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        pass
    with pytest.raises(NetworkSafetyError):
        with make_budget(tmp_path, create=True).locked():
            pytest.fail("existing budget reset")
    with pytest.raises(NetworkSafetyError, match="checkpoint"):
        with make_budget(tmp_path, maximum_download_bytes=12).locked():
            pytest.fail("changed cap accepted")


def test_unresolved_request_blocks_restart_even_with_no_response_bytes(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        request(budget)
    with pytest.raises(NetworkSafetyError, match="unresolved"):
        with make_budget(tmp_path).locked():
            pytest.fail("interrupted request resumed")


def test_fatal_stop_is_durable(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        budget.stop("malformed_payload")
        assert budget.halted
        with pytest.raises(NetworkSafetyError, match="stop"):
            request(budget)
    with pytest.raises(NetworkSafetyError, match="halted"):
        with make_budget(tmp_path).locked():
            pytest.fail("halted pilot resumed")


def test_concurrent_and_recursive_session_locks_fail_immediately(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        with pytest.raises(NetworkSafetyError, match="already locked"):
            with budget.locked():
                pytest.fail("recursive lock succeeded")
        with pytest.raises(NetworkSafetyError):
            with make_budget(tmp_path).locked():
                pytest.fail("concurrent lock succeeded")


@pytest.mark.parametrize("mutation", ["delete_tail", "extra_event", "modified_event"])
def test_checkpoint_detects_journal_tampering(tmp_path: Path, mutation: str) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        request(budget)
        budget.record_received(7)
        budget.finish_request()
    folder = journal(tmp_path, budget)
    if mutation == "delete_tail":
        (folder / "event-000002.json").unlink()
    elif mutation == "extra_event":
        (folder / "event-000003.json").write_bytes(b"{}")
    else:
        path = folder / "event-000001.json"
        value = json.loads(path.read_bytes())
        value["event"]["count"] = 0
        path.write_bytes(canonicalize(value))
    with pytest.raises(NetworkSafetyError):
        with make_budget(tmp_path).locked():
            pytest.fail("tampered journal accepted")


@pytest.mark.parametrize("kind", ["fifo", "symlink", "socket"])
def test_storage_inventory_rejects_nonregular_entries_without_reading(
    tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    budget = make_budget(tmp_path)
    nested = tmp_path / ".staging" / "nested"
    nested.mkdir()
    target = nested / "unexpected"
    if kind == "fifo":
        os.mkfifo(target)
    elif kind == "symlink":
        target.symlink_to(tmp_path)
    else:
        # The sandbox prohibits socket creation; model its lstat result exactly.
        target.write_bytes(b"")
        original_stat = os.stat

        def socket_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
            result = original_stat(path, *args, **kwargs)
            if path == "unexpected":
                values = list(result)
                values[0] = stat.S_IFSOCK | 0o600
                return os.stat_result(values)
            return result

        monkeypatch.setattr(os, "stat", socket_stat)
    with pytest.raises(NetworkSafetyError, match="non-regular"):
        budget.assert_storage_capacity(1)


def test_storage_cap_accounts_for_staging_and_reservation_before_cas_write(
    tmp_path: Path,
) -> None:
    budget = make_budget(tmp_path, maximum_storage_bytes=100_000)
    (tmp_path / ".staging" / "partial-response").write_bytes(b"x" * 80_000)
    with pytest.raises(NetworkSafetyError, match="storage cap"):
        budget.assert_storage_capacity(30_000)
    assert list(budget.store.objects_root.iterdir()) == []


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_unsafe_checkpoint_rejected_without_blocking(tmp_path: Path, kind: str) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        pass
    checkpoint = journal(tmp_path, budget) / "checkpoint.json"
    checkpoint.unlink()
    if kind == "fifo":
        os.mkfifo(checkpoint)
    else:
        checkpoint.symlink_to(tmp_path / "missing")
    with pytest.raises(NetworkSafetyError):
        with make_budget(tmp_path).locked():
            pytest.fail("unsafe checkpoint accepted")


def test_budget_operations_require_the_exclusive_lock(tmp_path: Path) -> None:
    budget = make_budget(tmp_path)
    for action in (
        lambda: request(budget),
        lambda: budget.record_received(1),
        budget.finish_request,
    ):
        with pytest.raises(NetworkSafetyError, match="session lock"):
            action()


@pytest.mark.parametrize("value", [True, -1, 1.0, float("nan"), float("inf"), "1"])
def test_invalid_counters_fail_closed(tmp_path: Path, value: object) -> None:
    with pytest.raises(NetworkSafetyError):
        make_budget(tmp_path, maximum_download_bytes=value)
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        request(budget)
        with pytest.raises(NetworkSafetyError):
            budget.record_received(value)
        with pytest.raises(NetworkSafetyError):
            budget.assert_storage_capacity(value)


def test_attempts_cannot_reset_or_exceed_four(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        for attempt in range(1, 5):
            request(budget, timestamp=f"2026-09-09T00:30:{(attempt - 1) * 5:02d}Z", attempt=attempt)
            budget.finish_request()
        with pytest.raises(NetworkSafetyError, match="attempt"):
            request(budget, timestamp="2026-09-09T00:30:20Z", attempt=5)
        with pytest.raises(NetworkSafetyError, match="sequence"):
            request(budget, timestamp="2026-09-09T00:30:20Z", attempt=1)


def test_symlink_ancestor_rejected_during_budget_operation(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        folder = tmp_path / ".network-budgets"
        moved = tmp_path / "moved-budget"
        folder.rename(moved)
        folder.symlink_to(moved, target_is_directory=True)
        with pytest.raises(NetworkSafetyError):
            request(budget)


@pytest.mark.parametrize("mutation", ["late_fifo", "directory_symlink"])
def test_storage_inventory_postpass_rejects_late_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    budget = make_budget(tmp_path)
    nested = tmp_path / ".staging" / "nested"
    nested.mkdir()
    (nested / "payload").write_bytes(b"already inventoried")
    nested_inode = nested.stat().st_ino
    original = budget_module._inventory
    changed = False

    def racing_inventory(descriptor: int) -> tuple[int, int]:
        nonlocal changed
        result = original(descriptor)
        if os.fstat(descriptor).st_ino == nested_inode and not changed:
            changed = True
            if mutation == "late_fifo":
                os.mkfifo(nested / "late")
            else:
                moved = nested.with_name("moved")
                nested.rename(moved)
                nested.symlink_to(moved, target_is_directory=True)
        return result

    monkeypatch.setattr(budget_module, "_inventory", racing_inventory)
    with pytest.raises(NetworkSafetyError, match="changed"):
        budget.assert_storage_capacity(0)
    assert changed


def test_budget_creation_checks_storage_before_any_new_journal_write(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, maximum_storage_bytes=1, create=True)
    with pytest.raises(NetworkSafetyError, match="storage cap"):
        with budget.locked():
            pytest.fail("creation exceeded retained cap")
    assert not (tmp_path / ".network-budgets").exists()


def test_payload_inventory_includes_all_store_files_and_exact_bytes(tmp_path: Path) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        request(budget)
        budget.record_received(17)
        budget.finish_request()
        digest = budget.store.put_bytes(b"exact\x00gzip\r\nbytes")
        budget.store.publish_bundle(
            "example", {"nested/data": b"report"}, metadata={"kind": "test"}
        )
        (tmp_path / ".staging" / "unexpected-partial").write_bytes(b"not ignored")
        (tmp_path / "collector.log").write_bytes(b"local audit")
        inventory = budget.payload_inventory()
        expected = {
            str(path.relative_to(tmp_path)): sha256_bytes(path.read_bytes())
            for path in tmp_path.rglob("*")
            if path.is_file()
        }
        assert inventory == expected
        assert inventory[f"objects/sha256/{digest[:2]}/{digest}"] == digest
        assert f".network-budgets/{budget.directory_name}/checkpoint.json" in inventory
        assert "collector.log" in inventory
        assert ".staging/unexpected-partial" in inventory


@pytest.mark.parametrize("mutation", ["late_fifo", "directory_symlink", "payload_replace"])
def test_payload_inventory_rejects_mutations_between_two_global_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        nested = tmp_path / ".staging" / "nested"
        nested.mkdir()
        payload = nested / "payload"
        payload.write_bytes(b"valid bytes")
        original_read = budget_module._read_regular_file_at_once
        changed = False

        def racing_read(directory: int, name: str, **kwargs: str) -> tuple[bytes, os.stat_result]:
            nonlocal changed
            result = original_read(directory, name, **kwargs)
            if name == "payload" and not changed:
                changed = True
                if mutation == "late_fifo":
                    os.mkfifo(nested / "late")
                elif mutation == "directory_symlink":
                    moved = nested.with_name("moved")
                    nested.rename(moved)
                    nested.symlink_to(moved, target_is_directory=True)
                else:
                    replacement = nested / "replacement"
                    replacement.write_bytes(b"forged bytes")
                    replacement.replace(payload)
            return result

        monkeypatch.setattr(budget_module, "_read_regular_file_at_once", racing_read)
        with pytest.raises(NetworkSafetyError, match="changed"):
            budget.payload_inventory()
        assert changed


def test_payload_inventory_requires_lock(tmp_path: Path) -> None:
    with pytest.raises(NetworkSafetyError, match="lock"):
        make_budget(tmp_path).payload_inventory()


@pytest.mark.parametrize("timestamp", [None, True, 1, "2026-09-09T00:30:00.000000Z", "bad"])
def test_request_timestamp_failures_are_project_specific(tmp_path: Path, timestamp: object) -> None:
    budget = make_budget(tmp_path, create=True)
    with budget.locked():
        with pytest.raises(NetworkSafetyError, match="UTC timestamp"):
            request(budget, timestamp=timestamp)
