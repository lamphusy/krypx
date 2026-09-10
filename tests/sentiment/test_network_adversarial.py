"""Independent-review regressions; all transport, time and payloads are synthetic."""

from contextlib import contextmanager
from datetime import timedelta

import pytest

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize
from crypto_ai.sentiment.exceptions import NetworkSafetyError
from crypto_ai.sentiment.network import RetrievalFailure, verify_retained_closeout
from crypto_ai.sentiment.receipts import sign_closeout

from .test_gdelt_gsg_network import KEY, PLAN, PUBLIC, RAW, Clock, Response, client_at, minute


@pytest.mark.parametrize("stage", ["budget", "receipts"])
def test_hydration_time_counts_toward_session_deadline(tmp_path, monkeypatch, stage):
    client, clock, transport = client_at(tmp_path, Response())
    if stage == "budget":
        original = client.budget.locked

        @contextmanager
        def slow_budget():
            with original() as budget:
                clock.seconds += 901
                yield budget

        monkeypatch.setattr(client.budget, "locked", slow_budget)
    else:
        original = client._load_receipts

        def slow_receipts():
            original()
            clock.seconds += 901

        monkeypatch.setattr(client, "_load_receipts", slow_receipts)
    with pytest.raises(NetworkSafetyError):
        with client:
            pytest.fail("late hydration activated a session")
    transport.open.assert_not_called()


def test_closeout_cannot_overlap_active_retrieval_lock(tmp_path):
    client, _, _ = client_at(tmp_path)
    with client:
        client._retrieval_lock.acquire()
        try:
            with pytest.raises(NetworkSafetyError, match="overlap"):
                client.closeout()
        finally:
            client._retrieval_lock.release()
        assert not (client.store.publications_root / f"{client._prefix}closeout").exists()


def test_gap_terminal_time_cannot_precede_raw_publication(tmp_path, monkeypatch):
    client, clock, _ = client_at(tmp_path, Response(b"not gzip"))
    original = client.store.put_bytes

    def delayed_publication(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.seconds += 1
        return result

    monkeypatch.setattr(client.store, "put_bytes", delayed_publication)
    with client:
        with pytest.raises(RetrievalFailure) as failure:
            client.retrieve(minute())
        result = failure.value.result
        assert result.gap_evidence.terminal_at == result.snapshot.receipt.raw_published_at
        assert result.gap_evidence.terminal_at > result.receipts[-1]["body"]["completed_at_utc"]


@pytest.mark.parametrize("mutation", ["cost", "mock_only"])
def test_transport_authority_change_during_read_fails_closed(tmp_path, monkeypatch, mutation):
    response = Response()
    client, _, transport = client_at(tmp_path, response)
    original = response.read

    def unsafe_read(*args, **kwargs):
        if mutation == "cost":
            transport.incremental_cost_usd = 0.01
        else:
            transport.mock_only = False
        return original(*args, **kwargs)

    monkeypatch.setattr(response, "read", unsafe_read)
    with client:
        with pytest.raises(NetworkSafetyError):
            client.retrieve(minute())
        assert client.budget.total_download_bytes == len(RAW)
        assert client.receipts == ()
    assert response.closed


def test_resume_waits_five_monotonic_seconds_despite_forward_utc_step(tmp_path):
    client, _, first_transport = client_at(tmp_path, Response())
    with client:
        client.retrieve(minute())
    second_clock = Clock()
    second_clock.start += timedelta(seconds=5)
    resumed, _, second_transport = client_at(
        tmp_path, Response(), clock=second_clock, create_pilot=False
    )
    with resumed:
        resumed.retrieve(minute(1))
    assert second_transport.starts[0] - first_transport.starts[0] >= 5


@pytest.mark.parametrize("foreign", [False, True])
def test_signed_closeout_cannot_claim_detached_or_foreign_receipt_chain(tmp_path, foreign):
    source, _, _ = client_at(tmp_path / "source", Response())
    with source:
        source.retrieve(minute())
        source_closeout = source.closeout()
        receipts = source.receipts
    target, _, _ = client_at(tmp_path / "target")
    with target:
        inventory = target.budget.payload_inventory()
        body = dict(source_closeout["body"])
        body["payload_inventory_sha256"] = canonical_sha256(inventory)
        if foreign:
            # Even an empty, correctly signed foreign pilot closeout is forbidden.
            body["pilot_id"] = "different-pilot"
            body["receipt_count"] = 0
            body["final_receipt_sha256"] = None
            receipts = ()
        closeout = sign_closeout(body, KEY)
        pin = canonical_sha256(closeout)
        target.store.publish_bundle(
            f"{target._prefix}closeout",
            {
                "signed-closeout.json": canonicalize(closeout),
                "payload-inventory.json": canonicalize(inventory),
            },
            metadata={"closeout_sha256": pin, "inventory_sha256": canonical_sha256(inventory)},
        )
        with pytest.raises(CryptoAIError):
            verify_retained_closeout(
                target.budget,
                public_key=PUBLIC,
                receipts=receipts,
                expected_closeout_sha256=pin,
                expected_plan_sha256=PLAN.plan_id,
            )


def test_closed_out_inventory_cannot_be_mutated_by_later_rejected_operations(tmp_path):
    client, _, _ = client_at(tmp_path, Response())
    with client:
        client.retrieve(minute())
        closeout = client.closeout()
        before = client.budget.payload_inventory()
        for operation in (lambda: client.retrieve(minute(1)), client.closeout):
            with pytest.raises(NetworkSafetyError):
                operation()
        assert client.budget.payload_inventory() == before
        verify_retained_closeout(
            client.budget,
            public_key=PUBLIC,
            receipts=client.receipts,
            expected_closeout_sha256=canonical_sha256(closeout),
            expected_plan_sha256=PLAN.plan_id,
        )


def test_caller_exception_after_closeout_does_not_rewrite_budget(tmp_path):
    client, _, _ = client_at(tmp_path)
    with pytest.raises(RuntimeError, match="caller exception"):
        with client:
            closeout = client.closeout()
            before = client.budget.payload_inventory()
            raise RuntimeError("caller exception")
    with client.budget.locked():
        assert client.budget.payload_inventory() == before
        verify_retained_closeout(
            client.budget,
            public_key=PUBLIC,
            receipts=client.receipts,
            expected_closeout_sha256=canonical_sha256(closeout),
            expected_plan_sha256=PLAN.plan_id,
        )


def test_deleted_terminal_gap_publication_rejects_restart(tmp_path):
    client, clock, _ = client_at(tmp_path, *(Response(b"busy", 503) for _ in range(4)))
    with client:
        result = client.retrieve(minute())
        gap = client.store.publications_root / client._gap_publication_id(result.gap_evidence)
    # Move the exact temporary fixture publication aside; no repository data touched.
    gap.rename(tmp_path / "removed-gap-evidence")
    resumed, _, transport = client_at(tmp_path, Response(), clock=clock, create_pilot=False)
    with pytest.raises(NetworkSafetyError):
        with resumed:
            pytest.fail("missing terminal gap evidence hydrated")
    transport.open.assert_not_called()


def test_literal_500_mb_counter_crossing_is_durable_without_allocating_payload(tmp_path):
    client, _, _ = client_at(tmp_path)
    with client.budget.locked():
        client.budget.record_request(
            "2026-09-09T00:45:00Z", filename_timestamp=minute(), attempt_number=1
        )
        # Synthetic preceding received-byte facts, not a real transfer or 500 MB allocation.
        client.budget.record_received(500_000_000)
        with pytest.raises(NetworkSafetyError, match="cumulative download"):
            client.budget.record_received(1)
        assert client.budget.total_download_bytes == 500_000_001


def test_literal_two_gb_logical_storage_limit_is_enforced_before_reading(tmp_path):
    client, _, transport = client_at(tmp_path, Response())
    path = tmp_path / ".staging" / "sparse-fixture"
    with path.open("wb") as stream:
        stream.truncate(2_000_000_001)
    with pytest.raises(NetworkSafetyError, match="storage cap"):
        client.budget.assert_storage_capacity(1)
    transport.open.assert_not_called()


def test_five_missed_reporting_slots_stop_before_any_new_request(tmp_path):
    clock = Clock()
    clock.start += timedelta(minutes=75)
    client, _, transport = client_at(tmp_path, Response(), clock=clock)
    with client:
        with pytest.raises(NetworkSafetyError, match="92/96 unattainable"):
            client.retrieve(minute(75))
        assert client.budget.halted
    transport.open.assert_not_called()
