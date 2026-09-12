"""Regression reproducers for the four Batch B P1 findings; synthetic I/O only."""

import base64
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp
from crypto_ai.sentiment.exceptions import NetworkSafetyError, ReceiptValidationError
from crypto_ai.sentiment.network import (
    RETRY_POLICY_VERSION,
    RetrievalFailure,
    verify_retained_closeout,
    verify_terminal_gap_evidence,
)
from crypto_ai.sentiment.providers.gdelt_gsg import GSGAdapter, expected_gsg_source_locator
from crypto_ai.sentiment.receipts import (
    RECEIPT_DOMAIN,
    RECEIPT_SCHEMA_VERSION,
    sign_closeout,
    signer_key_id,
)

from .test_gdelt_gsg_network import KEY, PLAN, PUBLIC, RAW, Response, client_at, minute


@pytest.mark.parametrize("delay_location", ["before_dispatch", "after_dispatch", "intent_write"])
def test_physical_starts_paced_despite_delayed_dispatch(tmp_path, monkeypatch, delay_location):
    client, clock, transport = client_at(tmp_path, Response(), Response())
    original_open = transport._open

    def delayed_open(url, *, timeout):
        if delay_location == "before_dispatch" and not transport.starts:
            clock.seconds += 1.0
        result = original_open(url, timeout=timeout)
        if delay_location == "after_dispatch" and len(transport.starts) == 1:
            clock.seconds += 1.0
        return result

    transport.open.side_effect = delayed_open
    if delay_location == "intent_write":
        original_request = client.budget.record_request

        def slow_intent(*args, **kwargs):
            original_request(*args, **kwargs)
            if not transport.starts:
                clock.seconds += 1.0

        monkeypatch.setattr(client.budget, "record_request", slow_intent)
    with client:
        client.retrieve(minute())
        client.retrieve(minute(1))
        assert transport.starts[1] - transport.starts[0] >= 5.0
        for physical, envelope in zip(transport.starts, client.receipts, strict=True):
            actual = clock.start + timedelta(seconds=physical - 100.0)
            body = envelope["body"]
            assert parse_utc_timestamp(body["requested_at_utc"], field="lower") <= actual
            assert actual <= parse_utc_timestamp(body["dispatch_confirmed_at_utc"], field="upper")
        closeout = client.closeout()
        verify_retained_closeout(
            client.budget,
            public_key=PUBLIC,
            receipts=client.receipts,
            expected_closeout_sha256=canonical_sha256(closeout),
            expected_plan_sha256=PLAN.plan_id,
        )


def test_rate_limit_remains_conservative_after_delayed_dispatch_restart(tmp_path):
    client, clock, first = client_at(tmp_path, Response())
    original = first._open

    def delayed(url, *, timeout):
        clock.seconds += 1.0
        return original(url, timeout=timeout)

    first.open.side_effect = delayed
    with client:
        client.retrieve(minute())
    resumed, _, second = client_at(tmp_path, Response(), clock=clock, create_pilot=False)
    with resumed:
        resumed.retrieve(minute(1))
        assert second.starts[0] - first.starts[0] >= 5.0


@pytest.mark.parametrize("status", [200, 429, 503])
@pytest.mark.parametrize("difference", [-1, 100])
def test_declared_transfer_mismatch_is_terminal_never_verified(tmp_path, status, difference):
    response = Response(RAW, status, {"Content-Length": str(len(RAW) + difference)})
    client, _, transport = client_at(tmp_path, response, Response())
    with client:
        with pytest.raises(RetrievalFailure) as failure:
            client.retrieve(minute())
        result = failure.value.result
        body = result.receipts[0]["body"]
        assert body["content_length"] == len(RAW) + difference
        assert body["transfer_complete"] is False
        assert client.store.get_bytes(body["raw_sha256"]) == RAW
        assert client.budget.total_download_bytes == len(RAW)
        assert client.budget.halted
        if status == 200:
            assert result.snapshot.state == "invalid"
            assert result.snapshot.observations == ()
        assert result.gap_evidence.final_terminal_disposition == "non_retryable"
        assert result.gap_evidence.attempt_count == 1
        verify_terminal_gap_evidence(
            result.gap_evidence,
            plan=PLAN,
            filename_timestamp=minute(),
            receipts=client.receipts,
            public_key=PUBLIC,
            protocol_sha256=client.context["protocol_sha256"],
        )
        closeout = client.closeout()
        verified = verify_retained_closeout(
            client.budget,
            public_key=PUBLIC,
            receipts=client.receipts,
            expected_closeout_sha256=canonical_sha256(closeout),
            expected_plan_sha256=PLAN.plan_id,
        )
        assert verified["slot_outcomes"][0]["outcome"] == "provider_gap"
        assert all(slot["outcome"] != "verified" for slot in verified["slot_outcomes"])
    transport.open.assert_called_once()
    assert response.closed


@pytest.mark.parametrize("declared", [None, len(RAW)])
def test_complete_known_or_unknown_length_transfers_verify(tmp_path, declared):
    headers = {} if declared is None else {"content-length": str(declared)}
    client, _, _ = client_at(tmp_path, Response(RAW, headers=headers))
    with client:
        result = client.retrieve(minute())
        assert result.snapshot.state == "complete"
        body = result.receipts[0]["body"]
        assert body["transfer_complete"] is True
        assert body["content_length"] == declared


def test_unknown_length_early_eof_does_not_masquerade_as_complete_gzip(tmp_path):
    response = Response(RAW + b"unreceived remainder")
    response.read = Mock(side_effect=[RAW, b""])
    client, _, _ = client_at(tmp_path, response)
    with client:
        with pytest.raises(RetrievalFailure) as failure:
            client.retrieve(minute())
        assert failure.value.result.snapshot.state == "invalid"
        assert failure.value.result.receipts[0]["body"]["transfer_complete"] is False
        assert client.budget.total_download_bytes == len(RAW)


def test_missing_explicit_completion_signal_fails_closed(tmp_path):
    response = Response()
    # Use a fixture wrapper without the required completion property.
    wrapper = Mock(spec=["read", "close", "url", "headers", "status", "clock"])

    def read(*args, **kwargs):
        response.clock = wrapper.clock
        return response.read(*args, **kwargs)

    wrapper.read = read
    wrapper.url, wrapper.headers, wrapper.status = "", {}, 200
    client, _, _ = client_at(tmp_path, wrapper)
    with client, pytest.raises(NetworkSafetyError, match="explicit transfer completion"):
        client.retrieve(minute())
    assert client.receipts == ()


@pytest.mark.parametrize(
    "length", ["", "-1", "+253", "0253", "253.0", " 253", "253 ", "NaN", "500000001", "1" * 1000]
)
def test_malformed_content_length_fails_closed(tmp_path, length):
    response = Response(headers={"Content-Length": length})
    client, _, transport = client_at(tmp_path, response)
    with client, pytest.raises(NetworkSafetyError, match="Content-Length"):
        client.retrieve(minute())
    assert response.offset == 0
    assert client.receipts == ()
    transport.open.assert_called_once()


@pytest.mark.parametrize(
    "headers",
    [{"Content-Length": "253", "content-length": "253"}, {"Transfer-Encoding": "chunked"}],
)
def test_ambiguous_http_framing_fails_closed(tmp_path, headers):
    client, _, _ = client_at(tmp_path, Response(headers=headers))
    with client, pytest.raises(NetworkSafetyError):
        client.retrieve(minute())
    assert client.receipts == ()


def test_read_exception_does_not_create_success_or_fabricated_terminal_gap(tmp_path):
    response = Response()
    response.read = Mock(side_effect=[RAW, OSError("fixture stream interrupted")])
    client, _, transport = client_at(tmp_path, response)
    with client, pytest.raises(NetworkSafetyError):
        client.retrieve(minute())
    assert client.budget.total_download_bytes == len(RAW)
    assert client.receipts == ()
    assert not list(client.store.publications_root.glob("gsg-network-gap-*"))
    transport.open.assert_called_once()


def test_retained_closeout_rejects_validly_signed_year2099_chain(tmp_path):
    client, _, _ = client_at(tmp_path)
    chain = []
    with client.budget.locked():
        for index in range(15):
            when = datetime(2099, 1, 1, tzinfo=UTC) + timedelta(seconds=5 * index)
            instant = format_utc_timestamp(when)
            url = expected_gsg_source_locator(PLAN.intervals[index])
            client.budget.record_request(
                instant, filename_timestamp=minute(index), attempt_number=1
            )
            client.budget.record_received(len(RAW))
            snapshot = GSGAdapter(client.store, clock=lambda instant=when: instant).ingest_snapshot(
                RAW,
                filename_timestamp=minute(index),
                ingested_at=instant,
                source_locator=url,
                collection_mode="prospective",
                input_class="synthetic_fixture",
            )
            body = {
                **client.context,
                "interval_index": 0,
                "filename_timestamp": minute(index),
                "source_locator": url,
                "retry_policy_version": RETRY_POLICY_VERSION,
                "input_class": "synthetic_fixture",
                "real_network_calls_prohibited": True,
                "attempt_number": 1,
                "http_status": 200,
                "bytes_received": len(RAW),
                "content_length": len(RAW),
                "transfer_complete": True,
                "raw_sha256": snapshot.receipt.raw_snapshot_sha256,
                "snapshot_id": snapshot.receipt.snapshot_id,
                "snapshot_state": "complete",
                "raw_published_at_utc": instant,
                "requested_at_utc": instant,
                "dispatch_confirmed_at_utc": instant,
                "completed_at_utc": instant,
                "retry_after_seconds": None,
                "previous_receipt_sha256": canonical_sha256(chain[-1]) if chain else None,
            }
            encoded = canonicalize(body)
            envelope = {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "body": body,
                "body_sha256": canonical_sha256(body),
                "algorithm": "Ed25519",
                "signer_key_id": signer_key_id(PUBLIC),
                "signature": base64.b64encode(KEY.sign(RECEIPT_DOMAIN + encoded)).decode(),
            }
            PUBLIC.verify(base64.b64decode(envelope["signature"]), RECEIPT_DOMAIN + encoded)
            client.store.publish_bundle(
                f"{client._prefix}attempt-{index+1:08d}",
                {"signed-receipt.json": canonicalize(envelope)},
                metadata={
                    "provider": "gdelt_gsg",
                    "pilot_id": client.context["pilot_id"],
                    "sequence": index + 1,
                },
            )
            chain.append(envelope)
            client.budget.finish_request()
        inventory = client.budget.payload_inventory()
        inventory_hash = canonical_sha256(inventory)
        closeout = sign_closeout(
            {
                **{k: v for k, v in client.context.items() if k != "plan_start_at_utc"},
                "final_receipt_sha256": canonical_sha256(chain[-1]),
                "receipt_count": 15,
                "slot_outcomes": [
                    {"interval_index": i, "outcome": "verified" if i == 0 else "uncovered"}
                    for i in range(96)
                ],
                "payload_inventory_sha256": inventory_hash,
            },
            KEY,
        )
        pin = canonical_sha256(closeout)
        client.store.publish_bundle(
            f"{client._prefix}closeout",
            {
                "signed-closeout.json": canonicalize(closeout),
                "payload-inventory.json": canonicalize(inventory),
            },
            metadata={"closeout_sha256": pin, "inventory_sha256": inventory_hash},
        )
        with pytest.raises(ReceiptValidationError, match="scheduled reporting slot"):
            verify_retained_closeout(
                client.budget,
                public_key=PUBLIC,
                receipts=chain,
                expected_closeout_sha256=pin,
                expected_plan_sha256=PLAN.plan_id,
            )
