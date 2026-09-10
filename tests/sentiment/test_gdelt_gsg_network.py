"""Mock-only network/CAS/signing integration and circuit-breaker adversarial probes."""

from __future__ import annotations

import gzip
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from crypto_ai.exceptions import CryptoAIError, NormalizationIntegrityError
from crypto_ai.sentiment.canonical import canonical_sha256, sha256_bytes
from crypto_ai.sentiment.exceptions import NetworkSafetyError, ReceiptValidationError
from crypto_ai.sentiment.network import (
    GAP_EVIDENCE_VERSION,
    GSGNetworkClient,
    RetrievalFailure,
    build_terminal_gap_evidence,
    verify_retained_closeout,
    verify_terminal_gap_evidence,
)
from crypto_ai.sentiment.providers.gdelt_gsg import (
    GSGRetryPolicy,
    TerminalGapEvidence,
    _validate_terminal_gap_evidence,
    plan_retrieval,
)
from crypto_ai.sentiment.receipts import receipt_sha256, verify_closeout, verify_receipt_chain
from crypto_ai.sentiment.storage import ContentAddressedStore

PLAN = plan_retrieval("2026-09-09T00:00:00Z", "2026-09-10T00:00:00Z")
KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
PUBLIC = KEY.public_key()
RAW = gzip.compress(
    (Path(__file__).parents[1] / "fixtures/gdelt_gsg/base.jsonl").read_bytes(), mtime=0
)


class Clock:
    def __init__(self) -> None:
        self.seconds = 0.0
        self.sleeps: list[float] = []
        self.start = datetime(2026, 9, 9, 0, 45, tzinfo=UTC)

    def utc(self) -> datetime:
        return self.start + timedelta(seconds=self.seconds)

    def monotonic(self) -> float:
        return self.seconds + 100.0

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.seconds += duration


class Response:
    def __init__(self, raw: bytes = RAW, status: int = 200, headers=None) -> None:
        self.status = status
        self.headers = headers or {}
        self.url = ""
        self.raw = raw
        self.closed = False
        self.offset = 0
        self.read_delay = 0.01
        self.read_timeouts: list[float] = []

    def read(self, maximum_bytes: int, *, timeout: float) -> bytes:
        self.clock.seconds += self.read_delay
        self.read_timeouts.append(timeout)
        chunk = self.raw[self.offset : self.offset + maximum_bytes]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class Transport:
    mock_only = True
    incremental_cost_usd = 0.0

    def __init__(self, clock: Clock, *responses: Response) -> None:
        self.clock = clock
        self.responses = iter(responses)
        self.starts: list[float] = []
        self.open = Mock(side_effect=self._open)

    def _open(self, url: str, *, timeout: float) -> Response:
        self.starts.append(self.clock.monotonic())
        response = next(self.responses)
        response.url = response.url or url
        response.clock = self.clock
        return response


def client_at(tmp_path: Path, *responses: Response, **kwargs):
    clock = kwargs.pop("clock", Clock())
    transport = kwargs.pop("transport", Transport(clock, *responses))
    store = ContentAddressedStore(tmp_path)
    client = GSGNetworkClient(
        store,
        pilot_id="offline-fixture-pilot",
        specification_id="phase2-batch-b-offline-implementation-v1",
        protocol_sha256="a" * 64,
        code_commit="b" * 40,
        plan=PLAN,
        transport=transport,
        private_key=KEY,
        public_key=PUBLIC,
        clock=clock.utc,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        create_pilot=kwargs.pop("create_pilot", True),
        **kwargs,
    )
    return client, clock, transport


def minute(index=0):
    return PLAN.intervals[index].filename_timestamp


def test_success_exact_gzip_cas_and_signed_receipt(tmp_path):
    response = Response()
    client, _, transport = client_at(tmp_path, response)
    with client:
        result = client.retrieve(minute())
        assert result.snapshot.state == "complete"
        assert result.gap_evidence is None
        assert client.store.get_bytes(sha256_bytes(RAW)) == RAW
        assert client.budget.total_download_bytes == len(RAW)
        body = verify_receipt_chain(client.receipts, PUBLIC)[0]
        assert body["raw_sha256"] == sha256_bytes(RAW)
        assert body["snapshot_id"] == result.snapshot.receipt.snapshot_id
        assert body["raw_published_at_utc"] == result.snapshot.receipt.raw_published_at
        assert body["real_network_calls_prohibited"] is True
    transport.open.assert_called_once_with(
        "https://data.gdeltproject.org/gdeltv3/gsg/20260909000000.gsg.json.gz", timeout=10.0
    )
    assert all(timeout == 10.0 for timeout in response.read_timeouts)
    assert response.closed


def test_cumulative_received_byte_cap_counts_crossing_chunk_and_aborts(tmp_path):
    # Lower test cap exercises the exact same hard-cap path without allocating 500 MB.
    client, _, transport = client_at(
        tmp_path, Response(), Response(), maximum_download_bytes=len(RAW)
    )
    with client:
        client.retrieve(minute())
        with pytest.raises(NetworkSafetyError, match="cumulative download"):
            client.retrieve(minute(1))
        assert client.budget.total_download_bytes == 2 * len(RAW)
        assert len(client.receipts) == 1
        assert client.budget.halted
    assert transport.open.call_count == 2
    resumed, _, resumed_transport = client_at(
        tmp_path, Response(), create_pilot=False, maximum_download_bytes=len(RAW)
    )
    with pytest.raises(NetworkSafetyError):
        with resumed:
            pass
    resumed_transport.open.assert_not_called()


def test_storage_cap_aborts_before_raw_cas_write(tmp_path, monkeypatch):
    client, _, transport = client_at(tmp_path, Response())
    put = Mock(wraps=client.store.put_bytes)
    monkeypatch.setattr(client.store, "put_bytes", put)
    with client:
        original = client.budget.assert_storage_capacity

        def reject_payload(additional):
            if additional >= 2 * len(RAW) + 128 * 1024:
                raise NetworkSafetyError("retained pilot storage cap would be exceeded")
            return original(additional)

        monkeypatch.setattr(client.budget, "assert_storage_capacity", reject_payload)
        with pytest.raises(NetworkSafetyError, match="retained pilot storage"):
            client.retrieve(minute())
    put.assert_not_called()
    assert transport.open.call_count == 1


def test_rate_limit_and_signed_chain_across_requests_and_restarts(tmp_path):
    client, clock, transport = client_at(tmp_path, Response(), Response())
    with client:
        client.retrieve(minute())
        client.retrieve(minute(1))
        old_head = receipt_sha256(client.receipts[-1])
    assert transport.starts[1] - transport.starts[0] >= 5
    resumed, _, resumed_transport = client_at(tmp_path, Response(), clock=clock, create_pilot=False)
    with resumed:
        resumed.retrieve(minute(2))
        assert resumed.receipts[-1]["body"]["previous_receipt_sha256"] == old_head
        assert resumed.budget.total_download_bytes == 3 * len(RAW)
    assert resumed_transport.starts[0] - transport.starts[-1] >= 5


@pytest.mark.parametrize("status", [429, 500, 502, 503, 599])
def test_transient_retry_success_exact_error_bytes_also_count(tmp_path, status):
    client, clock, transport = client_at(tmp_path, Response(b"error bytes", status), Response())
    with client:
        result = client.retrieve(minute())
        assert result.snapshot.state == "complete"
        assert len(result.receipts) == 2
        assert client.budget.total_download_bytes == len(b"error bytes") + len(RAW)
        assert client.store.get_bytes(sha256_bytes(b"error bytes")) == b"error bytes"
        assert [
            body["attempt_number"] for body in verify_receipt_chain(client.receipts, PUBLIC)
        ] == [1, 2]
    assert transport.starts[1] - transport.starts[0] >= 5
    assert clock.sleeps == [5.0]


def test_four_attempt_exhaustion_bound_to_versioned_terminal_evidence(tmp_path):
    client, clock, transport = client_at(tmp_path, *(Response(b"busy", 503) for _ in range(4)))
    with client:
        result = client.retrieve(minute())
        evidence = result.gap_evidence
        assert isinstance(evidence, TerminalGapEvidence)
        assert evidence.version == GAP_EVIDENCE_VERSION
        assert evidence.attempt_count == 4
        assert evidence.final_terminal_disposition == "retry_exhausted"
        bindings = dict(
            plan=PLAN,
            filename_timestamp=minute(),
            receipts=client.receipts,
            public_key=PUBLIC,
            protocol_sha256="a" * 64,
        )
        verify_terminal_gap_evidence(evidence, **bindings)
        for changed in (
            replace(evidence, attempt_count=3),
            replace(evidence, evidence_sha256="0" * 64),
            replace(evidence, terminal_at="2099-01-01T00:00:00Z"),
        ):
            with pytest.raises(NetworkSafetyError):
                verify_terminal_gap_evidence(changed, **bindings)
        with pytest.raises(NetworkSafetyError):
            build_terminal_gap_evidence(**{**bindings, "receipts": client.receipts[:-1]})
        with pytest.raises(NormalizationIntegrityError):
            _validate_terminal_gap_evidence(
                evidence,
                interval_start=minute(),
                interval_end_exclusive=minute(1),
                expected_source_locator=evidence.expected_source_locator,
                protocol_config_sha256="a" * 64,
                terminal_as_of=None,
            )
    assert transport.open.call_count == 4
    assert clock.sleeps == [5.0, 10.0, 20.0]
    assert GSGRetryPolicy.maximum_attempts == 3  # Accepted Batch A is untouched.


@pytest.mark.parametrize("status", [301, 400, 401, 402, 403, 404, 408])
def test_permanent_http_errors_do_not_retry_and_halt(tmp_path, status):
    client, _, transport = client_at(tmp_path, Response(b"no", status))
    with client:
        with pytest.raises(RetrievalFailure) as failure:
            client.retrieve(minute())
        assert failure.value.result.gap_evidence.attempt_count == 1
        assert client.budget.halted
    transport.open.assert_called_once()


@pytest.mark.parametrize("raw", [b"", b"not gzip", RAW[:-3], gzip.compress(b"{bad json}", mtime=0)])
def test_malformed_payload_keeps_exact_incident_snapshot_and_fails_closed(tmp_path, raw):
    client, _, transport = client_at(tmp_path, Response(raw))
    with client:
        with pytest.raises(RetrievalFailure) as failure:
            client.retrieve(minute())
        result = failure.value.result
        assert result.snapshot.state == "invalid"
        assert result.gap_evidence.observed_raw_snapshot_sha256 == sha256_bytes(raw)
        assert client.store.get_bytes(sha256_bytes(raw)) == raw
    transport.open.assert_called_once()


def test_longer_retry_after_is_honored(tmp_path):
    client, clock, _ = client_at(
        tmp_path, Response(b"busy", 429, {"Retry-After": "31"}), Response()
    )
    with client:
        client.retrieve(minute())
    assert clock.sleeps == [31.0]


@pytest.mark.parametrize(
    "header", ["bad", "NaN", "-1", "1.5", "901", "Sun, 10 Sep 2026 01:00:00 GMT"]
)
def test_malformed_or_unsupported_retry_after_fails_closed(tmp_path, header):
    client, _, transport = client_at(tmp_path, Response(b"busy", 429, {"Retry-After": header}))
    with client, pytest.raises(NetworkSafetyError, match="Retry-After"):
        client.retrieve(minute())
    transport.open.assert_called_once()


@pytest.mark.parametrize("cost", [None, float("nan"), float("inf"), True, -1.0, 0.01])
def test_positive_unknown_or_invalid_cost_stops_before_transport(tmp_path, cost):
    clock = Clock()
    transport = Transport(clock, Response())
    transport.incremental_cost_usd = cost
    with pytest.raises(NetworkSafetyError, match="cost"):
        client_at(tmp_path, clock=clock, transport=transport)
    transport.open.assert_not_called()


def test_real_network_transport_and_override_are_rejected(tmp_path):
    transport = Transport(Clock(), Response())
    transport.mock_only = False
    with pytest.raises(NetworkSafetyError, match="offline mock"):
        client_at(tmp_path, transport=transport)
    with pytest.raises(NetworkSafetyError, match="real network"):
        client_at(tmp_path, real_network_calls_prohibited=False)
    transport.open.assert_not_called()


def test_timeout_closes_stream_and_preserves_received_byte_count(tmp_path):
    response = Response()
    response.read_delay = 10.1
    client, _, transport = client_at(tmp_path, response)
    with client:
        with pytest.raises(NetworkSafetyError, match="HTTP read"):
            client.retrieve(minute())
        assert client.budget.total_download_bytes == len(RAW)
        assert client.budget.halted
    assert response.closed
    transport.open.assert_called_once()


def test_monotonic_and_fixed_worker_deadlines(tmp_path):
    client, clock, transport = client_at(tmp_path, Response())
    with client:
        clock.seconds = 900
        with pytest.raises(NetworkSafetyError, match="deadline"):
            client.retrieve(minute())
    transport.open.assert_not_called()


@pytest.mark.parametrize(
    "timestamp", ["2099-01-01T00:00:00Z", "http://publisher.test/article", minute(15)]
)
def test_unplanned_locator_or_wrong_reporting_worker_is_rejected(tmp_path, timestamp):
    client, _, transport = client_at(tmp_path, Response())
    with client, pytest.raises(NetworkSafetyError):
        client.retrieve(timestamp)
    transport.open.assert_not_called()


def test_redirected_response_is_rejected_without_following(tmp_path):
    response = Response()
    response.url = "https://publisher.test/private?token=SECRET"
    client, _, transport = client_at(tmp_path, response)
    with client, pytest.raises(NetworkSafetyError) as failure:
        client.retrieve(minute())
    assert "SECRET" not in str(failure.value)
    assert response.closed
    transport.open.assert_called_once()


def test_transport_error_is_redacted_and_never_retried(tmp_path):
    client, _, transport = client_at(tmp_path, Response())
    transport.open.side_effect = OSError("secret_token=PRIVATE")
    with client, pytest.raises(NetworkSafetyError) as failure:
        client.retrieve(minute())
    assert "PRIVATE" not in str(failure.value)
    assert failure.value.__cause__ is None
    transport.open.assert_called_once()


def test_closeout_binds_all_payloads_and_detects_truncated_chain(tmp_path):
    client, _, _ = client_at(tmp_path, Response(), Response())
    with client:
        client.retrieve(minute())
        client.retrieve(minute(1))
        closeout = client.closeout()
        pin = canonical_sha256(closeout)
        verified = verify_retained_closeout(
            client.budget,
            public_key=PUBLIC,
            receipts=client.receipts,
            expected_closeout_sha256=pin,
            expected_plan_sha256=PLAN.plan_id,
        )
        assert len(verified["slot_outcomes"]) == 96
        assert all(slot["outcome"] == "uncovered" for slot in verified["slot_outcomes"])
        with pytest.raises(ReceiptValidationError):
            verify_closeout(
                closeout,
                PUBLIC,
                client.receipts[:-1],
                expected_closeout_sha256=pin,
                expected_inventory_sha256=verified["payload_inventory_sha256"],
                expected_plan_sha256=PLAN.plan_id,
            )
        # An injected extra raw object must break the complete inventory commitment.
        client.store.put_bytes(b"unmanifested extra object")
        with pytest.raises(NetworkSafetyError, match="inventory"):
            verify_retained_closeout(
                client.budget,
                public_key=PUBLIC,
                receipts=client.receipts,
                expected_closeout_sha256=pin,
                expected_plan_sha256=PLAN.plan_id,
            )


def test_full_reporting_interval_requires_fifteen_verified_minute_files(tmp_path):
    client, _, _ = client_at(tmp_path, *(Response() for _ in range(15)))
    with client:
        for index in range(15):
            client.retrieve(minute(index))
        closeout = client.closeout()
        assert closeout["body"]["slot_outcomes"][0]["outcome"] == "verified"
        assert sum(slot["outcome"] == "verified" for slot in closeout["body"]["slot_outcomes"]) == 1


def test_deterministic_offline_rerun_and_no_mutable_receipt_alias(tmp_path):
    outputs = []
    for directory in (tmp_path / "one", tmp_path / "two"):
        client, _, _ = client_at(directory, Response())
        with client:
            result = client.retrieve(minute())
            exposed = client.receipts
            exposed[0]["body"]["pilot_id"] = "changed"
            assert client.receipts[0]["body"]["pilot_id"] == "offline-fixture-pilot"
            outputs.append((result.snapshot.receipt.to_dict(), client.receipts))
    assert outputs[0] == outputs[1]


def test_corrupted_receipt_restoration_raises_project_error(tmp_path, monkeypatch):
    client, clock, _ = client_at(tmp_path, Response())
    with client:
        client.retrieve(minute())
    resumed, _, transport = client_at(tmp_path, Response(), clock=clock, create_pilot=False)
    original = resumed.store.read_publication

    def corrupted(name, **kwargs):
        publication = original(name, **kwargs)
        if "attempt-" in name:
            publication.files["signed-receipt.json"] = b"{bad json}"
        return publication

    monkeypatch.setattr(resumed.store, "read_publication", corrupted)
    with pytest.raises(CryptoAIError):
        with resumed:
            pass
    transport.open.assert_not_called()


def test_signed_receipt_mutation_reorder_and_missing_predecessor_rejected(tmp_path):
    client, _, _ = client_at(tmp_path, Response(), Response())
    with client:
        client.retrieve(minute())
        client.retrieve(minute(1))
        receipts = client.receipts
        for changed in (receipts[::-1], (receipts[1],)):
            with pytest.raises(ReceiptValidationError):
                verify_receipt_chain(changed, PUBLIC)
        altered = deepcopy(receipts)
        altered[1]["body"]["previous_receipt_sha256"] = None
        with pytest.raises(ReceiptValidationError):
            verify_receipt_chain(altered, PUBLIC)
