"""Offline signature, schema, causal chain and independently pinned closeout probes."""

from __future__ import annotations

import base64
from copy import deepcopy
from datetime import timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp
from crypto_ai.sentiment.exceptions import ReceiptValidationError
from crypto_ai.sentiment.providers.gdelt_gsg import (
    MAX_COMPRESSED_BYTES,
    MAX_DECOMPRESSED_BYTES,
    MAX_JSON_LINES,
    _derive_snapshot_id,
    plan_retrieval,
)
from crypto_ai.sentiment.receipts import (
    CLOSEOUT_DOMAIN,
    RECEIPT_DOMAIN,
    RETRY_POLICY_VERSION,
    closeout_sha256,
    receipt_sha256,
    sign_closeout,
    sign_receipt,
    signer_key_id,
    verify_closeout,
    verify_receipt,
    verify_receipt_chain,
)

KEY = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
PUBLIC = KEY.public_key()
START = "2026-09-09T00:00:00Z"
PLAN = plan_retrieval(START, "2026-09-10T00:00:00Z", maximum_intervals=1_440)
INVENTORY = sha256_bytes(b"independently verified test payload inventory")


def _body(*, minute: int = 0, requested_seconds: int = 0, **changes: object) -> dict:
    timestamp = PLAN.intervals[minute].filename_timestamp
    request_at = format_utc_timestamp(
        parse_utc_timestamp(START, field="start")
        + timedelta(minutes=45 + (minute // 15) * 15, seconds=requested_seconds)
    )
    raw_hash = sha256_bytes(b"fixture gzip bytes")
    body = {
        "pilot_id": "offline-test-pilot",
        "specification_id": "phase2-batch-b-prospective-ingestion-v1",
        "protocol_sha256": "a" * 64,
        "code_commit": "b" * 40,
        "plan_sha256": PLAN.plan_id,
        "plan_start_at_utc": START,
        "interval_index": minute // 15,
        "filename_timestamp": timestamp,
        "source_locator": f"https://data.gdeltproject.org/{PLAN.intervals[minute].relative_path}",
        "retry_policy_version": RETRY_POLICY_VERSION,
        "input_class": "synthetic_fixture",
        "real_network_calls_prohibited": True,
        "attempt_number": 1,
        "http_status": 200,
        "bytes_received": len(b"fixture gzip bytes"),
        "content_length": None,
        "transfer_complete": True,
        "raw_sha256": raw_hash,
        "snapshot_id": _derive_snapshot_id(
            collection_mode="prospective",
            filename_timestamp=timestamp,
            input_class="synthetic_fixture",
            raw_snapshot_sha256=raw_hash,
            max_compressed_bytes=MAX_COMPRESSED_BYTES,
            max_decompressed_bytes=MAX_DECOMPRESSED_BYTES,
            max_json_lines=MAX_JSON_LINES,
        ),
        "snapshot_state": "complete",
        "retry_after_seconds": None,
        "requested_at_utc": request_at,
        "dispatch_confirmed_at_utc": request_at,
        "completed_at_utc": request_at,
        "raw_published_at_utc": request_at,
        "previous_receipt_sha256": None,
    }
    body.update(changes)
    return body


def _chain(count: int = 2) -> list[dict]:
    chain = []
    for minute in range(count):
        chain.append(
            sign_receipt(
                _body(
                    minute=minute,
                    requested_seconds=(minute % 15) * 5,
                    previous_receipt_sha256=receipt_sha256(chain[-1]) if chain else None,
                ),
                KEY,
            )
        )
    return chain


def _closeout(chain: list[dict], *, outcome: str = "uncovered", **changes: object) -> dict:
    context = {
        field: _body()[field]
        for field in (
            "pilot_id",
            "specification_id",
            "protocol_sha256",
            "code_commit",
            "plan_sha256",
        )
    }
    body = {
        **context,
        "final_receipt_sha256": receipt_sha256(chain[-1]) if chain else None,
        "receipt_count": len(chain),
        "slot_outcomes": [
            {"interval_index": index, "outcome": outcome if index == 0 else "uncovered"}
            for index in range(96)
        ],
        "payload_inventory_sha256": INVENTORY,
    }
    body.update(changes)
    return sign_closeout(body, KEY)


def _verify_closeout(closeout: dict, chain: list[dict], **changes: object) -> dict:
    pins = {
        "expected_closeout_sha256": closeout_sha256(closeout),
        "expected_inventory_sha256": INVENTORY,
        "expected_plan_sha256": PLAN.plan_id,
    }
    pins.update(changes)
    return verify_closeout(closeout, PUBLIC, chain, **pins)


def test_detached_signature_exact_domain_canonical_bytes_and_copy_isolation() -> None:
    body = _body()
    receipt = sign_receipt(body, KEY)
    PUBLIC.verify(base64.b64decode(receipt["signature"]), RECEIPT_DOMAIN + canonicalize(body))
    assert RECEIPT_DOMAIN == b"KrypX Batch B receipt v1\n"
    assert verify_receipt(receipt, PUBLIC) == body
    assert receipt["signer_key_id"] == signer_key_id(PUBLIC)
    body["pilot_id"] = "mutated"
    verified = verify_receipt(receipt, PUBLIC)
    verified["pilot_id"] = "also-mutated"
    assert verify_receipt(receipt, PUBLIC)["pilot_id"] == "offline-test-pilot"
    assert sign_receipt(_body(), KEY) == receipt
    assert issubclass(ReceiptValidationError, CryptoAIError)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pilot_id", ""),
        ("pilot_id", "https://secret@example.com"),
        ("specification_id", 1),
        ("protocol_sha256", "A" * 64),
        ("code_commit", "b" * 7),
        ("plan_sha256", "f" * 64),
        ("plan_start_at_utc", "2026-09-09T00:00:00+00:00"),
        ("interval_index", True),
        ("interval_index", 96),
        ("interval_index", 1),
        ("attempt_number", 0),
        ("attempt_number", 5),
        ("attempt_number", True),
        ("http_status", True),
        ("http_status", 600),
        ("bytes_received", -1),
        ("bytes_received", 500_000_001),
        ("bytes_received", float("inf")),
        ("bytes_received", float("nan")),
        ("raw_sha256", ["a"]),
        ("snapshot_id", "f" * 64),
        ("snapshot_state", "absent"),
        ("retry_after_seconds", -1),
        ("retry_after_seconds", float("nan")),
        ("retry_after_seconds", 20),
        ("retry_after_seconds", True),
        ("input_class", "provider_response"),
        ("real_network_calls_prohibited", False),
        ("retry_policy_version", "gdelt-gsg-retry-policy-v1"),
        ("source_locator", "http://data.gdeltproject.org/gdeltv2/20260909000000.gsg.jsonl.gz"),
        ("filename_timestamp", "2026-09-09T00:00:01Z"),
        ("requested_at_utc", "2026-09-09T00:29:59Z"),
        ("completed_at_utc", "2026-09-09T00:44:59Z"),
        ("raw_published_at_utc", "2026-09-09T00:44:59Z"),
        ("requested_at_utc", "2026-02-30T00:00:00Z"),
        ("requested_at_utc", "2026-09-09T00:45:00.0Z"),
        ("previous_receipt_sha256", "invalid"),
    ],
)
def test_malformed_receipt_fields_fail_with_project_error(field: str, value: object) -> None:
    with pytest.raises(ReceiptValidationError):
        sign_receipt(_body(**{field: value}), KEY)


@pytest.mark.parametrize("mutation", ["missing", "extra", "not_dict"])
def test_closed_world_body(mutation: str) -> None:
    body = _body()
    if mutation == "missing":
        del body["previous_receipt_sha256"]
    elif mutation == "extra":
        body["unsigned_approval"] = True
    else:
        body = []
    with pytest.raises(ReceiptValidationError):
        sign_receipt(body, KEY)


def test_null_http_requires_zero_bytes_and_absent_capture() -> None:
    body = _body(
        http_status=None,
        raw_sha256=None,
        bytes_received=0,
        snapshot_state="absent",
        snapshot_id=None,
        raw_published_at_utc=None,
        transfer_complete=False,
    )
    assert verify_receipt(sign_receipt(body, KEY), PUBLIC) == body
    body["bytes_received"] = 1
    with pytest.raises(ReceiptValidationError, match="absent response"):
        sign_receipt(body, KEY)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "future-version"),
        ("algorithm", "none"),
        ("signer_key_id", "e" * 64),
        ("body_sha256", "f" * 64),
        ("signature", "not-base64"),
        ("signature", ""),
        ("signature", "é"),
        ("signature", base64.b64encode(b"a" * 64).decode() + "\n"),
        ("signature", base64.b64encode(b"a" * 64).decode()),
    ],
)
def test_signature_and_envelope_tampering(field: str, value: object) -> None:
    receipt = sign_receipt(_body(), KEY)
    receipt[field] = value
    with pytest.raises(ReceiptValidationError):
        verify_receipt(receipt, PUBLIC)


def test_recomputed_dependent_hash_cannot_forge_signature() -> None:
    receipt = sign_receipt(_body(), KEY)
    receipt["body"]["bytes_received"] += 1
    receipt["body_sha256"] = sha256_bytes(canonicalize(receipt["body"]))
    with pytest.raises(ReceiptValidationError, match="signature verification"):
        verify_receipt(receipt, PUBLIC)


def test_unknown_outer_field_and_wrong_pinned_key_rejected() -> None:
    receipt = sign_receipt(_body(), KEY)
    receipt["approval"] = "secret unsigned metadata"
    with pytest.raises(ReceiptValidationError, match="envelope fields"):
        verify_receipt(receipt, PUBLIC)
    del receipt["approval"]
    with pytest.raises(ReceiptValidationError, match="pinned key"):
        verify_receipt(receipt, Ed25519PrivateKey.from_private_bytes(b"z" * 32).public_key())


@pytest.mark.parametrize(
    "mutation", ["reordered", "missing_first", "missing_middle", "duplicate", "missing_hash"]
)
def test_receipt_chain_rejects_missing_or_reordered_receipts(mutation: str) -> None:
    chain = _chain(3)
    if mutation == "reordered":
        chain = [chain[1], chain[0], chain[2]]
    elif mutation == "missing_first":
        chain = chain[1:]
    elif mutation == "missing_middle":
        chain = [chain[0], chain[2]]
    elif mutation == "duplicate":
        chain = [chain[0], chain[0], chain[1]]
    else:
        body = deepcopy(chain[1]["body"])
        body["previous_receipt_sha256"] = None
        chain[1] = sign_receipt(body, KEY)
    with pytest.raises(ReceiptValidationError, match="predecessor"):
        verify_receipt_chain(chain, PUBLIC)


def test_chain_checks_independent_head_and_context() -> None:
    chain = _chain(3)
    assert len(verify_receipt_chain(chain, PUBLIC, receipt_sha256(chain[-1]))) == 3
    with pytest.raises(ReceiptValidationError, match="pinned head"):
        verify_receipt_chain(chain[:-1], PUBLIC, receipt_sha256(chain[-1]))
    body = deepcopy(chain[1]["body"])
    body["pilot_id"] = "different-pilot"
    with pytest.raises(ReceiptValidationError, match="context"):
        verify_receipt_chain([chain[0], sign_receipt(body, KEY)], PUBLIC)


@pytest.mark.parametrize("seconds", [0, 4])
def test_chain_rejects_early_request_starts(seconds: int) -> None:
    first = sign_receipt(_body(), KEY)
    second = sign_receipt(
        _body(minute=1, requested_seconds=seconds, previous_receipt_sha256=receipt_sha256(first)),
        KEY,
    )
    with pytest.raises(ReceiptValidationError, match="rate limit"):
        verify_receipt_chain([first, second], PUBLIC)


def _retry_chain(statuses: tuple[int, ...] = (429, 503, 500, 502)) -> list[dict]:
    chain = []
    for index, status in enumerate(statuses):
        chain.append(
            sign_receipt(
                _body(
                    requested_seconds=(0, 5, 15, 35)[index],
                    attempt_number=index + 1,
                    http_status=status,
                    snapshot_state="absent",
                    snapshot_id=None,
                    previous_receipt_sha256=receipt_sha256(chain[-1]) if chain else None,
                ),
                KEY,
            )
        )
    return chain


def test_authenticated_retry_order_and_closeout_gap() -> None:
    chain = _retry_chain()
    assert len(verify_receipt_chain(chain, PUBLIC)) == 4
    closeout = _closeout(chain, outcome="provider_gap")
    assert _verify_closeout(closeout, chain)["slot_outcomes"][0]["outcome"] == "provider_gap"
    with pytest.raises(ReceiptValidationError, match="terminal attempt"):
        _verify_closeout(_closeout(chain[:3], outcome="provider_gap"), chain[:3])


def test_authenticated_retry_after_cannot_be_ignored() -> None:
    first = sign_receipt(
        _body(http_status=429, snapshot_state="absent", snapshot_id=None, retry_after_seconds=30),
        KEY,
    )
    second = sign_receipt(
        _body(requested_seconds=5, attempt_number=2, previous_receipt_sha256=receipt_sha256(first)),
        KEY,
    )
    with pytest.raises(ReceiptValidationError, match="backoff"):
        verify_receipt_chain([first, second], PUBLIC)


def test_nonretryable_or_skipped_attempts_fail_closed() -> None:
    first = sign_receipt(_body(), KEY)
    with pytest.raises(ReceiptValidationError, match="nonretryable"):
        verify_receipt_chain(
            [
                first,
                sign_receipt(
                    _body(
                        requested_seconds=5,
                        attempt_number=2,
                        previous_receipt_sha256=receipt_sha256(first),
                    ),
                    KEY,
                ),
            ],
            PUBLIC,
        )
    with pytest.raises(ReceiptValidationError, match="consecutive"):
        verify_receipt_chain([sign_receipt(_body(attempt_number=2), KEY)], PUBLIC)


def test_signed_closeout_verifies_all_fifteen_minute_receipts() -> None:
    chain = _chain(15)
    closeout = _closeout(chain, outcome="verified")
    PUBLIC.verify(
        base64.b64decode(closeout["signature"]), CLOSEOUT_DOMAIN + canonicalize(closeout["body"])
    )
    assert CLOSEOUT_DOMAIN == b"KrypX Batch B closeout v1\n"
    result = _verify_closeout(closeout, chain)
    assert len(result["slot_outcomes"]) == 96
    assert result["slot_outcomes"][0]["outcome"] == "verified"


def test_closeout_detects_truncated_chain_and_changed_signed_closeout() -> None:
    chain = _chain(3)
    closeout = _closeout(chain)
    with pytest.raises(ReceiptValidationError, match="pinned head"):
        _verify_closeout(closeout, chain[:-1])
    replacement = _closeout(chain[:-1])
    with pytest.raises(ReceiptValidationError, match="closeout pin"):
        _verify_closeout(
            replacement, chain[:-1], expected_closeout_sha256=closeout_sha256(closeout)
        )


@pytest.mark.parametrize(
    "field", ["expected_closeout_sha256", "expected_inventory_sha256", "expected_plan_sha256"]
)
def test_closeout_requires_independent_matching_pins(field: str) -> None:
    chain = _chain()
    with pytest.raises(ReceiptValidationError):
        _verify_closeout(_closeout(chain), chain, **{field: "f" * 64})


def test_missing_closeout_and_empty_chain_are_not_coverage_success() -> None:
    with pytest.raises(ReceiptValidationError, match="envelope fields"):
        verify_closeout(
            None,
            PUBLIC,
            [],
            expected_closeout_sha256="a" * 64,
            expected_inventory_sha256=INVENTORY,
            expected_plan_sha256=PLAN.plan_id,
        )
    assert _verify_closeout(_closeout([]), [])["receipt_count"] == 0
    with pytest.raises(ReceiptValidationError, match="15 successful"):
        _verify_closeout(_closeout([], outcome="verified"), [])


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "reordered", "extra", "bool_index", "bad_outcome"]
)
def test_closeout_requires_exact_96_ordered_slot_outcomes(mutation: str) -> None:
    body = _closeout([])["body"]
    slots = body["slot_outcomes"]
    if mutation == "missing":
        slots.pop()
    elif mutation == "duplicate":
        slots[1] = deepcopy(slots[0])
    elif mutation == "reordered":
        slots[0], slots[1] = slots[1], slots[0]
    elif mutation == "extra":
        slots[0]["verified_count"] = 1440
    elif mutation == "bool_index":
        slots[0]["interval_index"] = False
    else:
        slots[0]["outcome"] = "approved"
    with pytest.raises(ReceiptValidationError):
        sign_closeout(body, KEY)


def test_http_200_invalid_snapshot_does_not_count_as_verified() -> None:
    chain = _chain(15)
    body = deepcopy(chain[-1]["body"])
    body["snapshot_state"] = "invalid"
    chain[-1] = sign_receipt(body, KEY)
    with pytest.raises(ReceiptValidationError, match="15 successful"):
        _verify_closeout(_closeout(chain, outcome="verified"), chain)
    assert _verify_closeout(_closeout(chain, outcome="provider_gap"), chain)


def test_invalid_key_types_raise_project_specific_errors() -> None:
    with pytest.raises(ReceiptValidationError):
        sign_receipt(_body(), b"not a signing key")
    with pytest.raises(ReceiptValidationError):
        verify_receipt(sign_receipt(_body(), KEY), b"not a verification key")


def test_complete_snapshot_must_respect_frozen_compressed_bound() -> None:
    with pytest.raises(ReceiptValidationError, match="compressed-byte bound"):
        sign_receipt(_body(bytes_received=MAX_COMPRESSED_BYTES + 1), KEY)


def test_receipt_chain_enforces_cumulative_download_ceiling() -> None:
    first = sign_receipt(_body(bytes_received=250_000_001, snapshot_state="invalid"), KEY)
    second = sign_receipt(
        _body(
            minute=1,
            requested_seconds=5,
            bytes_received=250_000_000,
            snapshot_state="invalid",
            previous_receipt_sha256=receipt_sha256(first),
        ),
        KEY,
    )
    with pytest.raises(ReceiptValidationError, match="cumulative download cap"):
        verify_receipt_chain([first, second], PUBLIC)


def test_distinct_snapshot_publication_times_must_strictly_increase() -> None:
    first = sign_receipt(_body(raw_published_at_utc="2026-09-09T00:45:05Z"), KEY)
    second = sign_receipt(
        _body(minute=1, requested_seconds=5, previous_receipt_sha256=receipt_sha256(first)),
        KEY,
    )
    with pytest.raises(ReceiptValidationError, match="first-seen publication times"):
        verify_receipt_chain([first, second], PUBLIC)


def test_source_minutes_cannot_move_backwards_but_uncovered_minutes_are_allowed() -> None:
    first = sign_receipt(_body(minute=1), KEY)
    second = sign_receipt(
        _body(minute=0, requested_seconds=5, previous_receipt_sha256=receipt_sha256(first)), KEY
    )
    with pytest.raises(ReceiptValidationError, match="source-minute chronology"):
        verify_receipt_chain([first, second], PUBLIC)
    later = sign_receipt(
        _body(minute=4, requested_seconds=5, previous_receipt_sha256=receipt_sha256(first)), KEY
    )
    assert len(verify_receipt_chain([first, later], PUBLIC)) == 2


def test_plan_datetime_overflow_raises_project_specific_error() -> None:
    with pytest.raises(ReceiptValidationError, match="24-hour receipt plan"):
        sign_receipt(_body(plan_start_at_utc="9999-12-31T00:00:00Z"), KEY)


def _resign_unvalidated_receipt(receipt: dict, **changes: object) -> dict:
    """A legitimate key cannot make semantically false schedule evidence valid."""
    receipt = deepcopy(receipt)
    receipt["body"].update(changes)
    encoded = canonicalize(receipt["body"])
    receipt["body_sha256"] = sha256_bytes(encoded)
    receipt["signature"] = base64.b64encode(KEY.sign(RECEIPT_DOMAIN + encoded)).decode("ascii")
    PUBLIC.verify(base64.b64decode(receipt["signature"]), RECEIPT_DOMAIN + encoded)
    return receipt


@pytest.mark.parametrize(
    "field",
    ["requested_at_utc", "dispatch_confirmed_at_utc", "completed_at_utc", "raw_published_at_utc"],
)
@pytest.mark.parametrize(
    "outside", ["2026-09-09T00:44:59.999999Z", "2026-09-09T01:00:00Z", "2099-01-01T00:00:00Z"]
)
def test_every_receipt_timestamp_is_confined_to_authoritative_worker_slot(
    field: str, outside: str
) -> None:
    body = _body(**{field: outside})
    with pytest.raises(ReceiptValidationError, match="scheduled reporting slot"):
        sign_receipt(body, KEY)
    forged = _resign_unvalidated_receipt(sign_receipt(_body(), KEY), **{field: outside})
    with pytest.raises(ReceiptValidationError, match="scheduled reporting slot"):
        verify_receipt(forged, PUBLIC)
    with pytest.raises(ReceiptValidationError, match="scheduled reporting slot"):
        verify_receipt_chain([forged], PUBLIC)


@pytest.mark.parametrize("minute", [0, 14, 15, 30, 1_439])
@pytest.mark.parametrize("offset_microseconds", [0, 899_999_999])
def test_reporting_worker_bounds_are_lower_inclusive_and_upper_exclusive(
    minute: int, offset_microseconds: int
) -> None:
    start = parse_utc_timestamp(START, field="start")
    timestamp = format_utc_timestamp(
        start + timedelta(minutes=45 + (minute // 15) * 15, microseconds=offset_microseconds)
    )
    body = _body(
        minute=minute,
        requested_at_utc=timestamp,
        dispatch_confirmed_at_utc=timestamp,
        completed_at_utc=timestamp,
        raw_published_at_utc=timestamp,
    )
    assert verify_receipt(sign_receipt(body, KEY), PUBLIC) == body


def test_cryptographically_valid_future_receipts_cannot_certify_closeout_coverage() -> None:
    chain = _chain(15)
    future_chain = []
    future = parse_utc_timestamp("2099-01-01T00:00:00Z", field="future")
    for index, receipt in enumerate(chain):
        timestamp = format_utc_timestamp(future + timedelta(seconds=index * 5))
        future_chain.append(
            _resign_unvalidated_receipt(
                receipt,
                requested_at_utc=timestamp,
                dispatch_confirmed_at_utc=timestamp,
                completed_at_utc=timestamp,
                raw_published_at_utc=timestamp,
                previous_receipt_sha256=(
                    sha256_bytes(canonicalize(future_chain[-1])) if future_chain else None
                ),
            )
        )
    body = _closeout(chain, outcome="verified")["body"]
    body["final_receipt_sha256"] = sha256_bytes(canonicalize(future_chain[-1]))
    closeout = sign_closeout(body, KEY)
    PUBLIC.verify(base64.b64decode(closeout["signature"]), CLOSEOUT_DOMAIN + canonicalize(body))
    with pytest.raises(ReceiptValidationError, match="scheduled reporting slot"):
        _verify_closeout(closeout, future_chain)


@pytest.mark.parametrize("declared", [-1, 500_000_001, True, "18", 18.0, float("nan")])
def test_content_length_requires_a_bounded_exact_integer(declared: object) -> None:
    with pytest.raises(ReceiptValidationError, match="content_length"):
        sign_receipt(_body(content_length=declared), KEY)
    if isinstance(declared, float) and declared != declared:
        # NaN has no canonical JSON encoding to sign; reject even before hashing.
        forged = sign_receipt(_body(), KEY)
        forged["body"]["content_length"] = declared
    else:
        forged = _resign_unvalidated_receipt(sign_receipt(_body(), KEY), content_length=declared)
    with pytest.raises(ReceiptValidationError, match="content_length"):
        verify_receipt(forged, PUBLIC)


@pytest.mark.parametrize("delta", [-1, 1])
def test_declared_transfer_length_mismatch_cannot_be_signed_as_complete(delta: int) -> None:
    body = _body(content_length=len(b"fixture gzip bytes") + delta)
    with pytest.raises(ReceiptValidationError, match="Content-Length"):
        sign_receipt(body, KEY)
    forged = _resign_unvalidated_receipt(
        sign_receipt(_body(), KEY), content_length=body["content_length"]
    )
    with pytest.raises(ReceiptValidationError, match="Content-Length"):
        verify_receipt(forged, PUBLIC)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_transfer_complete_requires_an_exact_boolean(value: object) -> None:
    with pytest.raises(ReceiptValidationError, match="transfer_complete"):
        sign_receipt(_body(transfer_complete=value), KEY)


def test_transfer_completion_evidence_is_required_for_verified_coverage() -> None:
    with pytest.raises(ReceiptValidationError, match="verified complete transfer"):
        sign_receipt(_body(transfer_complete=False), KEY)
    forged = _resign_unvalidated_receipt(sign_receipt(_body(), KEY), transfer_complete=False)
    with pytest.raises(ReceiptValidationError, match="verified complete transfer"):
        verify_receipt(forged, PUBLIC)
    body = _body(
        content_length=len(b"fixture gzip bytes") + 100,
        transfer_complete=False,
        snapshot_state="invalid",
    )
    receipt = sign_receipt(body, KEY)
    assert verify_receipt(receipt, PUBLIC) == body
    assert _verify_closeout(_closeout([receipt], outcome="provider_gap"), [receipt])


def test_non200_incomplete_transfer_is_terminal_and_not_retryable() -> None:
    receipt = sign_receipt(
        _body(
            http_status=429,
            snapshot_id=None,
            snapshot_state="absent",
            content_length=100,
            transfer_complete=False,
        ),
        KEY,
    )
    assert _verify_closeout(_closeout([receipt], outcome="provider_gap"), [receipt])
    retry = sign_receipt(
        _body(
            requested_seconds=5,
            attempt_number=2,
            previous_receipt_sha256=receipt_sha256(receipt),
        ),
        KEY,
    )
    with pytest.raises(ReceiptValidationError, match="nonretryable"):
        verify_receipt_chain([receipt, retry], PUBLIC)


@pytest.mark.parametrize("dispatch", ["2026-09-09T00:45:00Z", "2026-09-09T00:45:02Z"])
def test_dispatch_confirmation_must_be_inside_request_completion_window(dispatch: str) -> None:
    body = _body(requested_seconds=1, dispatch_confirmed_at_utc=dispatch)
    with pytest.raises(ReceiptValidationError, match="dispatch confirmation"):
        sign_receipt(body, KEY)


def test_chain_paces_from_dispatch_upper_bound_not_request_intent() -> None:
    first = sign_receipt(
        _body(
            dispatch_confirmed_at_utc="2026-09-09T00:45:01Z",
            completed_at_utc="2026-09-09T00:45:01Z",
            raw_published_at_utc="2026-09-09T00:45:01Z",
        ),
        KEY,
    )
    early = sign_receipt(
        _body(minute=1, requested_seconds=5, previous_receipt_sha256=receipt_sha256(first)), KEY
    )
    with pytest.raises(ReceiptValidationError, match="five-second rate limit"):
        verify_receipt_chain([first, early], PUBLIC)
    on_time = sign_receipt(
        _body(minute=1, requested_seconds=6, previous_receipt_sha256=receipt_sha256(first)), KEY
    )
    assert len(verify_receipt_chain([first, on_time], PUBLIC)) == 2


def test_legacy_v1_receipts_cannot_silently_lack_transfer_or_dispatch_evidence() -> None:
    receipt = sign_receipt(_body(), KEY)
    assert receipt["schema_version"] == "batch-b-signed-receipt-v2"
    receipt["schema_version"] = "batch-b-signed-receipt-v1"
    for field in ("content_length", "transfer_complete", "dispatch_confirmed_at_utc"):
        del receipt["body"][field]
    receipt = _resign_unvalidated_receipt(receipt)
    with pytest.raises(ReceiptValidationError, match="unknown signed envelope schema"):
        verify_receipt(receipt, PUBLIC)
