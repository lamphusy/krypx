"""Strict, offline-verifiable Ed25519 collector attestations.

These signatures authenticate a collector, never GDELT or publisher rights. Keys
are supplied by the caller and are neither generated nor persisted by this module.
A receipt chain alone cannot prove completeness; acceptance requires an
independently pinned signed closeout and independently checked payload inventory.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from crypto_ai.exceptions import CanonicalizationError
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

RECEIPT_SCHEMA_VERSION = "batch-b-signed-receipt-v1"
CLOSEOUT_SCHEMA_VERSION = "batch-b-signed-closeout-v1"
RECEIPT_DOMAIN = b"KrypX Batch B receipt v1\n"
CLOSEOUT_DOMAIN = b"KrypX Batch B closeout v1\n"
RETRY_POLICY_VERSION = "batch-b-gsg-retry-policy-v1"
RECEIPT_BODY_FIELDS = frozenset(
    {
        "pilot_id",
        "specification_id",
        "protocol_sha256",
        "code_commit",
        "plan_sha256",
        "plan_start_at_utc",
        "interval_index",
        "filename_timestamp",
        "source_locator",
        "retry_policy_version",
        "input_class",
        "real_network_calls_prohibited",
        "attempt_number",
        "http_status",
        "bytes_received",
        "raw_sha256",
        "snapshot_id",
        "snapshot_state",
        "raw_published_at_utc",
        "retry_after_seconds",
        "requested_at_utc",
        "completed_at_utc",
        "previous_receipt_sha256",
    }
)
CLOSEOUT_BODY_FIELDS = frozenset(
    {
        "pilot_id",
        "specification_id",
        "protocol_sha256",
        "code_commit",
        "plan_sha256",
        "final_receipt_sha256",
        "receipt_count",
        "slot_outcomes",
        "payload_inventory_sha256",
    }
)
_ENVELOPE_FIELDS = frozenset(
    {"schema_version", "body", "body_sha256", "algorithm", "signer_key_id", "signature"}
)
_CONTEXT_FIELDS = ("pilot_id", "specification_id", "protocol_sha256", "code_commit", "plan_sha256")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptValidationError(message)


def _hash(value: object, field: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{field} must be lowercase SHA-256 hex",
    )


def _integer(value: object, field: str, low: int, high: int) -> None:
    _require(type(value) is int and low <= value <= high, f"{field} must be an integer in range")


def _timestamp(value: object, field: str) -> datetime:
    try:
        parsed = parse_utc_timestamp(value, field=field)
        if parsed is None or format_utc_timestamp(parsed) != value:
            raise ValueError("timestamp must use the canonical UTC representation")
        return parsed
    except (TypeError, ValueError) as exc:
        raise ReceiptValidationError(f"{field}: {exc}") from exc


def _canonical(value: object) -> bytes:
    try:
        return canonicalize(value)
    except (CanonicalizationError, TypeError, ValueError, RecursionError) as exc:
        raise ReceiptValidationError("receipt contains invalid canonical JSON") from exc


def _copy(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical(value))


def _context(body: dict[str, Any]) -> None:
    for field in ("pilot_id", "specification_id"):
        _require(
            isinstance(body[field], str) and _IDENTIFIER.fullmatch(body[field]) is not None,
            f"{field} must be a bounded identifier",
        )
    for field in ("protocol_sha256", "plan_sha256"):
        _hash(body[field], field)
    _require(
        isinstance(body["code_commit"], str) and _COMMIT.fullmatch(body["code_commit"]) is not None,
        "code_commit must be a full lowercase Git commit SHA",
    )


@lru_cache(maxsize=16)
def _plan_hash(start_at: str) -> str:
    start = _timestamp(start_at, "plan_start_at_utc")
    _require(start.second == 0 and start.microsecond == 0, "plan must start on a minute boundary")
    try:
        end = format_utc_timestamp(start + timedelta(days=1))
        return plan_retrieval(start_at, end, maximum_intervals=1_440).plan_id
    except (ValueError, OverflowError) as exc:
        raise ReceiptValidationError("invalid 24-hour receipt plan") from exc


def _receipt_body(body: object) -> dict[str, Any]:
    _require(
        type(body) is dict and body.keys() == RECEIPT_BODY_FIELDS, "invalid receipt body fields"
    )
    _context(body)
    _integer(body["interval_index"], "interval_index", 0, 95)
    _integer(body["attempt_number"], "attempt_number", 1, 4)
    _integer(body["bytes_received"], "bytes_received", 0, 500_000_000)
    if body["http_status"] is not None:
        _integer(body["http_status"], "http_status", 100, 599)
    _hash(body["raw_sha256"], "raw_sha256", nullable=True)
    _require(
        (body["http_status"] is None) == (body["raw_sha256"] is None),
        "a captured HTTP response must bind an exact raw SHA-256",
    )
    _hash(body["previous_receipt_sha256"], "previous_receipt_sha256", nullable=True)
    plan_start = _timestamp(body["plan_start_at_utc"], "plan_start_at_utc")
    _require(
        body["plan_sha256"] == _plan_hash(body["plan_start_at_utc"]), "receipt plan hash mismatch"
    )
    minute = _timestamp(body["filename_timestamp"], "filename_timestamp")
    _require(
        minute.second == 0 and minute.microsecond == 0, "filename timestamp must be minute aligned"
    )
    offset = int((minute - plan_start).total_seconds())
    _require(0 <= offset < 86_400, "receipt minute is outside the 24-hour plan")
    _require(
        offset // 900 == body["interval_index"], "receipt minute does not match its reporting slot"
    )
    _require(
        body["source_locator"]
        == f"https://data.gdeltproject.org/gdeltv3/gsg/{minute:%Y%m%d%H%M%S}.gsg.json.gz",
        "source_locator is not the canonical GSG minute locator",
    )
    _require(body["retry_policy_version"] == RETRY_POLICY_VERSION, "unknown receipt retry policy")
    _require(
        body["input_class"] == "synthetic_fixture", "only synthetic fixture receipts are permitted"
    )
    _require(body["real_network_calls_prohibited"] is True, "real network execution is prohibited")
    requested = _timestamp(body["requested_at_utc"], "requested_at_utc")
    completed = _timestamp(body["completed_at_utc"], "completed_at_utc")
    _require(
        requested >= minute + timedelta(minutes=30), "request precedes the GSG release allowance"
    )
    _require(completed >= requested, "receipt completion precedes request")
    retry_after = body["retry_after_seconds"]
    if retry_after is not None:
        _require(
            type(retry_after) in (int, float)
            and math.isfinite(retry_after)
            and 0 <= retry_after <= 900
            and body["http_status"] in (429, 503),
            "retry_after_seconds is invalid for the captured status",
        )
    _require(
        isinstance(body["snapshot_state"], str)
        and body["snapshot_state"] in {"complete", "invalid", "absent"},
        "invalid snapshot state",
    )
    _hash(body["snapshot_id"], "snapshot_id", nullable=True)
    if body["http_status"] is None:
        _require(body["bytes_received"] == 0, "absent response cannot contain received bytes")
        _require(
            body["raw_published_at_utc"] is None, "absent response cannot have a publication time"
        )
    else:
        published = _timestamp(body["raw_published_at_utc"], "raw_published_at_utc")
        _require(published >= completed, "raw publication precedes response completion")
        _require(
            body["bytes_received"] != 0 or body["raw_sha256"] == sha256_bytes(b""),
            "zero-byte response must bind the empty byte-string hash",
        )
    if body["http_status"] == 200:
        _require(body["snapshot_state"] != "absent", "HTTP 200 must record snapshot validation")
        _require(
            body["snapshot_state"] != "complete"
            or 0 < body["bytes_received"] <= MAX_COMPRESSED_BYTES,
            "complete gzip snapshot must satisfy the frozen compressed-byte bound",
        )
        _require(
            body["snapshot_id"]
            == _derive_snapshot_id(
                collection_mode="prospective",
                filename_timestamp=body["filename_timestamp"],
                input_class="synthetic_fixture",
                raw_snapshot_sha256=body["raw_sha256"],
                max_compressed_bytes=MAX_COMPRESSED_BYTES,
                max_decompressed_bytes=MAX_DECOMPRESSED_BYTES,
                max_json_lines=MAX_JSON_LINES,
            ),
            "snapshot identity does not match the accepted Batch A parser policy",
        )
    else:
        _require(
            body["snapshot_state"] == "absent" and body["snapshot_id"] is None,
            "non-200 response cannot contain a parsed GSG snapshot",
        )
    _canonical(body)
    return body


def _closeout_body(body: object) -> dict[str, Any]:
    _require(
        type(body) is dict and body.keys() == CLOSEOUT_BODY_FIELDS, "invalid closeout body fields"
    )
    _context(body)
    _hash(body["final_receipt_sha256"], "final_receipt_sha256", nullable=True)
    _hash(body["payload_inventory_sha256"], "payload_inventory_sha256")
    _integer(body["receipt_count"], "receipt_count", 0, 5_760)
    _require(
        (body["receipt_count"] == 0) == (body["final_receipt_sha256"] is None),
        "closeout head and receipt count disagree",
    )
    outcomes = body["slot_outcomes"]
    _require(
        type(outcomes) is list and len(outcomes) == 96, "closeout must account for exactly 96 slots"
    )
    for index, slot in enumerate(outcomes):
        _require(
            type(slot) is dict and slot.keys() == {"interval_index", "outcome"},
            "invalid closeout slot fields",
        )
        _integer(slot["interval_index"], "slot interval_index", 0, 95)
        _require(
            slot["interval_index"] == index, "closeout slots must be ordered without omissions"
        )
        _require(
            isinstance(slot["outcome"], str)
            and slot["outcome"] in {"verified", "provider_gap", "uncovered"},
            "unknown closeout slot outcome",
        )
    _canonical(body)
    return body


def signer_key_id(public_key: Ed25519PublicKey) -> str:
    """Identify an independently pinned Ed25519 key, not a self-declared identity."""
    _require(isinstance(public_key, Ed25519PublicKey), "an Ed25519 public key is required")
    return sha256_bytes(
        public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    )


def _sign(
    body: dict[str, Any], private_key: Ed25519PrivateKey, *, closeout: bool
) -> dict[str, Any]:
    _require(isinstance(private_key, Ed25519PrivateKey), "an Ed25519 private key is required")
    body = _copy(_closeout_body(body) if closeout else _receipt_body(body))
    encoded = _canonical(body)
    domain = CLOSEOUT_DOMAIN if closeout else RECEIPT_DOMAIN
    return {
        "schema_version": CLOSEOUT_SCHEMA_VERSION if closeout else RECEIPT_SCHEMA_VERSION,
        "body": body,
        "body_sha256": sha256_bytes(encoded),
        "algorithm": "Ed25519",
        "signer_key_id": signer_key_id(private_key.public_key()),
        "signature": base64.b64encode(private_key.sign(domain + encoded)).decode("ascii"),
    }


def sign_receipt(body: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Sign a strictly validated fixture receipt without retaining key material."""
    return _sign(body, private_key, closeout=False)


def sign_closeout(body: dict[str, Any], private_key: Ed25519PrivateKey) -> dict[str, Any]:
    """Sign a complete 96-slot inventory commitment; verification also needs pins."""
    return _sign(body, private_key, closeout=True)


def _envelope(envelope: object, *, closeout: bool) -> dict[str, Any]:
    _require(
        type(envelope) is dict and envelope.keys() == _ENVELOPE_FIELDS,
        "invalid signed envelope fields",
    )
    expected = CLOSEOUT_SCHEMA_VERSION if closeout else RECEIPT_SCHEMA_VERSION
    _require(envelope["schema_version"] == expected, "unknown signed envelope schema")
    _require(envelope["algorithm"] == "Ed25519", "signature algorithm must be Ed25519")
    body = _closeout_body(envelope["body"]) if closeout else _receipt_body(envelope["body"])
    _hash(envelope["body_sha256"], "body_sha256")
    _hash(envelope["signer_key_id"], "signer_key_id")
    _require(
        envelope["body_sha256"] == sha256_bytes(_canonical(body)), "receipt body hash mismatch"
    )
    signature = envelope["signature"]
    _require(isinstance(signature, str), "signature must be canonical base64")
    try:
        decoded = base64.b64decode(signature, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ReceiptValidationError("signature must be canonical base64") from exc
    _require(
        len(decoded) == 64 and base64.b64encode(decoded).decode("ascii") == signature,
        "signature must be canonical base64 encoding exactly 64 bytes",
    )
    return envelope


def _verify(envelope: object, public_key: Ed25519PublicKey, *, closeout: bool) -> dict[str, Any]:
    envelope = _envelope(envelope, closeout=closeout)
    _require(
        envelope["signer_key_id"] == signer_key_id(public_key),
        "receipt signer is not the pinned key",
    )
    domain = CLOSEOUT_DOMAIN if closeout else RECEIPT_DOMAIN
    try:
        public_key.verify(
            base64.b64decode(envelope["signature"], validate=True),
            domain + _canonical(envelope["body"]),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ReceiptValidationError("Ed25519 signature verification failed") from exc
    return _copy(envelope["body"])


def verify_receipt(envelope: object, public_key: Ed25519PublicKey) -> dict[str, Any]:
    """Verify a detached receipt against the caller's independently pinned key."""
    return _verify(envelope, public_key, closeout=False)


def receipt_sha256(envelope: object) -> str:
    """Hash the full canonical envelope; signature verification is a separate step."""
    return sha256_bytes(_canonical(_envelope(envelope, closeout=False)))


def closeout_sha256(envelope: object) -> str:
    """Hash the entire validated closeout envelope for independent pinning."""
    return sha256_bytes(_canonical(_envelope(envelope, closeout=True)))


def verify_receipt_chain(
    receipts: object,
    public_key: Ed25519PublicKey,
    expected_head: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Reject reordered, omitted, duplicated, cross-context or noncausal receipts.

    Without an independently supplied head this checks internal consistency only;
    it cannot distinguish a legitimate prefix from a truncated complete chain.
    """
    _require(type(receipts) in (list, tuple), "receipt chain must be a list or tuple")
    _require(len(receipts) <= 5_760, "receipt chain exceeds the 1,440-minute attempt cap")
    signer_key_id(public_key)
    _hash(expected_head, "expected_head", nullable=True)
    verified: list[dict[str, Any]] = []
    previous_hash = None
    previous_by_minute: dict[str, dict[str, Any]] = {}
    bytes_received = 0
    last_snapshot_publication: datetime | None = None
    for envelope in receipts:
        body = verify_receipt(envelope, public_key)
        bytes_received += body["bytes_received"]
        _require(bytes_received <= 500_000_000, "receipt chain exceeds the cumulative download cap")
        _require(
            body["previous_receipt_sha256"] == previous_hash, "receipt predecessor hash mismatch"
        )
        if verified:
            previous = verified[-1]
            _require(
                all(body[field] == verified[0][field] for field in _CONTEXT_FIELDS),
                "receipt chain context changed",
            )
            _require(
                _timestamp(body["requested_at_utc"], "requested_at_utc")
                >= _timestamp(
                    previous["raw_published_at_utc"] or previous["completed_at_utc"],
                    "previous publication/completion",
                ),
                "receipt chain chronology moved backwards",
            )
            _require(
                _timestamp(body["requested_at_utc"], "requested_at_utc")
                >= _timestamp(previous["requested_at_utc"], "requested_at_utc")
                + timedelta(seconds=5),
                "receipt request starts violate the five-second rate limit",
            )
            _require(
                _timestamp(body["filename_timestamp"], "filename_timestamp")
                >= _timestamp(previous["filename_timestamp"], "filename_timestamp"),
                "receipt source-minute chronology moved backwards",
            )
        if body["snapshot_state"] != "absent":
            published = _timestamp(body["raw_published_at_utc"], "raw_published_at_utc")
            _require(
                last_snapshot_publication is None or published > last_snapshot_publication,
                "snapshot first-seen publication times must strictly increase",
            )
            last_snapshot_publication = published
        prior = previous_by_minute.get(body["filename_timestamp"])
        _require(
            body["attempt_number"] == (1 if prior is None else prior["attempt_number"] + 1),
            "receipt attempts must start at one and remain consecutive",
        )
        if prior is not None:
            status = prior["http_status"]
            _require(
                status == 429 or (isinstance(status, int) and 500 <= status <= 599),
                "a nonretryable response cannot have another attempt",
            )
            backoff = max(
                (5, 10, 20)[body["attempt_number"] - 2], prior["retry_after_seconds"] or 0
            )
            _require(
                _timestamp(body["requested_at_utc"], "requested_at_utc")
                >= _timestamp(prior["completed_at_utc"], "completed_at_utc")
                + timedelta(seconds=backoff),
                "receipt retry precedes its authenticated backoff",
            )
        previous_by_minute[body["filename_timestamp"]] = body
        verified.append(body)
        previous_hash = receipt_sha256(envelope)
    if expected_head is not None:
        _require(previous_hash == expected_head, "receipt chain does not reach its pinned head")
    return tuple(verified)


def verify_closeout(
    envelope: object,
    public_key: Ed25519PublicKey,
    receipts: object,
    *,
    expected_closeout_sha256: str,
    expected_inventory_sha256: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Verify externally pinned completion, inventory, chain and evidenced slots."""
    for field, value in (
        ("expected_closeout_sha256", expected_closeout_sha256),
        ("expected_inventory_sha256", expected_inventory_sha256),
        ("expected_plan_sha256", expected_plan_sha256),
    ):
        _hash(value, field)
    body = _verify(envelope, public_key, closeout=True)
    _require(closeout_sha256(envelope) == expected_closeout_sha256, "closeout pin mismatch")
    _require(
        body["payload_inventory_sha256"] == expected_inventory_sha256, "payload inventory mismatch"
    )
    _require(body["plan_sha256"] == expected_plan_sha256, "closeout retrieval plan mismatch")
    verified = verify_receipt_chain(receipts, public_key, body["final_receipt_sha256"])
    _require(len(verified) == body["receipt_count"], "closeout receipt count mismatch")
    for receipt in verified:
        _require(
            all(receipt[field] == body[field] for field in _CONTEXT_FIELDS),
            "closeout and receipt context differ",
        )
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for receipt in verified:
        latest[(receipt["interval_index"], receipt["filename_timestamp"])] = receipt
    for slot in body["slot_outcomes"]:
        evidence = [item for (index, _), item in latest.items() if index == slot["interval_index"]]
        successful = sum(item["snapshot_state"] == "complete" for item in evidence)
        if slot["outcome"] == "verified":
            _require(successful == 15, "verified slot lacks all 15 successful minute receipts")
        if slot["outcome"] == "provider_gap":
            _require(
                any(
                    item["snapshot_state"] == "invalid"
                    or (
                        item["http_status"] is not None
                        and item["http_status"] not in (200, 429)
                        and not 500 <= item["http_status"] <= 599
                    )
                    or (
                        item["attempt_number"] == 4
                        and (item["http_status"] == 429 or 500 <= (item["http_status"] or 0) <= 599)
                    )
                    for item in evidence
                ),
                "provider gap lacks a bound terminal attempt sequence",
            )
    return body
