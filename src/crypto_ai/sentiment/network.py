"""Offline-tested GSG streaming client; no built-in live transport or launch path.

Only explicitly injected mock transports are accepted. The transport interface models
HTTPS without importing or creating a socket, HTTP session, credential or scheduler.
The four-attempt Batch B policy is deliberately separate from accepted Batch A state.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from crypto_ai.exceptions import CryptoAIError
from crypto_ai.sentiment.canonical import canonical_sha256, canonicalize
from crypto_ai.sentiment.contracts import format_utc_timestamp, parse_utc_timestamp
from crypto_ai.sentiment.exceptions import NetworkSafetyError
from crypto_ai.sentiment.network_budget import PilotBudget
from crypto_ai.sentiment.providers.gdelt_gsg import (
    MAX_COMPRESSED_BYTES,
    GapAttempt,
    GSGAdapter,
    RetrievalPlan,
    SnapshotResult,
    TerminalGapEvidence,
    expected_gsg_source_locator,
    plan_retrieval,
)
from crypto_ai.sentiment.receipts import (
    receipt_sha256,
    sign_closeout,
    sign_receipt,
    verify_closeout,
    verify_receipt_chain,
)
from crypto_ai.sentiment.storage import ContentAddressedStore

RETRY_POLICY_VERSION = "batch-b-gsg-retry-policy-v1"
GAP_EVIDENCE_VERSION = "batch-b-gsg-terminal-gap-evidence-v1"
HTTP_TIMEOUT_SECONDS = 10.0
SESSION_TIMEOUT_SECONDS = 900.0
MINIMUM_REQUEST_SPACING_SECONDS = 5.0
BACKOFF_SECONDS = (5.0, 10.0, 20.0)
MAXIMUM_ATTEMPTS = 4
CHUNK_BYTES = 64 * 1024
PUBLICATION_RESERVE_BYTES = 128 * 1024


class MockResponse(Protocol):
    """An injected fixture stream; ``read`` must honor its bound and timeout."""

    status: int
    headers: Mapping[str, str]
    url: str

    def read(self, maximum_bytes: int, *, timeout: float) -> bytes: ...

    def close(self) -> None: ...


class MockTransport(Protocol):
    """No default real implementation exists; production transport is forbidden."""

    mock_only: bool
    incremental_cost_usd: float

    def open(self, url: str, *, timeout: float) -> MockResponse: ...


@dataclass(frozen=True)
class RetrievalResult:
    snapshot: SnapshotResult | None
    receipts: tuple[dict[str, Any], ...]
    gap_evidence: TerminalGapEvidence | None


class RetrievalFailure(NetworkSafetyError):
    """A terminal client/payload error, with immutable facts retained for audit."""

    def __init__(self, result: RetrievalResult) -> None:
        super().__init__("GSG retrieval stopped on a terminal client or payload error")
        self.result = result


def _utc(value: str) -> datetime:
    try:
        instant = parse_utc_timestamp(value, field="collector timestamp")
        if format_utc_timestamp(instant) != value:
            raise ValueError("not canonical UTC")
        return instant
    except (CryptoAIError, TypeError, ValueError) as exc:
        raise NetworkSafetyError("invalid canonical UTC instant") from exc


def _transient(status: int | None) -> bool:
    return status == 429 or (status is not None and 500 <= status <= 599)


def _require_gzip_framing(snapshot: SnapshotResult, raw: bytes) -> SnapshotResult:
    # GzipFile accepts an empty input as an empty stream. An empty HTTP body has
    # no gzip member and is not proof of a complete, empty GSG minute file.
    if len(raw) < 18 or not raw.startswith(b"\x1f\x8b\x08"):
        return replace(
            snapshot, state="invalid", observations=(), json_line_count=0, error_code="invalid_gzip"
        )
    return snapshot


def build_terminal_gap_evidence(
    *,
    plan: RetrievalPlan,
    filename_timestamp: str,
    receipts: Sequence[dict[str, Any]],
    public_key: Ed25519PublicKey,
    protocol_sha256: str,
) -> TerminalGapEvidence:
    """Recompute a versioned Batch B terminal fact from authenticated actual attempts.

    The existing TerminalGapEvidence *shape* is reused, with a distinct version/policy.
    Batch A validators correctly reject this version; it must never enter state-v3 as v1.
    """
    if type(plan) is not RetrievalPlan:
        raise NetworkSafetyError("gap evidence requires a retrieval plan")
    bodies = verify_receipt_chain(receipts, public_key)
    if plan != plan_retrieval(plan.start_at, plan.end_at_exclusive):
        raise NetworkSafetyError("gap retrieval plan identity mismatch")
    intervals = {interval.filename_timestamp: interval for interval in plan.intervals}
    if filename_timestamp not in intervals:
        raise NetworkSafetyError("gap minute is outside the retrieval plan")
    selected = [body for body in bodies if body["filename_timestamp"] == filename_timestamp]
    if not selected or len(selected) > MAXIMUM_ATTEMPTS:
        raise NetworkSafetyError("terminal gap lacks bounded signed attempts")
    for index, body in enumerate(selected):
        if (
            body["plan_sha256"] != plan.plan_id
            or body["protocol_sha256"] != protocol_sha256
            or body["source_locator"] != expected_gsg_source_locator(intervals[filename_timestamp])
            or body["attempt_number"] != index + 1
            or _utc(body["requested_at_utc"]) < _utc(intervals[filename_timestamp].due_at)
        ):
            raise NetworkSafetyError("terminal gap attempt binding mismatch")
        if index:
            prior = selected[index - 1]
            delay = max(BACKOFF_SECONDS[index - 1], prior["retry_after_seconds"] or 0)
            if not _transient(prior["http_status"]) or _utc(body["requested_at_utc"]) < _utc(
                prior["completed_at_utc"]
            ) + timedelta(seconds=delay):
                raise NetworkSafetyError("terminal gap retries violate the Batch B policy")
    last = selected[-1]
    if last["snapshot_state"] == "complete" or last["http_status"] is None:
        raise NetworkSafetyError("a successful or unobserved retrieval is not a provider gap")
    if _transient(last["http_status"]) and len(selected) != MAXIMUM_ATTEMPTS:
        raise NetworkSafetyError("terminal retry exhaustion is not established")
    invalid = last["snapshot_state"] == "invalid"
    attempts = tuple(
        GapAttempt(
            attempt_number=body["attempt_number"],
            attempted_at=body["requested_at_utc"],
            http_status=None if body["snapshot_state"] == "invalid" else body["http_status"],
            error_kind="invalid_payload" if body["snapshot_state"] == "invalid" else None,
            retry_after_seconds=body["retry_after_seconds"],
            retry_disposition="retry" if index < len(selected) - 1 else "gap",
        )
        for index, body in enumerate(selected)
    )
    evidence = TerminalGapEvidence.create(
        interval_start=filename_timestamp,
        interval_end_exclusive=format_utc_timestamp(
            _utc(filename_timestamp) + timedelta(minutes=1)
        ),
        expected_source_locator=last["source_locator"],
        attempts=attempts,
        terminal_at=last["raw_published_at_utc"],
        protocol_config_sha256=protocol_sha256,
        retry_policy_version=RETRY_POLICY_VERSION,
        final_terminal_disposition=(
            "retry_exhausted" if _transient(last["http_status"]) else "non_retryable"
        ),
        observed_snapshot_id=last["snapshot_id"] if invalid else None,
        observed_raw_snapshot_sha256=last["raw_sha256"] if invalid else None,
    )
    evidence = replace(evidence, version=GAP_EVIDENCE_VERSION)
    identity = evidence.identity_payload()
    evidence_id = canonical_sha256(
        {"identity": identity, "identity_version": "batch-b-gsg-terminal-gap-identity-v1"}
    )
    return replace(
        evidence,
        evidence_id=evidence_id,
        evidence_sha256=canonical_sha256({**identity, "evidence_id": evidence_id}),
    )


def verify_terminal_gap_evidence(evidence: TerminalGapEvidence, **bindings: Any) -> None:
    """Reject forged counters, chronology, hashes, policy or receipt provenance."""
    expected = build_terminal_gap_evidence(**bindings)
    if type(evidence) is not TerminalGapEvidence or evidence != expected:
        raise NetworkSafetyError("terminal gap evidence differs from authenticated attempts")


def _gap_publication_name(pilot_id: str, evidence: TerminalGapEvidence) -> str:
    identity = {"pilot_id": pilot_id, "evidence_id": evidence.evidence_id}
    return f"gsg-network-gap-{canonical_sha256(identity)}"


def _load_verified_receipts(
    budget: PilotBudget, public_key: Ed25519PublicKey, context: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Verify closed-world receipt, CAS, parsed snapshot, gap and budget provenance."""
    if (
        context["pilot_id"] != budget.pilot_id
        or context["protocol_sha256"] != budget.protocol_sha256
    ):
        raise NetworkSafetyError("signed pilot context does not match the retained budget")
    store = budget.store
    prefix = f"gsg-network-{canonical_sha256(budget.pilot_id)}-"
    names = sorted(
        path.name
        for path in store.publications_root.iterdir()
        if path.name.startswith(f"{prefix}attempt-")
    )
    loaded = []
    for number, name in enumerate(names, start=1):
        if name != f"{prefix}attempt-{number:08d}":
            raise NetworkSafetyError("receipt publication sequence is incomplete")
        publication = store.read_publication(name)
        if set(publication.files) != {"signed-receipt.json"} or publication.manifest[
            "metadata"
        ] != {
            "provider": "gdelt_gsg",
            "pilot_id": budget.pilot_id,
            "sequence": number,
        }:
            raise NetworkSafetyError("receipt publication envelope mismatch")
        raw = publication.files["signed-receipt.json"]
        envelope = json.loads(raw)
        if canonicalize(envelope) != raw:
            raise NetworkSafetyError("receipt publication must be exact canonical bytes")
        loaded.append(envelope)
    bodies = verify_receipt_chain(loaded, public_key)
    attempts: dict[str, int] = {}
    final_by_minute = {}
    for body, envelope in zip(bodies, loaded, strict=True):
        if any(body[key] != value for key, value in context.items()):
            raise NetworkSafetyError("persisted receipt belongs to a different pilot context")
        attempts[body["filename_timestamp"]] = body["attempt_number"]
        final_by_minute[body["filename_timestamp"]] = (body, envelope)
        if body["raw_sha256"] is not None:
            raw = store.get_bytes(body["raw_sha256"])
            if len(raw) != body["bytes_received"]:
                raise NetworkSafetyError("persisted raw bytes disagree with signed receipt")
        if body["snapshot_id"] is not None:
            adapter = GSGAdapter(
                store, clock=lambda instant=body["plan_start_at_utc"]: _utc(instant)
            )
            receipt = adapter._load_existing_receipt(body["snapshot_id"])
            if (
                receipt is None
                or receipt.raw_snapshot_sha256 != body["raw_sha256"]
                or (receipt.raw_published_at != body["raw_published_at_utc"])
            ):
                raise NetworkSafetyError("snapshot receipt binding mismatch")
            parsed = _require_gzip_framing(adapter._parse_snapshot(receipt, raw), raw)
            if parsed.state != body["snapshot_state"]:
                raise NetworkSafetyError("signed snapshot state does not replay")
    if (
        attempts != budget.request_attempts
        or sum(body["bytes_received"] for body in bodies) != budget.total_download_bytes
    ):
        raise NetworkSafetyError("receipt suffix or raw byte count detached from durable budget")
    if bodies:
        start = bodies[0]["plan_start_at_utc"]
        plan = plan_retrieval(start, format_utc_timestamp(_utc(start) + timedelta(days=1)))
        for timestamp, (body, envelope) in final_by_minute.items():
            if body["snapshot_state"] == "complete":
                continue
            evidence = build_terminal_gap_evidence(
                plan=plan,
                filename_timestamp=timestamp,
                receipts=loaded,
                public_key=public_key,
                protocol_sha256=budget.protocol_sha256,
            )
            publication = store.read_publication(_gap_publication_name(budget.pilot_id, evidence))
            if publication.files != {
                "terminal-gap.json": evidence.canonical_bytes()
            } or publication.manifest["metadata"] != {
                "plan_sha256": plan.plan_id,
                "receipt_head": receipt_sha256(envelope),
            }:
                raise NetworkSafetyError("immutable terminal gap does not match signed attempts")
    return loaded


class GSGNetworkClient:
    """Session-scoped safety controls and signed provenance over mock-only streams."""

    def __init__(
        self,
        store: ContentAddressedStore,
        *,
        pilot_id: str,
        specification_id: str,
        protocol_sha256: str,
        code_commit: str,
        plan: RetrievalPlan,
        transport: MockTransport,
        private_key: Ed25519PrivateKey,
        public_key: Ed25519PublicKey,
        clock: Callable[[], datetime],
        monotonic: Callable[[], float],
        sleep: Callable[[float], None],
        create_pilot: bool = False,
        maximum_download_bytes: int = 500_000_000,
        maximum_storage_bytes: int = 2_000_000_000,
        real_network_calls_prohibited: bool = True,
    ) -> None:
        if real_network_calls_prohibited is not True:
            raise NetworkSafetyError("real network calls are prohibited")
        if (
            type(plan) is not RetrievalPlan
            or len(plan.intervals) != 1440
            or plan != plan_retrieval(plan.start_at, plan.end_at_exclusive)
        ):
            raise NetworkSafetyError("pilot requires an exact 24-hour, 1440-minute retrieval plan")
        if not isinstance(private_key, Ed25519PrivateKey) or not isinstance(
            public_key, Ed25519PublicKey
        ):
            raise NetworkSafetyError(
                "explicit in-memory signing and pinned verification keys required"
            )
        if private_key.public_key().public_bytes_raw() != public_key.public_bytes_raw():
            raise NetworkSafetyError("signing key does not match the pinned public key")
        for label, value, pattern in (
            ("pilot", pilot_id, r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"),
            ("specification", specification_id, r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"),
            ("protocol", protocol_sha256, r"[0-9a-f]{64}"),
            ("code commit", code_commit, r"[0-9a-f]{40}"),
        ):
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise NetworkSafetyError(f"invalid {label} identity")
        self.store = store
        self.plan = plan
        self.transport = transport
        self.private_key = private_key
        self.public_key = public_key
        self.clock = clock
        self.monotonic = monotonic
        self.sleep = sleep
        self.context = {
            "pilot_id": pilot_id,
            "specification_id": specification_id,
            "protocol_sha256": protocol_sha256,
            "code_commit": code_commit,
            "plan_sha256": plan.plan_id,
            "plan_start_at_utc": plan.start_at,
        }
        self.budget = PilotBudget(
            store,
            pilot_id,
            protocol_sha256,
            create=create_pilot,
            maximum_download_bytes=maximum_download_bytes,
            maximum_storage_bytes=maximum_storage_bytes,
        )
        self._prefix = f"gsg-network-{canonical_sha256(pilot_id)}-"
        self._receipts: list[dict[str, Any]] = []
        self._active = False
        self._closed_out = False
        self._retrieval_lock = Lock()
        self._last_monotonic: float | None = None
        self._last_request_monotonic: float | None = None
        self._started: float | None = None
        self._worker_end: datetime | None = None
        self._worker_index: int | None = None
        self._check_transport()

    def _check_transport(self) -> None:
        if getattr(self.transport, "mock_only", None) is not True:
            raise NetworkSafetyError(
                "only an explicitly injected offline mock transport is allowed"
            )
        cost = getattr(self.transport, "incremental_cost_usd", None)
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost != 0:
            raise NetworkSafetyError("positive or unknown incremental cost is prohibited")

    def _now(self) -> str:
        try:
            return format_utc_timestamp(self.clock())
        except (CryptoAIError, TypeError, ValueError, AttributeError) as exc:
            raise NetworkSafetyError(
                "collector clock must return a timezone-aware UTC instant"
            ) from exc

    def _tick(self) -> float:
        value = self.monotonic()
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise NetworkSafetyError("invalid monotonic clock")
        if self._last_monotonic is not None and value < self._last_monotonic:
            raise NetworkSafetyError("monotonic clock regression")
        self._last_monotonic = float(value)
        return float(value)

    def _remaining(self) -> float:
        if not self._active or self._closed_out or self._started is None:
            raise NetworkSafetyError("collector requires an open, unclosed mock session")
        remaining = SESSION_TIMEOUT_SECONDS - (self._tick() - self._started)
        if remaining <= 0:
            raise NetworkSafetyError("900-second monotonic session deadline exhausted")
        if self._worker_end is None:
            raise NetworkSafetyError("worker schedule is unavailable")
        calendar_remaining = (self._worker_end - _utc(self._now())).total_seconds()
        if calendar_remaining <= 0:
            raise NetworkSafetyError("fixed lag-adjusted worker deadline exhausted")
        return min(remaining, calendar_remaining)

    def _wait(self, duration: float) -> None:
        if duration <= 0:
            return
        if duration >= self._remaining():
            raise NetworkSafetyError("wait cannot fit the remaining session budget")
        before = self._tick()
        target = math.nextafter(before + duration, math.inf)
        self.sleep(duration)
        after = self._tick()
        if after < before + duration and after <= before:
            raise NetworkSafetyError("injected sleeper failed to honor the minimum delay")
        # Round toward a later start, never accept a tolerance that starts early.
        for _ in range(2):
            if after >= before + duration:
                break
            self.sleep(target - after)
            after = self._tick()
        if after < before + duration:
            raise NetworkSafetyError("injected sleeper failed to reach its bounded deadline")
        self._remaining()

    def __enter__(self) -> GSGNetworkClient:
        if self._active:
            raise NetworkSafetyError("collector session is already active")
        self._started = self._tick()
        self._budget_context = self.budget.locked()
        self._budget_context.__enter__()
        try:
            self._active = True
            first_start = _utc(self.plan.start_at) + timedelta(minutes=45)
            elapsed = (_utc(self._now()) - first_start).total_seconds()
            if not 0 <= elapsed < 86400:
                raise NetworkSafetyError("mock session is outside the frozen worker schedule")
            self._worker_index = int(elapsed // 900)
            self._worker_end = first_start + timedelta(seconds=(self._worker_index + 1) * 900)
            self._load_receipts()
            self._remaining()
            if self.budget.last_request_at_utc is not None:
                # UTC can jump between processes; a fresh monotonic clock cannot
                # prove how much time elapsed. Always wait a full five seconds.
                self._last_request_monotonic = self._tick()
            return self
        except Exception:
            self._active = False
            self._budget_context.__exit__(None, None, None)
            raise NetworkSafetyError("unable to hydrate a safe offline pilot session") from None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if exc_type is not None and not self._closed_out:
                self.budget.stop("session_failed")
        finally:
            self._active = False
            self._budget_context.__exit__(exc_type, exc, traceback)

    @property
    def receipts(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._receipts))

    def _load_receipts(self) -> None:
        if (self.store.publications_root / f"{self._prefix}closeout").exists():
            raise NetworkSafetyError("pilot already has an immutable closeout")
        self._receipts = _load_verified_receipts(self.budget, self.public_key, self.context)

    def retrieve(self, filename_timestamp: str) -> RetrievalResult:
        if not self._retrieval_lock.acquire(blocking=False):
            raise NetworkSafetyError("only one retrieval may be in flight")
        try:
            return self._retrieve(filename_timestamp)
        except NetworkSafetyError:
            if self._active and not self._closed_out:
                self.budget.stop("retrieval_failed")
            raise
        except Exception:
            if self._active and not self._closed_out:
                self.budget.stop("retrieval_failed")
            raise NetworkSafetyError(
                "collector failed closed; no provider diagnostic retained"
            ) from None
        finally:
            self._retrieval_lock.release()

    def _retrieve(self, filename_timestamp: str) -> RetrievalResult:
        self._remaining()
        self._check_transport()
        self._require_attainable_coverage()
        intervals = {item.filename_timestamp: item for item in self.plan.intervals}
        if filename_timestamp not in intervals:
            raise NetworkSafetyError("retrieval minute is outside the exact allowed plan")
        if (
            self._receipts
            and filename_timestamp <= self._receipts[-1]["body"]["filename_timestamp"]
        ):
            raise NetworkSafetyError(
                "completed minute identities cannot be replayed as new requests"
            )
        interval = intervals[filename_timestamp]
        if _utc(self._now()) < _utc(interval.due_at):
            raise NetworkSafetyError("GSG minute is not due under the 30-minute release allowance")
        url = expected_gsg_source_locator(interval)
        offset = int((_utc(filename_timestamp) - _utc(self.plan.start_at)).total_seconds() // 900)
        if offset != self._worker_index:
            raise NetworkSafetyError("minute is outside this fixed reporting worker slot")
        result_receipts = []
        snapshot = None
        for attempt in range(1, MAXIMUM_ATTEMPTS + 1):
            self._check_transport()
            self._pace()
            self.budget.assert_storage_capacity(PUBLICATION_RESERVE_BYTES)
            requested_at = self._now()
            self.budget.record_request(
                requested_at, filename_timestamp=filename_timestamp, attempt_number=attempt
            )
            self._last_request_monotonic = self._tick()
            timeout = min(HTTP_TIMEOUT_SECONDS, self._remaining())
            response = self.transport.open(url, timeout=timeout)
            try:
                self._check_transport()
                if self._tick() - self._last_request_monotonic > timeout:
                    raise NetworkSafetyError("HTTP open exceeded its timeout")
                self._remaining()
                if type(response.status) is not int or not 100 <= response.status <= 599:
                    raise NetworkSafetyError("invalid HTTP status")
                if response.url != url:
                    raise NetworkSafetyError(
                        "redirect or unexpected response locator is prohibited"
                    )
                status = response.status
                retry_after = self._retry_after(response.headers, status)
                body_bytes = self._read_body(response)
                completed_at = self._now()
            finally:
                response.close()
            self._remaining()
            self._check_transport()
            self.budget.assert_storage_capacity(2 * len(body_bytes) + PUBLICATION_RESERVE_BYTES)
            if status == 200:
                prior_times = [
                    item["body"]["raw_published_at_utc"]
                    for item in self._receipts
                    if item["body"]["raw_published_at_utc"] is not None
                ]
                if prior_times and _utc(self._now()) <= _utc(prior_times[-1]):
                    raise NetworkSafetyError(
                        "distinct snapshot publication times must strictly increase"
                    )
                snapshot = GSGAdapter(self.store, clock=self.clock).ingest_snapshot(
                    body_bytes,
                    filename_timestamp=filename_timestamp,
                    ingested_at=completed_at,
                    source_locator=url,
                    collection_mode="prospective",
                    input_class="synthetic_fixture",
                )
                snapshot = _require_gzip_framing(snapshot, body_bytes)
                raw_hash = snapshot.receipt.raw_snapshot_sha256
            else:
                raw_hash = self.store.put_bytes(body_bytes)
            raw_published_at = snapshot.receipt.raw_published_at if snapshot else self._now()
            body = {
                **self.context,
                "interval_index": offset,
                "filename_timestamp": filename_timestamp,
                "source_locator": url,
                "retry_policy_version": RETRY_POLICY_VERSION,
                "input_class": "synthetic_fixture",
                "real_network_calls_prohibited": True,
                "attempt_number": attempt,
                "http_status": status,
                "bytes_received": len(body_bytes),
                "raw_sha256": raw_hash,
                "snapshot_id": snapshot.receipt.snapshot_id if snapshot else None,
                "snapshot_state": snapshot.state if snapshot else "absent",
                "raw_published_at_utc": raw_published_at,
                "requested_at_utc": requested_at,
                "completed_at_utc": completed_at,
                "retry_after_seconds": retry_after,
                "previous_receipt_sha256": (
                    receipt_sha256(self._receipts[-1]) if self._receipts else None
                ),
            }
            envelope = sign_receipt(body, self.private_key)
            verify_receipt_chain([*self._receipts, envelope], self.public_key)
            self._remaining()
            encoded = canonicalize(envelope)
            self.budget.assert_storage_capacity(2 * len(encoded) + PUBLICATION_RESERVE_BYTES)
            number = len(self._receipts) + 1
            self.store.publish_bundle(
                f"{self._prefix}attempt-{number:08d}",
                {"signed-receipt.json": encoded},
                metadata={
                    "provider": "gdelt_gsg",
                    "pilot_id": self.context["pilot_id"],
                    "sequence": number,
                },
            )
            self._receipts.append(envelope)
            result_receipts.append(envelope)
            self.budget.finish_request()
            self._remaining()
            if snapshot is not None and snapshot.state == "complete":
                return RetrievalResult(snapshot, tuple(deepcopy(result_receipts)), None)
            if _transient(status) and attempt < MAXIMUM_ATTEMPTS:
                self._wait(max(BACKOFF_SECONDS[attempt - 1], retry_after or 0.0))
                continue
            evidence = build_terminal_gap_evidence(
                plan=self.plan,
                filename_timestamp=filename_timestamp,
                receipts=self._receipts,
                public_key=self.public_key,
                protocol_sha256=self.context["protocol_sha256"],
            )
            encoded = evidence.canonical_bytes()
            self.budget.assert_storage_capacity(2 * len(encoded) + PUBLICATION_RESERVE_BYTES)
            self.store.publish_bundle(
                self._gap_publication_id(evidence),
                {"terminal-gap.json": encoded},
                metadata={
                    "plan_sha256": self.plan.plan_id,
                    "receipt_head": receipt_sha256(envelope),
                },
            )
            self._remaining()
            result = RetrievalResult(snapshot, tuple(deepcopy(result_receipts)), evidence)
            if not _transient(status):
                raise RetrievalFailure(result)
            self._require_attainable_coverage()
            return result
        raise NetworkSafetyError("unreachable retry state")  # pragma: no cover

    def _gap_publication_id(self, evidence: TerminalGapEvidence) -> str:
        return _gap_publication_name(self.context["pilot_id"], evidence)

    def _require_attainable_coverage(self) -> None:
        """Five irrecoverable reporting slots make the fixed 92/96 gate impossible."""
        latest = {}
        for envelope in self._receipts:
            body = envelope["body"]
            latest[body["filename_timestamp"]] = body
        uncovered = 0
        for index in range(96):
            slot = [body for body in latest.values() if body["interval_index"] == index]
            complete = sum(body["snapshot_state"] == "complete" for body in slot)
            terminal_gap = any(
                body["snapshot_state"] == "invalid"
                or body["http_status"] is not None
                and body["http_status"] != 200
                and (not _transient(body["http_status"]) or body["attempt_number"] == 4)
                for body in slot
            )
            if terminal_gap or index < self._worker_index and complete < 15:
                uncovered += 1
        if uncovered >= 5:
            raise NetworkSafetyError("five reporting gaps or missed slots make 92/96 unattainable")

    def _pace(self) -> None:
        now = self._tick()
        delay = 0.0
        if self._last_request_monotonic is not None:
            delay = max(
                delay, MINIMUM_REQUEST_SPACING_SECONDS - (now - self._last_request_monotonic)
            )
        if self.budget.last_request_at_utc is not None:
            elapsed = (_utc(self._now()) - _utc(self.budget.last_request_at_utc)).total_seconds()
            if elapsed < 0:
                raise NetworkSafetyError("UTC request clock regression")
            delay = max(delay, MINIMUM_REQUEST_SPACING_SECONDS - elapsed)
        self._wait(delay)

    def _read_body(self, response: MockResponse) -> bytes:
        chunks = []
        total = 0
        while True:
            timeout = min(HTTP_TIMEOUT_SECONDS, self._remaining())
            before = self._tick()
            chunk = response.read(CHUNK_BYTES, timeout=timeout)
            if type(chunk) is not bytes:
                raise NetworkSafetyError("stream must return exact byte strings")
            if chunk:
                self.budget.record_received(len(chunk))
            self._check_transport()
            if self._tick() - before > timeout:
                raise NetworkSafetyError("HTTP read exceeded its timeout")
            self._remaining()
            if len(chunk) > CHUNK_BYTES:
                raise NetworkSafetyError("transport exceeded the bounded read size")
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_COMPRESSED_BYTES:
                raise NetworkSafetyError("response exceeds the frozen compressed parser cap")
            self.budget.assert_storage_capacity(2 * total + PUBLICATION_RESERVE_BYTES)
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _retry_after(headers: Mapping[str, str], status: int) -> float | None:
        if not isinstance(headers, Mapping) or any(
            type(k) is not str or type(v) is not str for k, v in headers.items()
        ):
            raise NetworkSafetyError("malformed response headers")
        lowered = {}
        for key, value in headers.items():
            if key.lower() in lowered:
                raise NetworkSafetyError("duplicate response header identity")
            lowered[key.lower()] = value
        if lowered.get("content-encoding", "identity").lower() != "identity":
            raise NetworkSafetyError(
                "unexpected content encoding would change gzip raw-byte identity"
            )
        if "retry-after" not in lowered:
            return None
        value = lowered["retry-after"]
        if status not in {429, 503} or not re.fullmatch(r"[0-9]{1,3}", value) or int(value) > 900:
            raise NetworkSafetyError("unsupported or malformed Retry-After")
        return float(value)

    def closeout(self) -> dict[str, Any]:
        """Sign an inventory-bound 96-slot closeout; this does not grant pilot acceptance.

        Caller must independently preserve the returned closeout hash and inventory hash.
        Missing minutes stay uncovered. No claimed state-v3 normalization is performed here.
        """
        if not self._retrieval_lock.acquire(blocking=False):
            raise NetworkSafetyError("closeout cannot overlap an active retrieval")
        try:
            return self._closeout()
        except NetworkSafetyError:
            if self._active and not self._closed_out:
                self.budget.stop("closeout_failed")
            raise
        except Exception:
            if self._active and not self._closed_out:
                self.budget.stop("closeout_failed")
            raise NetworkSafetyError("closeout verification or publication failed") from None
        finally:
            self._retrieval_lock.release()

    def _closeout(self) -> dict[str, Any]:
        self._remaining()
        self._load_receipts()
        bodies = verify_receipt_chain(self._receipts, self.public_key)
        outcomes = []
        for index in range(96):
            slot = [body for body in bodies if body["interval_index"] == index]
            complete = {
                body["filename_timestamp"] for body in slot if body["snapshot_state"] == "complete"
            }
            outcome = "verified" if len(complete) == 15 else "uncovered"
            if outcome != "verified" and any(
                body["snapshot_state"] == "invalid"
                or body["http_status"] is not None
                and body["http_status"] != 200
                and (not _transient(body["http_status"]) or body["attempt_number"] == 4)
                for body in slot
            ):
                outcome = "provider_gap"
            outcomes.append({"interval_index": index, "outcome": outcome})
        inventory = self.budget.payload_inventory()
        inventory_hash = canonical_sha256(inventory)
        closeout_body = {
            **{key: value for key, value in self.context.items() if key != "plan_start_at_utc"},
            "final_receipt_sha256": receipt_sha256(self._receipts[-1]) if self._receipts else None,
            "receipt_count": len(self._receipts),
            "slot_outcomes": outcomes,
            "payload_inventory_sha256": inventory_hash,
        }
        envelope = sign_closeout(closeout_body, self.private_key)
        closeout_hash = canonical_sha256(envelope)
        verify_closeout(
            envelope,
            self.public_key,
            self._receipts,
            expected_closeout_sha256=closeout_hash,
            expected_inventory_sha256=inventory_hash,
            expected_plan_sha256=self.plan.plan_id,
        )
        encoded = canonicalize(envelope)
        inventory_bytes = canonicalize(inventory)
        self.budget.assert_storage_capacity(
            2 * (len(encoded) + len(inventory_bytes)) + PUBLICATION_RESERVE_BYTES
        )
        self._remaining()
        self.store.publish_bundle(
            f"{self._prefix}closeout",
            {"signed-closeout.json": encoded, "payload-inventory.json": inventory_bytes},
            metadata={"closeout_sha256": closeout_hash, "inventory_sha256": inventory_hash},
        )
        self._remaining()
        self._closed_out = True
        return deepcopy(envelope)


def verify_retained_closeout(
    budget: PilotBudget,
    *,
    public_key: Ed25519PublicKey,
    receipts: Sequence[dict[str, Any]],
    expected_closeout_sha256: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Re-read the complete retained inventory against an independent closeout pin.

    Call while holding the budget lock. The inventory itself and signed closeout are
    excluded from their own inventory to avoid self-reference; all other bytes count.
    """
    name = f"gsg-network-{canonical_sha256(budget.pilot_id)}-closeout"
    publication = budget.store.read_publication(name)
    if set(publication.files) != {"signed-closeout.json", "payload-inventory.json"}:
        raise NetworkSafetyError("unexpected closeout publication payload")
    try:
        envelope = json.loads(publication.files["signed-closeout.json"])
        inventory = json.loads(publication.files["payload-inventory.json"])
        if canonicalize(envelope) != publication.files["signed-closeout.json"] or (
            canonicalize(inventory) != publication.files["payload-inventory.json"]
        ):
            raise NetworkSafetyError("closeout payloads are not canonical")
        actual = budget.payload_inventory()
        for filename in ("manifest.json", "signed-closeout.json", "payload-inventory.json"):
            actual.pop(f"publications/{name}/{filename}")
        if actual != inventory:
            raise NetworkSafetyError("retained payload inventory differs from signed closeout")
        inventory_hash = canonical_sha256(actual)
        if publication.manifest["metadata"] != {
            "closeout_sha256": expected_closeout_sha256,
            "inventory_sha256": inventory_hash,
        }:
            raise NetworkSafetyError("closeout manifest metadata differs from its independent pin")
        verified = verify_closeout(
            envelope,
            public_key,
            receipts,
            expected_closeout_sha256=expected_closeout_sha256,
            expected_inventory_sha256=inventory_hash,
            expected_plan_sha256=expected_plan_sha256,
        )
        context = {
            key: verified[key]
            for key in (
                "pilot_id",
                "specification_id",
                "protocol_sha256",
                "code_commit",
                "plan_sha256",
            )
        }
        retained = _load_verified_receipts(budget, public_key, context)
        if canonicalize(retained) != canonicalize(receipts):
            raise NetworkSafetyError("closeout receipt chain is not the retained receipt chain")
        return verified
    except (ValueError, TypeError, KeyError) as exc:
        raise NetworkSafetyError("malformed retained closeout") from exc
