"""Phase 2-only exception refinements for sentiment integrity checks."""

from dataclasses import dataclass

from crypto_ai.exceptions import SentimentError, SentimentStorageError


class StorageIntegrityError(SentimentStorageError):
    """A previously inventoried immutable storage tree changed during verification."""


class NetworkSafetyError(SentimentError):
    """An offline-tested collector violated its fail-closed safety contract."""


NETWORK_FAILURE_STAGES = frozenset(
    {
        "preflight",
        "storage_capacity",
        "request_intent",
        "tls_context",
        "tls_verification",
        "request_dispatch",
        "response_headers",
        "response_body",
        "response_close",
        "response_validation",
        "snapshot_storage",
        "receipt_signing",
        "receipt_publication",
        "request_completion",
        "gap_publication",
        "coverage_gate",
        "retry_backoff",
    }
)

NETWORK_BODY_ERROR_CATEGORIES = frozenset(
    {"tls_eof", "connection_reset", "timeout", "length_mismatch", "stream_error"}
)


@dataclass(frozen=True)
class NetworkFailureDiagnostic:
    """Bounded local failure evidence; never contains provider text or headers."""

    http_status: int
    content_length: int | None
    bytes_received: int
    error_category: str

    def __post_init__(self) -> None:
        if type(self.http_status) is not int or not 100 <= self.http_status <= 599:
            raise NetworkSafetyError("invalid diagnostic HTTP status")
        if self.content_length is not None and (
            type(self.content_length) is not int or not 0 <= self.content_length <= 500_000_000
        ):
            raise NetworkSafetyError("invalid diagnostic Content-Length")
        if type(self.bytes_received) is not int or not 0 <= self.bytes_received <= 500_000_000:
            raise NetworkSafetyError("invalid diagnostic received byte count")
        if (
            type(self.error_category) is not str
            or self.error_category not in NETWORK_BODY_ERROR_CATEGORIES
        ):
            raise NetworkSafetyError("invalid diagnostic error category")

    def as_event(self) -> dict[str, int | str | None]:
        return {
            "http_status": self.http_status,
            "content_length": self.content_length,
            "bytes_received": self.bytes_received,
            "error_category": self.error_category,
        }


class NetworkStageError(NetworkSafetyError):
    """A transport failure carrying only an allowlisted, non-sensitive stage code."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        diagnostic: NetworkFailureDiagnostic | None = None,
    ) -> None:
        if type(stage) is not str or stage not in NETWORK_FAILURE_STAGES:
            raise NetworkSafetyError("unsupported network failure stage")
        if diagnostic is not None and (
            stage != "response_body" or type(diagnostic) is not NetworkFailureDiagnostic
        ):
            raise NetworkSafetyError("failure diagnostic requires a response-body stage")
        super().__init__(message)
        self.stage = stage
        self.diagnostic = diagnostic


class ReceiptValidationError(SentimentError):
    """A signed receipt or closeout failed schema, signature or chain verification."""
