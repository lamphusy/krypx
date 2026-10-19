"""Phase 2-only exception refinements for sentiment integrity checks."""

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


class NetworkStageError(NetworkSafetyError):
    """A transport failure carrying only an allowlisted, non-sensitive stage code."""

    def __init__(self, stage: str, message: str) -> None:
        if type(stage) is not str or stage not in NETWORK_FAILURE_STAGES:
            raise NetworkSafetyError("unsupported network failure stage")
        super().__init__(message)
        self.stage = stage


class ReceiptValidationError(SentimentError):
    """A signed receipt or closeout failed schema, signature or chain verification."""
