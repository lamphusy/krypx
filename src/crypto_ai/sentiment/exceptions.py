"""Phase 2-only exception refinements for sentiment integrity checks."""

from crypto_ai.exceptions import SentimentError, SentimentStorageError


class StorageIntegrityError(SentimentStorageError):
    """A previously inventoried immutable storage tree changed during verification."""


class NetworkSafetyError(SentimentError):
    """An offline-tested collector violated its fail-closed safety contract."""


class ReceiptValidationError(SentimentError):
    """A signed receipt or closeout failed schema, signature or chain verification."""
