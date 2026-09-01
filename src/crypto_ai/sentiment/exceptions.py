"""Phase 2-only exception refinements for sentiment integrity checks."""

from crypto_ai.exceptions import SentimentStorageError


class StorageIntegrityError(SentimentStorageError):
    """A previously inventoried immutable storage tree changed during verification."""
