"""Project-specific exception hierarchy."""


class CryptoAIError(Exception):
    """Base project exception."""


class MarketDataError(CryptoAIError):
    """Base market-data exception."""


class MarketDataNetworkError(MarketDataError):
    """Network request failed after retries."""


class MarketDataExchangeError(MarketDataError):
    """Exchange rejected the market-data request."""


class MarketDataValidationError(MarketDataError):
    """Market data violated required invariants."""


class FeatureEngineeringError(CryptoAIError):
    """Feature computation failed."""


class LabelGenerationError(CryptoAIError):
    """Label generation failed."""


class DatasetSplitError(CryptoAIError):
    """Chronological split construction failed."""


class ModelTrainingError(CryptoAIError):
    """Model training failed."""


class BacktestError(CryptoAIError):
    """Backtest execution failed."""


class ArtifactError(CryptoAIError):
    """Artifact loading or saving failed."""


class SentimentError(CryptoAIError):
    """Base error for the offline Phase 2 sentiment foundation."""


class CanonicalizationError(SentimentError):
    """A value cannot be represented by the frozen canonical JSON contract."""


class ArticleValidationError(SentimentError):
    """An article envelope violates the frozen Phase 2 contract."""


class ScoreValidationError(SentimentError):
    """A score envelope violates the frozen Phase 2 contract."""


class SentimentStorageError(SentimentError):
    """Immutable sentiment storage failed an integrity check or write."""


class PublicationCollisionError(SentimentStorageError):
    """An immutable publication identifier already exists."""


class ProviderIngestionError(SentimentError):
    """A provider fixture or ingestion envelope violates the frozen adapter contract."""
