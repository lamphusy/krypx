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
