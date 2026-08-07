"""Machine-learning baseline factories."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_ai.config import settings


def create_logistic_pipeline(
    model_params: dict[str, object] | None = None,
) -> Pipeline:
    """Create a fold-local standardization and logistic-regression pipeline."""
    parameters = dict(settings.LOGISTIC_REGRESSION_PARAMS if model_params is None else model_params)
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**parameters)),
        ]
    )
