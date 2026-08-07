"""Preparation and atomic storage of inference and labeled datasets."""

import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from crypto_ai.config import settings
from crypto_ai.costs import minimum_gross_return_for_net_edge
from crypto_ai.data.storage import (
    get_raw_data_path,
    load_ohlcv_csv,
    sha256_file,
    symbol_to_slug,
)
from crypto_ai.exceptions import FeatureEngineeringError
from crypto_ai.features.build import compute_features, get_expected_feature_columns
from crypto_ai.features.labels import add_labels

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedDatasetResult:
    """Prepared inference/training frames and their exact raw-data provenance."""

    features: pd.DataFrame
    labeled: pd.DataFrame
    feature_path: Path
    labeled_path: Path
    source_snapshot_path: Path
    source_sha256: str
    feature_columns: tuple[str, ...]
    warmup_rows_removed: int
    unlabeled_rows_removed: int
    minimum_required_return: float


def load_feature_dataset(path: Path) -> pd.DataFrame:
    """Load the exact persisted inference-ready feature schema."""
    try:
        result = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise FeatureEngineeringError(f"Unable to load feature dataset {path}: {exc}") from exc
    feature_columns = get_expected_feature_columns()
    expected = [*settings.RAW_COLUMNS, *feature_columns]
    if result.columns.tolist() != expected:
        raise FeatureEngineeringError("Persisted feature columns do not match expected schema")
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
        for column in expected[1:]:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureEngineeringError(
            f"Persisted feature dataset contains invalid values: {exc}"
        ) from exc
    if result.empty or not np.isfinite(result[expected[1:]].to_numpy()).all():
        raise FeatureEngineeringError("Persisted feature dataset is empty or incomplete")
    return result


def load_labeled_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the exact persisted labeled-dataset schema."""
    try:
        result = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise FeatureEngineeringError(f"Unable to load labeled dataset {path}: {exc}") from exc

    feature_columns = get_expected_feature_columns()
    expected_columns = [*settings.RAW_COLUMNS, *feature_columns, *settings.LABEL_COLUMNS]
    if result.columns.tolist() != expected_columns:
        raise FeatureEngineeringError(
            "Persisted labeled columns do not match the ordered expected schema"
        )

    try:
        for column in ("timestamp", "entry_timestamp", "exit_timestamp"):
            result[column] = pd.to_datetime(result[column], utc=True, errors="raise")
        numeric_columns = [
            *settings.RAW_COLUMNS[1:],
            *feature_columns,
            "entry_open",
            "exit_open",
            "gross_forward_return",
        ]
        for column in numeric_columns:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
        labels = pd.to_numeric(result["label"], errors="raise")
    except (TypeError, ValueError, OverflowError) as exc:
        raise FeatureEngineeringError(
            f"Persisted labeled dataset {path} contains invalid values: {exc}"
        ) from exc

    if result.empty:
        raise FeatureEngineeringError(f"Persisted labeled dataset {path} is empty")
    if not labels.isin([0, 1]).all():
        raise FeatureEngineeringError("Persisted labels must contain only 0 and 1")
    result["label"] = labels.astype("int8")
    if not np.isfinite(result[numeric_columns].to_numpy(dtype="float64")).all():
        raise FeatureEngineeringError(
            "Persisted labeled dataset contains missing or infinite values"
        )
    if result["timestamp"].duplicated().any() or not result["timestamp"].is_monotonic_increasing:
        raise FeatureEngineeringError(
            "Persisted labeled decisions must be unique and chronological"
        )
    if (result["entry_timestamp"] <= result["timestamp"]).any():
        raise FeatureEngineeringError("Every entry_timestamp must follow its decision timestamp")
    if (result["exit_timestamp"] <= result["entry_timestamp"]).any():
        raise FeatureEngineeringError("Every exit_timestamp must follow its entry_timestamp")
    return result


def _atomic_write_dataframe(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        df.to_csv(
            temporary_path,
            index=False,
            date_format="%Y-%m-%dT%H:%M:%S.%fZ",
            float_format="%.17g",
            lineterminator="\n",
        )
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def prepare_datasets(
    symbol: str,
    timeframe: str,
    *,
    current_utc_time: datetime | pd.Timestamp | None = None,
    raw_dir: Path | None = None,
    snapshots_dir: Path | None = None,
    interim_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> PreparedDatasetResult:
    """Build and atomically store inference-ready and labeled datasets.

    The mutable latest raw file is used only to identify its content hash. Feature
    computation loads the corresponding immutable snapshot and verifies its exact
    bytes before doing any work.
    """
    raw_root = raw_dir if raw_dir is not None else settings.DATA_RAW_DIR
    if snapshots_dir is not None:
        snapshot_root = snapshots_dir
    elif raw_dir is not None:
        snapshot_root = raw_root / "snapshots"
    else:
        snapshot_root = settings.DATA_RAW_SNAPSHOTS_DIR
    interim_root = interim_dir if interim_dir is not None else settings.DATA_INTERIM_DIR
    processed_root = processed_dir if processed_dir is not None else settings.DATA_PROCESSED_DIR
    latest_path = get_raw_data_path(symbol, timeframe, raw_root)
    if not latest_path.exists():
        raise FeatureEngineeringError(
            f"Latest raw data does not exist at {latest_path}; run the fetch command first"
        )

    source_sha256 = sha256_file(latest_path)
    dataset_slug = f"{symbol_to_slug(symbol)}_{timeframe}"
    source_snapshot_path = snapshot_root / dataset_slug / f"{source_sha256}.csv"
    if not source_snapshot_path.exists():
        raise FeatureEngineeringError(
            f"Immutable raw snapshot for latest data is missing: {source_snapshot_path}"
        )
    snapshot_sha256 = sha256_file(source_snapshot_path)
    if snapshot_sha256 != source_sha256:
        raise FeatureEngineeringError(
            f"Raw snapshot hash mismatch at {source_snapshot_path}: {snapshot_sha256}"
        )

    raw_data = load_ohlcv_csv(
        source_snapshot_path,
        timeframe=timeframe,
        current_utc_time=current_utc_time,
    )
    features = compute_features(raw_data)
    minimum_required_return = minimum_gross_return_for_net_edge(
        fee_rate=settings.TAKER_FEE_RATE,
        slippage_bps_per_side=settings.SLIPPAGE_BPS_PER_SIDE,
        half_spread_bps_per_side=settings.HALF_SPREAD_BPS_PER_SIDE,
        minimum_net_edge_bps=settings.MIN_EDGE_BPS,
    )
    labeled = add_labels(
        features,
        horizon=settings.PREDICTION_HORIZON,
        minimum_required_return=minimum_required_return,
    )

    feature_path = interim_root / f"{dataset_slug}_features.csv"
    labeled_path = processed_root / f"{dataset_slug}_labeled.csv"
    _atomic_write_dataframe(features, feature_path)
    _atomic_write_dataframe(labeled, labeled_path)

    feature_columns = tuple(get_expected_feature_columns())
    result = PreparedDatasetResult(
        features=features,
        labeled=labeled,
        feature_path=feature_path,
        labeled_path=labeled_path,
        source_snapshot_path=source_snapshot_path,
        source_sha256=source_sha256,
        feature_columns=feature_columns,
        warmup_rows_removed=len(raw_data) - len(features),
        unlabeled_rows_removed=len(features) - len(labeled),
        minimum_required_return=minimum_required_return,
    )
    logger.info(
        "Prepared %s inference rows and %s labeled rows from raw snapshot %s",
        len(features),
        len(labeled),
        source_sha256,
    )
    return result
