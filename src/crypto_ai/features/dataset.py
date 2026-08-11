"""Preparation, provenance, and verified loading of prepared dataset bundles."""

import hashlib
import io
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai.config import settings
from crypto_ai.costs import minimum_gross_return_for_net_edge
from crypto_ai.data.storage import (
    get_raw_data_path,
    load_ohlcv_csv_bytes,
    sha256_file,
    symbol_to_slug,
)
from crypto_ai.exceptions import FeatureEngineeringError, MarketDataValidationError
from crypto_ai.features.build import compute_features, get_expected_feature_columns
from crypto_ai.features.labels import add_labels

logger = logging.getLogger(__name__)

_DATASET_MANIFEST_VERSION = 1
_LABEL_DEFINITION = "gross open[t+H+1] / open[t+1] - 1 > minimum_required_return"


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
    manifest_path: Path | None = None
    feature_sha256: str = ""
    labeled_sha256: str = ""


@dataclass(frozen=True)
class PreparedDatasetBundle:
    """A fully verified prepared dataset and its immutable provenance.

    ``manifest_bytes`` and ``manifest_sha256`` identify the exact manifest
    content that was parsed to produce ``manifest``.  Downstream artifact
    publication can therefore use the captured bytes instead of re-reading the
    mutable completion-marker path after validation.
    """

    features: pd.DataFrame
    labeled: pd.DataFrame
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_bytes: bytes
    manifest_sha256: str
    source_snapshot_path: Path
    source_snapshot_sha256: str
    feature_path: Path
    feature_sha256: str
    labeled_path: Path
    labeled_sha256: str
    feature_columns: tuple[str, ...]


def get_feature_dataset_path(
    symbol: str,
    timeframe: str,
    interim_dir: Path | None = None,
) -> Path:
    """Return the canonical inference-ready feature dataset path."""
    root = interim_dir if interim_dir is not None else settings.DATA_INTERIM_DIR
    return root / f"{symbol_to_slug(symbol)}_{timeframe}_features.csv"


def get_labeled_dataset_path(
    symbol: str,
    timeframe: str,
    processed_dir: Path | None = None,
) -> Path:
    """Return the canonical labeled model dataset path."""
    root = processed_dir if processed_dir is not None else settings.DATA_PROCESSED_DIR
    return root / f"{symbol_to_slug(symbol)}_{timeframe}_labeled.csv"


def get_prepared_dataset_manifest_path(
    symbol: str,
    timeframe: str,
    processed_dir: Path | None = None,
) -> Path:
    """Return the completion-marker path for a prepared dataset bundle."""
    root = processed_dir if processed_dir is not None else settings.DATA_PROCESSED_DIR
    return root / f"{symbol_to_slug(symbol)}_{timeframe}_dataset_manifest.json"


def _load_feature_dataset_bytes(content: bytes, description: str) -> pd.DataFrame:
    """Parse and validate one already captured inference-ready CSV byte sequence."""
    try:
        result = pd.read_csv(io.BytesIO(content))
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise FeatureEngineeringError(
            f"Unable to load feature dataset {description}: {exc}"
        ) from exc
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
    if result["timestamp"].duplicated().any() or not result["timestamp"].is_monotonic_increasing:
        raise FeatureEngineeringError(
            "Persisted feature decisions must be unique and chronological"
        )
    return result


def load_feature_dataset(path: Path) -> pd.DataFrame:
    """Load the exact persisted inference-ready feature schema."""
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise FeatureEngineeringError(f"Unable to load feature dataset {path}: {exc}") from exc
    return _load_feature_dataset_bytes(content, str(path))


def _load_labeled_dataset_bytes(content: bytes, description: str) -> pd.DataFrame:
    """Parse and validate one already captured labeled CSV byte sequence."""
    try:
        result = pd.read_csv(io.BytesIO(content))
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise FeatureEngineeringError(
            f"Unable to load labeled dataset {description}: {exc}"
        ) from exc

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
            f"Persisted labeled dataset {description} contains invalid values: {exc}"
        ) from exc

    if result.empty:
        raise FeatureEngineeringError(f"Persisted labeled dataset {description} is empty")
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


def load_labeled_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the exact persisted labeled-dataset schema."""
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise FeatureEngineeringError(f"Unable to load labeled dataset {path}: {exc}") from exc
    return _load_labeled_dataset_bytes(content, str(path))


def _feature_configuration() -> dict[str, Any]:
    """Return every setting that affects the persisted feature matrix."""
    return {
        "ema_short": settings.EMA_SHORT,
        "ema_long": settings.EMA_LONG,
        "macd_fast": settings.MACD_FAST,
        "macd_slow": settings.MACD_SLOW,
        "macd_signal": settings.MACD_SIGNAL,
        "rsi_period": settings.RSI_PERIOD,
        "stoch_rsi_period": settings.STOCH_RSI_PERIOD,
        "bb_period": settings.BB_PERIOD,
        "bb_std_dev": settings.BB_STD_DEV,
        "atr_period": settings.ATR_PERIOD,
        "volume_ma_period": settings.VOLUME_MA_PERIOD,
        "return_periods": list(settings.RETURN_PERIODS),
    }


def _feature_schema_hash(feature_columns: list[str] | tuple[str, ...]) -> str:
    payload = json.dumps(list(feature_columns), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fee_assumptions() -> dict[str, Any]:
    return {
        "taker_fee_rate": settings.TAKER_FEE_RATE,
        "charged_on_entry": True,
        "charged_on_exit": True,
    }


def _slippage_assumptions() -> dict[str, Any]:
    return {
        "slippage_bps_per_side": settings.SLIPPAGE_BPS_PER_SIDE,
        "adverse_on_entry_and_exit": True,
    }


def _spread_assumptions() -> dict[str, Any]:
    return {
        "half_spread_bps_per_side": settings.HALF_SPREAD_BPS_PER_SIDE,
        "adverse_on_entry_and_exit": True,
    }


def _utc_iso(value: datetime | pd.Timestamp) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise FeatureEngineeringError("Dataset provenance timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, sort_keys=True, allow_nan=False)
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes, str]:
    """Parse and fingerprint one exact read of a prepared manifest."""
    try:
        manifest_bytes = path.read_bytes()
        raw_payload = manifest_bytes.decode("utf-8")

        def reject_constant(value: str) -> None:
            raise ValueError(f"invalid JSON constant {value}")

        payload = json.loads(raw_payload, parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise FeatureEngineeringError(
            f"Unable to load prepared dataset manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FeatureEngineeringError(f"Prepared dataset manifest {path} must be a JSON object")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    return payload, manifest_bytes, manifest_sha256


def _manifest_absolute_path(manifest: dict[str, Any], key: str) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise FeatureEngineeringError(f"Prepared dataset manifest has invalid {key}")
    path = Path(value)
    if not path.is_absolute():
        raise FeatureEngineeringError(f"Prepared dataset manifest {key} must be absolute")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FeatureEngineeringError(
            f"Unable to resolve prepared dataset manifest {key} path {path}: {exc}"
        ) from exc


def _resolved_requested_path(path: Path, description: str) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FeatureEngineeringError(
            f"Unable to resolve requested prepared dataset {description} path {path}: {exc}"
        ) from exc


def _read_verified_bytes(path: Path, expected_sha256: object, description: str) -> bytes:
    """Read, hash, and return one exact bundle-member byte sequence."""
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise FeatureEngineeringError(
            f"Prepared dataset manifest has invalid {description} SHA-256"
        )
    if not path.is_file():
        raise FeatureEngineeringError(f"Prepared dataset {description} is missing: {path}")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise FeatureEngineeringError(
            f"Unable to read prepared dataset {description} {path}: {exc}"
        ) from exc
    actual_sha256 = hashlib.sha256(content).hexdigest()
    if actual_sha256 != expected_sha256:
        raise FeatureEngineeringError(
            f"Prepared dataset {description} hash mismatch at {path}: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return content


def _validate_manifest_configuration(
    manifest: dict[str, Any],
    symbol: str,
    timeframe: str,
    expected_feature_path: Path,
    expected_labeled_path: Path,
) -> tuple[Path, Path, Path, tuple[str, ...]]:
    required_fields = {
        "dataset_manifest_version",
        "created_at_utc",
        "symbol",
        "timeframe",
        "source_snapshot_path",
        "source_snapshot_sha256",
        "feature_file_path",
        "feature_file_sha256",
        "labeled_file_path",
        "labeled_file_sha256",
        "feature_columns",
        "feature_schema_hash",
        "feature_configuration",
        "prediction_horizon",
        "label_lookahead_rows",
        "minimum_required_return",
        "minimum_net_edge_bps",
        "label_definition",
        "fee_assumptions",
        "slippage_assumptions",
        "spread_assumptions",
        "warmup_rows_removed",
        "unlabeled_rows_removed",
        "feature_row_count",
        "labeled_row_count",
        "feature_start_timestamp",
        "feature_end_timestamp",
        "labeled_start_timestamp",
        "labeled_end_timestamp",
    }
    missing = sorted(required_fields.difference(manifest))
    if missing:
        raise FeatureEngineeringError(
            f"Prepared dataset manifest is incomplete; missing fields: {missing}"
        )
    if manifest["dataset_manifest_version"] != _DATASET_MANIFEST_VERSION:
        raise FeatureEngineeringError("Prepared dataset manifest version is unsupported")
    if manifest["symbol"] != symbol or manifest["timeframe"] != timeframe:
        raise FeatureEngineeringError(
            "Prepared dataset manifest symbol or timeframe does not match the requested bundle"
        )
    try:
        created_at = pd.Timestamp(manifest["created_at_utc"])
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(
            "Prepared dataset manifest created_at_utc is invalid"
        ) from exc
    if created_at.tzinfo is None:
        raise FeatureEngineeringError(
            "Prepared dataset manifest created_at_utc must be timezone-aware"
        )

    feature_path = _manifest_absolute_path(manifest, "feature_file_path")
    labeled_path = _manifest_absolute_path(manifest, "labeled_file_path")
    source_snapshot_path = _manifest_absolute_path(manifest, "source_snapshot_path")
    if feature_path != _resolved_requested_path(expected_feature_path, "feature"):
        raise FeatureEngineeringError(
            "Prepared dataset manifest feature path does not match the requested bundle"
        )
    if labeled_path != _resolved_requested_path(expected_labeled_path, "labeled"):
        raise FeatureEngineeringError(
            "Prepared dataset manifest labeled path does not match the requested bundle"
        )

    expected_feature_columns = tuple(get_expected_feature_columns())
    manifest_feature_columns = manifest["feature_columns"]
    if (
        not isinstance(manifest_feature_columns, list)
        or tuple(manifest_feature_columns) != expected_feature_columns
        or manifest["feature_schema_hash"] != _feature_schema_hash(expected_feature_columns)
    ):
        raise FeatureEngineeringError(
            "Prepared dataset manifest feature schema does not match current configuration"
        )
    if manifest["feature_configuration"] != _feature_configuration():
        raise FeatureEngineeringError(
            "Prepared dataset feature configuration does not match current configuration"
        )

    expected_minimum_return = minimum_gross_return_for_net_edge(
        fee_rate=settings.TAKER_FEE_RATE,
        slippage_bps_per_side=settings.SLIPPAGE_BPS_PER_SIDE,
        half_spread_bps_per_side=settings.HALF_SPREAD_BPS_PER_SIDE,
        minimum_net_edge_bps=settings.MIN_EDGE_BPS,
    )
    minimum_return = manifest["minimum_required_return"]
    if (
        not isinstance(minimum_return, (int, float))
        or isinstance(minimum_return, bool)
        or not math.isfinite(minimum_return)
        or not math.isclose(
            float(minimum_return), expected_minimum_return, rel_tol=1e-15, abs_tol=0.0
        )
        or manifest["prediction_horizon"] != settings.PREDICTION_HORIZON
        or manifest["label_lookahead_rows"] != settings.LABEL_LOOKAHEAD_ROWS
        or settings.LABEL_LOOKAHEAD_ROWS != settings.PREDICTION_HORIZON + 1
        or manifest["minimum_net_edge_bps"] != settings.MIN_EDGE_BPS
        or manifest["label_definition"] != _LABEL_DEFINITION
        or manifest["fee_assumptions"] != _fee_assumptions()
        or manifest["slippage_assumptions"] != _slippage_assumptions()
        or manifest["spread_assumptions"] != _spread_assumptions()
    ):
        raise FeatureEngineeringError(
            "Prepared dataset label or execution-cost configuration does not match current "
            "configuration"
        )

    for field in (
        "warmup_rows_removed",
        "unlabeled_rows_removed",
        "feature_row_count",
        "labeled_row_count",
    ):
        value = manifest[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise FeatureEngineeringError(f"Prepared dataset manifest has invalid {field}")
    return source_snapshot_path, feature_path, labeled_path, expected_feature_columns


def _manifest_timestamp(manifest: dict[str, Any], key: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(manifest[key])
    except (TypeError, ValueError) as exc:
        raise FeatureEngineeringError(f"Prepared dataset manifest has invalid {key}") from exc
    if timestamp.tzinfo is None:
        raise FeatureEngineeringError(f"Prepared dataset manifest {key} must include UTC")
    return timestamp.tz_convert("UTC")


def _validate_logical_bundle(
    raw_data: pd.DataFrame,
    features: pd.DataFrame,
    labeled: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    feature_columns = get_expected_feature_columns()
    horizon = settings.PREDICTION_HORIZON
    if manifest["feature_row_count"] != len(features):
        raise FeatureEngineeringError("Prepared dataset feature row count does not match manifest")
    if manifest["labeled_row_count"] != len(labeled):
        raise FeatureEngineeringError("Prepared dataset labeled row count does not match manifest")
    if manifest["warmup_rows_removed"] != len(raw_data) - len(features):
        raise FeatureEngineeringError("Prepared dataset warm-up row count does not reconcile")
    if manifest["unlabeled_rows_removed"] != len(features) - len(labeled):
        raise FeatureEngineeringError("Prepared dataset unlabeled row count does not reconcile")
    if len(features) - len(labeled) != settings.LABEL_LOOKAHEAD_ROWS:
        raise FeatureEngineeringError(
            "Prepared feature and labeled files do not have the configured label lookahead"
        )

    expected_timestamps = {
        "feature_start_timestamp": features["timestamp"].iloc[0],
        "feature_end_timestamp": features["timestamp"].iloc[-1],
        "labeled_start_timestamp": labeled["timestamp"].iloc[0],
        "labeled_end_timestamp": labeled["timestamp"].iloc[-1],
    }
    for key, value in expected_timestamps.items():
        if _manifest_timestamp(manifest, key) != value:
            raise FeatureEngineeringError(
                f"Prepared dataset {key} does not match the persisted data"
            )

    warmup_rows = manifest["warmup_rows_removed"]
    raw_tail = raw_data.iloc[warmup_rows:].reset_index(drop=True)
    feature_market = features[settings.RAW_COLUMNS].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            raw_tail[settings.RAW_COLUMNS],
            feature_market,
            check_exact=False,
            rtol=1e-13,
            atol=1e-13,
        )
        pd.testing.assert_frame_equal(
            features.iloc[: len(labeled)][[*settings.RAW_COLUMNS, *feature_columns]].reset_index(
                drop=True
            ),
            labeled[[*settings.RAW_COLUMNS, *feature_columns]].reset_index(drop=True),
            check_exact=False,
            rtol=1e-13,
            atol=1e-13,
        )
    except AssertionError as exc:
        raise FeatureEngineeringError(
            "Prepared feature, labeled, and raw snapshot rows do not belong to one bundle"
        ) from exc

    expected_entry_timestamps = (
        features["timestamp"].iloc[1 : len(labeled) + 1].reset_index(drop=True)
    )
    expected_exit_timestamps = (
        features["timestamp"].iloc[horizon + 1 : horizon + 1 + len(labeled)].reset_index(drop=True)
    )
    if not labeled["entry_timestamp"].reset_index(drop=True).equals(expected_entry_timestamps):
        raise FeatureEngineeringError("Prepared labels do not use next-open entry timestamps")
    if not labeled["exit_timestamp"].reset_index(drop=True).equals(expected_exit_timestamps):
        raise FeatureEngineeringError(
            "Prepared labels do not use the configured fixed-horizon exit timestamps"
        )

    expected_entry_open = features["open"].iloc[1 : len(labeled) + 1].to_numpy()
    expected_exit_open = features["open"].iloc[horizon + 1 : horizon + 1 + len(labeled)].to_numpy()
    expected_gross_return = expected_exit_open / expected_entry_open - 1.0
    if not np.allclose(
        labeled["entry_open"].to_numpy(), expected_entry_open, rtol=1e-13, atol=1e-13
    ):
        raise FeatureEngineeringError("Prepared labels do not use next-open entry prices")
    if not np.allclose(labeled["exit_open"].to_numpy(), expected_exit_open, rtol=1e-13, atol=1e-13):
        raise FeatureEngineeringError("Prepared labels do not use fixed-horizon exit prices")
    if not np.allclose(
        labeled["gross_forward_return"].to_numpy(),
        expected_gross_return,
        rtol=1e-13,
        atol=1e-13,
    ):
        raise FeatureEngineeringError("Prepared gross forward returns do not reconcile")
    expected_labels = (expected_gross_return > float(manifest["minimum_required_return"])).astype(
        "int8"
    )
    if not np.array_equal(labeled["label"].to_numpy(), expected_labels):
        raise FeatureEngineeringError(
            "Prepared labels do not match the recorded minimum required return"
        )


def load_prepared_dataset_bundle(
    symbol: str,
    timeframe: str,
    *,
    interim_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> PreparedDatasetBundle:
    """Load a prepared dataset only after verifying its complete provenance bundle.

    The manifest is the completion marker. Its file hashes bind the feature and
    labeled CSVs to each other and to the exact immutable raw snapshot used during
    preparation. The mutable latest raw-data file is intentionally not consulted.
    """
    feature_path = get_feature_dataset_path(symbol, timeframe, interim_dir)
    labeled_path = get_labeled_dataset_path(symbol, timeframe, processed_dir)
    manifest_path = get_prepared_dataset_manifest_path(symbol, timeframe, processed_dir)
    if not manifest_path.is_file():
        raise FeatureEngineeringError(
            f"Prepared dataset manifest is missing: {manifest_path}; run prepare first"
        )
    manifest, manifest_bytes, manifest_sha256 = _load_manifest(manifest_path)
    (
        source_snapshot_path,
        manifest_feature_path,
        manifest_labeled_path,
        feature_columns,
    ) = _validate_manifest_configuration(
        manifest,
        symbol,
        timeframe,
        feature_path,
        labeled_path,
    )
    source_bytes = _read_verified_bytes(
        source_snapshot_path,
        manifest["source_snapshot_sha256"],
        "source snapshot",
    )
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_snapshot_path.stem != source_sha256:
        raise FeatureEngineeringError(
            "Prepared dataset source snapshot path is not content-addressed by its SHA-256"
        )
    feature_bytes = _read_verified_bytes(
        manifest_feature_path,
        manifest["feature_file_sha256"],
        "feature file",
    )
    feature_sha256 = hashlib.sha256(feature_bytes).hexdigest()
    labeled_bytes = _read_verified_bytes(
        manifest_labeled_path,
        manifest["labeled_file_sha256"],
        "labeled file",
    )
    labeled_sha256 = hashlib.sha256(labeled_bytes).hexdigest()

    try:
        raw_data = load_ohlcv_csv_bytes(
            source_bytes,
            timeframe=timeframe,
            description=f"prepared dataset source snapshot {source_snapshot_path}",
        )
    except MarketDataValidationError as exc:
        raise FeatureEngineeringError(
            f"Prepared dataset source snapshot is invalid: {exc}"
        ) from exc
    features = _load_feature_dataset_bytes(feature_bytes, str(manifest_feature_path))
    labeled = _load_labeled_dataset_bytes(labeled_bytes, str(manifest_labeled_path))
    _validate_logical_bundle(raw_data, features, labeled, manifest)
    return PreparedDatasetBundle(
        features=features,
        labeled=labeled,
        manifest=manifest,
        manifest_path=_resolved_requested_path(manifest_path, "manifest"),
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
        source_snapshot_path=source_snapshot_path,
        source_snapshot_sha256=source_sha256,
        feature_path=manifest_feature_path,
        feature_sha256=feature_sha256,
        labeled_path=manifest_labeled_path,
        labeled_sha256=labeled_sha256,
        feature_columns=feature_columns,
    )


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
        with temporary_path.open("w", encoding="utf-8", newline="") as file_handle:
            df.to_csv(
                file_handle,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%S.%fZ",
                float_format="%.17g",
                lineterminator="\n",
            )
            file_handle.flush()
            os.fsync(file_handle.fileno())
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

    try:
        source_sha256 = sha256_file(latest_path)
    except OSError as exc:
        raise FeatureEngineeringError(
            f"Unable to hash latest raw data {latest_path}: {exc}"
        ) from exc
    dataset_slug = f"{symbol_to_slug(symbol)}_{timeframe}"
    source_snapshot_path = snapshot_root / dataset_slug / f"{source_sha256}.csv"
    if not source_snapshot_path.exists():
        raise FeatureEngineeringError(
            f"Immutable raw snapshot for latest data is missing: {source_snapshot_path}"
        )
    source_bytes = _read_verified_bytes(
        source_snapshot_path,
        source_sha256,
        "immutable raw snapshot",
    )
    raw_data = load_ohlcv_csv_bytes(
        source_bytes,
        timeframe=timeframe,
        current_utc_time=current_utc_time,
        description=f"immutable raw snapshot {source_snapshot_path}",
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

    feature_columns = tuple(get_expected_feature_columns())
    feature_path = get_feature_dataset_path(symbol, timeframe, interim_root)
    labeled_path = get_labeled_dataset_path(symbol, timeframe, processed_root)
    manifest_path = get_prepared_dataset_manifest_path(symbol, timeframe, processed_root)
    try:
        _atomic_write_dataframe(features, feature_path)
        _atomic_write_dataframe(labeled, labeled_path)
        feature_sha256 = sha256_file(feature_path)
        labeled_sha256 = sha256_file(labeled_path)
        manifest = {
            "dataset_manifest_version": _DATASET_MANIFEST_VERSION,
            "created_at_utc": _utc_iso(datetime.now(UTC).replace(microsecond=0)),
            "symbol": symbol,
            "timeframe": timeframe,
            "source_snapshot_path": str(source_snapshot_path.resolve(strict=True)),
            "source_snapshot_sha256": source_sha256,
            "feature_file_path": str(feature_path.resolve(strict=True)),
            "feature_file_sha256": feature_sha256,
            "labeled_file_path": str(labeled_path.resolve(strict=True)),
            "labeled_file_sha256": labeled_sha256,
            "feature_columns": list(feature_columns),
            "feature_schema_hash": _feature_schema_hash(feature_columns),
            "feature_configuration": _feature_configuration(),
            "prediction_horizon": settings.PREDICTION_HORIZON,
            "label_lookahead_rows": settings.LABEL_LOOKAHEAD_ROWS,
            "minimum_required_return": minimum_required_return,
            "minimum_net_edge_bps": settings.MIN_EDGE_BPS,
            "label_definition": _LABEL_DEFINITION,
            "fee_assumptions": _fee_assumptions(),
            "slippage_assumptions": _slippage_assumptions(),
            "spread_assumptions": _spread_assumptions(),
            "warmup_rows_removed": len(raw_data) - len(features),
            "unlabeled_rows_removed": len(features) - len(labeled),
            "feature_row_count": len(features),
            "labeled_row_count": len(labeled),
            "feature_start_timestamp": _utc_iso(features["timestamp"].iloc[0]),
            "feature_end_timestamp": _utc_iso(features["timestamp"].iloc[-1]),
            "labeled_start_timestamp": _utc_iso(labeled["timestamp"].iloc[0]),
            "labeled_end_timestamp": _utc_iso(labeled["timestamp"].iloc[-1]),
        }
        # This completion marker is deliberately replaced only after both finalized
        # CSVs have been hashed. A partially replaced bundle is therefore rejected.
        _atomic_write_json(manifest, manifest_path)
    except (OSError, TypeError, ValueError) as exc:
        raise FeatureEngineeringError(
            f"Unable to persist prepared dataset bundle for {symbol} {timeframe}: {exc}"
        ) from exc

    bundle = load_prepared_dataset_bundle(
        symbol,
        timeframe,
        interim_dir=interim_root,
        processed_dir=processed_root,
    )
    result = PreparedDatasetResult(
        features=bundle.features,
        labeled=bundle.labeled,
        feature_path=bundle.feature_path,
        labeled_path=bundle.labeled_path,
        source_snapshot_path=bundle.source_snapshot_path,
        source_sha256=bundle.source_snapshot_sha256,
        feature_columns=bundle.feature_columns,
        warmup_rows_removed=int(bundle.manifest["warmup_rows_removed"]),
        unlabeled_rows_removed=int(bundle.manifest["unlabeled_rows_removed"]),
        minimum_required_return=float(bundle.manifest["minimum_required_return"]),
        manifest_path=bundle.manifest_path,
        feature_sha256=bundle.feature_sha256,
        labeled_sha256=bundle.labeled_sha256,
    )
    logger.info(
        "Prepared %s inference rows and %s labeled rows from raw snapshot %s",
        len(bundle.features),
        len(bundle.labeled),
        bundle.source_snapshot_sha256,
    )
    return result
