"""Phase 1 development, holdout evaluation, and production workflows."""

import hashlib
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_ai.artifacts.manifest import (
    atomic_write_json,
    environment_metadata,
    git_metadata,
    utc_now_iso,
)
from crypto_ai.artifacts.registry import (
    claim_holdout_evaluation,
    copy_verified_snapshot,
    create_run_directory,
    generate_run_id,
    load_xgboost_model,
    resolve_artifact_subdirectory,
    save_xgboost_model,
    update_holdout_claim,
)
from crypto_ai.backtesting.baselines import (
    buy_and_hold_backtest,
    cash_baseline,
    random_exposure_summary,
    rule_backtest,
    run_cost_sensitivity,
)
from crypto_ai.backtesting.engine import run_backtest
from crypto_ai.backtesting.metrics import calculate_backtest_metrics
from crypto_ai.config import settings
from crypto_ai.costs import CostConfig
from crypto_ai.data.storage import sha256_file
from crypto_ai.data.validation import timeframe_to_milliseconds
from crypto_ai.exceptions import ArtifactError, CryptoAIError
from crypto_ai.features.build import compute_features
from crypto_ai.features.dataset import load_prepared_dataset_bundle
from crypto_ai.features.labels import add_labels
from crypto_ai.modeling.splits import create_split_plan, save_split_metadata
from crypto_ai.modeling.train import (
    evaluate_logistic_walk_forward,
    evaluate_walk_forward,
    feature_importance_frame,
    feature_schema_hash,
    train_evaluation_model,
    train_production_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DevelopmentRunResult:
    """Saved development validation outputs and frozen evaluation model."""

    run_id: str
    run_directory: Path
    evaluation_directory: Path
    xgboost_metrics: dict[str, Any]
    logistic_metrics: dict[str, Any]
    development_summary: dict[str, Any]


def _base_cost() -> CostConfig:
    return CostConfig(
        settings.TAKER_FEE_RATE,
        settings.SLIPPAGE_BPS_PER_SIDE,
        settings.HALF_SPREAD_BPS_PER_SIDE,
    )


def _feature_configuration() -> dict[str, Any]:
    """Return every setting that affects the persisted Phase 1 feature matrix."""
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
        "return_periods": settings.RETURN_PERIODS,
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file_handle:
            frame.to_csv(
                file_handle,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%S.%fZ",
                lineterminator="\n",
            )
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.link(temporary_path, path)
    except FileExistsError as exc:
        raise ArtifactError(f"Artifact already exists: {path}") from exc
    except OSError as exc:
        raise ArtifactError(f"Unable to write CSV artifact {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _write_text(path: Path, content: str) -> None:
    """Write a new text artifact without silently replacing prior evidence."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file_handle:
            file_handle.write(content)
    except FileExistsError as exc:
        raise ArtifactError(f"Artifact already exists: {path}") from exc
    except OSError as exc:
        raise ArtifactError(f"Unable to write text artifact {path}: {exc}") from exc


def _write_verified_bytes(path: Path, content: bytes, expected_sha256: str) -> None:
    """Persist captured immutable bytes without consulting their mutable source path."""
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ArtifactError(f"Captured artifact bytes do not match their SHA-256: {path}")
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(content)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        os.link(temporary_path, path)
        _verified_hash(path, expected_sha256, path.name)
    except ArtifactError:
        raise
    except FileExistsError as exc:
        raise ArtifactError(f"Artifact already exists: {path}") from exc
    except OSError as exc:
        raise ArtifactError(f"Unable to write binary artifact {path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def _read_json_artifact(path: Path, description: str) -> dict[str, Any]:
    """Load one required JSON-object artifact as a project-specific failure."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except FileNotFoundError as exc:
        raise ArtifactError(f"Missing {description}: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactError(f"Unable to load {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"{description.capitalize()} must contain a JSON object: {path}")
    return payload


def _require_artifacts(paths: dict[str, Path]) -> None:
    """Require every named artifact before an irreversible workflow action."""
    missing = [
        f"{name}: {path}" for name, path in paths.items() if not path.is_file() or path.is_symlink()
    ]
    if missing:
        raise ArtifactError("Required artifacts are missing: " + "; ".join(missing))


def _artifact_hash(path: Path, description: str) -> str:
    """Hash an artifact and normalize filesystem failures."""
    try:
        return sha256_file(path)
    except OSError as exc:
        raise ArtifactError(f"Unable to hash {description} {path}: {exc}") from exc


def _verified_hash(path: Path, expected_sha256: str, description: str) -> None:
    """Verify a file digest and normalize filesystem failures."""
    actual = _artifact_hash(path, description)
    if actual != expected_sha256:
        raise ArtifactError(
            f"{description.capitalize()} hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )


def _positive_integer(value: Any, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ArtifactError(f"Frozen {description} must be a positive integer")
    return value


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"Frozen {description} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArtifactError(f"Frozen {description} must be finite")
    return numeric


def _validated_evaluation_contract(
    run_id: str,
    manifest: dict[str, Any],
    config: dict[str, Any],
    model_metadata: dict[str, Any],
    saved_schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate every frozen metadata value knowable without reading holdout rows."""
    try:
        data_hash = manifest["data_hash"]
        feature_columns = list(saved_schema["feature_columns"])
        frozen_artifact_hashes = dict(manifest["frozen_artifact_hashes"])
        timeframe = manifest["timeframe"]
        timeframe_delta = pd.to_timedelta(timeframe_to_milliseconds(timeframe), unit="ms")
        horizon = _positive_integer(config["prediction_horizon"], "prediction horizon")
        lookahead = _positive_integer(config["label_lookahead_rows"], "label lookahead")
        signal_threshold = _finite_float(config["signal_threshold"], "signal threshold")
        minimum_required_return = _finite_float(
            config["minimum_required_return"], "minimum required return"
        )
        initial_capital = _finite_float(config["initial_capital"], "initial capital")
        holdout_ratio = _finite_float(config["final_holdout_ratio"], "final holdout ratio")
        random_simulations = _positive_integer(
            config["random_baseline_simulations"], "random baseline simulation count"
        )
        random_seed = config["random_seed"]
        if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
            raise ArtifactError("Frozen random seed must be a non-negative integer")

        walk_forward = dict(config["walk_forward_configuration"])
        n_splits = _positive_integer(walk_forward["n_splits"], "walk-forward split count")
        gap_rows = _positive_integer(walk_forward["gap_rows"], "walk-forward gap")
        test_size_rows = _positive_integer(walk_forward["test_size_rows"], "walk-forward test size")
        test_ratio = _finite_float(walk_forward["test_ratio"], "walk-forward test ratio")
        base_cost = CostConfig(**dict(config["base_cost"]))
        cost_scenarios = dict(config["cost_scenarios"])
        if not cost_scenarios:
            raise ArtifactError("Frozen cost-sensitivity scenarios must not be empty")
        scenario_costs = {
            name: CostConfig(**dict(values)) for name, values in cost_scenarios.items()
        }

        development_boundary = dict(manifest["development_boundary"])
        holdout_boundary = dict(manifest["holdout_boundary"])
        training_start = pd.Timestamp(model_metadata["training_start"])
        training_end = pd.Timestamp(model_metadata["training_end"])
        training_exit_end = pd.Timestamp(model_metadata["training_exit_end"])
        holdout_start = pd.Timestamp(model_metadata["holdout_start"])
        development_start = pd.Timestamp(development_boundary["start_timestamp"])
        development_end = pd.Timestamp(development_boundary["end_timestamp"])
        expected_holdout_start = pd.Timestamp(holdout_boundary["start_timestamp"])
        training_row_count = _positive_integer(
            model_metadata["training_row_count"], "evaluation-model training row count"
        )
    except ArtifactError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, CryptoAIError) as exc:
        raise ArtifactError(f"Holdout preflight metadata is invalid: {exc}") from exc

    valid_digest = (
        isinstance(data_hash, str)
        and len(data_hash) == 64
        and all(character in "0123456789abcdef" for character in data_hash)
    )
    valid_schema = (
        bool(feature_columns)
        and all(isinstance(column, str) and column for column in feature_columns)
        and len(feature_columns) == len(set(feature_columns))
    )
    if not valid_digest or not valid_schema:
        raise ArtifactError("Holdout preflight data hash or feature schema is invalid")
    if not isinstance(timeframe, str):
        raise ArtifactError("Frozen timeframe must be a string")
    if horizon != settings.PREDICTION_HORIZON or lookahead != settings.LABEL_LOOKAHEAD_ROWS:
        raise ArtifactError("Holdout preflight feature/label horizon has changed")
    if lookahead != horizon + 1 or gap_rows < lookahead:
        raise ArtifactError("Holdout preflight purge and label lookahead are inconsistent")
    if not 0.0 <= signal_threshold <= 1.0:
        raise ArtifactError("Frozen signal threshold must be within [0, 1]")
    if minimum_required_return <= -1.0 or initial_capital <= 0.0:
        raise ArtifactError("Frozen return threshold or initial capital is invalid")
    if not 0.0 < holdout_ratio < 1.0 or not 0.0 < test_ratio < 1.0:
        raise ArtifactError("Frozen holdout and walk-forward ratios must be within (0, 1)")
    try:
        expected_test_size_rows = max(1, int(training_row_count * test_ratio))
    except (OverflowError, ValueError) as exc:
        raise ArtifactError("Frozen walk-forward geometry is numerically invalid") from exc
    if test_size_rows != expected_test_size_rows:
        raise ArtifactError("Frozen walk-forward test size does not match its ratio")
    if training_row_count - n_splits * test_size_rows - gap_rows <= 0:
        raise ArtifactError("Frozen walk-forward geometry leaves no initial training rows")
    if any(cost.fee_rate != base_cost.fee_rate for cost in scenario_costs.values()):
        raise ArtifactError("Cost sensitivity must keep the frozen fee assumption unchanged")
    if scenario_costs.get("base") != base_cost:
        raise ArtifactError("Frozen base-cost sensitivity scenario is inconsistent")
    boundary_timestamps = (
        training_start,
        training_end,
        training_exit_end,
        holdout_start,
        development_start,
        development_end,
        expected_holdout_start,
    )
    if any(pd.isna(timestamp) or timestamp.tzinfo is None for timestamp in boundary_timestamps):
        raise ArtifactError("Frozen training and holdout boundaries must be timezone-aware")
    try:
        expected_training_end = training_start + (training_row_count - 1) * timeframe_delta
        expected_training_exit_end = training_end + lookahead * timeframe_delta
        expected_purged_holdout_start = training_end + (lookahead + 1) * timeframe_delta
    except (OverflowError, TypeError, ValueError) as exc:
        raise ArtifactError("Frozen training boundary geometry is numerically invalid") from exc
    if training_end != expected_training_end:
        raise ArtifactError("Frozen training row count does not match its timestamp span")
    if not training_start <= training_end < training_exit_end < holdout_start:
        raise ArtifactError("Frozen training and holdout boundaries are not chronological")
    if training_exit_end != expected_training_exit_end:
        raise ArtifactError("Frozen training exit does not match the label lookahead")
    if holdout_start != expected_purged_holdout_start:
        raise ArtifactError("Frozen holdout start does not follow the boundary purge")

    configuration_valid = (
        manifest.get("run_id") == run_id
        and all(manifest.get(key) == value for key, value in config.items())
        and config.get("feature_configuration") == _feature_configuration()
        and feature_columns == config.get("feature_columns")
        and feature_schema_hash(feature_columns) == config.get("feature_schema_hash")
        and model_metadata.get("feature_schema_hash") == config.get("feature_schema_hash")
        and model_metadata.get("feature_columns") == feature_columns
        and model_metadata.get("model_parameters") == config.get("model_parameters")
        and model_metadata.get("prediction_horizon") == horizon
        and model_metadata.get("label_threshold") == minimum_required_return
        and model_metadata.get("signal_threshold") == signal_threshold
        and model_metadata.get("data_hash") == data_hash
        and model_metadata.get("model_version") == run_id
        and model_metadata.get("model_type") == "XGBClassifier"
        and training_start == development_start
        and training_end == development_end
        and training_row_count == development_boundary.get("row_count")
        and holdout_start == expected_holdout_start
        and training_exit_end < holdout_start
    )
    if not configuration_valid:
        raise ArtifactError("Holdout preflight failed: frozen configuration is inconsistent")
    return {
        "data_hash": data_hash,
        "feature_columns": feature_columns,
        "frozen_artifact_hashes": frozen_artifact_hashes,
        "timeframe": timeframe,
        "horizon": horizon,
        "lookahead": lookahead,
        "signal_threshold": signal_threshold,
        "minimum_required_return": minimum_required_return,
        "initial_capital": initial_capital,
        "holdout_ratio": holdout_ratio,
        "random_simulations": random_simulations,
        "random_seed": random_seed,
        "walk_forward": walk_forward,
        "n_splits": n_splits,
        "gap_rows": gap_rows,
        "test_size_rows": test_size_rows,
        "test_ratio": test_ratio,
        "base_cost": base_cost,
        "cost_scenarios": cost_scenarios,
        "training_exit_end": training_exit_end,
        "holdout_start": holdout_start,
    }


def _validate_prepared_provenance(
    prepared_manifest: dict[str, Any],
    run_manifest: dict[str, Any],
    config: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Cross-check the frozen prepared bundle against the development contract."""
    try:
        prepared_base_cost = {
            "fee_rate": prepared_manifest["fee_assumptions"]["taker_fee_rate"],
            "slippage_bps_per_side": prepared_manifest["slippage_assumptions"][
                "slippage_bps_per_side"
            ],
            "half_spread_bps_per_side": prepared_manifest["spread_assumptions"][
                "half_spread_bps_per_side"
            ],
        }
        valid = (
            prepared_manifest["symbol"] == run_manifest["symbol"]
            and prepared_manifest["timeframe"] == contract["timeframe"]
            and prepared_manifest["source_snapshot_sha256"] == contract["data_hash"]
            and prepared_manifest["feature_columns"] == contract["feature_columns"]
            and prepared_manifest["feature_schema_hash"] == config["feature_schema_hash"]
            and prepared_manifest["feature_configuration"] == config["feature_configuration"]
            and prepared_manifest["prediction_horizon"] == contract["horizon"]
            and prepared_manifest["label_lookahead_rows"] == contract["lookahead"]
            and prepared_manifest["minimum_required_return"] == contract["minimum_required_return"]
            and prepared_manifest["label_definition"] == config["label_definition"]
            and prepared_base_cost == config["base_cost"]
            and prepared_manifest["feature_file_sha256"] == run_manifest["feature_file_sha256"]
            and prepared_manifest["labeled_file_sha256"] == run_manifest["labeled_file_sha256"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"Frozen prepared-dataset provenance is invalid: {exc}") from exc
    if not valid:
        raise ArtifactError("Frozen prepared-dataset provenance is inconsistent")


def _trading_suite(
    features: pd.DataFrame,
    probabilities: pd.Series,
    labels: pd.Series,
    timeframe: str,
    *,
    include_random: bool = True,
    horizon: int | None = None,
    signal_threshold: float | None = None,
    initial_capital: float | None = None,
    base_cost: CostConfig | None = None,
    random_simulations: int | None = None,
    random_seed: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    horizon = settings.PREDICTION_HORIZON if horizon is None else horizon
    signal_threshold = settings.SIGNAL_THRESHOLD if signal_threshold is None else signal_threshold
    initial_capital = settings.INITIAL_CAPITAL if initial_capital is None else initial_capital
    cost = _base_cost() if base_cost is None else base_cost
    random_seed = settings.RANDOM_SEED if random_seed is None else random_seed
    model_backtest = run_backtest(
        features,
        probabilities,
        labels,
        horizon,
        timeframe,
        signal_threshold,
        initial_capital,
        cost,
    )
    model_metrics = calculate_backtest_metrics(model_backtest, timeframe)
    baselines: dict[str, Any] = {
        "cash": calculate_backtest_metrics(
            cash_baseline(
                features,
                probabilities.index,
                labels,
                horizon,
                timeframe,
                cost,
                initial_capital=initial_capital,
            ),
            timeframe,
        ),
        "buy_and_hold": calculate_backtest_metrics(
            buy_and_hold_backtest(
                features,
                probabilities.index,
                horizon,
                cost,
                initial_capital=initial_capital,
            ),
            timeframe,
        ),
        "ema": calculate_backtest_metrics(
            rule_backtest(
                features,
                probabilities.index,
                features["ema_short"] > features["ema_long"],
                labels,
                horizon,
                timeframe,
                cost,
                initial_capital=initial_capital,
            ),
            timeframe,
        ),
        "momentum": calculate_backtest_metrics(
            rule_backtest(
                features,
                probabilities.index,
                features["return_24"] > 0,
                labels,
                horizon,
                timeframe,
                cost,
                initial_capital=initial_capital,
            ),
            timeframe,
        ),
    }
    if include_random:
        baselines["random"] = random_exposure_summary(
            features,
            probabilities,
            labels,
            horizon,
            timeframe,
            cost,
            model_metrics["total_return"],
            random_simulations,
            signal_threshold=signal_threshold,
            initial_capital=initial_capital,
            random_seed=random_seed,
        )
    buy_hold_return = baselines["buy_and_hold"]["total_return"]
    model_metrics["buy_and_hold_total_return"] = buy_hold_return
    model_metrics["excess_return_vs_buy_and_hold"] = model_metrics["total_return"] - buy_hold_return
    model_metrics["excess_return_vs_cash"] = model_metrics["total_return"]
    return model_backtest, {"model": model_metrics, "baselines": baselines}


def _report_number(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    return f"{numeric:.2%}" if percent else f"{numeric:.6f}"


def _fold_table(title: str, metrics: tuple[dict[str, Any], ...]) -> str:
    lines = [
        f"### {title}",
        "",
        "| Fold | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Log loss | Brier | "
        "Positive labels | Predicted positive |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics:
        lines.append(
            f"| {row['fold_number']} | {_report_number(row['accuracy'])} | "
            f"{_report_number(row['balanced_accuracy'])} | "
            f"{_report_number(row['roc_auc'])} | {_report_number(row['pr_auc'])} | "
            f"{_report_number(row['log_loss'])} | {_report_number(row['brier_score'])} | "
            f"{_report_number(row['positive_label_rate'], percent=True)} | "
            f"{_report_number(row['predicted_positive_rate'], percent=True)} |"
        )
    lines.extend(
        [
            "",
            "| Fold | Precision 1 | Recall 1 | F1 1 | Precision 0 | Recall 0 | F1 0 | "
            "Confusion matrix |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['fold_number']} | {_report_number(row['precision_class_1'])} | "
            f"{_report_number(row['recall_class_1'])} | "
            f"{_report_number(row['f1_class_1'])} | "
            f"{_report_number(row['precision_class_0'])} | "
            f"{_report_number(row['recall_class_0'])} | "
            f"{_report_number(row['f1_class_0'])} | `{row['confusion_matrix']}` |"
        )
    return "\n".join(lines)


def _undefined_classification_warnings(
    model_name: str,
    fold_metrics: tuple[dict[str, Any], ...],
    aggregate_metrics: dict[str, Any],
) -> list[str]:
    """Describe undefined classification results without inventing values."""
    warnings: list[str] = []
    for row in fold_metrics:
        undefined = [
            name
            for name, key in (("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"))
            if row.get(key) is None
        ]
        if undefined:
            warnings.append(
                f"{model_name} fold {row['fold_number']}: {' and '.join(undefined)} "
                "undefined because the validation labels contain one class"
            )
    aggregate_undefined = [
        name
        for name, key in (("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"))
        if aggregate_metrics.get(key) is None
    ]
    if aggregate_undefined:
        warnings.append(
            f"{model_name} aggregate: {' and '.join(aggregate_undefined)} undefined because "
            "the out-of-fold labels contain one class"
        )
    return warnings


def _fold_stability_commentary(
    xgboost_metrics: tuple[dict[str, Any], ...],
    logistic_metrics: tuple[dict[str, Any], ...],
) -> str:
    def span(rows: tuple[dict[str, Any], ...], metric: str) -> str:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if not values:
            return "undefined in every fold"
        return f"{min(values):.6f} to {max(values):.6f}"

    return (
        f"XGBoost fold accuracy ranged from {span(xgboost_metrics, 'accuracy')} and PR-AUC "
        f"ranged from {span(xgboost_metrics, 'pr_auc')}. Logistic-regression fold accuracy "
        f"ranged from {span(logistic_metrics, 'accuracy')} and PR-AUC ranged from "
        f"{span(logistic_metrics, 'pr_auc')}. These ranges describe development-only "
        "variation; they are not a holdout or profitability conclusion."
    )


def _development_warnings(xgboost: Any, logistic: Any, trading: dict[str, Any]) -> list[str]:
    """Collect model, classification, and strategy caveats for every run artifact."""
    strategy_display_names = {
        "cash": "Cash",
        "buy_and_hold": "Buy & Hold",
        "ema": "EMA",
        "momentum": "Momentum",
    }
    warnings = [
        *(f"XGBoost: {warning}" for warning in xgboost.warnings),
        *(f"Logistic regression: {warning}" for warning in logistic.warnings),
        *_undefined_classification_warnings(
            "XGBoost", xgboost.fold_metrics, xgboost.aggregate_metrics
        ),
        *_undefined_classification_warnings(
            "Logistic regression", logistic.fold_metrics, logistic.aggregate_metrics
        ),
        *(f"XGBoost OOF: {warning}" for warning in trading["model"].get("warnings", [])),
    ]
    for strategy_name, metrics in trading["baselines"].items():
        if strategy_name == "random":
            continue
        warnings.extend(
            f"{strategy_display_names[strategy_name]}: {warning}"
            for warning in metrics.get("warnings", [])
        )
    return warnings


def _development_report(
    run_id: str,
    plan: Any,
    xgboost: Any,
    logistic: Any,
    trading: dict[str, Any],
    importance: pd.DataFrame,
    config: dict[str, Any],
) -> str:
    """Render all required development-only evidence without exposing holdout outcomes."""
    aggregate_lines = [
        "| Model | Accuracy | Balanced accuracy | ROC-AUC | PR-AUC | Log loss | Brier |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in (
        ("XGBoost", xgboost.aggregate_metrics),
        ("Logistic regression", logistic.aggregate_metrics),
    ):
        aggregate_lines.append(
            f"| {name} | {_report_number(metrics['accuracy'])} | "
            f"{_report_number(metrics['balanced_accuracy'])} | "
            f"{_report_number(metrics['roc_auc'])} | "
            f"{_report_number(metrics['pr_auc'])} | "
            f"{_report_number(metrics['log_loss'])} | "
            f"{_report_number(metrics['brier_score'])} |"
        )

    strategy_lines = [
        "| Strategy | Total return | Sharpe | Maximum drawdown | Exposure | Trades |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    strategy_rows = {"XGBoost OOF": trading["model"], **trading["baselines"]}
    random_result = strategy_rows.pop("random")
    strategy_display_names = {
        "XGBoost OOF": "XGBoost OOF",
        "cash": "Cash",
        "buy_and_hold": "Buy & Hold",
        "ema": "EMA",
        "momentum": "Momentum",
    }
    for name, metrics in strategy_rows.items():
        strategy_lines.append(
            f"| {strategy_display_names[name]} | "
            f"{_report_number(metrics['total_return'], percent=True)} | "
            f"{_report_number(metrics['sharpe_ratio'])} | "
            f"{_report_number(metrics['maximum_drawdown'], percent=True)} | "
            f"{_report_number(metrics['market_exposure'], percent=True)} | "
            f"{metrics['num_trades']} |"
        )

    importance_lines = [
        "| Feature | Gain | Weight | Cover |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in importance.sort_values(["gain", "weight"], ascending=False).head(10).itertuples():
        importance_lines.append(
            f"| {row.feature} | {row.gain:.6f} | {row.weight:.6f} | {row.cover:.6f} |"
        )

    base_cost = config["base_cost"]
    warnings = _development_warnings(xgboost, logistic, trading)
    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None recorded."
    aggregate_table = "\n".join(aggregate_lines)
    strategy_table = "\n".join(strategy_lines)
    importance_table = "\n".join(importance_lines)
    return (
        f"# Development Report — {run_id}\n\n"
        "## Chronological boundaries\n\n"
        "| Partition | Start | End | Rows |\n"
        "| --- | --- | --- | ---: |\n"
        f"| Development | {plan.development['timestamp'].iloc[0]} | "
        f"{plan.development['timestamp'].iloc[-1]} | {len(plan.development)} |\n"
        f"| Boundary purge | {plan.boundary_purge['timestamp'].iloc[0]} | "
        f"{plan.boundary_purge['timestamp'].iloc[-1]} | {len(plan.boundary_purge)} |\n"
        f"| Untouched holdout | {plan.holdout['timestamp'].iloc[0]} | "
        f"{plan.holdout['timestamp'].iloc[-1]} | {len(plan.holdout)} |\n\n"
        "Only holdout boundaries and row count are shown; no holdout labels or returns were "
        "inspected.\n\n"
        "## Per-fold classification results\n\n"
        f"{_fold_table('XGBoost', xgboost.fold_metrics)}\n\n"
        f"{_fold_table('Logistic regression', logistic.fold_metrics)}\n\n"
        "## Aggregate model comparison\n\n"
        f"{aggregate_table}\n\n"
        "## Development OOF trading comparison\n\n"
        f"{strategy_table}\n\n"
        "### Random-exposure baseline\n\n"
        f"- Simulations: {random_result['simulations']}\n"
        "- Signal probability: "
        f"{_report_number(random_result['signal_probability'], percent=True)}\n"
        f"- Total-return median / p05 / p95: "
        f"{_report_number(random_result['total_return']['median'], percent=True)} / "
        f"{_report_number(random_result['total_return']['p05'], percent=True)} / "
        f"{_report_number(random_result['total_return']['p95'], percent=True)}\n"
        f"- Sharpe median / p05 / p95: "
        f"{_report_number(random_result['sharpe_ratio']['median'])} / "
        f"{_report_number(random_result['sharpe_ratio']['p05'])} / "
        f"{_report_number(random_result['sharpe_ratio']['p95'])}\n"
        f"- Maximum-drawdown median / p05 / p95: "
        f"{_report_number(random_result['maximum_drawdown']['median'], percent=True)} / "
        f"{_report_number(random_result['maximum_drawdown']['p05'], percent=True)} / "
        f"{_report_number(random_result['maximum_drawdown']['p95'], percent=True)}\n"
        f"- Fraction with return at least the model: "
        f"{_report_number(random_result['fraction_return_at_least_model'], percent=True)}\n\n"
        "## Global XGBoost feature importance\n\n"
        f"{importance_table}\n\n"
        "These are global model importances and do not explain individual signals.\n\n"
        "## Fold stability\n\n"
        f"{_fold_stability_commentary(xgboost.fold_metrics, logistic.fold_metrics)}\n\n"
        "## Frozen execution and cost assumptions\n\n"
        f"- Execution: {config['execution_policy']}\n"
        f"- Prediction horizon: {config['prediction_horizon']} candles\n"
        f"- Signal threshold: {config['signal_threshold']}\n"
        f"- Entry/exit fee rate: {base_cost['fee_rate']} per side\n"
        f"- Slippage: {base_cost['slippage_bps_per_side']} bps per side\n"
        f"- Half-spread: {base_cost['half_spread_bps_per_side']} bps per side\n"
        "- Official result convention: the frozen base-cost scenario\n\n"
        "## Warnings\n\n"
        f"{warning_lines}\n\n"
        "## Limitations\n\n"
        "Historical OHLCV is the only input; fills and transaction costs are modeled; the "
        "random baseline is descriptive; global feature importance is not a local explanation; "
        "and development performance does not establish live profitability.\n\n"
        "## Final holdout status\n\n"
        "The final holdout was not evaluated by this development run. Evaluating it remains a "
        "separate, explicit, one-time human research decision.\n"
    )


def _load_completed_evaluation_authorization(evaluation_run_id: str) -> dict[str, Any]:
    """Verify completed evaluation evidence before production data can be loaded."""
    run_directory = resolve_artifact_subdirectory(settings.RUNS_DIR, evaluation_run_id)
    evaluation_directory = resolve_artifact_subdirectory(
        settings.EVALUATIONS_DIR, evaluation_run_id
    )
    if not run_directory.is_dir() or not evaluation_directory.is_dir():
        raise ArtifactError(f"Unknown evaluation run: {evaluation_run_id}")

    run_paths = {
        "development manifest": run_directory / "manifest.json",
        "frozen configuration": run_directory / "config.json",
        "prepared dataset manifest": run_directory / "prepared_dataset_manifest.json",
        "holdout claim": run_directory / "holdout_evaluation_claim.json",
    }
    evaluation_paths = {
        "verified input snapshot": evaluation_directory / "input_data_snapshot.csv",
        "evaluation model": evaluation_directory / "evaluation_model.json",
        "model metadata": evaluation_directory / "model_metadata.json",
        "feature schema": evaluation_directory / "feature_columns.json",
        "holdout predictions": evaluation_directory / "holdout_predictions.csv",
        "trade ledger": evaluation_directory / "trade_ledger.csv",
        "equity curve": evaluation_directory / "equity_curve.csv",
        "strategy metrics": evaluation_directory / "metrics.json",
        "baseline metrics": evaluation_directory / "baseline_metrics.json",
        "cost sensitivity": evaluation_directory / "cost_sensitivity.json",
        "evaluation manifest": evaluation_directory / "evaluation_manifest.json",
    }
    _require_artifacts({**run_paths, **evaluation_paths})

    run_manifest = _read_json_artifact(run_paths["development manifest"], "run manifest")
    config = _read_json_artifact(run_paths["frozen configuration"], "frozen configuration")
    claim = _read_json_artifact(run_paths["holdout claim"], "holdout-evaluation claim")
    evaluation_manifest = _read_json_artifact(
        evaluation_paths["evaluation manifest"], "evaluation manifest"
    )
    feature_schema = _read_json_artifact(evaluation_paths["feature schema"], "feature schema")
    model_metadata = _read_json_artifact(evaluation_paths["model metadata"], "model metadata")
    prepared_manifest = _read_json_artifact(
        run_paths["prepared dataset manifest"], "prepared dataset manifest"
    )
    evaluation_payloads = {
        description: _read_json_artifact(evaluation_paths[description], description)
        for description in ("strategy metrics", "baseline metrics", "cost sensitivity")
    }

    if claim.get("status") != "completed":
        raise ArtifactError(
            f"Evaluation run {evaluation_run_id} is not completed; claim status is "
            f"{claim.get('status')!r}"
        )
    if evaluation_manifest.get("evaluation_artifacts_status") != "complete":
        raise ArtifactError("Evaluation manifest does not record complete immutable artifacts")
    if evaluation_manifest.get("holdout_evaluation_claim_status") != "claimed_at_manifest_write":
        raise ArtifactError("Evaluation manifest has an invalid irreversible-claim marker")
    if (
        run_manifest.get("run_id") != evaluation_run_id
        or evaluation_manifest.get("run_id") != evaluation_run_id
    ):
        raise ArtifactError("Evaluation run identifiers are inconsistent")
    data_hash = run_manifest.get("data_hash")
    if not isinstance(data_hash, str) or evaluation_manifest.get("data_hash") != data_hash:
        raise ArtifactError("Evaluation snapshot provenance is inconsistent")
    if not all(evaluation_manifest.get(key) == value for key, value in config.items()):
        raise ArtifactError("Evaluation manifest does not match the frozen configuration")
    if evaluation_manifest.get("frozen_artifact_hashes") != run_manifest.get(
        "frozen_artifact_hashes"
    ):
        raise ArtifactError("Evaluation manifest does not match the frozen input inventory")
    contract = _validated_evaluation_contract(
        evaluation_run_id, run_manifest, config, model_metadata, feature_schema
    )
    _validate_prepared_provenance(prepared_manifest, run_manifest, config, contract)
    frozen_paths = {
        "config.json": run_paths["frozen configuration"],
        "evaluation_model.json": evaluation_paths["evaluation model"],
        "model_metadata.json": evaluation_paths["model metadata"],
        "feature_columns.json": evaluation_paths["feature schema"],
        "input_data_snapshot.csv": evaluation_paths["verified input snapshot"],
        "prepared_dataset_manifest.json": run_paths["prepared dataset manifest"],
    }
    frozen_hashes = contract["frozen_artifact_hashes"]
    if set(frozen_hashes) != set(frozen_paths):
        raise ArtifactError("Evaluation frozen-artifact hash inventory is incomplete")
    if run_manifest.get("prepared_dataset_manifest_sha256") != frozen_hashes.get(
        "prepared_dataset_manifest.json"
    ):
        raise ArtifactError("Evaluation prepared-dataset provenance is inconsistent")
    for name, path in frozen_paths.items():
        expected_hash = frozen_hashes.get(name)
        if not isinstance(expected_hash, str):
            raise ArtifactError(f"Evaluation frozen-artifact hash is invalid for {name}")
        _verified_hash(path, expected_hash, name)
    _verified_hash(evaluation_paths["verified input snapshot"], data_hash, "evaluation snapshot")
    artifact_paths = {
        "input_data_snapshot.csv": evaluation_paths["verified input snapshot"],
        "evaluation_model.json": evaluation_paths["evaluation model"],
        "model_metadata.json": evaluation_paths["model metadata"],
        "feature_columns.json": evaluation_paths["feature schema"],
        "holdout_predictions.csv": evaluation_paths["holdout predictions"],
        "trade_ledger.csv": evaluation_paths["trade ledger"],
        "equity_curve.csv": evaluation_paths["equity curve"],
        "metrics.json": evaluation_paths["strategy metrics"],
        "baseline_metrics.json": evaluation_paths["baseline metrics"],
        "cost_sensitivity.json": evaluation_paths["cost sensitivity"],
    }
    try:
        artifact_hashes = dict(evaluation_manifest["evaluation_artifact_hashes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("Evaluation artifact hash inventory is invalid") from exc
    if set(artifact_hashes) != set(artifact_paths):
        raise ArtifactError("Evaluation artifact hash inventory is incomplete")
    for name, path in artifact_paths.items():
        expected_hash = artifact_hashes.get(name)
        if not isinstance(expected_hash, str):
            raise ArtifactError(f"Evaluation artifact hash is invalid for {name}")
        _verified_hash(path, expected_hash, name)
    embedded_evaluation_payloads = {
        "strategy metrics": evaluation_manifest.get("strategy_metrics"),
        "baseline metrics": evaluation_manifest.get("baseline_metrics"),
        "cost sensitivity": evaluation_manifest.get("cost_sensitivity"),
    }
    if embedded_evaluation_payloads != evaluation_payloads:
        raise ArtifactError("Evaluation manifest metrics do not match their artifacts")
    try:
        evaluated_at = pd.Timestamp(evaluation_manifest["evaluated_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("Evaluation manifest timestamp is invalid") from exc
    if evaluated_at.tzinfo is None:
        raise ArtifactError("Evaluation manifest timestamp must be timezone-aware")

    return {
        "run_id": evaluation_run_id,
        "run_directory": run_directory,
        "evaluation_directory": evaluation_directory,
        "run_manifest": run_manifest,
        "config": config,
        "evaluation_manifest": evaluation_manifest,
        "evaluation_manifest_sha256": _artifact_hash(
            evaluation_paths["evaluation manifest"], "evaluation manifest"
        ),
    }


def _verify_production_compatibility(
    authorization: dict[str, Any],
    bundle: Any,
    symbol: str,
    timeframe: str,
) -> None:
    """Require production inputs to match the accepted Phase 1 research contract."""
    run_manifest = authorization["run_manifest"]
    config = authorization["config"]
    prepared = bundle.manifest
    prepared_base_cost = {
        "fee_rate": prepared["fee_assumptions"]["taker_fee_rate"],
        "slippage_bps_per_side": prepared["slippage_assumptions"]["slippage_bps_per_side"],
        "half_spread_bps_per_side": prepared["spread_assumptions"]["half_spread_bps_per_side"],
    }
    compatible = (
        run_manifest.get("symbol") == symbol
        and run_manifest.get("timeframe") == timeframe
        and config.get("feature_columns") == list(bundle.feature_columns)
        and config.get("feature_schema_hash") == prepared.get("feature_schema_hash")
        and config.get("feature_configuration") == prepared.get("feature_configuration")
        and config.get("prediction_horizon") == prepared.get("prediction_horizon")
        and config.get("label_lookahead_rows") == prepared.get("label_lookahead_rows")
        and config.get("minimum_required_return") == prepared.get("minimum_required_return")
        and config.get("label_definition") == prepared.get("label_definition")
        and config.get("base_cost") == prepared_base_cost
        and config.get("model_parameters") == settings.XGBOOST_PARAMS
        and config.get("signal_threshold") == settings.SIGNAL_THRESHOLD
    )
    if not compatible:
        raise ArtifactError(
            "Prepared production dataset is incompatible with the accepted evaluation contract"
        )


def run_development_validation(
    symbol: str = settings.SYMBOL,
    timeframe: str = settings.TIMEFRAME,
) -> DevelopmentRunResult:
    """Run development-only walk-forward validation and freeze an evaluation model."""
    bundle = load_prepared_dataset_bundle(symbol, timeframe)
    features = bundle.features
    labeled = bundle.labeled
    plan = create_split_plan(labeled)
    feature_columns = list(bundle.feature_columns)
    xgb = evaluate_walk_forward(
        plan.development,
        feature_columns,
        settings.XGBOOST_PARAMS,
        settings.N_WALK_FORWARD_SPLITS,
        plan.test_size_rows,
        plan.gap_rows,
    )
    logistic = evaluate_logistic_walk_forward(
        plan.development,
        feature_columns,
        settings.N_WALK_FORWARD_SPLITS,
        plan.test_size_rows,
        plan.gap_rows,
    )
    evaluation_model = train_evaluation_model(
        plan.development, feature_columns, settings.XGBOOST_PARAMS
    )
    development_scores = pd.Series(
        xgb.predictions["probability_score"], index=xgb.predictions.index
    )
    development_labels = plan.development.loc[development_scores.index, "label"]
    _, development_trading = _trading_suite(
        features,
        development_scores,
        development_labels,
        timeframe,
    )

    git = git_metadata(settings.BASE_DIR)
    run_id = generate_run_id(symbol, timeframe, str(git["git_commit"]))
    run_directory = create_run_directory(settings.RUNS_DIR, run_id)
    evaluation_directory = create_run_directory(settings.EVALUATIONS_DIR, run_id)
    snapshot = bundle.source_snapshot_path
    data_hash = bundle.source_snapshot_sha256
    prepared_manifest_hash = bundle.manifest_sha256
    copy_verified_snapshot(snapshot, evaluation_directory / "input_data_snapshot.csv", data_hash)
    _write_verified_bytes(
        run_directory / "prepared_dataset_manifest.json",
        bundle.manifest_bytes,
        prepared_manifest_hash,
    )
    save_xgboost_model(evaluation_model, evaluation_directory / "evaluation_model.json")

    importance = feature_importance_frame(evaluation_model, feature_columns)
    _write_frame(run_directory / "oof_predictions.csv", xgb.predictions)
    _write_frame(run_directory / "feature_importance.csv", importance)
    atomic_write_json(run_directory / "feature_columns.json", {"feature_columns": feature_columns})
    atomic_write_json(
        run_directory / "fold_metrics.json",
        {"xgboost": xgb.fold_metrics, "logistic": logistic.fold_metrics},
    )
    atomic_write_json(
        run_directory / "classification_report.json",
        {"xgboost": xgb.aggregate_metrics, "logistic_regression": logistic.aggregate_metrics},
    )
    atomic_write_json(run_directory / "development_strategy_metrics.json", development_trading)
    save_split_metadata(plan, run_directory / "split_metadata.json")

    minimum_return = float(bundle.manifest["minimum_required_return"])
    fee_assumptions = bundle.manifest["fee_assumptions"]
    slippage_assumptions = bundle.manifest["slippage_assumptions"]
    spread_assumptions = bundle.manifest["spread_assumptions"]
    config = {
        "feature_columns": feature_columns,
        "feature_schema_hash": bundle.manifest["feature_schema_hash"],
        "feature_configuration": bundle.manifest["feature_configuration"],
        "prediction_horizon": bundle.manifest["prediction_horizon"],
        "label_lookahead_rows": bundle.manifest["label_lookahead_rows"],
        "minimum_required_return": minimum_return,
        "label_definition": bundle.manifest["label_definition"],
        "signal_threshold": settings.SIGNAL_THRESHOLD,
        "model_parameters": settings.XGBOOST_PARAMS,
        "walk_forward_configuration": {
            "n_splits": settings.N_WALK_FORWARD_SPLITS,
            "test_ratio": settings.WALK_FORWARD_TEST_RATIO,
            "test_size_rows": plan.test_size_rows,
            "gap_rows": plan.gap_rows,
        },
        "final_holdout_ratio": settings.FINAL_HOLDOUT_RATIO,
        "initial_capital": settings.INITIAL_CAPITAL,
        "random_seed": settings.RANDOM_SEED,
        "random_baseline_simulations": settings.RANDOM_BASELINE_SIMULATIONS,
        "base_cost": {
            "fee_rate": fee_assumptions["taker_fee_rate"],
            "slippage_bps_per_side": slippage_assumptions["slippage_bps_per_side"],
            "half_spread_bps_per_side": spread_assumptions["half_spread_bps_per_side"],
        },
        "cost_scenarios": settings.COST_SCENARIOS,
        "execution_policy": "close decision; next-open entry; fixed H; t+H+1 open exit; no overlap",
        "baseline_definitions": ["cash", "buy_and_hold", "ema", "momentum", "random"],
        "metric_definitions": "IMPLEMENTATION_PLAN.md sections 11-13",
    }
    atomic_write_json(run_directory / "config.json", config)
    atomic_write_json(
        evaluation_directory / "feature_columns.json", {"feature_columns": feature_columns}
    )
    atomic_write_json(
        evaluation_directory / "model_metadata.json",
        {
            "model_version": run_id,
            "model_type": "XGBClassifier",
            "training_start": plan.development["timestamp"].iloc[0],
            "training_end": plan.development["timestamp"].iloc[-1],
            "training_exit_end": plan.development["exit_timestamp"].max(),
            "training_row_count": len(plan.development),
            "holdout_start": plan.holdout["timestamp"].iloc[0],
            "feature_columns": feature_columns,
            "feature_schema_hash": config["feature_schema_hash"],
            "model_parameters": settings.XGBOOST_PARAMS,
            "prediction_horizon": config["prediction_horizon"],
            "label_threshold": minimum_return,
            "signal_threshold": settings.SIGNAL_THRESHOLD,
            "data_hash": data_hash,
            "prepared_dataset_manifest_sha256": prepared_manifest_hash,
            "feature_file_sha256": bundle.feature_sha256,
            "labeled_file_sha256": bundle.labeled_sha256,
            "code_commit": git["git_commit"],
            "created_at_utc": utc_now_iso(),
        },
    )
    frozen_artifact_paths = {
        "config.json": run_directory / "config.json",
        "evaluation_model.json": evaluation_directory / "evaluation_model.json",
        "model_metadata.json": evaluation_directory / "model_metadata.json",
        "feature_columns.json": evaluation_directory / "feature_columns.json",
        "input_data_snapshot.csv": evaluation_directory / "input_data_snapshot.csv",
        "prepared_dataset_manifest.json": run_directory / "prepared_dataset_manifest.json",
    }
    frozen_artifact_hashes = {
        name: _artifact_hash(path, name) for name, path in frozen_artifact_paths.items()
    }

    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        **environment_metadata(),
        **git,
        "random_seed": settings.RANDOM_SEED,
        "exchange": settings.EXCHANGE_ID,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_path": bundle.labeled_path,
        "prepared_dataset_manifest_path": bundle.manifest_path,
        "prepared_dataset_manifest_copy": run_directory / "prepared_dataset_manifest.json",
        "prepared_dataset_manifest_sha256": prepared_manifest_hash,
        "feature_file_path": bundle.feature_path,
        "feature_file_sha256": bundle.feature_sha256,
        "labeled_file_path": bundle.labeled_path,
        "labeled_file_sha256": bundle.labeled_sha256,
        "fee_assumptions": fee_assumptions,
        "slippage_assumptions": slippage_assumptions,
        "spread_assumptions": spread_assumptions,
        "immutable_snapshot_path": snapshot,
        "data_hash": data_hash,
        "data_start_timestamp": features["timestamp"].iloc[0],
        "data_end_timestamp": features["timestamp"].iloc[-1],
        "row_count": len(features),
        "development_boundary": plan.partition_metadata.development,
        "boundary_purge": plan.partition_metadata.boundary_purge,
        "holdout_boundary": plan.partition_metadata.holdout,
        "purge_gap": plan.gap_rows,
        "classification_metrics": {
            "xgboost": xgb.aggregate_metrics,
            "logistic": logistic.aggregate_metrics,
        },
        "strategy_metrics": development_trading,
        "holdout_evaluation_claim_status": "not_claimed",
        "frozen_artifact_hashes": frozen_artifact_hashes,
        "warnings": _development_warnings(xgb, logistic, development_trading),
        **config,
    }
    atomic_write_json(run_directory / "manifest.json", manifest)
    atomic_write_json(evaluation_directory / "development_manifest.json", manifest)
    report = _development_report(
        run_id, plan, xgb, logistic, development_trading, importance, config
    )
    _write_text(run_directory / "development_report.md", report)
    _write_text(
        run_directory / "logs.txt",
        f"{utc_now_iso()} Development validation completed; final holdout not evaluated.\n",
    )
    return DevelopmentRunResult(
        run_id,
        run_directory,
        evaluation_directory,
        xgb.aggregate_metrics,
        logistic.aggregate_metrics,
        {
            "development_start": plan.development["timestamp"].iloc[0],
            "development_end": plan.development["timestamp"].iloc[-1],
            "development_rows": len(plan.development),
            "development_positive_label_rate": float(plan.development["label"].mean()),
            "boundary_purge_start": plan.boundary_purge["timestamp"].iloc[0],
            "boundary_purge_end": plan.boundary_purge["timestamp"].iloc[-1],
            "boundary_purge_rows": len(plan.boundary_purge),
            "holdout_start": plan.holdout["timestamp"].iloc[0],
            "holdout_end": plan.holdout["timestamp"].iloc[-1],
            "holdout_rows": len(plan.holdout),
        },
    )


def evaluate_final_holdout(run_id: str) -> dict[str, Any]:
    """Claim and evaluate one untouched holdout under a frozen development run."""
    run_directory = resolve_artifact_subdirectory(settings.RUNS_DIR, run_id)
    evaluation_directory = resolve_artifact_subdirectory(settings.EVALUATIONS_DIR, run_id)
    if not run_directory.is_dir() or not evaluation_directory.is_dir():
        raise ArtifactError(f"Unknown development run: {run_id}")
    inputs = {
        "run manifest": run_directory / "manifest.json",
        "frozen configuration": run_directory / "config.json",
        "prepared dataset manifest": run_directory / "prepared_dataset_manifest.json",
        "evaluation model": evaluation_directory / "evaluation_model.json",
        "model metadata": evaluation_directory / "model_metadata.json",
        "feature schema": evaluation_directory / "feature_columns.json",
        "verified input snapshot": evaluation_directory / "input_data_snapshot.csv",
    }
    outputs = {
        "holdout predictions": evaluation_directory / "holdout_predictions.csv",
        "trade ledger": evaluation_directory / "trade_ledger.csv",
        "equity curve": evaluation_directory / "equity_curve.csv",
        "strategy metrics": evaluation_directory / "metrics.json",
        "baseline metrics": evaluation_directory / "baseline_metrics.json",
        "cost sensitivity": evaluation_directory / "cost_sensitivity.json",
        "evaluation manifest": evaluation_directory / "evaluation_manifest.json",
    }
    _require_artifacts(inputs)
    existing_outputs = [
        f"{name}: {path}" for name, path in outputs.items() if path.exists() or path.is_symlink()
    ]
    if existing_outputs:
        raise ArtifactError(
            "Holdout preflight found existing evaluation artifacts: " + "; ".join(existing_outputs)
        )

    manifest = _read_json_artifact(inputs["run manifest"], "run manifest")
    config = _read_json_artifact(inputs["frozen configuration"], "frozen configuration")
    prepared_manifest = _read_json_artifact(
        inputs["prepared dataset manifest"], "prepared dataset manifest"
    )
    model_metadata = _read_json_artifact(inputs["model metadata"], "model metadata")
    saved_schema = _read_json_artifact(inputs["feature schema"], "feature schema")
    contract = _validated_evaluation_contract(
        run_id, manifest, config, model_metadata, saved_schema
    )
    _validate_prepared_provenance(prepared_manifest, manifest, config, contract)
    data_hash = contract["data_hash"]
    feature_columns = contract["feature_columns"]
    frozen_artifact_hashes = contract["frozen_artifact_hashes"]
    frozen_paths = {
        "config.json": inputs["frozen configuration"],
        "evaluation_model.json": inputs["evaluation model"],
        "model_metadata.json": inputs["model metadata"],
        "feature_columns.json": inputs["feature schema"],
        "input_data_snapshot.csv": inputs["verified input snapshot"],
        "prepared_dataset_manifest.json": inputs["prepared dataset manifest"],
    }
    if set(frozen_artifact_hashes) != set(frozen_paths):
        raise ArtifactError("Holdout preflight frozen-artifact inventory is incomplete")
    if manifest.get("prepared_dataset_manifest_sha256") != frozen_artifact_hashes.get(
        "prepared_dataset_manifest.json"
    ):
        raise ArtifactError("Holdout preflight prepared-dataset provenance is inconsistent")
    for name, path in frozen_paths.items():
        expected_hash = frozen_artifact_hashes.get(name)
        if not isinstance(expected_hash, str):
            raise ArtifactError(f"Holdout preflight has no valid hash for {name}")
        _verified_hash(path, expected_hash, name)
    _verified_hash(inputs["verified input snapshot"], data_hash, "evaluation snapshot")

    claim_path = claim_holdout_evaluation(run_directory)
    try:
        raw = pd.read_csv(inputs["verified input snapshot"])
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        for column in settings.RAW_COLUMNS[1:]:
            raw[column] = raw[column].astype("float64")
        features = compute_features(raw)
        horizon = contract["horizon"]
        labeled = add_labels(
            features,
            horizon,
            contract["minimum_required_return"],
        )
        plan = create_split_plan(
            labeled,
            holdout_ratio=contract["holdout_ratio"],
            label_lookahead_rows=contract["lookahead"],
            n_splits=contract["n_splits"],
            test_ratio=contract["test_ratio"],
            gap_rows=contract["gap_rows"],
        )
        if (
            plan.holdout["timestamp"].iloc[0] != contract["holdout_start"]
            or plan.development["exit_timestamp"].max() != contract["training_exit_end"]
            or plan.test_size_rows != contract["test_size_rows"]
        ):
            raise ArtifactError("Recreated holdout boundaries do not match the frozen run")
        model = load_xgboost_model(inputs["evaluation model"])
        probabilities = pd.Series(
            model.predict_proba(plan.holdout[feature_columns])[:, 1],
            index=plan.holdout.index,
            name="probability_score",
        )
        labels = plan.holdout["label"]
        signal_threshold = contract["signal_threshold"]
        initial_capital = contract["initial_capital"]
        base_cost = contract["base_cost"]
        backtest, trading = _trading_suite(
            features,
            probabilities,
            labels,
            contract["timeframe"],
            horizon=horizon,
            signal_threshold=signal_threshold,
            initial_capital=initial_capital,
            base_cost=base_cost,
            random_simulations=contract["random_simulations"],
            random_seed=contract["random_seed"],
        )
        metrics = trading["model"]
        baseline_results = trading["baselines"]
        sensitivity = run_cost_sensitivity(
            features,
            probabilities,
            labels,
            horizon,
            contract["timeframe"],
            cost_scenarios=contract["cost_scenarios"],
            signal_threshold=signal_threshold,
            initial_capital=initial_capital,
        )
        prediction_frame = plan.holdout[["timestamp", "label"]].copy()
        prediction_frame["probability_score"] = probabilities
        prediction_frame["predicted_label"] = (probabilities >= signal_threshold).astype("int8")
        _write_frame(outputs["holdout predictions"], prediction_frame)
        _write_frame(outputs["trade ledger"], backtest.trade_ledger)
        _write_frame(outputs["equity curve"], backtest.equity_curve)
        atomic_write_json(outputs["strategy metrics"], metrics, exclusive=True)
        atomic_write_json(outputs["baseline metrics"], baseline_results, exclusive=True)
        atomic_write_json(outputs["cost sensitivity"], sensitivity, exclusive=True)
        evaluation_artifact_paths = {
            "input_data_snapshot.csv": inputs["verified input snapshot"],
            "evaluation_model.json": inputs["evaluation model"],
            "model_metadata.json": inputs["model metadata"],
            "feature_columns.json": inputs["feature schema"],
            "holdout_predictions.csv": outputs["holdout predictions"],
            "trade_ledger.csv": outputs["trade ledger"],
            "equity_curve.csv": outputs["equity curve"],
            "metrics.json": outputs["strategy metrics"],
            "baseline_metrics.json": outputs["baseline metrics"],
            "cost_sensitivity.json": outputs["cost sensitivity"],
        }
        evaluation_artifact_hashes = {
            name: _artifact_hash(path, name) for name, path in evaluation_artifact_paths.items()
        }
        atomic_write_json(
            outputs["evaluation manifest"],
            {
                **manifest,
                "holdout_evaluation_claim_status": "claimed_at_manifest_write",
                "evaluation_artifacts_status": "complete",
                "evaluation_artifact_hashes": evaluation_artifact_hashes,
                "strategy_metrics": metrics,
                "baseline_metrics": baseline_results,
                "cost_sensitivity": sensitivity,
                "evaluated_at_utc": utc_now_iso(),
            },
            exclusive=True,
        )
        update_holdout_claim(claim_path, "completed")
        return {"metrics": metrics, "baselines": baseline_results, "cost_sensitivity": sensitivity}
    except Exception as exc:
        failure = (
            exc
            if isinstance(exc, CryptoAIError)
            else ArtifactError(f"Final holdout evaluation failed after claim creation: {exc}")
        )
        try:
            update_holdout_claim(claim_path, "failed", error=str(failure))
        except ArtifactError as claim_error:
            logger.error(
                "Unable to mark irreversible holdout claim failed; claimed state remains: %s",
                claim_error,
            )
        if failure is exc:
            raise
        raise failure from exc


def train_versioned_production_model(
    evaluation_run_id: str,
    *,
    symbol: str = settings.SYMBOL,
    timeframe: str = settings.TIMEFRAME,
) -> Path:
    """Train a versioned production model after a completed evaluation is accepted."""
    authorization = _load_completed_evaluation_authorization(evaluation_run_id)
    bundle = load_prepared_dataset_bundle(symbol, timeframe)
    _verify_production_compatibility(authorization, bundle, symbol, timeframe)

    labeled = bundle.labeled
    features = list(bundle.feature_columns)
    model = train_production_model(labeled, features, settings.XGBOOST_PARAMS)
    _verified_hash(
        bundle.manifest_path,
        bundle.manifest_sha256,
        "prepared dataset manifest after production fitting",
    )
    git = git_metadata(settings.BASE_DIR)
    version = generate_run_id(symbol, timeframe, str(git["git_commit"]))
    version_directory = create_run_directory(settings.PRODUCTION_DIR / "versions", version)
    save_xgboost_model(model, version_directory / "model.json")
    atomic_write_json(version_directory / "feature_columns.json", {"feature_columns": features})
    prepared_manifest_hash = bundle.manifest_sha256
    prepared_manifest_copy = version_directory / "prepared_dataset_manifest.json"
    _write_verified_bytes(
        prepared_manifest_copy,
        bundle.manifest_bytes,
        prepared_manifest_hash,
    )
    _verified_hash(
        bundle.manifest_path,
        prepared_manifest_hash,
        "prepared dataset manifest before production manifest commit",
    )
    atomic_write_json(
        version_directory / "manifest.json",
        {
            "model_version": version,
            "model_type": "XGBClassifier",
            "training_start": labeled["timestamp"].iloc[0],
            "training_end": labeled["timestamp"].iloc[-1],
            "training_row_count": len(labeled),
            "feature_columns": features,
            "feature_schema_hash": feature_schema_hash(features),
            "model_parameters": settings.XGBOOST_PARAMS,
            "prediction_horizon": bundle.manifest["prediction_horizon"],
            "label_threshold": bundle.manifest["minimum_required_return"],
            "signal_threshold": settings.SIGNAL_THRESHOLD,
            "data_hash": bundle.source_snapshot_sha256,
            "immutable_snapshot_path": bundle.source_snapshot_path,
            "prepared_dataset_manifest_path": bundle.manifest_path,
            "prepared_dataset_manifest_copy": prepared_manifest_copy,
            "prepared_dataset_manifest_sha256": prepared_manifest_hash,
            "feature_file_path": bundle.feature_path,
            "feature_file_sha256": bundle.feature_sha256,
            "labeled_file_path": bundle.labeled_path,
            "labeled_file_sha256": bundle.labeled_sha256,
            "authorized_evaluation_run_id": evaluation_run_id,
            "authorized_evaluation_manifest_path": (
                authorization["evaluation_directory"] / "evaluation_manifest.json"
            ),
            "authorized_evaluation_manifest_sha256": authorization["evaluation_manifest_sha256"],
            "authorized_evaluation_data_hash": authorization["run_manifest"]["data_hash"],
            "code_commit": git["git_commit"],
            "created_at_utc": utc_now_iso(),
            **git,
        },
    )
    return version_directory
