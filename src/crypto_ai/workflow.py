"""Phase 1 development, holdout evaluation, and production workflows."""

import json
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
from crypto_ai.costs import CostConfig, minimum_gross_return_for_net_edge
from crypto_ai.data.storage import get_raw_data_path, sha256_file, symbol_to_slug
from crypto_ai.exceptions import ArtifactError
from crypto_ai.features.build import compute_features, get_expected_feature_columns
from crypto_ai.features.dataset import load_feature_dataset, load_labeled_dataset
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


@dataclass(frozen=True)
class DevelopmentRunResult:
    """Saved development validation outputs and frozen evaluation model."""

    run_id: str
    run_directory: Path
    evaluation_directory: Path
    xgboost_metrics: dict[str, Any]
    logistic_metrics: dict[str, Any]
    development_summary: dict[str, Any]


def _paths(symbol: str, timeframe: str) -> tuple[Path, Path]:
    slug = f"{symbol_to_slug(symbol)}_{timeframe}"
    return (
        settings.DATA_INTERIM_DIR / f"{slug}_features.csv",
        settings.DATA_PROCESSED_DIR / f"{slug}_labeled.csv",
    )


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
    if path.exists():
        raise ArtifactError(f"Artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")


def _raw_snapshot(symbol: str, timeframe: str) -> tuple[Path, str]:
    latest = get_raw_data_path(symbol, timeframe)
    if not latest.exists():
        raise ArtifactError("Raw latest data is missing; run fetch and prepare first")
    digest = sha256_file(latest)
    snapshot = (
        settings.DATA_RAW_SNAPSHOTS_DIR / f"{symbol_to_slug(symbol)}_{timeframe}" / f"{digest}.csv"
    )
    if not snapshot.exists() or sha256_file(snapshot) != digest:
        raise ArtifactError("Matching immutable raw snapshot is missing or corrupt")
    return snapshot, digest


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


def run_development_validation(
    symbol: str = settings.SYMBOL,
    timeframe: str = settings.TIMEFRAME,
) -> DevelopmentRunResult:
    """Run development-only walk-forward validation and freeze an evaluation model."""
    feature_path, labeled_path = _paths(symbol, timeframe)
    features = load_feature_dataset(feature_path)
    labeled = load_labeled_dataset(labeled_path)
    plan = create_split_plan(labeled)
    feature_columns = get_expected_feature_columns()
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
    snapshot, data_hash = _raw_snapshot(symbol, timeframe)
    copy_verified_snapshot(snapshot, evaluation_directory / "input_data_snapshot.csv", data_hash)
    save_xgboost_model(evaluation_model, evaluation_directory / "evaluation_model.json")

    _write_frame(run_directory / "oof_predictions.csv", xgb.predictions)
    _write_frame(
        run_directory / "feature_importance.csv",
        feature_importance_frame(evaluation_model, feature_columns),
    )
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

    minimum_return = minimum_gross_return_for_net_edge(
        settings.TAKER_FEE_RATE,
        settings.SLIPPAGE_BPS_PER_SIDE,
        settings.HALF_SPREAD_BPS_PER_SIDE,
        settings.MIN_EDGE_BPS,
    )
    config = {
        "feature_columns": feature_columns,
        "feature_schema_hash": feature_schema_hash(feature_columns),
        "feature_configuration": _feature_configuration(),
        "prediction_horizon": settings.PREDICTION_HORIZON,
        "label_lookahead_rows": settings.LABEL_LOOKAHEAD_ROWS,
        "minimum_required_return": minimum_return,
        "label_definition": "gross open[t+H+1] / open[t+1] - 1 > minimum_required_return",
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
            "fee_rate": settings.TAKER_FEE_RATE,
            "slippage_bps_per_side": settings.SLIPPAGE_BPS_PER_SIDE,
            "half_spread_bps_per_side": settings.HALF_SPREAD_BPS_PER_SIDE,
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
            "prediction_horizon": settings.PREDICTION_HORIZON,
            "label_threshold": minimum_return,
            "signal_threshold": settings.SIGNAL_THRESHOLD,
            "data_hash": data_hash,
            "code_commit": git["git_commit"],
            "created_at_utc": utc_now_iso(),
        },
    )

    manifest = {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        **environment_metadata(),
        **git,
        "random_seed": settings.RANDOM_SEED,
        "exchange": settings.EXCHANGE_ID,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_path": labeled_path,
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
        "warnings": [
            *(f"XGBoost: {warning}" for warning in xgb.warnings),
            *(f"Logistic regression: {warning}" for warning in logistic.warnings),
        ],
        **config,
    }
    atomic_write_json(run_directory / "manifest.json", manifest)
    atomic_write_json(evaluation_directory / "development_manifest.json", manifest)
    report = (
        f"# Development Report — {run_id}\n\n"
        "## Frozen partition\n\n"
        f"- Development rows: {len(plan.development)}\n"
        f"- Boundary-purge rows: {len(plan.boundary_purge)}\n"
        f"- Untouched holdout rows: {len(plan.holdout)}\n\n"
        "## Model comparison\n\n"
        "| Model | Accuracy | Balanced accuracy | PR-AUC | Log loss | Brier |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |\n"
        f"| XGBoost | {xgb.aggregate_metrics['accuracy']:.6f} | "
        f"{xgb.aggregate_metrics['balanced_accuracy']:.6f} | "
        f"{xgb.aggregate_metrics['pr_auc']} | {xgb.aggregate_metrics['log_loss']:.6f} | "
        f"{xgb.aggregate_metrics['brier_score']:.6f} |\n"
        f"| Logistic regression | {logistic.aggregate_metrics['accuracy']:.6f} | "
        f"{logistic.aggregate_metrics['balanced_accuracy']:.6f} | "
        f"{logistic.aggregate_metrics['pr_auc']} | "
        f"{logistic.aggregate_metrics['log_loss']:.6f} | "
        f"{logistic.aggregate_metrics['brier_score']:.6f} |\n\n"
        "Per-fold metrics, out-of-fold predictions, development trading baselines, and global "
        "feature importances are saved beside this report.\n\n"
        "## Limitations\n\n"
        "Historical OHLCV is the only input; fills and costs are modeled; the random baseline "
        "is descriptive; global feature importance does not explain individual signals; and "
        "research results do not guarantee live performance. The final holdout remains untouched.\n"
    )
    (run_directory / "development_report.md").write_text(report, encoding="utf-8")
    (run_directory / "logs.txt").write_text(
        f"{utc_now_iso()} Development validation completed; final holdout not evaluated.\n",
        encoding="utf-8",
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
    run_directory = settings.RUNS_DIR / run_id
    evaluation_directory = settings.EVALUATIONS_DIR / run_id
    if not run_directory.exists() or not evaluation_directory.exists():
        raise ArtifactError(f"Unknown development run: {run_id}")
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    model_path = evaluation_directory / "evaluation_model.json"
    metadata_path = evaluation_directory / "model_metadata.json"
    feature_schema_path = evaluation_directory / "feature_columns.json"
    snapshot_path = evaluation_directory / "input_data_snapshot.csv"
    if (
        not snapshot_path.exists()
        or sha256_file(snapshot_path) != manifest["data_hash"]
        or not model_path.exists()
        or not metadata_path.exists()
        or not feature_schema_path.exists()
    ):
        raise ArtifactError("Holdout preflight failed: frozen model or data snapshot is invalid")
    model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_schema = json.loads(feature_schema_path.read_text(encoding="utf-8"))
    feature_columns = list(saved_schema["feature_columns"])
    if (
        config["feature_configuration"] != _feature_configuration()
        or int(config["prediction_horizon"]) != settings.PREDICTION_HORIZON
        or int(config["label_lookahead_rows"]) != settings.LABEL_LOOKAHEAD_ROWS
        or feature_columns != config["feature_columns"]
        or feature_schema_hash(feature_columns) != config["feature_schema_hash"]
        or model_metadata["feature_schema_hash"] != config["feature_schema_hash"]
        or model_metadata["data_hash"] != manifest["data_hash"]
        or pd.Timestamp(model_metadata["training_exit_end"])
        >= pd.Timestamp(model_metadata["holdout_start"])
    ):
        raise ArtifactError("Holdout preflight failed: frozen configuration is inconsistent")

    claim_path = claim_holdout_evaluation(run_directory)
    try:
        raw = pd.read_csv(snapshot_path)
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        for column in settings.RAW_COLUMNS[1:]:
            raw[column] = raw[column].astype("float64")
        features = compute_features(raw)
        horizon = int(config["prediction_horizon"])
        labeled = add_labels(
            features,
            horizon,
            float(config["minimum_required_return"]),
        )
        walk_forward = config["walk_forward_configuration"]
        plan = create_split_plan(
            labeled,
            holdout_ratio=float(config["final_holdout_ratio"]),
            label_lookahead_rows=int(config["label_lookahead_rows"]),
            n_splits=int(walk_forward["n_splits"]),
            test_ratio=float(walk_forward["test_ratio"]),
            gap_rows=int(walk_forward["gap_rows"]),
        )
        if plan.holdout["timestamp"].iloc[0] != pd.Timestamp(
            model_metadata["holdout_start"]
        ) or plan.development["exit_timestamp"].max() != pd.Timestamp(
            model_metadata["training_exit_end"]
        ):
            raise ArtifactError("Recreated holdout boundaries do not match the frozen run")
        model = load_xgboost_model(model_path)
        probabilities = pd.Series(
            model.predict_proba(plan.holdout[feature_columns])[:, 1],
            index=plan.holdout.index,
            name="probability_score",
        )
        labels = plan.holdout["label"]
        signal_threshold = float(config["signal_threshold"])
        initial_capital = float(config["initial_capital"])
        base_cost = CostConfig(**config["base_cost"])
        backtest, trading = _trading_suite(
            features,
            probabilities,
            labels,
            manifest["timeframe"],
            horizon=horizon,
            signal_threshold=signal_threshold,
            initial_capital=initial_capital,
            base_cost=base_cost,
            random_simulations=int(config["random_baseline_simulations"]),
            random_seed=int(config["random_seed"]),
        )
        metrics = trading["model"]
        baseline_results = trading["baselines"]
        sensitivity = run_cost_sensitivity(
            features,
            probabilities,
            labels,
            horizon,
            manifest["timeframe"],
            cost_scenarios=config["cost_scenarios"],
            signal_threshold=signal_threshold,
            initial_capital=initial_capital,
        )
        prediction_frame = plan.holdout[["timestamp", "label"]].copy()
        prediction_frame["probability_score"] = probabilities
        prediction_frame["predicted_label"] = (probabilities >= signal_threshold).astype("int8")
        _write_frame(evaluation_directory / "holdout_predictions.csv", prediction_frame)
        _write_frame(evaluation_directory / "trade_ledger.csv", backtest.trade_ledger)
        _write_frame(evaluation_directory / "equity_curve.csv", backtest.equity_curve)
        atomic_write_json(evaluation_directory / "metrics.json", metrics, exclusive=True)
        atomic_write_json(
            evaluation_directory / "baseline_metrics.json", baseline_results, exclusive=True
        )
        atomic_write_json(
            evaluation_directory / "cost_sensitivity.json", sensitivity, exclusive=True
        )
        atomic_write_json(
            evaluation_directory / "evaluation_manifest.json",
            {
                **manifest,
                "holdout_evaluation_claim_status": "completed",
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
        update_holdout_claim(claim_path, "failed", error=str(exc))
        raise


def train_versioned_production_model(
    symbol: str = settings.SYMBOL,
    timeframe: str = settings.TIMEFRAME,
) -> Path:
    """Train and save a new immutable production model version."""
    _, labeled_path = _paths(symbol, timeframe)
    labeled = load_labeled_dataset(labeled_path)
    features = get_expected_feature_columns()
    model = train_production_model(labeled, features, settings.XGBOOST_PARAMS)
    git = git_metadata(settings.BASE_DIR)
    version = generate_run_id(symbol, timeframe, str(git["git_commit"]))
    version_directory = create_run_directory(settings.PRODUCTION_DIR / "versions", version)
    save_xgboost_model(model, version_directory / "model.json")
    atomic_write_json(version_directory / "feature_columns.json", {"feature_columns": features})
    snapshot, data_hash = _raw_snapshot(symbol, timeframe)
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
            "prediction_horizon": settings.PREDICTION_HORIZON,
            "label_threshold": minimum_gross_return_for_net_edge(
                settings.TAKER_FEE_RATE,
                settings.SLIPPAGE_BPS_PER_SIDE,
                settings.HALF_SPREAD_BPS_PER_SIDE,
                settings.MIN_EDGE_BPS,
            ),
            "signal_threshold": settings.SIGNAL_THRESHOLD,
            "data_hash": data_hash,
            "immutable_snapshot_path": snapshot,
            "code_commit": git["git_commit"],
            "created_at_utc": utc_now_iso(),
            **git,
        },
    )
    return version_directory
