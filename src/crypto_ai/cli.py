"""Command-line entry point for KrypX."""

import argparse
import logging
from collections.abc import Sequence

from crypto_ai import __version__
from crypto_ai.config import settings
from crypto_ai.data.storage import load_or_update_ohlcv
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.features.dataset import prepare_datasets
from crypto_ai.logging_config import configure_logging
from crypto_ai.workflow import (
    evaluate_final_holdout,
    run_development_validation,
    train_versioned_production_model,
)

logger = logging.getLogger(__name__)


def _metric_text(value: object, *, percent: bool = False) -> str:
    """Format an optional numeric metric for compact console tables."""
    if value is None:
        return "n/a"
    numeric = float(value)
    return f"{numeric:.2%}" if percent else f"{numeric:.4f}"


def _run_fetch(args: argparse.Namespace) -> int:
    """Fetch, validate, and persist the requested market data."""
    configure_logging()
    try:
        result = load_or_update_ohlcv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            lookback_days=args.lookback_days,
        )
    except CryptoAIError as exc:
        logger.error("Market-data update failed: %s", exc)
        return 1

    first_timestamp = result.data["timestamp"].iloc[0]
    last_timestamp = result.data["timestamp"].iloc[-1]
    print(
        f"Stored {len(result.data)} closed candles from {first_timestamp} through {last_timestamp}"
    )
    print(f"Latest data: {result.latest_path}")
    print(f"Immutable snapshot: {result.snapshot_path}")
    print(f"SHA-256: {result.sha256}")
    return 0


def _run_prepare(args: argparse.Namespace) -> int:
    """Build inference-ready features and executable training labels."""
    configure_logging()
    try:
        result = prepare_datasets(symbol=args.symbol, timeframe=args.timeframe)
    except CryptoAIError as exc:
        logger.error("Dataset preparation failed: %s", exc)
        return 1

    first_timestamp = result.features["timestamp"].iloc[0]
    last_timestamp = result.features["timestamp"].iloc[-1]
    print(
        f"Prepared {len(result.features)} inference rows from {first_timestamp} "
        f"through {last_timestamp}"
    )
    print(f"Labeled decision rows: {len(result.labeled)}")
    print(f"Feature count: {len(result.feature_columns)}")
    print(f"Warm-up rows removed: {result.warmup_rows_removed}")
    print(f"Unrealizable tail rows removed: {result.unlabeled_rows_removed}")
    print(f"Raw snapshot: {result.source_snapshot_path}")
    print(f"Raw SHA-256: {result.source_sha256}")
    print(f"Inference features: {result.feature_path}")
    print(f"Labeled dataset: {result.labeled_path}")
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    """Run development-only validation and freeze an evaluation model."""
    configure_logging()
    try:
        result = run_development_validation(args.symbol, args.timeframe)
    except CryptoAIError as exc:
        logger.error("Development validation failed: %s", exc)
        return 1
    print(f"Development run: {result.run_id}")
    print(f"Run artifacts: {result.run_directory}")
    print(f"Evaluation model: {result.evaluation_directory / 'evaluation_model.json'}")
    summary = result.development_summary
    print(
        "Development period: "
        f"{summary['development_start']} through {summary['development_end']} "
        f"({summary['development_rows']} rows)"
    )
    print(
        "Boundary purge: "
        f"{summary['boundary_purge_start']} through {summary['boundary_purge_end']} "
        f"({summary['boundary_purge_rows']} rows)"
    )
    print(
        "Frozen holdout: "
        f"{summary['holdout_start']} through {summary['holdout_end']} "
        f"({summary['holdout_rows']} rows; outcomes not inspected)"
    )
    print(
        "Development label distribution: "
        f"positive={summary['development_positive_label_rate']:.2%}, "
        f"negative={1 - summary['development_positive_label_rate']:.2%}"
    )
    print("Model                  Accuracy  Balanced  PR-AUC   Log loss  Brier")
    for name, metrics in (
        ("XGBoost", result.xgboost_metrics),
        ("Logistic regression", result.logistic_metrics),
    ):
        print(
            f"{name:<22} "
            f"{_metric_text(metrics['accuracy']):>8}  "
            f"{_metric_text(metrics['balanced_accuracy']):>8}  "
            f"{_metric_text(metrics['pr_auc']):>7}  "
            f"{_metric_text(metrics['log_loss']):>8}  "
            f"{_metric_text(metrics['brier_score']):>6}"
        )
    print("Final holdout has not been evaluated.")
    return 0


def _run_evaluate_holdout(args: argparse.Namespace) -> int:
    """Evaluate a frozen model on its one-time final holdout."""
    configure_logging()
    print("You are evaluating the final holdout.")
    print("Do not use these results for iterative model tuning.")
    try:
        result = evaluate_final_holdout(args.run_id)
    except CryptoAIError as exc:
        logger.error("Final holdout evaluation failed: %s", exc)
        return 1
    rows = {"XGBoost": result["metrics"], **result["baselines"]}
    print("Strategy               Return    Sharpe   Max DD    Exposure  Trades")
    for name, metrics in rows.items():
        if name == "random":
            continue
        display_name = name if name == "XGBoost" else name.replace("_", " ").title()
        print(
            f"{display_name:<22} "
            f"{_metric_text(metrics['total_return'], percent=True):>8}  "
            f"{_metric_text(metrics['sharpe_ratio']):>7}  "
            f"{_metric_text(metrics['maximum_drawdown'], percent=True):>8}  "
            f"{_metric_text(metrics['market_exposure'], percent=True):>8}  "
            f"{metrics['num_trades']:>6}"
        )
    if "random" in result["baselines"]:
        random_result = result["baselines"]["random"]
        print(
            "Random exposure median return: "
            f"{_metric_text(random_result['total_return']['median'], percent=True)} "
            f"across {random_result['simulations']} simulations"
        )
    return 0


def _run_train_production(args: argparse.Namespace) -> int:
    """Train a separately versioned production model on all labeled rows."""
    configure_logging()
    try:
        path = train_versioned_production_model(args.symbol, args.timeframe)
    except CryptoAIError as exc:
        logger.error("Production training failed: %s", exc)
        return 1
    print(f"Production model version: {path}")
    print("The new version was not automatically activated.")
    return 0


def _run_development(args: argparse.Namespace) -> int:
    """Fetch, prepare, and validate development data without touching holdout results."""
    if _run_fetch(args) != 0 or _run_prepare(args) != 0:
        return 1
    return _run_validate(args)


def build_parser() -> argparse.ArgumentParser:
    """Build the Phase 1 command-line parser."""
    parser = argparse.ArgumentParser(
        prog="krypx",
        description="KrypX Phase 1 cryptocurrency research pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="fetch, validate, and store closed OHLCV candles",
    )
    fetch_parser.add_argument("--symbol", default=settings.SYMBOL)
    fetch_parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    fetch_parser.add_argument("--lookback-days", type=int, default=settings.LOOKBACK_DAYS)
    fetch_parser.set_defaults(handler=_run_fetch)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="build inference features and executable training labels",
    )
    prepare_parser.add_argument("--symbol", default=settings.SYMBOL)
    prepare_parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    prepare_parser.set_defaults(handler=_run_prepare)

    validate_parser = subparsers.add_parser(
        "validate", help="run development-only walk-forward validation"
    )
    validate_parser.add_argument("--symbol", default=settings.SYMBOL)
    validate_parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    validate_parser.set_defaults(handler=_run_validate)

    holdout_parser = subparsers.add_parser(
        "evaluate-holdout", help="evaluate a frozen model on its one-time final holdout"
    )
    holdout_parser.add_argument("--run-id", required=True)
    holdout_parser.set_defaults(handler=_run_evaluate_holdout)

    production_parser = subparsers.add_parser(
        "train-production", help="train a new versioned production model"
    )
    production_parser.add_argument("--symbol", default=settings.SYMBOL)
    production_parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    production_parser.set_defaults(handler=_run_train_production)

    development_parser = subparsers.add_parser(
        "run-development", help="run fetch, prepare, and development validation"
    )
    development_parser.add_argument("--symbol", default=settings.SYMBOL)
    development_parser.add_argument("--timeframe", default=settings.TIMEFRAME)
    development_parser.add_argument("--lookback-days", type=int, default=settings.LOOKBACK_DAYS)
    development_parser.set_defaults(handler=_run_development)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
