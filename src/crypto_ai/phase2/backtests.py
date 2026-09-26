"""Verified synthetic-only Milestone 7 development backtests and report buffers.

The public runner accepts only an immutable Milestone 6 run identifier. It reads
and replays that publication and its prepared Milestone 5 parent before using the
unchanged Phase 1 backtesting arithmetic. No network, model, or real-data seam is
available here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai.backtesting.baselines import buy_and_hold_backtest
from crypto_ai.backtesting.engine import BacktestResult, run_backtest
from crypto_ai.backtesting.metrics import calculate_backtest_metrics
from crypto_ai.costs import CostConfig
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2 import dataset as datasets
from crypto_ai.phase2 import experiments
from crypto_ai.sentiment.canonical import canonicalize, sha256_bytes
from crypto_ai.sentiment.contracts import format_utc_timestamp
from crypto_ai.sentiment.storage import (
    _open_directory_at,
    _open_directory_path,
    _read_regular_file_at_once,
    _stat_identity,
)

SPECIFICATION_ID = "phase2-milestone7-offline-development-backtests-v1"
HORIZON = 4
TIMEFRAME = "1h"
INITIAL_CAPITAL = 10_000.0
THRESHOLD = 0.5
RANDOM_SEED = 42
RANDOM_SIMULATIONS = 1_000
SCENARIOS = {
    "low": CostConfig(0.001, 1.0, 0.5),
    "base": CostConfig(0.001, 2.0, 1.0),
    "high": CostConfig(0.001, 5.0, 2.0),
}
SCENARIO_ORDER = ("low", "base", "high")
METRIC_KEYS = (
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "maximum_drawdown",
    "maximum_drawdown_duration",
    "calmar_ratio",
    "profit_factor",
    "num_trades",
    "win_rate",
    "market_exposure",
    "turnover",
    "average_holding_period",
    "total_estimated_costs",
)
PAYLOAD_FILES = (
    "strategy_metrics.json",
    "cost_sensitivity.json",
    "baseline_metrics.json",
    "ablation_report.json",
    "development_report.md",
)


class Phase2BacktestError(CryptoAIError):
    """The synthetic Milestone 7 backtest contract failed."""


class BacktestInputError(Phase2BacktestError):
    """A prediction, market context, or execution parameter is invalid."""


class BacktestIntegrityError(Phase2BacktestError):
    """Verified evidence or deterministic report replay did not match."""


class BacktestAuthorizationError(Phase2BacktestError):
    """An unsupported execution seam was requested."""


@dataclass(frozen=True, slots=True)
class BacktestBundle:
    """Exact candidate report buffers and parent-bound publication metadata."""

    files: tuple[tuple[str, bytes], ...]
    metadata: dict[str, Any]

    def json(self, filename: str) -> Any:
        if filename not in PAYLOAD_FILES[:-1]:
            raise BacktestInputError("unknown JSON report payload")
        try:
            return datasets._json(dict(self.files)[filename])
        except (CryptoAIError, KeyError, TypeError, ValueError) as exc:
            raise BacktestIntegrityError("invalid JSON report buffer") from exc


@dataclass(frozen=True, slots=True)
class _VerifiedInputs:
    source: experiments.ExperimentArtifact
    prepared: datasets.PreparedDatasetArtifact
    source_manifest_bytes: bytes
    market: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    labeled: pd.DataFrame
    start: int
    end: int
    decision_identity: tuple[dict[str, Any], ...]


def _timestamp(value: Any) -> str:
    if not isinstance(value, pd.Timestamp) or value.tz is None:
        raise BacktestIntegrityError("timestamp must be timezone-aware")
    return format_utc_timestamp(value.to_pydatetime())


def _market_invariants(market: pd.DataFrame) -> None:
    if (
        type(market) is not pd.DataFrame
        or market.empty
        or not market.index.equals(pd.RangeIndex(len(market)))
        or "timestamp" not in market
        or "open" not in market
        or str(market["timestamp"].dtype) != "datetime64[ns, UTC]"
        or str(market["open"].dtype) != "float64"
    ):
        raise BacktestInputError("complete original-ordinal hourly market context required")
    times = market["timestamp"]
    prices = market["open"].to_numpy(dtype=np.float64)
    if (
        times.isna().any()
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not np.isfinite(prices).all()
        or (prices <= 0.0).any()
        or not (times.diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    ):
        raise BacktestInputError("market opens must be positive, finite, and hourly-contiguous")


def _scores_invariants(
    market: pd.DataFrame, scores: pd.Series, labels: pd.Series | None, start: int, end: int
) -> None:
    if (
        type(scores) is not pd.Series
        or scores.empty
        or not scores.index.is_unique
        or any(type(value) not in (int, np.int64) for value in scores.index)
        or list(scores.index) != sorted(scores.index)
        or type(start) is not int
        or type(end) is not int
        or start < 1
        or end >= len(market)
        or start >= end
    ):
        raise BacktestInputError("ordered original-ordinal decisions and window required")
    values = scores.to_numpy()
    if (
        values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
        or ((values < 0.0) | (values > 1.0)).any()
        or scores.index[0] + 1 != start
        or scores.index[-1] + HORIZON + 1 != end
    ):
        raise BacktestInputError("finite probabilities and exact common window required")
    if labels is not None and (
        type(labels) is not pd.Series
        or not labels.index.equals(scores.index)
        or labels.to_numpy().dtype.kind not in "iu"
        or not np.isin(labels.to_numpy(), (0, 1)).all()
    ):
        raise BacktestInputError("binary labels must exactly match OOF decision ordinals")
    for ordinal in scores.index:
        if ordinal < 0 or ordinal + HORIZON + 1 >= len(market):
            raise BacktestInputError("OOF decision lacks its next-open entry or final exit")


def _simulate(
    market_df: pd.DataFrame,
    scores: pd.Series,
    labels: pd.Series | None,
    cost_config: CostConfig,
    *,
    expected_start: int,
    expected_end: int,
) -> BacktestResult:
    """Call unchanged Phase 1 engine only after strict M7 ordinal/window checks."""
    try:
        _market_invariants(market_df)
        _scores_invariants(market_df, scores, labels, expected_start, expected_end)
        if type(cost_config) is not CostConfig or cost_config not in SCENARIOS.values():
            raise BacktestInputError("only frozen cost scenarios are permitted")
        result = run_backtest(
            market_df,
            scores,
            labels,
            HORIZON,
            TIMEFRAME,
            THRESHOLD,
            INITIAL_CAPITAL,
            cost_config,
        )
        if not result.equity_curve.index.equals(market_df.index[expected_start : expected_end + 1]):
            raise BacktestIntegrityError("Phase 1 engine changed the common performance window")
        return result
    except Phase2BacktestError:
        raise
    except (
        CryptoAIError,
        ValueError,
        TypeError,
        IndexError,
        AttributeError,
        OverflowError,
        RuntimeError,
    ) as exc:
        raise BacktestIntegrityError("synthetic strategy backtest failed") from exc


def _outer_source_manifest(
    store: experiments.ExperimentStore, run_id: str, source: experiments.ExperimentArtifact
) -> bytes:
    """Capture the source completion manifest through its pinned directory inode."""
    parent = _open_directory_path(
        store.root, description="source experiment runs", expected_identity=store._identity
    )
    try:
        descriptor = _open_directory_at(parent, run_id, description="source experiment run")
        try:
            identity = _stat_identity(os.fstat(descriptor))
            raw, _ = _read_regular_file_at_once(
                descriptor, "manifest.json", description="source experiment manifest"
            )
            store._check_attachment(run_id, identity, dict(source.files))
            return raw
        finally:
            os.close(descriptor)
    finally:
        os.close(parent)


def _verified_inputs(store: experiments.ExperimentStore, run_id: str) -> _VerifiedInputs:
    if type(store) is not experiments.ExperimentStore:
        raise BacktestAuthorizationError("exact verified synthetic experiment store required")
    try:
        source = store.get(run_id)
        source_manifest = _outer_source_manifest(store, run_id, source)
        outer = datasets._json(source_manifest)
        if (
            type(outer) is not dict
            or outer.get("metadata")
            != {"schema_version": experiments.SCHEMA, "experiment_id": source.experiment_id}
            or outer.get("run_id") != run_id
        ):
            raise BacktestIntegrityError("source outer manifest changed after verification")
        prepared = datasets.DatasetStore(store.parents).get(source.manifest["dataset_id"])
        if prepared is None or prepared.dataset_id != source.manifest["dataset_id"]:
            raise BacktestIntegrityError("verified synthetic prepared parent disappeared")
        prepared_files = dict(prepared.files)
        market = datasets._frame(prepared_files["market-price-context.json"], datasets.RAW_COLUMNS)
        _market_invariants(market)
        labeled = prepared.labeled.set_index("market_ordinal", drop=False)
        if not labeled.index.is_unique:
            raise BacktestIntegrityError("prepared decision ordinals are not unique")
        predictions = {cell: source.predictions(cell) for cell in experiments.CELLS}
        reference = predictions["A"]
        identity_fields = (
            "market_ordinal",
            "decision_at",
            "entry_timestamp",
            "exit_timestamp",
            "fold_number",
            "actual_label",
        )
        if reference.empty or reference.fold_number.nunique() != experiments.FOLD_COUNT:
            raise BacktestIntegrityError("five continuous OOF folds are required")
        for cell, frame in predictions.items():
            if not frame[list(identity_fields)].equals(reference[list(identity_fields)]):
                raise BacktestIntegrityError(f"cell {cell} has a mismatched OOF decision set")
            probabilities = frame.probability_score.to_numpy(dtype=np.float64)
            if (
                not np.isfinite(probabilities).all()
                or ((probabilities < 0.0) | (probabilities > 1.0)).any()
                or not np.array_equal(
                    (probabilities >= THRESHOLD).astype(np.int8),
                    frame.signal.to_numpy(dtype=np.int8),
                )
            ):
                raise BacktestIntegrityError("invalid OOF probabilities or frozen signals")
        ordinals = reference.market_ordinal.to_numpy(dtype=np.int64)
        if (ordinals < 0).any() or (np.diff(ordinals) <= 0).any():
            raise BacktestIntegrityError("OOF market ordinals must strictly increase")
        start, end = int(ordinals[0]) + 1, int(ordinals[-1]) + HORIZON + 1
        if end >= len(market):
            raise BacktestIntegrityError("final OOF decision lacks retained scheduled exit open")
        rows = []
        for row in reference.itertuples(index=False):
            ordinal = int(row.market_ordinal)
            if ordinal not in labeled.index:
                raise BacktestIntegrityError("OOF decision absent from prepared shared index")
            opened = market.iloc[ordinal].timestamp
            expected_decision = opened + pd.Timedelta(hours=1)
            if (
                row.decision_at != expected_decision
                or row.entry_timestamp != market.iloc[ordinal + 1].timestamp
                or row.exit_timestamp != market.iloc[ordinal + HORIZON + 1].timestamp
                or row.actual_label != int(labeled.loc[ordinal, "label"])
                or labeled.loc[ordinal, "decision_at"] != row.decision_at
                or not 1 <= row.fold_number <= experiments.FOLD_COUNT
            ):
                raise BacktestIntegrityError("OOF row and execution-price chronology differ")
            rows.append(
                {
                    "market_ordinal": ordinal,
                    "decision_at": _timestamp(row.decision_at),
                    "entry_timestamp": _timestamp(row.entry_timestamp),
                    "exit_timestamp": _timestamp(row.exit_timestamp),
                    "fold_number": int(row.fold_number),
                    "actual_label": int(row.actual_label),
                }
            )
        return _VerifiedInputs(
            source,
            prepared,
            source_manifest,
            market,
            predictions,
            labeled,
            start,
            end,
            tuple(rows),
        )
    except Phase2BacktestError:
        raise
    except (
        CryptoAIError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        AttributeError,
        OverflowError,
        RuntimeError,
    ) as exc:
        raise BacktestIntegrityError("verified synthetic parent evidence failed") from exc


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise BacktestIntegrityError("non-finite report number")
        return number
    if isinstance(value, pd.Timestamp):
        return _timestamp(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return [_scalar(item) for item in value]
    if isinstance(value, dict):
        return {key: _scalar(item) for key, item in value.items()}
    raise BacktestIntegrityError("unsupported report value")


def _undefined_metric_warnings(result: BacktestResult, metrics: dict[str, Any]) -> list[str]:
    """Explain every Phase 1 null in the M7 payload without changing Phase 1."""
    trades = result.trade_ledger
    no_trades = trades.empty
    no_wins = no_trades or not bool((trades["net_return"] > 0.0).any())
    no_losses = no_trades or not bool((trades["pnl"] < 0.0).any())
    reasons = {
        "annualized_return": "there are no open-to-open intervals or capital was exhausted",
        "annualized_volatility": "fewer than two open-to-open intervals are available",
        "sharpe_ratio": (
            "fewer than two open-to-open intervals are available"
            if result.n_intervals < 2
            else "net-return volatility is zero"
        ),
        "sortino_ratio": (
            "fewer than two open-to-open intervals are available"
            if result.n_intervals < 2
            else "downside deviation is zero"
        ),
        "calmar_ratio": (
            "annualized return is undefined"
            if metrics["annualized_return"] is None
            else "maximum drawdown is zero"
        ),
        "profit_factor": (
            "there are no completed trades and therefore no losing trades"
            if no_trades
            else "there are no losing trades"
        ),
        "win_rate": "there are no completed trades",
        "average_trade_return": "there are no completed trades",
        "median_trade_return": "there are no completed trades",
        "average_winning_return": "there are no winning trades" if no_wins else "no value exists",
        "average_losing_return": "there are no losing trades" if no_losses else "no value exists",
        "largest_winning_trade": "there are no winning trades" if no_wins else "no value exists",
        "largest_losing_trade": "there are no losing trades" if no_losses else "no value exists",
        "average_holding_period": "there are no completed trades",
        "actual_positive_label_rate": "the baseline has no actual-label series",
    }
    warnings = []
    for key, value in metrics.items():
        if value is None:
            if key not in reasons:
                raise BacktestIntegrityError(f"undefined metric {key} lacks an M7 reason")
            warnings.append(f"{key} is undefined because {reasons[key]}")
    return warnings


def _reconciliation(
    result: BacktestResult,
    timestamps_to_ordinals: dict[pd.Timestamp, int],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Publish independently checkable final-equity and transaction-mark proofs."""
    relative_tolerance = 1e-10
    absolute_tolerance_usd = 1e-8
    initial = float(result.initial_capital)
    curve = result.equity_curve
    ledger = result.trade_ledger
    final = float(curve["equity"].iloc[-1])
    interval_compounded = initial * float(np.prod(1.0 + result.interval_returns.to_numpy()))
    trade_compounded = (
        initial * float(np.prod(1.0 + ledger["net_return"].to_numpy(dtype=np.float64)))
        if not ledger.empty
        else initial
    )
    ledger_pnl = float(ledger["pnl"].sum()) if not ledger.empty else 0.0
    pnl_equity = initial + ledger_pnl
    final_differences = [
        abs(final - interval_compounded),
        abs(final - trade_compounded),
        abs(final - pnl_equity),
    ]
    if not all(
        math.isclose(final, candidate, rel_tol=relative_tolerance, abs_tol=absolute_tolerance_usd)
        for candidate in (interval_compounded, trade_compounded, pnl_equity)
    ):
        raise BacktestIntegrityError("net equity, intervals, and trade ledger do not reconcile")

    entry_fees: list[float] = []
    exit_fees: list[float] = []
    entry_execution: list[float] = []
    exit_execution: list[float] = []
    entry_curve_deductions: list[float] = []
    exit_curve_deductions: list[float] = []
    transaction_differences: list[float] = []
    for trade in ledger.to_dict("records"):
        entry_ordinal = timestamps_to_ordinals[pd.Timestamp(trade["entry_timestamp"])]
        exit_ordinal = timestamps_to_ordinals[pd.Timestamp(trade["exit_timestamp"])]
        entry_mark = float(curve.loc[entry_ordinal, "equity"])
        exit_mark = float(curve.loc[exit_ordinal, "equity"])
        entry_fee = float(trade["entry_fee"])
        exit_fee = float(trade["exit_fee"])
        entry_execution_cost = float(trade["entry_execution_cost"])
        exit_execution_cost = float(trade["exit_execution_cost"])
        entry_deduction = float(trade["equity_before_entry"]) - entry_mark
        exit_deduction = (
            float(trade["position_quantity"]) * float(trade["exit_market_price"]) - exit_mark
        )
        entry_cost = math.fsum((entry_fee, entry_execution_cost))
        exit_cost = math.fsum((exit_fee, exit_execution_cost))
        if not (
            math.isclose(
                entry_cost,
                entry_deduction,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance_usd,
            )
            and math.isclose(
                exit_cost,
                exit_deduction,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance_usd,
            )
        ):
            raise BacktestIntegrityError("ledger costs and transaction equity marks differ")
        entry_fees.append(entry_fee)
        exit_fees.append(exit_fee)
        entry_execution.append(entry_execution_cost)
        exit_execution.append(exit_execution_cost)
        entry_curve_deductions.append(entry_deduction)
        exit_curve_deductions.append(exit_deduction)
        transaction_differences.extend(
            (abs(entry_cost - entry_deduction), abs(exit_cost - exit_deduction))
        )
    ledger_total = math.fsum(entry_fees + exit_fees + entry_execution + exit_execution)
    curve_total = math.fsum(entry_curve_deductions + exit_curve_deductions)
    metric_total = float(metrics["total_estimated_costs"])
    if not (
        math.isclose(
            ledger_total, curve_total, rel_tol=relative_tolerance, abs_tol=absolute_tolerance_usd
        )
        and math.isclose(
            ledger_total, metric_total, rel_tol=relative_tolerance, abs_tol=absolute_tolerance_usd
        )
    ):
        raise BacktestIntegrityError("estimated costs do not reconcile with equity deductions")
    return {
        "relative_tolerance": relative_tolerance,
        "absolute_tolerance_usd": absolute_tolerance_usd,
        "initial_capital_usd": initial,
        "final_equity_curve_usd": final,
        "compounded_interval_returns_usd": interval_compounded,
        "compounded_trade_net_returns_usd": trade_compounded,
        "initial_capital_plus_trade_pnl_usd": pnl_equity,
        "maximum_final_equity_difference_usd": max(final_differences),
        "ledger_entry_fees_usd": math.fsum(entry_fees),
        "ledger_exit_fees_usd": math.fsum(exit_fees),
        "ledger_entry_execution_costs_usd": math.fsum(entry_execution),
        "ledger_exit_execution_costs_usd": math.fsum(exit_execution),
        "ledger_total_estimated_costs_usd": ledger_total,
        "equity_curve_entry_deductions_usd": math.fsum(entry_curve_deductions),
        "equity_curve_exit_deductions_usd": math.fsum(exit_curve_deductions),
        "equity_curve_total_cost_deductions_usd": curve_total,
        "metric_total_estimated_costs_usd": metric_total,
        "maximum_transaction_mark_cost_difference_usd": max(transaction_differences, default=0.0),
        "checks": {
            "interval_returns_compound_to_final_equity": True,
            "trade_net_returns_compound_to_final_equity": True,
            "trade_pnl_sums_to_final_equity": True,
            "entry_costs_match_equity_marks": True,
            "exit_costs_match_equity_marks": True,
            "estimated_costs_match_curve_deductions": True,
        },
        "passed": True,
        "verified": True,
    }


def _result_payload(result: BacktestResult) -> dict[str, Any]:
    """Serialize one already-reconciled Phase 1 run without changing its arithmetic."""
    curve_frame = result.equity_curve
    if curve_frame.empty:
        raise BacktestIntegrityError("a strategy omitted its common-window equity marks")
    timestamps_to_ordinals = {
        pd.Timestamp(row["timestamp"]): int(ordinal) for ordinal, row in curve_frame.iterrows()
    }
    if len(timestamps_to_ordinals) != len(curve_frame):
        raise BacktestIntegrityError("equity marks have duplicate UTC timestamps")
    final_equity = float(curve_frame["equity"].iloc[-1])
    interval_returns = result.interval_returns.to_numpy(dtype=np.float64)
    compounded = result.initial_capital * float(np.prod(1.0 + interval_returns))
    if not math.isclose(final_equity, compounded, rel_tol=1e-10, abs_tol=1e-8):
        raise BacktestIntegrityError("open-to-open net returns do not reconcile")
    if result.trade_ledger.empty:
        if not math.isclose(final_equity, result.initial_capital, rel_tol=1e-10, abs_tol=1e-8):
            raise BacktestIntegrityError("a no-trade path changed cash equity")
    else:
        net_returns = result.trade_ledger["net_return"].to_numpy(dtype=np.float64)
        trade_compounded = result.initial_capital * float(np.prod(1.0 + net_returns))
        trade_pnl = float(result.trade_ledger["pnl"].sum())
        if not (
            math.isclose(final_equity, trade_compounded, rel_tol=1e-10, abs_tol=1e-8)
            and math.isclose(
                final_equity - result.initial_capital, trade_pnl, rel_tol=1e-10, abs_tol=1e-8
            )
        ):
            raise BacktestIntegrityError("trade ledger and net equity do not reconcile")
    metrics = calculate_backtest_metrics(result, TIMEFRAME)
    if any(key not in metrics for key in METRIC_KEYS):
        raise BacktestIntegrityError("Phase 1 metric contract changed")
    metrics["warnings"] = _undefined_metric_warnings(result, metrics)
    reconciliation = _reconciliation(result, timestamps_to_ordinals, metrics)
    ledger = []
    for raw in result.trade_ledger.to_dict("records"):
        row = {key: _scalar(value) for key, value in raw.items()}
        # The unchanged Phase 1 engine labels a signal by its candle OPEN.
        # Milestone 7 reports the causally correct CLOSE while retaining that
        # original open for auditability.
        source_open = pd.Timestamp(raw["signal_timestamp"])
        row["signal_candle_open_timestamp"] = _timestamp(source_open)
        row["signal_timestamp"] = _timestamp(source_open + pd.Timedelta(hours=1))
        entry_ordinal = timestamps_to_ordinals[pd.Timestamp(raw["entry_timestamp"])]
        exit_ordinal = timestamps_to_ordinals[pd.Timestamp(raw["exit_timestamp"])]
        if exit_ordinal <= entry_ordinal or exit_ordinal - entry_ordinal != raw["holding_candles"]:
            raise BacktestIntegrityError("trade ledger lost original market ordinal alignment")
        row["signal_market_ordinal"] = entry_ordinal - 1
        row["entry_market_ordinal"] = entry_ordinal
        row["exit_market_ordinal"] = exit_ordinal
        ledger.append(row)
    curve = []
    for ordinal, row in result.equity_curve.iterrows():
        curve.append(
            {
                "market_ordinal": int(ordinal),
                "timestamp": _timestamp(row["timestamp"]),
                "equity": _scalar(row["equity"]),
                "position_open": _scalar(row["position_open"]),
                "market_exposure": (
                    None if pd.isna(row["market_exposure"]) else _scalar(row["market_exposure"])
                ),
                "period_return": (
                    None if pd.isna(row["period_return"]) else _scalar(row["period_return"])
                ),
            }
        )
    payload = {
        "metrics": _scalar(metrics),
        "reconciliation": _scalar(reconciliation),
        "trade_ledger": ledger,
        "equity_curve": curve,
    }
    canonicalize(payload)
    return payload


def _window(market: pd.DataFrame, start: int, end: int, decisions: int) -> dict[str, Any]:
    return {
        "first_open_ordinal": start,
        "final_open_ordinal": end,
        "first_open_at": _timestamp(market.iloc[start].timestamp),
        "final_open_at": _timestamp(market.iloc[end].timestamp),
        "equity_marks": end - start + 1,
        "open_to_open_intervals": end - start,
        "oof_decisions": decisions,
    }


def _result_window(result: BacktestResult, market: pd.DataFrame, start: int, end: int) -> None:
    if not result.equity_curve.index.equals(market.index[start : end + 1]):
        raise BacktestIntegrityError("a baseline used a different performance window")
    if result.n_intervals != end - start:
        raise BacktestIntegrityError("a baseline omitted an open-to-open interval")


def _baseline_scores(
    predictions: dict[str, pd.DataFrame], labeled: pd.DataFrame
) -> tuple[pd.Series, pd.Series]:
    reference = predictions["A"]
    ordinals = pd.Index(reference.market_ordinal.to_numpy(dtype=np.int64))
    if not ordinals.isin(labeled.index).all():
        raise BacktestIntegrityError("baseline features are absent from prepared decisions")
    prepared_rows = labeled.loc[ordinals]
    for field in ("ema_short", "ema_long", "return_24"):
        values = prepared_rows[field].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise BacktestIntegrityError("non-finite point-in-time baseline indicator")
    ema = pd.Series(
        (prepared_rows.ema_short.to_numpy() > prepared_rows.ema_long.to_numpy()).astype(np.float64),
        index=ordinals,
    )
    momentum = pd.Series(
        (prepared_rows.return_24.to_numpy() > 0.0).astype(np.float64), index=ordinals
    )
    return ema, momentum


def _summary(values: list[float | None]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(finite):
        return {"median": None, "p05": None, "p95": None}
    if not np.isfinite(finite).all():
        raise BacktestIntegrityError("non-finite random baseline metric")
    return {
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def _random_summaries(
    market: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    labels: pd.Series,
    model_metrics: dict[str, dict[str, dict[str, Any]]],
    start: int,
    end: int,
    *,
    simulations: int,
) -> dict[str, Any]:
    if type(simulations) is not int or not 1 <= simulations <= RANDOM_SIMULATIONS:
        raise BacktestInputError("random simulation count is outside the bounded fixture range")
    output = {}
    for cell in experiments.CELLS:
        frame = predictions[cell]
        ordinals = pd.Index(frame.market_ordinal.to_numpy(dtype=np.int64))
        positive = frame.probability_score.to_numpy(dtype=np.float64) >= THRESHOLD
        probability = float(positive.mean())
        samples: dict[str, dict[str, list[float | None]]] = {
            scenario: {key: [] for key in ("total_return", "sharpe_ratio", "maximum_drawdown")}
            for scenario in SCENARIO_ORDER
        }
        for simulation in range(simulations):
            rng = np.random.default_rng(RANDOM_SEED + simulation)
            draws = pd.Series(
                rng.binomial(1, probability, len(ordinals)).astype(np.float64), index=ordinals
            )
            for scenario in SCENARIO_ORDER:
                result = _simulate(
                    market,
                    draws,
                    labels,
                    SCENARIOS[scenario],
                    expected_start=start,
                    expected_end=end,
                )
                metrics = calculate_backtest_metrics(result, TIMEFRAME)
                for key in samples[scenario]:
                    samples[scenario][key].append(metrics[key])
        output[cell] = {}
        for scenario in SCENARIO_ORDER:
            model_return = model_metrics[cell][scenario]["total_return"]
            returns = samples[scenario]["total_return"]
            if any(value is None for value in returns):
                raise BacktestIntegrityError("random total return must be defined")
            output[cell][scenario] = {
                "simulations": simulations,
                "seed_base": RANDOM_SEED,
                "signal_probability": probability,
                "total_return": _summary(returns),
                "sharpe_ratio": _summary(samples[scenario]["sharpe_ratio"]),
                "maximum_drawdown": _summary(samples[scenario]["maximum_drawdown"]),
                "fraction_return_at_least_model": float(
                    np.mean(np.asarray(returns, dtype=np.float64) >= model_return)
                ),
            }
    return output


def _delta_metrics(augmented: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for key in METRIC_KEYS:
        left, right = augmented[key], control[key]
        if left is None or right is None:
            deltas[key] = None
        elif key in ("total_return", "annualized_return"):
            deltas[key] = 100.0 * (left - right)
        else:
            deltas[key] = left - right
    return deltas


def _comparison_metrics(augmented: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    deltas = _delta_metrics(augmented, control)
    return {
        key: {
            "control_metric": control[key],
            "augmented_metric": augmented[key],
            "delta": deltas[key],
        }
        for key in METRIC_KEYS
    }


def _fold_consistency(values: list[float]) -> dict[str, Any]:
    positives = [value for value in values if value > 0.0]
    total = math.fsum(positives)
    return {
        "positive_fold_count": len(positives),
        "median_return_delta_percentage_points": float(np.median(values)),
        "largest_positive_contribution_share": max(positives) / total if total > 0 else None,
    }


def _calculate(
    market: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    labeled: pd.DataFrame,
    *,
    start: int,
    end: int,
    random_simulations: int = RANDOM_SIMULATIONS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Pure deterministic arithmetic on already-verified synthetic parent frames.

    This private function exists for independent small fixture tests. The public
    engine obtains all three inputs exclusively from verified M5/M6 stores and
    always runs the exact frozen 1,000-simulation random baseline.
    """
    try:
        _market_invariants(market)
        if type(predictions) is not dict or set(predictions) != set(experiments.CELLS):
            raise BacktestInputError("exact A/B/C/D OOF predictions required")
        if type(labeled) is not pd.DataFrame or not labeled.index.is_unique:
            raise BacktestInputError("unique verified prepared decision rows required")
        reference = predictions["A"]
        if type(reference) is not pd.DataFrame or reference.empty:
            raise BacktestInputError("nonempty OOF decisions required")
        identity = [
            "market_ordinal",
            "decision_at",
            "entry_timestamp",
            "exit_timestamp",
            "fold_number",
            "actual_label",
        ]
        for cell, frame in predictions.items():
            if (
                type(frame) is not pd.DataFrame
                or not set(identity + ["probability_score", "signal"]) <= set(frame.columns)
                or not frame[identity].equals(reference[identity])
                or str(frame.probability_score.dtype) != "float64"
                or frame.signal.to_numpy().dtype.kind not in "iu"
                or not np.isfinite(frame.probability_score.to_numpy(dtype=np.float64)).all()
                or not np.array_equal(
                    (frame.probability_score.to_numpy(dtype=np.float64) >= THRESHOLD).astype(
                        np.int8
                    ),
                    frame.signal.to_numpy(dtype=np.int8),
                )
            ):
                raise BacktestInputError(f"cell {cell} does not share frozen OOF identities")
        ordinals = pd.Index(reference.market_ordinal.to_numpy(dtype=np.int64))
        labels = pd.Series(reference.actual_label.to_numpy(dtype=np.int8), index=ordinals)
        _scores_invariants(
            market,
            pd.Series(reference.probability_score.to_numpy(dtype=np.float64), index=ordinals),
            labels,
            start,
            end,
        )
        common_window = _window(market, start, end, len(reference))
        strategy: dict[str, Any] = {
            "schema_version": "phase2-strategy-metrics-v1",
            "synthetic": True,
            "scope": "engineering_only_no_research_outcome",
            "common_window": common_window,
            "cells": {},
        }
        model_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        trade_schedules: dict[str, tuple[tuple[str, str], ...]] = {}
        for cell in experiments.CELLS:
            frame = predictions[cell]
            scores = pd.Series(frame.probability_score.to_numpy(dtype=np.float64), index=ordinals)
            strategy["cells"][cell] = {}
            model_metrics[cell] = {}
            for scenario in SCENARIO_ORDER:
                result = _simulate(
                    market,
                    scores,
                    labels,
                    SCENARIOS[scenario],
                    expected_start=start,
                    expected_end=end,
                )
                payload = _result_payload(result)
                strategy["cells"][cell][scenario] = payload
                model_metrics[cell][scenario] = payload["metrics"]
                schedule = tuple(
                    (trade["entry_timestamp"], trade["exit_timestamp"])
                    for trade in payload["trade_ledger"]
                )
                if cell in trade_schedules and schedule != trade_schedules[cell]:
                    raise BacktestIntegrityError("cost scenario changed trade timing")
                trade_schedules[cell] = schedule
            low = model_metrics[cell]["low"]["total_return"]
            base = model_metrics[cell]["base"]["total_return"]
            high = model_metrics[cell]["high"]["total_return"]
            if low < base or base < high:
                raise BacktestIntegrityError("higher execution costs improved a fixed strategy")

        costs = {
            "schema_version": "phase2-cost-sensitivity-v1",
            "synthetic": True,
            "official_scenario": "base",
            "common_window": common_window,
            "scenarios": {
                name: {
                    "fee_rate": config.fee_rate,
                    "slippage_bps_per_side": config.slippage_bps_per_side,
                    "half_spread_bps_per_side": config.half_spread_bps_per_side,
                    "one_side_execution_rate": config.one_side_execution_rate,
                }
                for name, config in SCENARIOS.items()
            },
            "cells": model_metrics,
            "signals_and_trade_ordinals_frozen_across_scenarios": True,
            "returns_monotone_as_costs_increase": True,
        }

        ema_scores, momentum_scores = _baseline_scores(predictions, labeled)
        zeros = pd.Series(0.0, index=ordinals)
        baselines: dict[str, Any] = {
            "schema_version": "phase2-baseline-metrics-v1",
            "synthetic": True,
            "common_window": common_window,
            "deterministic": {},
            "random_exposure": {},
        }
        for scenario in SCENARIO_ORDER:
            config = SCENARIOS[scenario]
            cash = _simulate(market, zeros, labels, config, expected_start=start, expected_end=end)
            buy_hold = buy_and_hold_backtest(
                market, ordinals, HORIZON, config, initial_capital=INITIAL_CAPITAL
            )
            _result_window(buy_hold, market, start, end)
            ema = _simulate(
                market, ema_scores, labels, config, expected_start=start, expected_end=end
            )
            momentum = _simulate(
                market, momentum_scores, labels, config, expected_start=start, expected_end=end
            )
            baselines["deterministic"][scenario] = {
                "cash": _result_payload(cash),
                "buy_and_hold": _result_payload(buy_hold),
                "ema_9_21": _result_payload(ema),
                "momentum_24": _result_payload(momentum),
            }
        baselines["random_exposure"] = _random_summaries(
            market,
            predictions,
            labels,
            model_metrics,
            start,
            end,
            simulations=random_simulations,
        )
        for name in baselines["deterministic"]["base"]:
            paths = [baselines["deterministic"][scenario][name] for scenario in SCENARIO_ORDER]
            schedules = [
                tuple(
                    (trade["entry_market_ordinal"], trade["exit_market_ordinal"])
                    for trade in path["trade_ledger"]
                )
                for path in paths
            ]
            returns = [path["metrics"]["total_return"] for path in paths]
            if schedules[0] != schedules[1] or schedules[1] != schedules[2]:
                raise BacktestIntegrityError("cost scenario changed baseline trade timing")
            if returns[0] < returns[1] or returns[1] < returns[2]:
                raise BacktestIntegrityError("higher costs improved a fixed baseline")
        for cell in experiments.CELLS:
            for percentile in ("p05", "median", "p95"):
                returns = [
                    baselines["random_exposure"][cell][scenario]["total_return"][percentile]
                    for scenario in SCENARIO_ORDER
                ]
                if any(value is None for value in returns) or not (
                    returns[0] >= returns[1] >= returns[2]
                ):
                    raise BacktestIntegrityError("random baseline cost ordering changed")

        direct = {}
        for scenario in SCENARIO_ORDER:
            direct[scenario] = {
                "linear_C_minus_A": _comparison_metrics(
                    model_metrics["C"][scenario], model_metrics["A"][scenario]
                ),
                "nonlinear_D_minus_B": _comparison_metrics(
                    model_metrics["D"][scenario], model_metrics["B"][scenario]
                ),
            }
        folds = []
        for number in range(1, experiments.FOLD_COUNT + 1):
            mask = reference.fold_number == number
            fold_reference = reference.loc[mask]
            if fold_reference.empty:
                raise BacktestIntegrityError("every one of five validation folds must be present")
            fold_ordinals = pd.Index(fold_reference.market_ordinal.to_numpy(dtype=np.int64))
            fold_labels = pd.Series(
                fold_reference.actual_label.to_numpy(dtype=np.int8), index=fold_ordinals
            )
            fold_start = int(fold_ordinals[0]) + 1
            fold_end = int(fold_ordinals[-1]) + HORIZON + 1
            fold_cells = {}
            for cell in experiments.CELLS:
                frame = predictions[cell].loc[mask]
                scores = pd.Series(
                    frame.probability_score.to_numpy(dtype=np.float64), index=fold_ordinals
                )
                result = _simulate(
                    market,
                    scores,
                    fold_labels,
                    SCENARIOS["base"],
                    expected_start=fold_start,
                    expected_end=fold_end,
                )
                metrics = calculate_backtest_metrics(result, TIMEFRAME)
                fold_cells[cell] = {
                    "total_return": metrics["total_return"],
                    "completed_trades": metrics["num_trades"],
                }
            folds.append(
                {
                    "fold_number": number,
                    "common_window": _window(market, fold_start, fold_end, len(fold_reference)),
                    "cells": fold_cells,
                    "linear_C_minus_A_return_percentage_points": 100.0
                    * (fold_cells["C"]["total_return"] - fold_cells["A"]["total_return"]),
                    "nonlinear_D_minus_B_return_percentage_points": 100.0
                    * (fold_cells["D"]["total_return"] - fold_cells["B"]["total_return"]),
                }
            )
        ablation = {
            "schema_version": "phase2-ablation-report-v1",
            "synthetic": True,
            "research_gates_evaluated": False,
            "common_window": common_window,
            "direct_deltas": direct,
            "folds": folds,
            "fold_consistency": {
                "linear_C_minus_A": _fold_consistency(
                    [row["linear_C_minus_A_return_percentage_points"] for row in folds]
                ),
                "nonlinear_D_minus_B": _fold_consistency(
                    [row["nonlinear_D_minus_B_return_percentage_points"] for row in folds]
                ),
            },
        }
        for value in (strategy, costs, baselines, ablation):
            canonicalize(value)
        return strategy, costs, baselines, ablation
    except Phase2BacktestError:
        raise
    except (
        CryptoAIError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        AttributeError,
        OverflowError,
        RuntimeError,
    ) as exc:
        raise BacktestIntegrityError("synthetic backtest calculation failed") from exc


def _cost_configuration() -> dict[str, Any]:
    return {
        "scenario_order": list(SCENARIO_ORDER),
        "official_scenario": "base",
        "scenarios": {
            name: {
                "fee_rate": config.fee_rate,
                "slippage_bps_per_side": config.slippage_bps_per_side,
                "half_spread_bps_per_side": config.half_spread_bps_per_side,
            }
            for name, config in SCENARIOS.items()
        },
    }


def _baseline_configuration() -> dict[str, Any]:
    return {
        "deterministic": ["cash", "buy_and_hold", "ema_9_21", "momentum_24"],
        "ema_rule": "ema_short_greater_than_ema_long",
        "momentum_rule": "return_24_greater_than_zero",
        "random_simulations": RANDOM_SIMULATIONS,
        "random_seed_base": RANDOM_SEED,
        "random_signal_probability": "matched_cell_positive_oof_signal_fraction",
        "random_replay_across_cost_scenarios": True,
    }


def _metric_configuration() -> dict[str, Any]:
    return {
        "horizon_complete_candles": HORIZON,
        "signal_threshold": THRESHOLD,
        "initial_capital_usd": INITIAL_CAPITAL,
        "timeframe": TIMEFRAME,
        "annual_periods": 8760,
        "annual_risk_free_rate": 0.0,
        "metrics": list(METRIC_KEYS),
        "source": "unchanged_Phase1_backtesting_engine_and_metrics",
    }


def _implementation_source_hash() -> str:
    repository = Path(__file__).resolve().parents[3]
    sources = (
        "src/crypto_ai/phase2/backtests.py",
        "src/crypto_ai/phase2/artifacts.py",
        "src/crypto_ai/phase2/experiments.py",
        "src/crypto_ai/phase2/dataset.py",
        "src/crypto_ai/backtesting/engine.py",
        "src/crypto_ai/backtesting/baselines.py",
        "src/crypto_ai/backtesting/metrics.py",
        "src/crypto_ai/costs.py",
        "src/crypto_ai/sentiment/canonical.py",
        "src/crypto_ai/sentiment/storage.py",
        "src/crypto_ai/sentiment/contracts.py",
    )
    return sha256_bytes(
        canonicalize({name: sha256_bytes((repository / name).read_bytes()) for name in sources})
    )


def _metadata(inputs: _VerifiedInputs, source_run_id: str) -> dict[str, Any]:
    source_files = dict(inputs.source.files)
    source_manifest = inputs.source.manifest
    return {
        "specification_id": SPECIFICATION_ID,
        "synthetic": True,
        "source_run_id": source_run_id,
        "source_manifest_sha256": sha256_bytes(inputs.source_manifest_bytes),
        "source_experiment_id": inputs.source.experiment_id,
        "source_folds_sha256": source_manifest["folds_sha256"],
        "source_prediction_sha256": {
            cell: sha256_bytes(source_files[f"cells/{cell}/predictions.json"])
            for cell in experiments.CELLS
        },
        "prepared_dataset_id": inputs.prepared.dataset_id,
        "market_price_context_sha256": inputs.prepared.manifest["market_price_context_sha256"],
        "decision_window_sha256": sha256_bytes(
            canonicalize(
                {
                    "decisions": list(inputs.decision_identity),
                    "window": _window(
                        inputs.market, inputs.start, inputs.end, len(inputs.decision_identity)
                    ),
                }
            )
        ),
        "cost_config_sha256": sha256_bytes(canonicalize(_cost_configuration())),
        "baseline_config_sha256": sha256_bytes(canonicalize(_baseline_configuration())),
        "metric_config_sha256": sha256_bytes(canonicalize(_metric_configuration())),
        "implementation_source_sha256": _implementation_source_hash(),
        "dependency_lock_sha256": sha256_bytes(
            (Path(__file__).resolve().parents[3] / "requirements-phase2.txt").read_bytes()
        ),
    }


def _as_percent(value: float | None) -> str:
    return "undefined" if value is None else f"{100.0 * value:.4f}%"


def _as_number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _as_ablation_value(metric: str, value: float | int | None, *, delta: bool) -> str:
    if value is None:
        return "undefined"
    if metric in ("total_return", "annualized_return"):
        return f"{value:.4f} pp" if delta else _as_percent(value)
    if metric in ("num_trades", "maximum_drawdown_duration"):
        return str(value)
    return _as_number(value)


def _render_report(
    inputs: _VerifiedInputs,
    metadata: dict[str, Any],
    strategy: dict[str, Any],
    costs: dict[str, Any],
    baselines: dict[str, Any],
    ablation: dict[str, Any],
    hashes: dict[str, str],
) -> bytes:
    window = strategy["common_window"]
    lines = [
        "# Phase 2 Milestone 7 — Synthetic Development Engineering Report",
        "",
        "Engineering Status: **PASS** (deterministic contract verification).",
        "Research Outcome: **NOT_EVALUATED_SYNTHETIC_ONLY**.",
        "No live collection, real-data backtest, research gate, model choice, "
        "or holdout evaluation occurred.",
        "",
        "## Evidence boundary and immutable parents",
        "",
        f"- Source Milestone 6 run: `{metadata['source_run_id']}`.",
        f"- Source outer manifest SHA-256: `{metadata['source_manifest_sha256']}`.",
        f"- Source experiment ID: `{metadata['source_experiment_id']}`.",
        f"- Prepared synthetic dataset ID: `{metadata['prepared_dataset_id']}`.",
        f"- Complete market-price-context SHA-256: `{metadata['market_price_context_sha256']}`.",
        f"- Shared decision/window SHA-256: `{metadata['decision_window_sha256']}`.",
        "",
        "## Common OOF window, folds, and coverage exclusions",
        "",
        f"- {window['oof_decisions']} identical OOF decision rows across A/B/C/D; "
        "five validation folds.",
        f"- Original market opens: {window['first_open_ordinal']} "
        f"through {window['final_open_ordinal']} "
        f"({window['first_open_at']} through {window['final_open_at']}).",
        f"- {window['open_to_open_intervals']} open-to-open intervals; "
        "provider-gap decision rows are "
        "excluded by the verified prepared parent, never backfilled or compressed.",
        "",
        "## Four-cell classification context",
        "",
        "The accepted Milestone 6 OOF probabilities and fixed 0.50 threshold were reused "
        "without refitting or selecting a model.",
        "",
        "| Cell | F1 class 1 | ROC-AUC | Log loss |",
        "|---|---:|---:|---:|",
    ]
    for cell in experiments.CELLS:
        context = strategy["classification_context"][cell]
        lines.append(
            f"| {cell} | {_as_number(context['f1_class_1'])} | "
            f"{_as_number(context['roc_auc'])} | {_as_number(context['log_loss'])} |"
        )
    lines.extend(
        [
            "",
            "## Base-cost strategy, risk, and trading metrics",
            "",
            "| Cell | Total return | Sharpe | Max drawdown | Completed trades |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in experiments.CELLS:
        metrics = strategy["cells"][cell]["base"]["metrics"]
        sharpe = metrics["sharpe_ratio"]
        lines.append(
            f"| {cell} | {_as_percent(metrics['total_return'])} | "
            f"{'undefined' if sharpe is None else f'{sharpe:.4f}'} | "
            f"{_as_percent(metrics['maximum_drawdown'])} | {metrics['num_trades']} |"
        )
    lines.extend(
        [
            "",
            "## Low/base/high cost sensitivity",
            "",
            "The fee is 10 bps per side throughout; adverse slippage plus half-spread is "
            "1.5, 3.0, or 7.0 bps per side. Fixed signals and trade ordinals are identical.",
            "",
            "| Cell | Low return | Base return | High return |",
            "|---|---:|---:|---:|",
        ]
    )
    for cell in experiments.CELLS:
        metrics = costs["cells"][cell]
        lines.append(
            f"| {cell} | {_as_percent(metrics['low']['total_return'])} | "
            f"{_as_percent(metrics['base']['total_return'])} | "
            f"{_as_percent(metrics['high']['total_return'])} |"
        )
    lines.extend(
        [
            "",
            "## Five trading baselines",
            "",
            "Cash, cost-aware buy-and-hold, EMA 9/21, momentum 24, and 1,000-seed "
            "matched random exposure use the identical common window. Random summaries "
            "are cell-specific; all draws are replayed across cost scenarios.",
            "",
            "| Baseline | Base-cost total return |",
            "|---|---:|",
        ]
    )
    for name, value in baselines["deterministic"]["base"].items():
        lines.append(f"| {name} | {_as_percent(value['metrics']['total_return'])} |")
    lines.extend(
        [
            "",
            "Base-cost random-exposure total-return distributions (1,000 simulations "
            "per matched cell):",
            "",
            "| Cell | 5th percentile | Median | 95th percentile | " "Fraction at least model |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cell in experiments.CELLS:
        random = baselines["random_exposure"][cell]["base"]
        distribution = random["total_return"]
        lines.append(
            f"| {cell} | {_as_percent(distribution['p05'])} | "
            f"{_as_percent(distribution['median'])} | "
            f"{_as_percent(distribution['p95'])} | "
            f"{_as_percent(random['fraction_return_at_least_model'])} |"
        )
    lines.extend(
        [
            "",
            "## Direct ablation and fold consistency",
            "",
            "Matched control, augmented, and incremental values for every frozen metric "
            "and cost scenario. Total and annualized returns use percentages with "
            "percentage-point deltas; other values use native metric units (costs in USD).",
            "",
            "| Comparison | Scenario | Metric | Control | Augmented | Delta |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIO_ORDER:
        for label, key in (
            ("Linear C–A", "linear_C_minus_A"),
            ("Nonlinear D–B", "nonlinear_D_minus_B"),
        ):
            for metric in METRIC_KEYS:
                comparison = ablation["direct_deltas"][scenario][key][metric]
                lines.append(
                    f"| {label} | {scenario.title()} | {metric} | "
                    f"{_as_ablation_value(metric, comparison['control_metric'], delta=False)} | "
                    f"{_as_ablation_value(metric, comparison['augmented_metric'], delta=False)} | "
                    f"{_as_ablation_value(metric, comparison['delta'], delta=True)} |"
                )
    lines.extend(
        [
            "",
            "| Comparison | Positive folds / 5 | Median fold delta (percentage points) | "
            "Largest positive contribution share |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("Linear C–A", "linear_C_minus_A"),
        ("Nonlinear D–B", "nonlinear_D_minus_B"),
    ):
        summary = ablation["fold_consistency"][key]
        lines.append(
            f"| {label} | {summary['positive_fold_count']} | "
            f"{summary['median_return_delta_percentage_points']:.4f} | "
            f"{_as_percent(summary['largest_positive_contribution_share'])} |"
        )
    lines.extend(
        [
            "",
            "These synthetic deltas are descriptive engineering outputs, not an "
            "approved research gate or evidence of live alpha.",
            "",
            "## Exact payload hashes and integrity checks",
            "",
        ]
    )
    for name, digest in sorted(hashes.items()):
        lines.append(f"- `{name}` SHA-256: `{digest}`.")
    lines.extend(
        [
            "",
            "## Limitations and required next authorization",
            "",
            "All observations and probabilities were synthetic verified fixtures. "
            "The live pilot remains deferred without backfill. Real scoring, research "
            "backtests, model selection, numerical research gates, and holdout access "
            "require separate human authorization.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


class OfflineBacktestEngine:
    """No arbitrary frames: replay one verified, immutable synthetic M6 run."""

    def __init__(self, store: experiments.ExperimentStore):
        if type(store) is not experiments.ExperimentStore:
            raise BacktestAuthorizationError("exact synthetic Milestone 6 store required")
        self.store = store

    def run(self, source_run_id: str) -> BacktestBundle:
        try:
            inputs = _verified_inputs(self.store, source_run_id)
            strategy, costs, baselines, ablation = _calculate(
                inputs.market,
                inputs.predictions,
                inputs.labeled,
                start=inputs.start,
                end=inputs.end,
                random_simulations=RANDOM_SIMULATIONS,
            )
            if (
                _outer_source_manifest(self.store, source_run_id, inputs.source)
                != inputs.source_manifest_bytes
            ):
                raise BacktestIntegrityError("source experiment changed during report generation")
            prepared_again = datasets.DatasetStore(self.store.parents).get(
                inputs.prepared.dataset_id
            )
            if prepared_again is None or dict(prepared_again.files) != dict(inputs.prepared.files):
                raise BacktestIntegrityError("prepared parent changed during report generation")
            strategy["source_run_id"] = source_run_id
            strategy["source_experiment_id"] = inputs.source.experiment_id
            strategy["classification_context"] = {
                cell: inputs.source.metrics(cell)["aggregate"] for cell in experiments.CELLS
            }
            ablation["source_run_id"] = source_run_id
            metadata = _metadata(inputs, source_run_id)
            values = {
                "strategy_metrics.json": strategy,
                "cost_sensitivity.json": costs,
                "baseline_metrics.json": baselines,
                "ablation_report.json": ablation,
            }
            files = {name: canonicalize(value) for name, value in values.items()}
            hashes = {name: sha256_bytes(raw) for name, raw in files.items()}
            files["development_report.md"] = _render_report(
                inputs, metadata, strategy, costs, baselines, ablation, hashes
            )
            if set(files) != set(PAYLOAD_FILES):
                raise BacktestIntegrityError("incomplete Milestone 7 report inventory")
            return BacktestBundle(tuple(sorted(files.items())), metadata)
        except Phase2BacktestError:
            raise
        except (
            CryptoAIError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            AttributeError,
            OverflowError,
            RuntimeError,
        ) as exc:
            raise BacktestIntegrityError("offline report generation failed") from exc

    def publish(self, source_run_id: str, *, report_run_id: str | None = None) -> str:
        """Generate and atomically publish a distinct immutable synthetic report run."""
        from crypto_ai.phase2.artifacts import ArtifactStore

        bundle = self.run(source_run_id)
        return ArtifactStore(self.store.root, self.store).publish(
            dict(bundle.files), bundle.metadata, run_id=report_run_id
        )


def verify_report(
    store: experiments.ExperimentStore, files: dict[str, bytes], metadata: dict[str, Any]
) -> None:
    """Replay verified synthetic parents and compare every exact report byte."""
    if type(metadata) is not dict or type(metadata.get("source_run_id")) is not str:
        raise BacktestIntegrityError("report source identity is missing")
    expected = OfflineBacktestEngine(store).run(metadata["source_run_id"])
    if dict(expected.files) != files or expected.metadata != metadata:
        raise BacktestIntegrityError("report bytes or parent-bound metadata failed replay")
