"""Trading baselines and execution-cost sensitivity."""

from typing import Any

import numpy as np
import pandas as pd

from crypto_ai.backtesting.engine import BacktestResult, run_backtest
from crypto_ai.backtesting.metrics import calculate_backtest_metrics
from crypto_ai.config import settings
from crypto_ai.costs import CostConfig


def cash_baseline(
    market_df: pd.DataFrame,
    decision_index: pd.Index,
    actual_labels: pd.Series | None,
    horizon: int,
    timeframe: str,
    cost_config: CostConfig,
    *,
    initial_capital: float = settings.INITIAL_CAPITAL,
) -> BacktestResult:
    """Return the no-trade cash baseline over the common decision window."""
    return run_backtest(
        market_df,
        pd.Series(0.0, index=decision_index),
        actual_labels,
        horizon,
        timeframe,
        0.5,
        initial_capital,
        cost_config,
    )


def buy_and_hold_backtest(
    market_df: pd.DataFrame,
    decision_index: pd.Index,
    horizon: int,
    cost_config: CostConfig,
    *,
    initial_capital: float = settings.INITIAL_CAPITAL,
) -> BacktestResult:
    """Enter once at the common window's first open and exit at its final open."""
    positions = {index: position for position, index in enumerate(market_df.index)}
    decision_positions = [positions[index] for index in decision_index]
    valid = [position for position in decision_positions if position + horizon + 1 < len(market_df)]
    start = min(decision_positions) + 1
    end = max(valid) + horizon + 1
    entry_market = float(market_df.iloc[start]["open"])
    exit_market = float(market_df.iloc[end]["open"])
    execution = cost_config.one_side_execution_rate
    entry_fill = entry_market * (1 + execution)
    entry_fee = initial_capital * cost_config.fee_rate
    quantity = (initial_capital - entry_fee) / entry_fill
    exit_fill = exit_market * (1 - execution)
    gross_exit = quantity * exit_fill
    exit_fee = gross_exit * cost_config.fee_rate
    final_equity = gross_exit - exit_fee
    rows: list[dict[str, object]] = []
    previous = float(initial_capital)
    indexes = market_df.index[start : end + 1]
    for position in range(start, end + 1):
        if position == start:
            equity = quantity * entry_market
        elif position == end:
            equity = final_equity
        else:
            equity = quantity * float(market_df.iloc[position]["open"])
        rows.append(
            {
                "timestamp": market_df.iloc[position]["timestamp"],
                "equity": equity,
                "period_return": equity / previous - 1.0,
                "position_open": position < end,
                "market_exposure": 1.0 if position < end else 0.0,
            }
        )
        previous = equity
    ledger = pd.DataFrame(
        [
            {
                "trade_id": 1,
                "signal_timestamp": market_df.iloc[start - 1]["timestamp"],
                "entry_timestamp": market_df.iloc[start]["timestamp"],
                "exit_timestamp": market_df.iloc[end]["timestamp"],
                "entry_market_price": entry_market,
                "entry_fill_price": entry_fill,
                "position_quantity": quantity,
                "equity_before_entry": initial_capital,
                "entry_fee": entry_fee,
                "entry_execution_cost": quantity * (entry_fill - entry_market),
                "exit_market_price": exit_market,
                "exit_fill_price": exit_fill,
                "exit_fee": exit_fee,
                "exit_execution_cost": quantity * (exit_market - exit_fill),
                "equity_after_exit": final_equity,
                "holding_candles": end - start,
                "gross_market_return": exit_market / entry_market - 1,
                "filled_gross_return": exit_fill / entry_fill - 1,
                "total_fee_rate": 1 - (1 - cost_config.fee_rate) ** 2,
                "total_execution_cost_rate": 1 - (1 - execution) / (1 + execution),
                "net_return": final_equity / initial_capital - 1,
                "pnl": final_equity - initial_capital,
                "probability_score": 1.0,
                "winning_trade": final_equity > initial_capital,
            }
        ]
    )
    scores = pd.Series(1.0, index=decision_index)
    return BacktestResult(ledger, pd.DataFrame(rows, index=indexes), initial_capital, scores, None)


def rule_backtest(
    market_df: pd.DataFrame,
    decision_index: pd.Index,
    buy_rule: pd.Series,
    actual_labels: pd.Series | None,
    horizon: int,
    timeframe: str,
    cost_config: CostConfig,
    *,
    initial_capital: float = settings.INITIAL_CAPITAL,
) -> BacktestResult:
    """Backtest a Boolean rule with the common fixed-horizon execution policy."""
    scores = buy_rule.loc[decision_index].astype("float64")
    return run_backtest(
        market_df,
        scores,
        actual_labels,
        horizon,
        timeframe,
        0.5,
        initial_capital,
        cost_config,
    )


def random_exposure_summary(
    market_df: pd.DataFrame,
    model_scores: pd.Series,
    actual_labels: pd.Series | None,
    horizon: int,
    timeframe: str,
    cost_config: CostConfig,
    model_total_return: float,
    simulations: int | None = None,
    *,
    signal_threshold: float = settings.SIGNAL_THRESHOLD,
    initial_capital: float = settings.INITIAL_CAPITAL,
    random_seed: int = settings.RANDOM_SEED,
) -> dict[str, Any]:
    """Run deterministic Bernoulli exposure simulations matching model positive rate."""
    simulations = settings.RANDOM_BASELINE_SIMULATIONS if simulations is None else simulations
    probability = float((model_scores >= signal_threshold).mean())
    rows: list[dict[str, float | None]] = []
    for simulation in range(simulations):
        rng = np.random.default_rng(random_seed + simulation)
        scores = pd.Series(
            rng.binomial(1, probability, len(model_scores)).astype("float64"),
            index=model_scores.index,
        )
        result = run_backtest(
            market_df,
            scores,
            actual_labels,
            horizon,
            timeframe,
            0.5,
            initial_capital,
            cost_config,
        )
        rows.append(calculate_backtest_metrics(result, timeframe))

    def summary(metric: str) -> dict[str, float | None]:
        values = np.array([row[metric] for row in rows if row[metric] is not None], dtype="float64")
        if not len(values):
            return {"median": None, "p05": None, "p95": None}
        return {
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

    total_returns = np.array([float(row["total_return"]) for row in rows])
    return {
        "simulations": simulations,
        "signal_probability": probability,
        "total_return": summary("total_return"),
        "sharpe_ratio": summary("sharpe_ratio"),
        "maximum_drawdown": summary("maximum_drawdown"),
        "fraction_return_at_least_model": float((total_returns >= model_total_return).mean()),
    }


def run_cost_sensitivity(
    market_df: pd.DataFrame,
    scores: pd.Series,
    actual_labels: pd.Series | None,
    horizon: int,
    timeframe: str,
    *,
    cost_scenarios: dict[str, dict[str, float]] | None = None,
    signal_threshold: float = settings.SIGNAL_THRESHOLD,
    initial_capital: float = settings.INITIAL_CAPITAL,
) -> dict[str, dict[str, Any]]:
    """Backtest frozen predictions under every configured execution-cost scenario."""
    results: dict[str, dict[str, Any]] = {}
    scenarios = settings.COST_SCENARIOS if cost_scenarios is None else cost_scenarios
    for name, values in scenarios.items():
        cost = CostConfig(**values)
        backtest = run_backtest(
            market_df,
            scores,
            actual_labels,
            horizon,
            timeframe,
            signal_threshold,
            initial_capital,
            cost,
        )
        results[name] = calculate_backtest_metrics(backtest, timeframe)
    return results
