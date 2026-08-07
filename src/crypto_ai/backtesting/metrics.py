"""Strategy, risk, trading, and classification-context metrics."""

import math
from typing import Any

import numpy as np
import pandas as pd

from crypto_ai.backtesting.engine import BacktestResult
from crypto_ai.data.validation import timeframe_to_milliseconds


def periods_per_year(timeframe: str) -> float:
    """Calculate annual candle periods for a fixed timeframe."""
    return 365.0 * 24.0 * 60.0 * 60.0 * 1000.0 / timeframe_to_milliseconds(timeframe)


def _drawdown_duration(equity: pd.Series) -> int:
    below_peak = equity < equity.cummax()
    longest = current = 0
    for below in below_peak:
        current = current + 1 if below else 0
        longest = max(longest, current)
    return longest


def calculate_backtest_metrics(
    result: BacktestResult,
    timeframe: str,
    annual_risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Calculate the complete Phase 1 strategy metric contract."""
    curve = result.equity_curve
    ledger = result.trade_ledger
    returns = curve["period_return"].astype("float64")
    periods = periods_per_year(timeframe)
    n_periods = len(returns)
    final_equity = float(curve["equity"].iloc[-1])
    total_return = final_equity / result.initial_capital - 1.0
    annualized_return = (
        (1.0 + total_return) ** (periods / n_periods) - 1.0
        if n_periods and total_return > -1
        else None
    )
    risk_free_period = (1.0 + annual_risk_free_rate) ** (1.0 / periods) - 1.0
    volatility = float(returns.std(ddof=1)) if n_periods > 1 else math.nan
    sharpe = None
    if n_periods > 1 and np.isfinite(volatility) and volatility > 0:
        sharpe = float((returns.mean() - risk_free_period) / volatility * math.sqrt(periods))
    excess = returns - risk_free_period
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))) if n_periods else 0.0
    sortino = float(excess.mean() / downside * math.sqrt(periods)) if downside > 0 else None
    drawdowns = curve["equity"] / curve["equity"].cummax() - 1.0
    maximum_drawdown = float(drawdowns.min())
    calmar = (
        float(annualized_return / abs(maximum_drawdown))
        if annualized_return is not None and maximum_drawdown < 0
        else None
    )

    trade_returns = ledger["net_return"] if not ledger.empty else pd.Series(dtype="float64")
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    positive_pnl = float(ledger.loc[ledger["pnl"] > 0, "pnl"].sum()) if not ledger.empty else 0.0
    negative_pnl = float(ledger.loc[ledger["pnl"] < 0, "pnl"].sum()) if not ledger.empty else 0.0
    average_equity = float(curve["equity"].mean())
    notionals = 0.0
    total_costs = 0.0
    if not ledger.empty:
        notionals = float(
            (ledger["position_quantity"] * ledger["entry_market_price"]).sum()
            + (ledger["position_quantity"] * ledger["exit_market_price"]).sum()
        )
        total_costs = float(
            ledger[["entry_fee", "exit_fee", "entry_execution_cost", "exit_execution_cost"]]
            .sum()
            .sum()
        )

    scores = result.probability_scores
    metric_warnings: list[str] = []
    if len(ledger) and negative_pnl >= 0:
        metric_warnings.append("Profit factor is undefined because there are no losing trades")
    if sharpe is None:
        metric_warnings.append(
            "Sharpe ratio is undefined because observations are insufficient or volatility is zero"
        )
    if sortino is None:
        metric_warnings.append(
            "Sortino ratio is undefined because observations are insufficient or downside "
            "deviation is zero"
        )
    metrics: dict[str, Any] = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": (
            float(volatility * math.sqrt(periods)) if np.isfinite(volatility) else None
        ),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": maximum_drawdown,
        "maximum_drawdown_duration": _drawdown_duration(curve["equity"]),
        "calmar_ratio": calmar,
        "num_trades": len(ledger),
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else None,
        "average_trade_return": float(trade_returns.mean()) if len(trade_returns) else None,
        "median_trade_return": float(trade_returns.median()) if len(trade_returns) else None,
        "average_winning_return": float(wins.mean()) if len(wins) else None,
        "average_losing_return": float(losses.mean()) if len(losses) else None,
        "largest_winning_trade": float(wins.max()) if len(wins) else None,
        "largest_losing_trade": float(losses.min()) if len(losses) else None,
        "profit_factor": positive_pnl / abs(negative_pnl) if negative_pnl < 0 else None,
        "market_exposure": float(curve["market_exposure"].mean()) if n_periods else 0.0,
        "turnover": notionals / average_equity if average_equity > 0 else 0.0,
        "average_holding_period": (
            float(ledger["holding_candles"].mean()) if not ledger.empty else None
        ),
        "total_estimated_costs": total_costs,
        "positive_prediction_rate": float((scores >= 0.5).mean()),
        "average_buy_probability": float(scores.mean()),
        "actual_positive_label_rate": (
            float(result.actual_labels.mean()) if result.actual_labels is not None else None
        ),
        "warnings": metric_warnings,
    }
    return metrics
