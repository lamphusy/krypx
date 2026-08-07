"""Fixed-horizon, next-open long-or-cash backtest state machine."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_ai.costs import CostConfig
from crypto_ai.exceptions import BacktestError


@dataclass(frozen=True)
class BacktestResult:
    """Completed trades, candle-level equity, and classification context."""

    trade_ledger: pd.DataFrame
    equity_curve: pd.DataFrame
    initial_capital: float
    probability_scores: pd.Series
    actual_labels: pd.Series | None


def _validate_inputs(
    market_df: pd.DataFrame,
    probability_scores: pd.Series,
    actual_labels: pd.Series | None,
    horizon: int,
    signal_threshold: float,
    initial_capital: float,
) -> dict[object, int]:
    if market_df.empty or not market_df.index.is_unique:
        raise BacktestError("market_df must be non-empty with a unique index")
    for column in ("timestamp", "open"):
        if column not in market_df:
            raise BacktestError(f"market_df is missing {column}")
    if not market_df["timestamp"].is_monotonic_increasing:
        raise BacktestError("market_df must be chronological")
    if not probability_scores.index.is_unique or probability_scores.empty:
        raise BacktestError("probability_scores must have a non-empty unique index")
    if not probability_scores.index.isin(market_df.index).all():
        raise BacktestError("probability score indexes must be a subset of market_df")
    positions = {index: position for position, index in enumerate(market_df.index)}
    score_positions = [positions[index] for index in probability_scores.index]
    if score_positions != sorted(score_positions):
        raise BacktestError("probability score indexes must be chronologically ordered")
    values = probability_scores.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise BacktestError("probability scores must be finite and within [0, 1]")
    if actual_labels is not None and not actual_labels.index.equals(probability_scores.index):
        raise BacktestError("actual_labels must exactly align with probability_scores")
    if not isinstance(horizon, int) or horizon <= 0:
        raise BacktestError("horizon must be a positive integer")
    if not 0.0 <= signal_threshold <= 1.0:
        raise BacktestError("signal_threshold must be within [0, 1]")
    if not np.isfinite(initial_capital) or initial_capital <= 0.0:
        raise BacktestError("initial_capital must be positive and finite")
    return positions


def run_backtest(
    market_df: pd.DataFrame,
    probability_scores: pd.Series,
    actual_labels: pd.Series | None,
    horizon: int,
    timeframe: str,
    signal_threshold: float,
    initial_capital: float,
    cost_config: CostConfig,
) -> BacktestResult:
    """Simulate the fixed-horizon long-or-cash strategy."""
    del timeframe  # Annualization is calculated from the returned curve by metrics.
    positions = _validate_inputs(
        market_df, probability_scores, actual_labels, horizon, signal_threshold, initial_capital
    )
    decision_positions = [positions[index] for index in probability_scores.index]
    valid_decisions = [
        position for position in decision_positions if position + horizon + 1 < len(market_df)
    ]
    if not valid_decisions:
        raise BacktestError("No decision row has sufficient future execution-price context")
    window_start = min(decision_positions) + 1
    window_end = max(valid_decisions) + horizon + 1

    score_by_position = {
        positions[index]: float(probability_scores.loc[index]) for index in probability_scores.index
    }
    index_by_position = list(market_df.index)
    execution_rate = cost_config.one_side_execution_rate
    cash = float(initial_capital)
    quantity = 0.0
    pending_entry: int | None = None
    scheduled_exit: int | None = None
    active_trade: dict[str, object] | None = None
    ledger_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    previous_equity = float(initial_capital)

    for position in range(len(market_df)):
        row = market_df.iloc[position]
        market_open = float(row["open"])

        if scheduled_exit == position:
            if active_trade is None or quantity <= 0.0:
                raise BacktestError("Scheduled exit has no open position")
            exit_fill = market_open * (1.0 - execution_rate)
            gross_exit_value = quantity * exit_fill
            exit_fee = gross_exit_value * cost_config.fee_rate
            cash = gross_exit_value - exit_fee
            equity_before = float(active_trade["equity_before_entry"])
            ledger_rows.append(
                {
                    **active_trade,
                    "trade_id": len(ledger_rows) + 1,
                    "exit_timestamp": row["timestamp"],
                    "exit_market_price": market_open,
                    "exit_fill_price": exit_fill,
                    "exit_fee": exit_fee,
                    "exit_execution_cost": quantity * (market_open - exit_fill),
                    "equity_after_exit": cash,
                    "holding_candles": position - int(active_trade["entry_position"]),
                    "gross_market_return": market_open / float(active_trade["entry_market_price"])
                    - 1.0,
                    "filled_gross_return": exit_fill / float(active_trade["entry_fill_price"])
                    - 1.0,
                    "total_fee_rate": 1.0 - (1.0 - cost_config.fee_rate) ** 2,
                    "total_execution_cost_rate": 1.0
                    - (1.0 - execution_rate) / (1.0 + execution_rate),
                    "net_return": cash / equity_before - 1.0,
                    "pnl": cash - equity_before,
                    "winning_trade": cash > equity_before,
                }
            )
            quantity = 0.0
            scheduled_exit = None
            active_trade = None

        if pending_entry == position:
            if active_trade is not None:
                raise BacktestError("Pending entry encountered while a position is open")
            equity_before = cash
            entry_fill = market_open * (1.0 + execution_rate)
            entry_fee = equity_before * cost_config.fee_rate
            investable = equity_before - entry_fee
            quantity = investable / entry_fill
            signal_position = position - 1
            active_trade = {
                "signal_timestamp": market_df.iloc[signal_position]["timestamp"],
                "entry_timestamp": row["timestamp"],
                "entry_market_price": market_open,
                "entry_fill_price": entry_fill,
                "position_quantity": quantity,
                "equity_before_entry": equity_before,
                "entry_fee": entry_fee,
                "entry_execution_cost": quantity * (entry_fill - market_open),
                "entry_position": position,
                "probability_score": score_by_position[signal_position],
            }
            cash = 0.0
            pending_entry = None

        equity = cash if active_trade is None else quantity * market_open
        if window_start <= position <= window_end:
            period_return = equity / previous_equity - 1.0
            equity_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "equity": equity,
                    "period_return": period_return,
                    "position_open": active_trade is not None,
                    "market_exposure": (
                        1.0 if active_trade is not None and position < window_end else 0.0
                    ),
                }
            )
            previous_equity = equity

        if position in score_by_position and active_trade is None and pending_entry is None:
            if (
                score_by_position[position] >= signal_threshold
                and position + horizon + 1 <= window_end
            ):
                pending_entry = position + 1
                scheduled_exit = position + horizon + 1

    if active_trade is not None or pending_entry is not None or scheduled_exit is not None:
        raise BacktestError("Backtest ended with an unresolved position or pending order")

    ledger = pd.DataFrame(ledger_rows)
    if not ledger.empty:
        ledger = ledger.drop(columns="entry_position")
    equity_curve = pd.DataFrame(equity_rows, index=index_by_position[window_start : window_end + 1])
    if equity_curve.empty or not np.isfinite(equity_curve["equity"]).all():
        raise BacktestError("Backtest produced an invalid equity curve")
    return BacktestResult(
        ledger, equity_curve, initial_capital, probability_scores.copy(), actual_labels
    )
