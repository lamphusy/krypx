"""Fixed-horizon, next-open long-or-cash backtest state machine."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_ai.costs import CostConfig
from crypto_ai.exceptions import BacktestError


@dataclass(frozen=True)
class BacktestResult:
    """Completed trades, open-time equity marks, and classification context.

    ``equity_curve`` contains one mark at every market open in the common
    performance window.  Consequently, there are ``N + 1`` marks for ``N``
    open-to-open intervals.  Interval values are aligned to the interval's ending
    open: the first mark has no ``period_return`` or ``market_exposure``, while
    every later mark has exactly one of each.
    """

    trade_ledger: pd.DataFrame
    equity_curve: pd.DataFrame
    initial_capital: float
    probability_scores: pd.Series
    actual_labels: pd.Series | None

    @property
    def n_equity_marks(self) -> int:
        """Return the number of open-time equity observations."""
        return len(self.equity_curve)

    @property
    def n_intervals(self) -> int:
        """Return the number of open-to-open performance intervals."""
        return max(self.n_equity_marks - 1, 0)

    @property
    def interval_returns(self) -> pd.Series:
        """Return the net return for each interval, aligned to its ending open."""
        values = self.equity_curve["period_return"]
        if values.empty or pd.notna(values.iloc[0]) or values.iloc[1:].isna().any():
            raise BacktestError("Equity curve does not contain one return per interval")
        return values.iloc[1:].astype("float64")

    @property
    def interval_exposure(self) -> pd.Series:
        """Return held/not-held state for each interval, aligned to its ending open."""
        values = self.equity_curve["market_exposure"]
        if values.empty or pd.notna(values.iloc[0]) or values.iloc[1:].isna().any():
            raise BacktestError("Equity curve does not contain one exposure value per interval")
        return values.iloc[1:].astype("float64")


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
    open_prices = market_df["open"].to_numpy(dtype="float64")
    if not np.isfinite(open_prices).all() or (open_prices <= 0.0).any():
        raise BacktestError("market open prices must be positive and finite")
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


def resolve_performance_window(
    market_df: pd.DataFrame,
    decision_index: pd.Index,
    horizon: int,
) -> tuple[int, int]:
    """Resolve the common inclusive open-mark window for every strategy.

    The returned endpoints are positional indexes into ``market_df``.  The first
    mark is the next open after the first decision.  The final mark is the latest
    required exit open for the final decision that has sufficient price context.
    """
    if market_df.empty or not market_df.index.is_unique:
        raise BacktestError("market_df must be non-empty with a unique index")
    for column in ("timestamp", "open"):
        if column not in market_df:
            raise BacktestError(f"market_df is missing {column}")
    if not market_df["timestamp"].is_monotonic_increasing:
        raise BacktestError("market_df must be chronological")
    open_prices = market_df["open"].to_numpy(dtype="float64")
    if not np.isfinite(open_prices).all() or (open_prices <= 0.0).any():
        raise BacktestError("market open prices must be positive and finite")
    if not isinstance(horizon, int) or horizon <= 0:
        raise BacktestError("horizon must be a positive integer")
    if decision_index.empty or not decision_index.is_unique:
        raise BacktestError("decision_index must be non-empty with a unique index")
    if not decision_index.isin(market_df.index).all():
        raise BacktestError("decision indexes must be a subset of market_df")

    positions = {index: position for position, index in enumerate(market_df.index)}
    decision_positions = [positions[index] for index in decision_index]
    if decision_positions != sorted(decision_positions):
        raise BacktestError("decision indexes must be chronologically ordered")
    valid_decisions = [
        position for position in decision_positions if position + horizon + 1 < len(market_df)
    ]
    if not valid_decisions:
        raise BacktestError("No decision row has sufficient future execution-price context")
    return min(decision_positions) + 1, max(valid_decisions) + horizon + 1


def _is_close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-8))


def _add_interval_accounting(
    equity_curve: pd.DataFrame,
    initial_capital: float,
) -> pd.DataFrame:
    """Attach exactly one compounded net return to each open-to-open interval.

    Costs at an interval endpoint are included in the return ending at that open.
    Costs paid at the initial mark have no preceding interval, so they are folded
    into the first interval return.  This preserves transaction timing without
    inventing a zero-duration return observation.
    """
    result = equity_curve.copy()
    result["period_return"] = np.nan
    if len(result) > 1:
        equities = result["equity"].to_numpy(dtype="float64")
        returns = np.empty(len(equities) - 1, dtype="float64")
        returns[0] = equities[1] / initial_capital - 1.0
        if len(returns) > 1:
            returns[1:] = equities[2:] / equities[1:-1] - 1.0
        result.iloc[1:, result.columns.get_loc("period_return")] = returns
    return result


def _validate_reconciliation(
    ledger: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
    window_start: int,
) -> None:
    """Prove ledger cash flows, transaction marks, and final equity agree."""
    interval_returns = equity_curve["period_return"].iloc[1:].to_numpy(dtype="float64")
    if not np.isfinite(interval_returns).all():
        raise BacktestError("Backtest produced an invalid interval return")
    final_equity = float(equity_curve["equity"].iloc[-1])
    compounded_intervals = initial_capital * float(np.prod(1.0 + interval_returns))
    if not _is_close(final_equity, compounded_intervals):
        raise BacktestError("Interval returns do not reconcile with final equity")

    if ledger.empty:
        if not _is_close(final_equity, initial_capital):
            raise BacktestError("No-trade equity does not reconcile with initial capital")
        return

    expected_before = float(initial_capital)
    for expected_trade_id, trade in ledger.iterrows():
        entry_position = int(trade["entry_position"])
        exit_position = int(trade["exit_position"])
        quantity = float(trade["position_quantity"])
        equity_before = float(trade["equity_before_entry"])
        equity_after = float(trade["equity_after_exit"])
        if int(trade["trade_id"]) != expected_trade_id + 1:
            raise BacktestError("Trade identifiers are not sequential")
        if not _is_close(equity_before, expected_before):
            raise BacktestError("Trade equity does not chain from the previous trade")
        expected_entry_fee = equity_before * float(trade["fee_rate"])
        expected_entry_execution = quantity * (
            float(trade["entry_fill_price"]) - float(trade["entry_market_price"])
        )
        expected_exit_execution = quantity * (
            float(trade["exit_market_price"]) - float(trade["exit_fill_price"])
        )
        expected_exit_fee = quantity * float(trade["exit_fill_price"]) * float(trade["fee_rate"])
        expected_after = quantity * float(trade["exit_fill_price"]) - expected_exit_fee
        expected_entry_mark = quantity * float(trade["entry_market_price"])
        entry_mark = float(equity_curve.iloc[entry_position - window_start]["equity"])
        exit_mark = float(equity_curve.iloc[exit_position - window_start]["equity"])
        checks = (
            (float(trade["entry_fee"]), expected_entry_fee),
            (float(trade["entry_execution_cost"]), expected_entry_execution),
            (float(trade["exit_fee"]), expected_exit_fee),
            (float(trade["exit_execution_cost"]), expected_exit_execution),
            (equity_after, expected_after),
            (entry_mark, expected_entry_mark),
            (exit_mark, equity_after),
            (float(trade["pnl"]), equity_after - equity_before),
            (float(trade["net_return"]), equity_after / equity_before - 1.0),
        )
        if not all(_is_close(actual, expected) for actual, expected in checks):
            raise BacktestError("Trade ledger does not reconcile with transaction cash flows")
        expected_before = equity_after

    compounded_trades = initial_capital * float(
        np.prod(1.0 + ledger["net_return"].to_numpy(dtype="float64"))
    )
    if not _is_close(final_equity, compounded_trades):
        raise BacktestError("Trade ledger does not reconcile with final equity")
    if not _is_close(float(ledger["pnl"].sum()), final_equity - initial_capital):
        raise BacktestError("Trade ledger PnL does not reconcile with final equity")


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
    window_start, window_end = resolve_performance_window(
        market_df, probability_scores.index, horizon
    )

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

    for position in range(len(market_df)):
        row = market_df.iloc[position]
        market_open = float(row["open"])
        held_during_prior_interval = active_trade is not None

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
                "exit_position": scheduled_exit,
                "fee_rate": cost_config.fee_rate,
                "probability_score": score_by_position[signal_position],
            }
            cash = 0.0
            pending_entry = None

        equity = cash if active_trade is None else quantity * market_open
        if window_start <= position <= window_end:
            equity_rows.append(
                {
                    "timestamp": row["timestamp"],
                    "equity": equity,
                    "position_open": active_trade is not None,
                    "market_exposure": (
                        np.nan if position == window_start else float(held_during_prior_interval)
                    ),
                }
            )

        if position in score_by_position and active_trade is None and pending_entry is None:
            if (
                score_by_position[position] >= signal_threshold
                and position + horizon + 1 <= window_end
            ):
                pending_entry = position + 1
                scheduled_exit = position + horizon + 1

    if active_trade is not None or pending_entry is not None or scheduled_exit is not None:
        raise BacktestError("Backtest ended with an unresolved position or pending order")

    ledger_with_positions = pd.DataFrame(ledger_rows)
    equity_curve = pd.DataFrame(equity_rows, index=index_by_position[window_start : window_end + 1])
    if equity_curve.empty or not np.isfinite(equity_curve["equity"]).all():
        raise BacktestError("Backtest produced an invalid equity curve")
    equity_curve = _add_interval_accounting(equity_curve, initial_capital)
    _validate_reconciliation(ledger_with_positions, equity_curve, initial_capital, window_start)
    ledger = ledger_with_positions
    if not ledger.empty:
        ledger = ledger.drop(columns=["entry_position", "exit_position", "fee_rate"])
    return BacktestResult(
        ledger, equity_curve, initial_capital, probability_scores.copy(), actual_labels
    )
