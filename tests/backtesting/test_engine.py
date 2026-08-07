"""Manual execution and state-machine tests."""

import numpy as np
import pandas as pd
import pytest

from crypto_ai.backtesting.baselines import buy_and_hold_backtest
from crypto_ai.backtesting.engine import run_backtest
from crypto_ai.backtesting.metrics import calculate_backtest_metrics
from crypto_ai.costs import CostConfig


@pytest.fixture
def market_context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
            "open": np.arange(100.0, 112.0),
        },
        index=np.arange(10, 22),
    )


def test_signal_enters_next_open_and_holds_exact_horizon(market_context: pd.DataFrame) -> None:
    scores = pd.Series([0.9, 0.9, 0.0], index=[10, 11, 12])
    result = run_backtest(
        market_context,
        scores,
        None,
        2,
        "1h",
        0.5,
        10_000,
        CostConfig(0.0, 0.0, 0.0),
    )
    trade = result.trade_ledger.iloc[0]
    assert trade["entry_timestamp"] == market_context.loc[11, "timestamp"]
    assert trade["exit_timestamp"] == market_context.loc[13, "timestamp"]
    assert trade["holding_candles"] == 2
    assert len(result.trade_ledger) == 1


def test_trade_return_and_all_costs_reconcile(market_context: pd.DataFrame) -> None:
    cost = CostConfig(0.001, 2.0, 1.0)
    scores = pd.Series([0.9, 0.0, 0.0], index=[10, 11, 12])
    result = run_backtest(market_context, scores, None, 2, "1h", 0.5, 10_000, cost)
    trade = result.trade_ledger.iloc[0]
    execution = cost.one_side_execution_rate
    expected = (103 / 101) * (1 - execution) / (1 + execution) * (1 - cost.fee_rate) ** 2 - 1
    assert trade["net_return"] == pytest.approx(expected)
    assert trade["entry_fee"] > 0
    assert trade["exit_fee"] > 0
    assert trade["entry_execution_cost"] > 0
    assert trade["exit_execution_cost"] > 0
    assert result.equity_curve["equity"].iloc[-1] == pytest.approx(10_000 * (1 + expected))


def test_cash_backtest_has_no_trades_or_infinite_metrics(market_context: pd.DataFrame) -> None:
    scores = pd.Series(0.0, index=[10, 11, 12])
    result = run_backtest(market_context, scores, None, 2, "1h", 0.5, 10_000, CostConfig(0, 0, 0))
    metrics = calculate_backtest_metrics(result, "1h")
    assert result.trade_ledger.empty
    assert metrics["total_return"] == 0
    assert metrics["num_trades"] == 0
    assert metrics["market_exposure"] == 0
    assert metrics["sharpe_ratio"] is None
    assert metrics["maximum_drawdown"] <= 0


def test_exit_is_processed_before_exit_candle_signal(
    market_context: pd.DataFrame,
) -> None:
    scores = pd.Series([0.9, 0.9, 0.9], index=[10, 11, 13])
    result = run_backtest(
        market_context,
        scores,
        None,
        2,
        "1h",
        0.5,
        10_000,
        CostConfig(0, 0, 0),
    )
    assert result.trade_ledger["signal_timestamp"].tolist() == [
        market_context.loc[10, "timestamp"],
        market_context.loc[13, "timestamp"],
    ]
    assert result.trade_ledger["entry_timestamp"].tolist() == [
        market_context.loc[11, "timestamp"],
        market_context.loc[14, "timestamp"],
    ]
    assert not result.trade_ledger["entry_timestamp"].duplicated().any()


def test_second_trade_invests_all_updated_equity(market_context: pd.DataFrame) -> None:
    scores = pd.Series([0.9, 0.9], index=[10, 13])
    result = run_backtest(
        market_context,
        scores,
        None,
        2,
        "1h",
        0.5,
        10_000,
        CostConfig(0, 0, 0),
    )
    first, second = result.trade_ledger.iloc[0], result.trade_ledger.iloc[1]
    assert second["equity_before_entry"] == pytest.approx(first["equity_after_exit"])
    assert second["position_quantity"] * second["entry_fill_price"] == pytest.approx(
        second["equity_before_entry"]
    )


def test_tail_context_is_used_but_never_generates_a_signal(
    market_context: pd.DataFrame,
) -> None:
    scores = pd.Series([0.9, 0.9], index=[16, 19])
    result = run_backtest(
        market_context,
        scores,
        None,
        2,
        "1h",
        0.5,
        10_000,
        CostConfig(0, 0, 0),
    )
    assert len(result.trade_ledger) == 1
    assert result.trade_ledger.iloc[0]["exit_timestamp"] == market_context.loc[19, "timestamp"]
    assert result.equity_curve.index.equals(pd.Index([17, 18, 19]))
    assert np.isfinite(result.equity_curve["equity"]).all()


def test_buy_and_hold_applies_entry_and_exit_costs(market_context: pd.DataFrame) -> None:
    cost = CostConfig(0.001, 2.0, 1.0)
    result = buy_and_hold_backtest(market_context, pd.Index([10, 11, 12]), 2, cost)
    trade = result.trade_ledger.iloc[0]
    assert trade["entry_fee"] > 0
    assert trade["exit_fee"] > 0
    assert trade["entry_execution_cost"] > 0
    assert trade["exit_execution_cost"] > 0
    assert len(result.trade_ledger) == 1
