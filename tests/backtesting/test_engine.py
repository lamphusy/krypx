"""Manual execution and state-machine tests."""

import numpy as np
import pandas as pd
import pytest

from crypto_ai.backtesting.baselines import buy_and_hold_backtest
from crypto_ai.backtesting.engine import BacktestResult, run_backtest
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


def _backtest(
    market_context: pd.DataFrame,
    scores: pd.Series,
    *,
    horizon: int = 2,
    costs: CostConfig | None = None,
) -> BacktestResult:
    return run_backtest(
        market_context,
        scores,
        None,
        horizon,
        "1h",
        0.5,
        10_000,
        costs or CostConfig(0.0, 0.0, 0.0),
    )


def test_signal_enters_at_next_open(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9], index=[10]))
    assert result.trade_ledger.iloc[0]["entry_timestamp"] == market_context.loc[11, "timestamp"]


def test_trade_holds_exact_horizon(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9], index=[10]), horizon=3)
    trade = result.trade_ledger.iloc[0]
    assert trade["exit_timestamp"] == market_context.loc[14, "timestamp"]
    assert trade["holding_candles"] == 3


def test_backtest_uses_price_context_beyond_final_decision_row(
    market_context: pd.DataFrame,
) -> None:
    result = _backtest(market_context, pd.Series([0.9], index=[16]))
    assert result.trade_ledger.iloc[0]["exit_timestamp"] == market_context.loc[19, "timestamp"]
    assert result.equity_curve.index[-1] == 19


def test_market_context_rows_do_not_generate_signals(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.0], index=[16]))
    assert result.trade_ledger.empty
    assert result.probability_scores.index.equals(pd.Index([16]))


def test_trade_invests_full_current_equity(market_context: pd.DataFrame) -> None:
    costs = CostConfig(0.001, 2.0, 1.0)
    result = _backtest(market_context, pd.Series([0.9], index=[10]), costs=costs)
    trade = result.trade_ledger.iloc[0]
    assert trade["position_quantity"] * trade["entry_fill_price"] + trade[
        "entry_fee"
    ] == pytest.approx(trade["equity_before_entry"])


def test_new_signals_are_ignored_while_position_open(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9, 0.9, 0.9], index=[10, 11, 12]))
    assert result.trade_ledger["signal_timestamp"].tolist() == [market_context.loc[10, "timestamp"]]


def test_entry_fee_is_charged(market_context: pd.DataFrame) -> None:
    result = _backtest(
        market_context,
        pd.Series([0.9], index=[10]),
        costs=CostConfig(0.001, 0.0, 0.0),
    )
    assert result.trade_ledger.iloc[0]["entry_fee"] == pytest.approx(10.0)


def test_exit_fee_is_charged(market_context: pd.DataFrame) -> None:
    result = _backtest(
        market_context,
        pd.Series([0.9], index=[10]),
        costs=CostConfig(0.001, 0.0, 0.0),
    )
    trade = result.trade_ledger.iloc[0]
    assert trade["exit_fee"] == pytest.approx(
        trade["position_quantity"] * trade["exit_fill_price"] * 0.001
    )


def test_slippage_is_adverse_on_entry(market_context: pd.DataFrame) -> None:
    result = _backtest(
        market_context,
        pd.Series([0.9], index=[10]),
        costs=CostConfig(0.0, 2.0, 0.0),
    )
    trade = result.trade_ledger.iloc[0]
    assert trade["entry_fill_price"] > trade["entry_market_price"]


def test_slippage_is_adverse_on_exit(market_context: pd.DataFrame) -> None:
    result = _backtest(
        market_context,
        pd.Series([0.9], index=[10]),
        costs=CostConfig(0.0, 2.0, 0.0),
    )
    trade = result.trade_ledger.iloc[0]
    assert trade["exit_fill_price"] < trade["exit_market_price"]


def test_spread_is_adverse_on_both_sides(market_context: pd.DataFrame) -> None:
    result = _backtest(
        market_context,
        pd.Series([0.9], index=[10]),
        costs=CostConfig(0.0, 0.0, 1.0),
    )
    trade = result.trade_ledger.iloc[0]
    assert trade["entry_fill_price"] > trade["entry_market_price"]
    assert trade["exit_fill_price"] < trade["exit_market_price"]


def test_no_fee_when_no_trade_occurs(market_context: pd.DataFrame) -> None:
    result = _backtest(
        market_context,
        pd.Series([0.0], index=[10]),
        costs=CostConfig(0.001, 2.0, 1.0),
    )
    metrics = calculate_backtest_metrics(result, "1h")
    assert result.trade_ledger.empty
    assert metrics["total_estimated_costs"] == 0.0


def test_no_overlapping_trades(market_context: pd.DataFrame) -> None:
    scores = pd.Series(0.9, index=[10, 11, 12, 13, 14, 15, 16])
    result = _backtest(market_context, scores)
    exits = result.trade_ledger["exit_timestamp"].iloc[:-1].reset_index(drop=True)
    next_entries = result.trade_ledger["entry_timestamp"].iloc[1:].reset_index(drop=True)
    assert (next_entries > exits).all()


def test_final_open_position_is_not_left_unresolved(market_context: pd.DataFrame) -> None:
    # The final score lacks enough context for a full holding period and is ignored;
    # the earlier accepted position still exits at its pre-scheduled open.
    result = _backtest(market_context, pd.Series([0.9, 0.9], index=[10, 21]))
    assert len(result.trade_ledger) == 1
    assert not result.equity_curve.iloc[-1]["position_open"]


def test_num_trades_matches_trade_ledger(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9, 0.9], index=[10, 13]))
    metrics = calculate_backtest_metrics(result, "1h")
    assert metrics["num_trades"] == len(result.trade_ledger) == 2


def test_trade_return_matches_manual_calculation(market_context: pd.DataFrame) -> None:
    costs = CostConfig(0.001, 2.0, 1.0)
    result = _backtest(market_context, pd.Series([0.9], index=[10]), costs=costs)
    execution = costs.one_side_execution_rate
    expected = (103.0 / 101.0) * (1.0 - execution) / (1.0 + execution)
    expected *= (1.0 - costs.fee_rate) ** 2
    assert result.trade_ledger.iloc[0]["net_return"] == pytest.approx(expected - 1.0)


def test_equity_curve_has_no_nan(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9], index=[10]))
    assert not result.equity_curve[["timestamp", "equity", "position_open"]].isna().any().any()
    # The initial mark deliberately has no preceding interval; all actual interval
    # observations are complete.
    assert not result.interval_returns.isna().any()
    assert not result.interval_exposure.isna().any()


def test_equity_curve_has_no_infinity(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9], index=[10]))
    assert np.isfinite(result.equity_curve["equity"]).all()
    assert np.isfinite(result.interval_returns).all()
    assert np.isfinite(result.interval_exposure).all()


def test_equity_index_matches_market_index(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9, 0.0, 0.0], index=[10, 11, 12]))
    assert result.equity_curve.index.equals(market_context.loc[11:15].index)
    assert result.interval_returns.index.equals(market_context.loc[12:15].index)


def test_zero_volatility_sharpe_is_not_infinite(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.0], index=[10]))
    assert calculate_backtest_metrics(result, "1h")["sharpe_ratio"] is None


def test_max_drawdown_is_zero_or_negative(market_context: pd.DataFrame) -> None:
    result = _backtest(market_context, pd.Series([0.9], index=[10]))
    assert calculate_backtest_metrics(result, "1h")["maximum_drawdown"] <= 0.0


def test_buy_hold_uses_entry_and_exit_costs(market_context: pd.DataFrame) -> None:
    result = buy_and_hold_backtest(
        market_context,
        pd.Index([10, 11, 12]),
        2,
        CostConfig(0.001, 2.0, 1.0),
    )
    trade = result.trade_ledger.iloc[0]
    assert (
        trade[["entry_fee", "exit_fee", "entry_execution_cost", "exit_execution_cost"]]
        .gt(0.0)
        .all()
    )


def test_buy_hold_exposure_is_one(market_context: pd.DataFrame) -> None:
    result = buy_and_hold_backtest(
        market_context,
        pd.Index([10, 11, 12]),
        2,
        CostConfig(0.0, 0.0, 0.0),
    )
    assert calculate_backtest_metrics(result, "1h")["market_exposure"] == 1.0


def test_ledger_reconciles_with_equity_curve(market_context: pd.DataFrame) -> None:
    result = _backtest(
        market_context,
        pd.Series([0.9, 0.9], index=[10, 13]),
        costs=CostConfig(0.001, 2.0, 1.0),
    )
    ledger = result.trade_ledger
    final_equity = result.equity_curve["equity"].iloc[-1]
    compounded_trades = 10_000 * np.prod(1.0 + ledger["net_return"])
    compounded_intervals = 10_000 * np.prod(1.0 + result.interval_returns)
    assert final_equity == pytest.approx(compounded_trades)
    assert final_equity == pytest.approx(compounded_intervals)
    assert ledger["pnl"].sum() == pytest.approx(final_equity - 10_000)

    for _, trade in ledger.iterrows():
        entry_equity = result.equity_curve.loc[
            result.equity_curve["timestamp"] == trade["entry_timestamp"], "equity"
        ].iloc[0]
        exit_equity = result.equity_curve.loc[
            result.equity_curve["timestamp"] == trade["exit_timestamp"], "equity"
        ].iloc[0]
        assert trade["equity_before_entry"] - entry_equity == pytest.approx(
            trade["entry_fee"] + trade["entry_execution_cost"]
        )
        assert trade["position_quantity"] * trade["exit_market_price"] - exit_equity == (
            pytest.approx(trade["exit_fee"] + trade["exit_execution_cost"])
        )
