"""Adversarial synthetic checks for the frozen Milestone 7 trading contract."""

from __future__ import annotations

import math
import socket
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from crypto_ai.backtesting.metrics import calculate_backtest_metrics
from crypto_ai.costs import CostConfig
from crypto_ai.exceptions import CryptoAIError
from crypto_ai.phase2.backtests import (
    SCENARIOS,
    BacktestIntegrityError,
    _baseline_scores,
    _random_summaries,
    _result_payload,
    _simulate,
)


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Milestone 7 tests must stay wholly offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def market(*, hours: int = 24, prices: list[float] | None = None) -> pd.DataFrame:
    if prices is None:
        prices = [100.0] * hours
    assert len(prices) == hours
    stamps = pd.date_range(datetime(2026, 9, 1, tzinfo=UTC), periods=hours, freq="h")
    return pd.DataFrame(
        {
            "market_ordinal": np.arange(hours, dtype=np.int64),
            "timestamp": stamps,
            "open": np.asarray(prices, dtype=np.float64),
        },
        index=pd.RangeIndex(hours),
    )


def simulate(
    bars: pd.DataFrame,
    score_values: dict[int, float],
    *,
    cost: CostConfig | None = None,
    expected_start: int | None = None,
    expected_end: int | None = None,
):
    scores = pd.Series(score_values, dtype="float64")
    scores.index = pd.Index(score_values, dtype="int64")
    labels = pd.Series(0, index=scores.index, dtype="int8")
    start = min(score_values) + 1 if expected_start is None else expected_start
    end = max(score_values) + 5 if expected_end is None else expected_end
    return _simulate(
        bars,
        scores,
        labels,
        cost or CostConfig(0.001, 2.0, 1.0),
        expected_start=start,
        expected_end=end,
    )


def test_next_open_four_candles_multiplicative_costs_and_ledger_reconcile() -> None:
    prices = [100.0] * 20
    prices[5] = 104.0
    result = simulate(market(hours=20, prices=prices), {0: 0.5})
    ledger = result.trade_ledger
    curve = result.equity_curve
    assert len(ledger) == 1
    trade = ledger.iloc[0]
    assert trade.entry_timestamp == datetime(2026, 9, 1, 1, tzinfo=UTC)
    assert trade.exit_timestamp == datetime(2026, 9, 1, 5, tzinfo=UTC)
    assert trade.holding_candles == 4
    assert trade.entry_market_price == 100.0
    assert trade.exit_market_price == 104.0
    assert trade.entry_fill_price == pytest.approx(100.0 * 1.0003)
    assert trade.exit_fill_price == pytest.approx(104.0 * 0.9997)
    expected_growth = (104.0 * 0.9997 / (100.0 * 1.0003)) * 0.999**2
    assert trade.net_return == pytest.approx(expected_growth - 1.0)
    assert trade.position_quantity == pytest.approx(10_000.0 * 0.999 / (100.0 * 1.0003))
    assert curve.equity.iloc[-1] == pytest.approx(10_000.0 * expected_growth)
    assert curve.equity.iloc[-1] == pytest.approx(10_000.0 + ledger.pnl.sum())
    assert curve.equity.iloc[-1] == pytest.approx(
        10_000.0 * math.prod(1.0 + result.interval_returns.to_numpy())
    )
    diagnostics = _result_payload(result)["reconciliation"]
    assert diagnostics["passed"] is True
    assert diagnostics["verified"] is True
    assert all(diagnostics["checks"].values())
    assert diagnostics["relative_tolerance"] == 1e-10
    assert diagnostics["absolute_tolerance_usd"] == 1e-8
    assert diagnostics["final_equity_curve_usd"] == pytest.approx(10_000.0 * expected_growth)
    assert diagnostics["compounded_trade_net_returns_usd"] == pytest.approx(
        10_000.0 * expected_growth
    )
    assert diagnostics["compounded_interval_returns_usd"] == pytest.approx(
        10_000.0 * expected_growth
    )
    assert diagnostics["initial_capital_plus_trade_pnl_usd"] == pytest.approx(
        10_000.0 * expected_growth
    )
    expected_entry_deduction = 10_000.0 - float(curve.equity.iloc[0])
    expected_exit_deduction = trade.position_quantity * 104.0 - float(curve.equity.iloc[-1])
    expected_total_costs = expected_entry_deduction + expected_exit_deduction
    assert diagnostics["equity_curve_entry_deductions_usd"] == pytest.approx(
        expected_entry_deduction
    )
    assert diagnostics["equity_curve_exit_deductions_usd"] == pytest.approx(expected_exit_deduction)
    assert diagnostics["ledger_entry_fees_usd"] == pytest.approx(trade.entry_fee)
    assert diagnostics["ledger_exit_fees_usd"] == pytest.approx(trade.exit_fee)
    assert diagnostics["ledger_entry_execution_costs_usd"] == pytest.approx(
        trade.entry_execution_cost
    )
    assert diagnostics["ledger_exit_execution_costs_usd"] == pytest.approx(
        trade.exit_execution_cost
    )
    assert diagnostics["equity_curve_total_cost_deductions_usd"] == pytest.approx(
        expected_total_costs
    )
    assert diagnostics["ledger_total_estimated_costs_usd"] == pytest.approx(expected_total_costs)
    assert diagnostics["metric_total_estimated_costs_usd"] == pytest.approx(expected_total_costs)


def test_exit_at_open_precedes_next_close_signal_and_new_entry_one_open_later() -> None:
    # Signals while a position is held are ignored; a signal on the exit candle
    # can only schedule a new entry at the following open, not at the exit open.
    signals = {0: 1.0, 1: 1.0, 4: 1.0, 5: 1.0, 9: 1.0, 10: 1.0}
    result = simulate(market(), signals)
    ledger = result.trade_ledger
    assert ledger.entry_timestamp.tolist() == [
        datetime(2026, 9, 1, hour, tzinfo=UTC) for hour in (1, 6, 11)
    ]
    assert ledger.exit_timestamp.tolist() == [
        datetime(2026, 9, 1, hour, tzinfo=UTC) for hour in (5, 10, 15)
    ]
    assert ledger.holding_candles.tolist() == [4, 4, 4]
    assert (ledger.entry_timestamp.shift(-1).dropna() > ledger.exit_timestamp.iloc[:-1]).all()
    growth = (0.9997 / 1.0003) * 0.999**2
    assert result.equity_curve.equity.iloc[-1] == pytest.approx(10_000.0 * growth**3)


def test_transaction_mark_reconciliation_rejects_forged_ledger_cost() -> None:
    result = simulate(market(), {0: 1.0})
    forged_ledger = result.trade_ledger.copy()
    forged_ledger.loc[0, "entry_fee"] += 1.0
    with pytest.raises(BacktestIntegrityError, match="ledger costs"):
        _result_payload(replace(result, trade_ledger=forged_ledger))


def test_provider_gap_in_decisions_never_compresses_original_holding_period() -> None:
    # Decision ordinals 1..7 are absent, but all original market opens remain.
    result = simulate(market(), {0: 1.0, 8: 1.0})
    ledger = result.trade_ledger
    assert ledger.holding_candles.tolist() == [4, 4]
    assert ledger.entry_timestamp.tolist() == [
        datetime(2026, 9, 1, hour, tzinfo=UTC) for hour in (1, 9)
    ]
    assert ledger.exit_timestamp.tolist() == [
        datetime(2026, 9, 1, hour, tzinfo=UTC) for hour in (5, 13)
    ]
    assert len(result.equity_curve) == 13
    assert result.n_intervals == 12


def test_terminal_decision_without_its_scheduled_exit_fails_closed() -> None:
    # The Phase 1 helper can otherwise silently discard this late decision.
    with pytest.raises(CryptoAIError):
        simulate(market(hours=15), {10: 1.0}, expected_start=11, expected_end=15)


@pytest.mark.parametrize("mutation", ["missing_ordinal", "timestamp_gap", "bad_price"])
def test_market_context_must_be_contiguous_positive_and_hourly(mutation: str) -> None:
    bars = market()
    if mutation == "missing_ordinal":
        bars = bars.drop(index=8)
    elif mutation == "timestamp_gap":
        bars.loc[8, "timestamp"] += pd.Timedelta(hours=1)
    else:
        bars.loc[8, "open"] = 0.0
    with pytest.raises(CryptoAIError):
        simulate(bars, {0: 1.0, 10: 1.0})


@pytest.mark.parametrize("expected_start,expected_end", [(2, 5), (1, 6)])
def test_performance_window_cannot_be_shifted(expected_start: int, expected_end: int) -> None:
    with pytest.raises(CryptoAIError):
        simulate(market(), {0: 1.0}, expected_start=expected_start, expected_end=expected_end)


def test_higher_costs_reduce_return_without_changing_trade_schedule() -> None:
    bars = market(hours=22)
    signals = {0: 1.0, 5: 1.0, 10: 1.0}
    low = simulate(bars, signals, cost=CostConfig(0.001, 1.0, 0.5))
    base = simulate(bars, signals, cost=CostConfig(0.001, 2.0, 1.0))
    high = simulate(bars, signals, cost=CostConfig(0.001, 5.0, 2.0))
    for right in (base, high):
        assert (
            right.trade_ledger.entry_timestamp.tolist() == low.trade_ledger.entry_timestamp.tolist()
        )
        assert (
            right.trade_ledger.exit_timestamp.tolist() == low.trade_ledger.exit_timestamp.tolist()
        )
        assert right.equity_curve.index.equals(low.equity_curve.index)
    assert low.equity_curve.equity.iloc[-1] > base.equity_curve.equity.iloc[-1]
    assert base.equity_curve.equity.iloc[-1] > high.equity_curve.equity.iloc[-1]


def test_cash_path_has_zero_return_no_trades_and_no_invented_risk_ratios() -> None:
    result = simulate(market(), {0: 0.0, 4: 0.49, 10: 0.0})
    metrics = calculate_backtest_metrics(result, "1h")
    assert result.trade_ledger.empty
    assert result.equity_curve.equity.eq(10_000.0).all()
    assert result.interval_exposure.eq(0.0).all()
    assert metrics["total_return"] == 0.0
    assert metrics["num_trades"] == 0
    assert metrics["market_exposure"] == 0.0
    assert metrics["total_estimated_costs"] == 0.0
    assert metrics["sharpe_ratio"] is None
    assert metrics["sortino_ratio"] is None
    assert metrics["profit_factor"] is None
    assert metrics["win_rate"] is None
    payload = _result_payload(result)
    warnings = payload["metrics"]["warnings"]
    for key, value in payload["metrics"].items():
        if value is None:
            assert any(warning.startswith(f"{key} is undefined because ") for warning in warnings)
    assert any(
        "profit_factor is undefined because there are no completed trades" in warning
        for warning in warnings
    )
    assert any(
        "calmar_ratio is undefined because maximum drawdown is zero" in warning
        for warning in warnings
    )
    assert payload["reconciliation"]["verified"] is True
    assert payload["reconciliation"]["passed"] is True
    assert payload["reconciliation"]["ledger_total_estimated_costs_usd"] == 0.0
    assert payload["reconciliation"]["equity_curve_total_cost_deductions_usd"] == 0.0


@pytest.mark.parametrize("malformed", [float("nan"), float("inf"), -0.01, 1.01])
def test_malformed_probability_fails_closed(malformed: float) -> None:
    with pytest.raises(CryptoAIError):
        simulate(market(), {0: malformed})


def prediction_frames(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordinals = list(range(30, 45))
    probabilities = {
        "A": [0.9 if index % 2 else 0.1 for index in range(len(ordinals))],
        "B": [0.1 if index % 2 else 0.9 for index in range(len(ordinals))],
        "C": [0.9] * len(ordinals),
        "D": [0.9] * len(ordinals),
    }
    output = {}
    for cell, scores in probabilities.items():
        rows = []
        for offset, ordinal in enumerate(ordinals):
            signal = int(scores[offset] >= 0.5)
            rows.append(
                {
                    "market_ordinal": ordinal,
                    "decision_at": bars.iloc[ordinal + 1].timestamp,
                    "entry_timestamp": bars.iloc[ordinal + 1].timestamp,
                    "exit_timestamp": bars.iloc[ordinal + 5].timestamp,
                    "fold_number": offset // 3 + 1,
                    "actual_label": offset % 2,
                    "probability_score": scores[offset],
                    "predicted_label": signal,
                    "signal": signal,
                }
            )
        output[cell] = pd.DataFrame(rows)
    return output


def labeled_decisions(bars: pd.DataFrame) -> pd.DataFrame:
    ordinals = np.arange(30, 45, dtype=np.int64)
    return pd.DataFrame(
        {
            "market_ordinal": ordinals,
            "decision_at": bars.iloc[ordinals + 1].timestamp.to_numpy(),
            "ema_short": np.full(len(ordinals), 101.0),
            "ema_long": np.full(len(ordinals), 100.0),
            "return_24": np.where(ordinals % 2 == 0, 0.1, 0.0),
            "label": np.arange(len(ordinals)) % 2,
        }
    ).set_index("market_ordinal", drop=False)


def test_ema_is_a_state_rule_and_momentum_cutoff_is_strictly_positive() -> None:
    bars = market(hours=60)
    frames = prediction_frames(bars)
    labeled = labeled_decisions(bars)
    ema, momentum = _baseline_scores(frames, labeled)
    assert ema.index.tolist() == list(range(30, 45))
    assert ema.tolist() == [1.0] * 15
    assert momentum.tolist() == [float(value % 2 == 0) for value in range(30, 45)]


def test_random_exposure_uses_42_plus_simulation_index_and_is_replayable() -> None:
    bars = market(hours=60)
    frames = prediction_frames(bars)
    labels = pd.Series(0, index=pd.Index(range(30, 45)), dtype="int8")
    model_metrics = {
        cell: {scenario: {"total_return": 0.0} for scenario in SCENARIOS} for cell in frames
    }
    first = _random_summaries(bars, frames, labels, model_metrics, 31, 49, simulations=5)
    second = _random_summaries(bars, frames, labels, model_metrics, 31, 49, simulations=5)
    assert first == second
    assert first["A"]["base"]["simulations"] == 5
    assert first["A"]["base"]["seed_base"] == 42
    probability = sum(frames["A"].signal) / 15
    expected_returns = []
    for simulation in range(5):
        rng = np.random.default_rng(42 + simulation)
        draws = pd.Series(
            rng.binomial(1, probability, 15).astype(np.float64),
            index=pd.Index(range(30, 45)),
        )
        result = _simulate(
            bars,
            draws,
            labels,
            SCENARIOS["base"],
            expected_start=31,
            expected_end=49,
        )
        expected_returns.append(calculate_backtest_metrics(result, "1h")["total_return"])
    assert first["A"]["base"]["total_return"]["median"] == pytest.approx(
        float(np.median(expected_returns))
    )
    assert first["A"]["base"]["total_return"]["p05"] == pytest.approx(
        float(np.percentile(expected_returns, 5))
    )
    assert first["A"]["base"]["total_return"]["p95"] == pytest.approx(
        float(np.percentile(expected_returns, 95))
    )
