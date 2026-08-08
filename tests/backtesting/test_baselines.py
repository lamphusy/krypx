"""Deterministic baseline and common-window tests."""

import numpy as np
import pandas as pd
import pytest

import crypto_ai.backtesting.baselines as baseline_module
from crypto_ai.backtesting.baselines import (
    buy_and_hold_backtest,
    cash_baseline,
    random_exposure_summary,
    rule_backtest,
    run_cost_sensitivity,
)
from crypto_ai.backtesting.engine import run_backtest
from crypto_ai.backtesting.metrics import calculate_backtest_metrics
from crypto_ai.costs import CostConfig
from crypto_ai.exceptions import BacktestError


@pytest.fixture
def baseline_context() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    index = pd.RangeIndex(50, 74)
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(index), freq="h", tz="UTC"),
            "open": 100.0 + np.sin(np.arange(len(index)) / 2.0) + np.arange(len(index)) * 0.2,
        },
        index=index,
    )
    scores = pd.Series([0.8, 0.2, 0.9, 0.1, 0.7, 0.3], index=index[2:8])
    labels = pd.Series([1, 0, 1, 0, 1, 0], index=scores.index, dtype="int8")
    return market, scores, labels


def test_random_exposure_baseline_is_reproducible(
    baseline_context: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    market, scores, labels = baseline_context
    costs = CostConfig(0.001, 2.0, 1.0)
    kwargs = {
        "market_df": market,
        "model_scores": scores,
        "actual_labels": labels,
        "horizon": 2,
        "timeframe": "1h",
        "cost_config": costs,
        "model_total_return": 0.01,
        "simulations": 12,
        "random_seed": 20260808,
    }
    assert random_exposure_summary(**kwargs) == random_exposure_summary(**kwargs)


def test_all_strategies_use_the_same_common_performance_window(
    baseline_context: tuple[pd.DataFrame, pd.Series, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market, scores, labels = baseline_context
    costs = CostConfig(0.001, 2.0, 1.0)
    model = run_backtest(market, scores, labels, 2, "1h", 0.5, 10_000, costs)
    expected_index = model.equity_curve.index
    captured_indexes: list[pd.Index] = []
    original_run_backtest = baseline_module.run_backtest

    def capture_window(*args: object, **kwargs: object) -> object:
        result = original_run_backtest(*args, **kwargs)
        captured_indexes.append(result.equity_curve.index)
        return result

    monkeypatch.setattr(baseline_module, "run_backtest", capture_window)
    cash = cash_baseline(market, scores.index, labels, 2, "1h", costs)
    ema = rule_backtest(
        market,
        scores.index,
        pd.Series(True, index=market.index),
        labels,
        2,
        "1h",
        costs,
    )
    momentum = rule_backtest(
        market,
        scores.index,
        pd.Series([position % 2 == 0 for position in range(len(market))], index=market.index),
        labels,
        2,
        "1h",
        costs,
    )
    buy_hold = buy_and_hold_backtest(market, scores.index, 2, costs)
    random_exposure_summary(
        market,
        scores,
        labels,
        2,
        "1h",
        costs,
        calculate_backtest_metrics(model, "1h")["total_return"],
        simulations=4,
    )

    for result in (cash, ema, momentum, buy_hold):
        assert result.equity_curve.index.equals(expected_index)
        assert result.n_intervals == model.n_intervals
    assert captured_indexes
    assert all(index.equals(expected_index) for index in captured_indexes)


def test_cost_sensitivity_changes_only_execution_costs(
    baseline_context: tuple[pd.DataFrame, pd.Series, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market, scores, labels = baseline_context
    observed_inputs: list[tuple[pd.Series, pd.Series | None, CostConfig]] = []
    original_run_backtest = baseline_module.run_backtest

    def capture_inputs(*args: object, **kwargs: object) -> object:
        observed_inputs.append((args[1], args[2], args[7]))
        return original_run_backtest(*args, **kwargs)

    monkeypatch.setattr(baseline_module, "run_backtest", capture_inputs)
    scenarios = {
        "low": {
            "fee_rate": 0.001,
            "slippage_bps_per_side": 1.0,
            "half_spread_bps_per_side": 0.5,
        },
        "base": {
            "fee_rate": 0.001,
            "slippage_bps_per_side": 2.0,
            "half_spread_bps_per_side": 1.0,
        },
        "high": {
            "fee_rate": 0.001,
            "slippage_bps_per_side": 5.0,
            "half_spread_bps_per_side": 2.0,
        },
    }
    results = run_cost_sensitivity(
        market,
        scores,
        labels,
        2,
        "1h",
        cost_scenarios=scenarios,
        initial_capital=10_000,
    )

    assert len(observed_inputs) == len(scenarios)
    assert all(observed_scores is scores for observed_scores, _, _ in observed_inputs)
    assert all(observed_labels is labels for _, observed_labels, _ in observed_inputs)
    assert {cost.fee_rate for _, _, cost in observed_inputs} == {0.001}
    assert [cost.one_side_execution_rate for _, _, cost in observed_inputs] == sorted(
        cost.one_side_execution_rate for _, _, cost in observed_inputs
    )
    assert len({metrics["num_trades"] for metrics in results.values()}) == 1
    assert len({metrics["market_exposure"] for metrics in results.values()}) == 1
    assert results["low"]["total_estimated_costs"] < results["high"]["total_estimated_costs"]


def test_cost_sensitivity_does_not_regenerate_labels_or_retrain_model(
    baseline_context: tuple[pd.DataFrame, pd.Series, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market, scores, labels = baseline_context
    calls = 0
    original_run_backtest = baseline_module.run_backtest

    def frozen_prediction_backtest(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        assert args[1] is scores
        assert args[2] is labels
        return original_run_backtest(*args, **kwargs)

    monkeypatch.setattr(baseline_module, "run_backtest", frozen_prediction_backtest)
    scenarios = {
        "low": {
            "fee_rate": 0.001,
            "slippage_bps_per_side": 0.0,
            "half_spread_bps_per_side": 0.0,
        },
        "high": {
            "fee_rate": 0.001,
            "slippage_bps_per_side": 5.0,
            "half_spread_bps_per_side": 2.0,
        },
    }
    run_cost_sensitivity(
        market,
        scores,
        labels,
        2,
        "1h",
        cost_scenarios=scenarios,
    )
    assert calls == len(scenarios)


def test_cost_sensitivity_rejects_changed_fee_assumption(
    baseline_context: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    market, scores, labels = baseline_context
    with pytest.raises(BacktestError, match="fee assumption unchanged"):
        run_cost_sensitivity(
            market,
            scores,
            labels,
            2,
            "1h",
            cost_scenarios={
                "low": {
                    "fee_rate": 0.0,
                    "slippage_bps_per_side": 1.0,
                    "half_spread_bps_per_side": 0.5,
                },
                "high": {
                    "fee_rate": 0.001,
                    "slippage_bps_per_side": 5.0,
                    "half_spread_bps_per_side": 2.0,
                },
            },
        )
