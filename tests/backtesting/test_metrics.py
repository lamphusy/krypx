"""Open-to-open annualization, exposure, and drawdown metric tests."""

import numpy as np
import pandas as pd
import pytest

from crypto_ai.backtesting.engine import BacktestResult, run_backtest
from crypto_ai.backtesting.metrics import calculate_backtest_metrics, periods_per_year
from crypto_ai.costs import CostConfig


def _result(
    equities: list[float],
    *,
    initial_capital: float = 100.0,
    exposures: list[float] | None = None,
) -> BacktestResult:
    index = pd.RangeIndex(100, 100 + len(equities))
    timestamps = pd.date_range("2026-01-01", periods=len(equities), freq="h", tz="UTC")
    period_returns = np.full(len(equities), np.nan, dtype="float64")
    if len(equities) > 1:
        period_returns[1] = equities[1] / initial_capital - 1.0
    if len(equities) > 2:
        period_returns[2:] = np.asarray(equities[2:]) / np.asarray(equities[1:-1]) - 1.0
    interval_exposure = [np.nan] + (exposures or [0.0] * (len(equities) - 1))
    curve = pd.DataFrame(
        {
            "timestamp": timestamps,
            "equity": equities,
            "position_open": False,
            "market_exposure": interval_exposure,
            "period_return": period_returns,
        },
        index=index,
    )
    scores = pd.Series([0.0], index=[index[0]])
    return BacktestResult(pd.DataFrame(), curve, initial_capital, scores, None)


def test_annualization_uses_open_to_open_intervals() -> None:
    result = _result([100.0, 100.5, 101.0])
    metrics = calculate_backtest_metrics(result, "1d")
    expected = (101.0 / 100.0) ** (periods_per_year("1d") / 2) - 1.0
    assert result.n_equity_marks == 3
    assert result.n_intervals == 2
    assert metrics["n_intervals"] == 2
    assert metrics["annualized_return"] == pytest.approx(expected)


def test_compounded_interval_returns_reconcile_with_total_return() -> None:
    result = _result([99.0, 103.0, 98.0, 104.0])
    compounded = np.prod(1.0 + result.interval_returns) - 1.0
    metrics = calculate_backtest_metrics(result, "1h")
    assert compounded == pytest.approx(104.0 / 100.0 - 1.0)
    assert metrics["total_return"] == pytest.approx(compounded)


def test_cash_exposure_is_zero() -> None:
    metrics = calculate_backtest_metrics(_result([100.0, 100.0, 100.0]), "1h")
    assert metrics["market_exposure"] == 0.0


def test_partial_exposure_uses_intervals_not_equity_marks() -> None:
    result = _result([100.0, 100.0, 100.0, 100.0], exposures=[1.0, 0.0, 1.0])
    metrics = calculate_backtest_metrics(result, "1h")
    assert metrics["market_exposure"] == pytest.approx(2.0 / 3.0)


def test_drawdown_includes_initial_capital_and_flat_market_costs() -> None:
    market = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
            "open": 100.0,
        }
    )
    result = run_backtest(
        market,
        pd.Series([0.9], index=[0]),
        None,
        2,
        "1h",
        0.5,
        100.0,
        CostConfig(0.001, 2.0, 1.0),
    )
    metrics = calculate_backtest_metrics(result, "1h")
    assert metrics["maximum_drawdown"] == pytest.approx(result.trade_ledger.iloc[0]["net_return"])
    assert metrics["maximum_drawdown_duration"] == 2


def test_monotonically_rising_equity_has_no_drawdown() -> None:
    metrics = calculate_backtest_metrics(_result([100.0, 100.01, 100.02]), "1h")
    assert metrics["maximum_drawdown"] == 0.0
    assert metrics["maximum_drawdown_duration"] == 0


def test_recovered_drawdown_duration_counts_intervals() -> None:
    metrics = calculate_backtest_metrics(_result([100.0, 80.0, 100.0]), "1h")
    assert metrics["maximum_drawdown"] == pytest.approx(-0.2)
    assert metrics["maximum_drawdown_duration"] == 1


def test_unrecovered_terminal_drawdown_duration_counts_intervals() -> None:
    metrics = calculate_backtest_metrics(_result([100.0, 90.0, 80.0]), "1h")
    assert metrics["maximum_drawdown"] == pytest.approx(-0.2)
    assert metrics["maximum_drawdown_duration"] == 2


def test_cash_only_drawdown_is_zero() -> None:
    metrics = calculate_backtest_metrics(_result([100.0, 100.0, 100.0]), "1h")
    assert metrics["maximum_drawdown"] == 0.0
    assert metrics["maximum_drawdown_duration"] == 0
