"""Small independent synthetic four-cell cost, baseline, and ablation checks."""

from __future__ import annotations

import socket
from typing import Any

import pytest

from crypto_ai.phase2.backtests import METRIC_KEYS, _calculate, _render_report
from crypto_ai.sentiment.canonical import canonicalize
from phase2.test_backtests import labeled_decisions, market, prediction_frames


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("synthetic Milestone 7 calculations must remain offline")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_four_cell_results_share_window_and_have_complete_cost_baseline_matrix() -> None:
    bars = market(hours=60)
    frames = prediction_frames(bars)
    strategy, costs, baselines, ablation = _calculate(
        bars, frames, labeled_decisions(bars), start=31, end=49, random_simulations=5
    )
    window = strategy["common_window"]
    assert window["first_open_ordinal"] == 31
    assert window["final_open_ordinal"] == 49
    assert window["equity_marks"] == 19
    assert window["open_to_open_intervals"] == 18
    assert set(strategy["cells"]) == set("ABCD")
    assert set(costs["scenarios"]) == {"low", "base", "high"}
    for cell in "ABCD":
        assert set(strategy["cells"][cell]) == {"low", "base", "high"}
        assert set(baselines["random_exposure"][cell]) == {"low", "base", "high"}
        assert baselines["random_exposure"][cell]["base"]["simulations"] == 5
        ledgers = [
            strategy["cells"][cell][scenario]["trade_ledger"]
            for scenario in ("low", "base", "high")
        ]
        assert [
            [(row["entry_timestamp"], row["exit_timestamp"]) for row in ledger]
            for ledger in ledgers
        ] == [[(row["entry_timestamp"], row["exit_timestamp"]) for row in ledgers[0]]] * 3
        returns = [
            costs["cells"][cell][scenario]["total_return"] for scenario in ("low", "base", "high")
        ]
        assert returns[0] >= returns[1] >= returns[2]
        for scenario in ("low", "base", "high"):
            assert len(strategy["cells"][cell][scenario]["equity_curve"]) == 19
    assert set(baselines["deterministic"]["base"]) == {
        "cash",
        "buy_and_hold",
        "ema_9_21",
        "momentum_24",
    }
    for scenario in ("low", "base", "high"):
        for name in baselines["deterministic"][scenario]:
            assert len(baselines["deterministic"][scenario][name]["equity_curve"]) == 19
    assert len(ablation["folds"]) == 5
    assert [row["fold_number"] for row in ablation["folds"]] == [1, 2, 3, 4, 5]
    assert all(row["common_window"]["equity_marks"] == 7 for row in ablation["folds"])


def test_direct_ablation_uses_percentage_point_return_deltas_and_null_safety() -> None:
    bars = market(hours=60)
    strategy, _, _, ablation = _calculate(
        bars,
        prediction_frames(bars),
        labeled_decisions(bars),
        start=31,
        end=49,
        random_simulations=5,
    )
    for scenario in ("low", "base", "high"):
        metrics = {cell: strategy["cells"][cell][scenario]["metrics"] for cell in "ABCD"}
        direct = ablation["direct_deltas"][scenario]
        assert direct["linear_C_minus_A"]["total_return"]["delta"] == pytest.approx(
            100.0 * (metrics["C"]["total_return"] - metrics["A"]["total_return"])
        )
        assert direct["nonlinear_D_minus_B"]["total_return"]["delta"] == pytest.approx(
            100.0 * (metrics["D"]["total_return"] - metrics["B"]["total_return"])
        )
        assert direct["linear_C_minus_A"]["annualized_return"]["delta"] == pytest.approx(
            100.0 * (metrics["C"]["annualized_return"] - metrics["A"]["annualized_return"])
        )
        assert direct["nonlinear_D_minus_B"]["annualized_return"]["delta"] == pytest.approx(
            100.0 * (metrics["D"]["annualized_return"] - metrics["B"]["annualized_return"])
        )
        for key, control, augmented in (
            ("linear_C_minus_A", "A", "C"),
            ("nonlinear_D_minus_B", "B", "D"),
        ):
            assert set(direct[key]) == set(METRIC_KEYS)
            for metric in METRIC_KEYS:
                comparison = direct[key][metric]
                assert set(comparison) == {"control_metric", "augmented_metric", "delta"}
                assert comparison["control_metric"] == metrics[control][metric]
                assert comparison["augmented_metric"] == metrics[augmented][metric]
                if comparison["control_metric"] is None or comparison["augmented_metric"] is None:
                    assert comparison["delta"] is None
                elif metric in ("total_return", "annualized_return"):
                    assert comparison["delta"] == pytest.approx(
                        100.0 * (metrics[augmented][metric] - metrics[control][metric])
                    )
                else:
                    assert comparison["delta"] == pytest.approx(
                        metrics[augmented][metric] - metrics[control][metric]
                    )
        if metrics["C"]["profit_factor"] is None or metrics["A"]["profit_factor"] is None:
            assert direct["linear_C_minus_A"]["profit_factor"]["delta"] is None
    folds = ablation["folds"]
    assert ablation["fold_consistency"]["linear_C_minus_A"]["positive_fold_count"] == sum(
        row["linear_C_minus_A_return_percentage_points"] > 0 for row in folds
    )
    assert ablation["research_gates_evaluated"] is False


def test_ablation_markdown_shows_all_matched_scenario_metrics_and_replays_exactly() -> None:
    bars = market(hours=60)
    frames = prediction_frames(bars)
    labeled = labeled_decisions(bars)
    strategy, costs, baselines, ablation = _calculate(
        bars, frames, labeled, start=31, end=49, random_simulations=5
    )
    strategy["classification_context"] = {
        cell: {"f1_class_1": 0.5, "roc_auc": 0.5, "log_loss": 0.7} for cell in "ABCD"
    }
    metadata = {
        "source_run_id": "synthetic-source",
        "source_manifest_sha256": "a" * 64,
        "source_experiment_id": "synthetic-experiment",
        "prepared_dataset_id": "synthetic-prepared",
        "market_price_context_sha256": "b" * 64,
        "decision_window_sha256": "c" * 64,
    }
    first = _render_report(None, metadata, strategy, costs, baselines, ablation, {})
    second = _render_report(None, metadata, strategy, costs, baselines, ablation, {})
    assert first == second
    markdown = first.decode("utf-8")
    assert "| Comparison | Scenario | Metric | Control | Augmented | Delta |" in markdown
    for scenario in ("low", "base", "high"):
        for label in ("Linear C–A", "Nonlinear D–B"):
            for metric in METRIC_KEYS:
                prefix = f"| {label} | {scenario.title()} | {metric} | "
                assert sum(line.startswith(prefix) for line in markdown.splitlines()) == 1
        comparison = ablation["direct_deltas"][scenario]["linear_C_minus_A"]["total_return"]
        assert (
            f"| Linear C–A | {scenario.title()} | total_return | "
            f"{100.0 * comparison['control_metric']:.4f}% | "
            f"{100.0 * comparison['augmented_metric']:.4f}% | "
            f"{comparison['delta']:.4f} pp |"
        ) in markdown
    strategy_again, _, _, ablation_again = _calculate(
        bars, frames, labeled, start=31, end=49, random_simulations=5
    )
    assert canonicalize(ablation) == canonicalize(ablation_again)
    strategy.pop("classification_context")
    assert canonicalize(strategy) == canonicalize(strategy_again)
