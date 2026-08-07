"""Tests for shared multiplicative execution-cost calculations."""

import pytest

from crypto_ai.costs import (
    CostConfig,
    bps_to_rate,
    minimum_gross_return_for_net_edge,
    net_return_after_costs,
)


def test_bps_to_rate_converts_basis_points() -> None:
    """One basis point is one ten-thousandth in decimal-rate form."""
    assert bps_to_rate(5.0) == pytest.approx(0.0005)


def test_minimum_gross_return_reconciles_with_net_edge() -> None:
    """The threshold leaves exactly the requested edge after multiplicative costs."""
    config = CostConfig(
        fee_rate=0.001,
        slippage_bps_per_side=2.0,
        half_spread_bps_per_side=1.0,
    )
    threshold = minimum_gross_return_for_net_edge(
        fee_rate=config.fee_rate,
        slippage_bps_per_side=config.slippage_bps_per_side,
        half_spread_bps_per_side=config.half_spread_bps_per_side,
        minimum_net_edge_bps=5.0,
    )

    assert net_return_after_costs(threshold, config) == pytest.approx(bps_to_rate(5.0))


def test_zero_cost_threshold_equals_requested_edge() -> None:
    """Without execution costs, gross and net required returns are identical."""
    assert minimum_gross_return_for_net_edge(0.0, 0.0, 0.0, 10.0) == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("fee", "slippage", "spread"),
    [(-0.1, 1.0, 1.0), (1.0, 1.0, 1.0), (0.001, -1.0, 1.0), (0.001, 1.0, -1.0)],
)
def test_invalid_cost_config_is_rejected(fee: float, slippage: float, spread: float) -> None:
    """Fee and each adverse execution assumption must remain in their valid domain."""
    with pytest.raises(ValueError):
        CostConfig(fee, slippage, spread)


def test_invalid_gross_market_return_is_rejected() -> None:
    """A return cannot lose more than all invested market value."""
    with pytest.raises(ValueError, match="greater than -1.0"):
        net_return_after_costs(-1.0, CostConfig(0.001, 2.0, 1.0))
