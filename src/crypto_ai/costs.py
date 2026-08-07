"""Shared multiplicative transaction-cost contracts."""

import math
from dataclasses import dataclass


def _validate_rate(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be finite, non-negative, and strictly below 1.0")


@dataclass(frozen=True)
class CostConfig:
    """Per-side fee and adverse execution assumptions."""

    fee_rate: float
    slippage_bps_per_side: float
    half_spread_bps_per_side: float

    def __post_init__(self) -> None:
        """Validate the cost assumptions when the immutable configuration is created."""
        _validate_rate("fee_rate", self.fee_rate)
        slippage_rate = bps_to_rate(self.slippage_bps_per_side)
        half_spread_rate = bps_to_rate(self.half_spread_bps_per_side)
        execution_rate = slippage_rate + half_spread_rate
        _validate_rate("one_side_execution_rate", execution_rate)

    @property
    def one_side_execution_rate(self) -> float:
        """Return combined per-side slippage and half-spread as a decimal rate."""
        return bps_to_rate(self.slippage_bps_per_side + self.half_spread_bps_per_side)


def bps_to_rate(bps: float) -> float:
    """Convert non-negative basis points to a decimal rate."""
    if not math.isfinite(bps) or bps < 0.0:
        raise ValueError("bps must be finite and non-negative")
    return bps / 10_000.0


def minimum_gross_return_for_net_edge(
    fee_rate: float,
    slippage_bps_per_side: float,
    half_spread_bps_per_side: float,
    minimum_net_edge_bps: float,
) -> float:
    """Return the gross market return required to retain a requested net edge."""
    config = CostConfig(
        fee_rate=fee_rate,
        slippage_bps_per_side=slippage_bps_per_side,
        half_spread_bps_per_side=half_spread_bps_per_side,
    )
    minimum_net_edge_rate = bps_to_rate(minimum_net_edge_bps)
    _validate_rate("minimum_net_edge_rate", minimum_net_edge_rate)
    execution_rate = config.one_side_execution_rate
    return (1.0 + minimum_net_edge_rate) * (1.0 + execution_rate) / (
        (1.0 - execution_rate) * (1.0 - config.fee_rate) ** 2
    ) - 1.0


def net_return_after_costs(gross_market_return: float, cost_config: CostConfig) -> float:
    """Apply adverse entry/exit execution and both fees to a gross market return."""
    if not math.isfinite(gross_market_return) or gross_market_return <= -1.0:
        raise ValueError("gross_market_return must be finite and greater than -1.0")
    execution_rate = cost_config.one_side_execution_rate
    net_growth = (
        (1.0 + gross_market_return)
        * (1.0 - execution_rate)
        / (1.0 + execution_rate)
        * (1.0 - cost_config.fee_rate) ** 2
    )
    return net_growth - 1.0
