"""Decision-time proportional turnover costs for portfolio return objectives.

This is a first-order forecast adjustment, not a second accounting charge.
Execution still funds and charges actual trades at their eventual fill prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields
from investment_tool.domain.market_time import observable_history
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.entities.security_action import SecurityAction


@dataclass(frozen=True)
class TradingCosts:
    rate: float
    current_weights: np.ndarray
    outside_weight: float = 0.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.current_weights, dtype=float).copy()
        if (
            not np.isfinite(self.rate)
            or self.rate < 0
            or weights.ndim != 1
            or not np.isfinite(weights).all()
            or (weights < 0).any()
            or not np.isfinite(self.outside_weight)
            or self.outside_weight < 0
            or weights.sum() + self.outside_weight > 1 + 1e-8
        ):
            raise ValueError("Invalid decision-time trading-cost state")
        weights.setflags(write=False)
        object.__setattr__(self, "current_weights", weights)

    def fraction(self, weights: np.ndarray) -> float:
        """Buys plus sells, including liquidation outside the candidate set."""
        if weights.shape != self.current_weights.shape:
            raise ValueError("Trading-cost weights must align with candidate weights")
        return float(
            self.rate
            * (np.abs(weights - self.current_weights).sum() + self.outside_weight)
        )

    def restrict(self, indices: np.ndarray) -> TradingCosts:
        """A fixed support must also pay to liquidate omitted holdings."""
        retained = self.current_weights[indices]
        return TradingCosts(
            self.rate,
            retained,
            max(
                0.0,
                self.outside_weight
                + float(self.current_weights.sum() - retained.sum()),
            ),
        )

    def record(self) -> dict[str, Any]:
        return {
            "version": 1,
            "rate": self.rate,
            "current_weights": self.current_weights.tolist(),
            "outside_weight": self.outside_weight,
            "return_basis": "expected_return_less_estimated_turnover_cost",
        }


def expected_net_return(
    mu: np.ndarray, weights: np.ndarray, costs: TradingCosts | None = None
) -> float:
    gross = float(weights @ mu)
    return gross if costs is None else gross - costs.fraction(weights)


def net_return_gradient(
    mu: np.ndarray, weights: np.ndarray, costs: TradingCosts | None = None
) -> np.ndarray:
    """Use the zero subgradient at the absolute-value turnover kink."""
    return (
        mu
        if costs is None
        else mu - costs.rate * np.sign(weights - costs.current_weights)
    )


def decision_trading_costs(
    portfolio: Portfolio,
    stocks: Sequence[Stock],
    period: Period | None,
    assets: pd.Index,
    events: Sequence[SecurityAction] = (),
) -> TradingCosts | None:
    """Mark held quantities using only completed, available prior closes.

    Project already known entitlements on a copy. Neither future opening quotes
    nor an action first available after the decision can change this estimate.
    An unpriced held security is an error, not silently free liquidation.
    """
    if portfolio.transaction_cost_rate == 0:
        return None
    if period is None or period.decision_time is None:
        raise ValueError("Cost-aware optimization requires an explicit decision cutoff")
    state = deepcopy(portfolio)
    state.apply_security_actions(events, period.decision_time)
    held_values: dict[str, float] = {}
    raw = {stock.symbol: stock for stock in stocks}
    for symbol, quantity in state.positions.items():
        if not quantity:
            continue
        if symbol not in raw:
            raise ValueError(f"No decision-time history for held asset {symbol}")
        history = observable_history(
            raw[symbol].data, period.trading_date, period.decision_time
        )
        closes = history[StockFields.CLOSE]
        closes = closes[np.isfinite(closes) & (closes > 0)]
        if closes.empty:
            raise ValueError(f"No decision-time price for held asset {symbol}")
        held_values[symbol] = quantity * float(closes.iloc[-1])
    wealth = state.cash + sum(held_values.values())
    if not np.isfinite(wealth) or wealth <= 0 or state.cash < -1e-8:
        raise ValueError("Cost-aware optimization requires positive unlevered wealth")
    return TradingCosts(
        state.transaction_cost_rate,
        np.array([held_values.get(str(symbol), 0.0) / wealth for symbol in assets]),
        sum(value for symbol, value in held_values.items() if symbol not in assets)
        / wealth,
    )
