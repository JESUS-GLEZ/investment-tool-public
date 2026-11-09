"""Value supplied holdings and estimate a rebalance without recording any fills."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields
from investment_tool.domain.market_time import observable_history
from investment_tool.domain.model_data.model_data import ModelData


def reference_prices(
    model: ModelData, symbols: set[str]
) -> tuple[pd.Series, dict[str, dict[str, Any]]]:
    """Use completed, available closes, with provenance for every priced symbol."""
    period = model.instance.current_trading_period
    if period is None or period.decision_time is None:
        raise ValueError("A recommendation requires an explicit decision time")
    stocks = {stock.symbol: stock for stock in model.stocks}
    prices: dict[str, float] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for symbol in sorted(symbols):
        if symbol not in stocks:
            raise ValueError(f"No usable price history for holding/target {symbol}")
        history = observable_history(
            stocks[symbol].data, period.trading_date, period.decision_time
        )
        if history.empty:
            raise ValueError(f"No observable price for holding/target {symbol}")
        row = history.iloc[-1]
        price = float(row[StockFields.CLOSE])
        if not np.isfinite(price) or price <= 0:
            raise ValueError(f"Invalid latest completed price for {symbol}")
        prices[symbol] = price
        provenance[symbol] = {
            "price_period_end": (pd.Timestamp(row["period_end"]) - pd.Timedelta(days=1))
            .date()
            .isoformat(),
            "price_available_at": pd.Timestamp(row["available_at"]).isoformat(),
        }
    return pd.Series(prices, dtype=float), provenance


def initialize_holdings(model: ModelData, holdings: dict[str, float]) -> None:
    """Imported quantities already reflect events before the decision instant."""
    if any(not np.isfinite(q) or q <= 0 for q in holdings.values()):
        raise ValueError("Holding quantities must be finite and positive")
    prices, _ = reference_prices(model, set(holdings))
    wealth = model.portfolio.cash + sum(q * prices[s] for s, q in holdings.items())
    if (
        not np.isfinite(model.portfolio.cash)
        or model.portfolio.cash < 0
        or not np.isfinite(wealth)
        or wealth <= 0
    ):
        raise ValueError(
            "A recommendation requires nonnegative cash and positive total value"
        )
    if holdings:
        period = model.instance.current_trading_period
        assert period is not None and period.decision_time is not None
        if model.portfolio.positions or model.portfolio.history:
            raise ValueError("Initial holdings require a new portfolio state")
        # Account for past events while empty, then import today's quantities.
        # Otherwise a past split/merger could be applied to already-adjusted units.
        model.portfolio.apply_security_actions(
            model.corporate_action_events, period.decision_time
        )
        model.portfolio.positions = dict(holdings)


def estimate_rebalance(model: ModelData, weights: pd.Series) -> dict[str, Any]:
    """Reuse portfolio funding/fee arithmetic on a copy at reference prices."""
    if (
        weights.index.has_duplicates
        or not np.isfinite(weights.to_numpy()).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError(
            "Recommendation weights must be finite, long-only and sum to one"
        )
    current = model.portfolio
    symbols = set(current.positions) | set(weights[weights > 0].index)
    prices, provenance = reference_prices(model, symbols)
    value = current.calculate_portfolio_value(prices)
    preview = deepcopy(current)
    # Share-count dust is inappropriate for fractional crypto recommendations.
    preview.min_position = np.finfo(float).tiny
    preview.update(weights, execution_prices=prices)
    after = preview.calculate_portfolio_value(prices)
    if not np.isfinite(after) or after <= 0:
        raise ValueError("Estimated costs leave no positive portfolio value")
    rows = []
    for symbol in sorted(symbols):
        quantity = current.positions.get(symbol, 0.0)
        estimated = preview.positions.get(symbol, 0.0)
        trade = estimated - quantity
        price = float(prices[symbol])
        rows.append(
            {
                "symbol": symbol,
                "current_quantity": quantity,
                "reference_price": price,
                **provenance[symbol],
                "current_value": quantity * price,
                "current_weight": quantity * price / value,
                "target_weight": float(weights.get(symbol, 0.0)),
                "estimated_quantity": estimated,
                "estimated_weight": estimated * price / after,
                "trade_quantity": trade,
                "trade_value": trade * price,
                "estimated_cost": abs(trade * price) * current.transaction_cost_rate,
                "action": "hold"
                if abs(trade) <= 1e-12 * max(quantity, estimated, 1e-12)
                else "buy"
                if trade > 0
                else "sell",
            }
        )
    return {
        "currency": "USD",
        "price_basis": "latest_completed_available_close",
        "current_cash": current.cash,
        "current_value": value,
        "estimated_cash": preview.cash,
        "estimated_value": after,
        "estimated_cost": sum(row["estimated_cost"] for row in rows),
        "rows": rows,
    }
