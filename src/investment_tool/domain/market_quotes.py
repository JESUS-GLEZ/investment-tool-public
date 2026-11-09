"""Exact-period execution and valuation quotes shared by audit and accounting."""

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from investment_tool.constants import StockFields
from investment_tool.domain.model_data.entities.security_action import SecurityAction
from investment_tool.domain.model_data.entities.period import Period


def period_prices(
    period: Period, field: StockFields = StockFields.CLOSE
) -> dict[str, float]:
    """No fallback to another period, forward fill, or stale quote."""
    if period.market_data.empty:
        return {}
    if field not in period.market_data.columns.get_level_values(0):
        return {}
    rows = period.market_data.xs(field, level=0, axis=1)
    if period.trading_date not in rows.index:
        return {}
    return {
        str(symbol): float(price)
        for symbol, price in rows.loc[period.trading_date].items()
        if np.isfinite(price) and price > 0
    }


def due_settlements(
    events: Sequence[SecurityAction], cutoff: datetime | None
) -> set[str]:
    return {
        event.symbol
        for event in events
        if not event.retain_original
        and cutoff is not None
        and max(event.effective_at, event.published_at) <= cutoff
    }


def execution_prices(
    period: Period, crypto_symbols: set[str], events: Sequence[SecurityAction]
) -> dict[str, float]:
    prices = period_prices(period, StockFields.OPEN)
    if (
        period.execution_time is not None
        and "bar_start" in period.market_data.columns.get_level_values(0)
    ):
        starts = period.market_data["bar_start"].iloc[0]
        prices = {
            symbol: price
            for symbol, price in prices.items()
            if symbol in starts and starts[symbol] <= period.execution_time
        }
    for symbol in crypto_symbols:
        prices.pop(symbol, None)
    prices.update(
        (symbol, price)
        for symbol, price in period.crypto_execution_prices.items()
        if np.isfinite(price) and price > 0
    )
    # A reviewed redemption effective before the open cannot be bought again,
    # even if its old symbol still has a stale-looking vendor bar.
    for symbol in due_settlements(events, period.execution_time):
        prices.pop(symbol, None)
    return prices


def valuation_time(period: Period) -> datetime | None:
    if "bar_end" not in period.market_data.columns.get_level_values(0):
        return None
    ends = period.market_data["bar_end"].iloc[0].dropna()
    return None if ends.empty else ends.max().to_pydatetime()
