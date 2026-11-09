"""
Period entity for time-based market data.

This module defines the Period class which represents a single trading
period (day, week, or month) with associated market data.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from investment_tool.domain.market_time import utc_instant


class Period:
    """A single trading period with associated market data.

    Represents one time unit (day, week, month) in the backtesting or
    live trading timeline.

    Parameters
    ----------
    trading_date : datetime
        The date/time of this trading period.
    market_data : pd.DataFrame
        OHLCV data for all assets in this period.
    period_end_date : date | None, optional
        End date of the period (for weekly/monthly grouping).

    Attributes
    ----------
    trading_date : datetime
    period_end_date : date | None
    market_data : pd.DataFrame
    """

    def __init__(
        self,
        trading_date: datetime,
        market_data: pd.DataFrame,
        period_end_date: date | None = None,
        *,
        execution_time: datetime | None = None,
        decision_time: datetime | None = None,
        crypto_execution_prices: dict[str, float] | None = None,
        eligible_symbols: frozenset[str] | None = None,
        minimum_return_periods: int | None = None,
    ) -> None:
        self.trading_date = trading_date
        self.period_end_date = period_end_date
        self.market_data = market_data
        # Candidate membership is separate from quotes needed to sell holdings.
        self.eligible_symbols = eligible_symbols
        # Execution availability constrains fills, never historical features.
        self.execution_eligible_symbols: frozenset[str] | None = None
        # Legacy manifest name: monthly-return count, hence a calendar-month
        # duration even when the run uses daily/weekly observations.
        self.minimum_return_periods = minimum_return_periods
        self.decision_time = (
            utc_instant(decision_time).to_pydatetime()
            if decision_time is not None
            else None
        )
        # Separate execution quotes from monthly features used for prediction.
        self.execution_time = (
            utc_instant(execution_time).to_pydatetime()
            if execution_time is not None
            else None
        )
        self.crypto_execution_prices = crypto_execution_prices or {}
        self.crypto_execution_times: dict[str, datetime] = {}

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"Period(date={self.trading_date.date()})"
