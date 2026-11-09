"""
Instance entity for execution configuration.

This module defines the Instance class which holds configuration settings
and trading period references for a single execution run.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.domain.model_data.entities.period import Period


class Instance:
    """Execution instance configuration and trading period state.

    Holds the date range configuration for historical data and references
    to key trading periods (initial, previous, current) used during
    backtesting and live execution.

    Parameters
    ----------
    historical_start_date : date
        Start date of historical data range.
    historical_end_date : date
        End date of historical data range.
    frequency : Frequency
        Data frequency (DAILY, WEEKLY, MONTHLY).
    initial_trading_period : Period | None
        First trading period in the execution.
    previous_trading_period : Period | None
        Trading period immediately before current.
    current_trading_period : Period | None
        Current trading period being processed.

    Attributes
    ----------
    historical_start_date : date
    historical_end_date : date
    frequency : Frequency
    initial_trading_period : Period | None
    previous_trading_period : Period | None
    current_trading_period : Period | None
    """

    def __init__(
        self,
        historical_start_date: date,
        historical_end_date: date,
        frequency: Frequency,
        initial_trading_period: Period | None,
        previous_trading_period: Period | None,
        current_trading_period: Period | None,
    ) -> None:
        self.historical_start_date = historical_start_date
        self.historical_end_date = historical_end_date
        self.frequency = frequency
        self.initial_trading_period = initial_trading_period
        self.previous_trading_period = previous_trading_period
        self.current_trading_period = current_trading_period
        # Explicit per-asset calendars for daily history validation. References
        # are shared between equities and continuous assets, not copied per row.
        self.history_calendars: dict[str, pd.DatetimeIndex] = {}
        self.history_calendar_end: date | None = None

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"Instance(start={self.historical_start_date}, "
            f"end={self.historical_end_date}, freq={self.frequency})"
        )
