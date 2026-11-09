"""Shared dated-universe history rules, independent of the data provider.

Legacy universe manifests specify a minimum number of monthly returns. Preserve
that calendar duration when using weekly or daily bars, rather than admitting a
new listing after only twelve weeks/days. Daily calendars are explicit schedules,
not business-day guesses or the observations of the candidate being checked.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from investment_tool.constants import Frequency, StockFields
from investment_tool.domain.market_time import period_code


def history_labels(
    trading_date: datetime,
    minimum_months: int,
    frequency: Frequency,
    calendar: pd.DatetimeIndex | None = None,
    calendar_end: date | None = None,
) -> pd.DatetimeIndex:
    """Prior bar closes spanning the lookback, including its preceding close.

        Weekly windows round outward to whole weeks. Daily windows follow the supplied
        asset calendar, including holidays or continuous trading. An empty daily window
        means the schedule does not extend far enough to establish the preceding close.
    ``calendar_end`` records coverage of closed days after the final session; without
    it, coverage is conservatively limited to the last scheduled session.
    """
    current = pd.Timestamp(trading_date).to_period(period_code(frequency))
    lower = current.start_time - pd.DateOffset(months=minimum_months)
    if frequency != Frequency.DAILY:
        first = lower.to_period(period_code(frequency))
        return pd.period_range(first - 1, current - 1).start_time
    if calendar is None:
        raise ValueError("Daily return history requires an explicit session calendar")
    covered_until = (
        pd.Timestamp(calendar_end)
        if calendar_end is not None
        else (calendar[-1] if len(calendar) else pd.NaT)
    )
    if pd.isna(covered_until) or covered_until < current.start_time - pd.Timedelta(
        days=1
    ):
        raise ValueError(
            "Daily return history calendar does not cover the decision window"
        )
    first_index = int(calendar.searchsorted(lower))
    last_index = int(calendar.searchsorted(current.start_time))
    if first_index == 0:
        return pd.DatetimeIndex([])
    return calendar[first_index - 1 : last_index]


def has_return_history(
    data: pd.DataFrame,
    trading_date: datetime,
    minimum_months: int,
    frequency: Frequency,
    calendar: pd.DatetimeIndex | None = None,
    calendar_end: date | None = None,
) -> bool:
    """Require every expected, positive close and finite adjacent return.

    Callers first restrict data to completed bars available at the decision cutoff.
    Reindexing detects absent observations without borrowing later data or filling.
    """
    labels = history_labels(
        trading_date, minimum_months, frequency, calendar, calendar_end
    )
    if len(labels) < 2:
        return False
    closes = data[StockFields.CLOSE].reindex(labels)
    returns = closes.pct_change(fill_method=None).iloc[1:]
    return bool(
        np.isfinite(closes).all() and (closes > 0).all() and np.isfinite(returns).all()
    )


def adjacent_observations(
    labels: pd.DatetimeIndex,
    frequency: Frequency,
    calendar: pd.DatetimeIndex | None = None,
) -> np.ndarray:
    """Identify single-period returns; older holes must not enter estimators."""
    if labels.empty:
        return np.array([], dtype=bool)
    if frequency == Frequency.DAILY:
        if calendar is None:
            raise ValueError(
                "Daily return history requires an explicit session calendar"
            )
        positions = calendar.get_indexer(labels)
        adjacent = (np.diff(positions) == 1) & (positions[:-1] >= 0)
        adjacent &= positions[1:] >= 0
    else:
        positions = np.asarray(
            labels.to_period(period_code(frequency)).astype("int64"), dtype=np.int64
        )
        adjacent = np.diff(positions) == 1
    return np.asarray(np.r_[False, adjacent], dtype=bool)
