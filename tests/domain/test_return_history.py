"""History duration and gap detection must agree across data frequencies."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency
from investment_tool.domain.return_history import (
    adjacent_observations,
    has_return_history,
    history_labels,
)


@pytest.mark.parametrize(
    "frequency", [Frequency.MONTHLY, Frequency.WEEKLY, Frequency.DAILY]
)
def test_window_preserves_calendar_duration_and_preceding_close(frequency):
    date = (
        datetime(2025, 1, 6) if frequency != Frequency.MONTHLY else datetime(2025, 1, 1)
    )
    calendar = pd.bdate_range("2023-01-01", "2026-01-01")
    labels = history_labels(date, 12, frequency, calendar)
    assert labels.max() < pd.Timestamp(date)
    assert labels.min() < pd.Timestamp(date) - pd.DateOffset(months=12)
    assert len(labels) > 12  # Twelve daily/weekly observations are insufficient.
    if frequency == Frequency.MONTHLY:
        assert labels.equals(pd.date_range("2023-12-01", "2024-12-01", freq="MS"))
    if frequency == Frequency.WEEKLY:
        assert labels.min() == pd.Timestamp("2023-12-25")
        assert labels.max() == pd.Timestamp("2024-12-30")
    data = pd.DataFrame({"close": np.linspace(100, 120, len(labels))}, index=labels)
    assert has_return_history(data, date, 12, frequency, calendar)
    assert not has_return_history(data.iloc[-12:], date, 12, frequency, calendar)
    for gap in [0, len(labels) // 2, len(labels) - 1]:
        assert not has_return_history(
            data.drop(labels[gap]), date, 12, frequency, calendar
        )
    # Adding perfect future observations cannot repair missing earlier history.
    future = pd.DataFrame({"close": [1e9]}, index=[pd.Timestamp("2026-01-01")])
    assert not has_return_history(
        pd.concat([data.iloc[-12:], future]), date, 12, frequency, calendar
    )


@pytest.mark.parametrize("bad", [0, -1, np.nan, np.inf])
def test_nonfinite_or_nonpositive_close_cannot_establish_eligibility(bad):
    date = datetime(2025, 1, 6)
    labels = history_labels(date, 12, Frequency.WEEKLY)
    data = pd.DataFrame({"close": 100.0}, index=labels)
    data.loc[labels[10], "close"] = bad
    assert not has_return_history(data, date, 12, Frequency.WEEKLY)


def test_daily_sessions_respect_holidays_and_continuous_crypto():
    days = pd.date_range("2023-01-01", "2025-01-20")
    sessions = pd.bdate_range("2023-01-01", "2025-01-20").difference(
        pd.DatetimeIndex(["2024-12-25", "2025-01-01"])
    )
    date = datetime(2025, 1, 6)
    stocks = pd.DataFrame({"close": 100.0}, index=sessions)
    crypto = pd.DataFrame({"close": 100.0}, index=days)
    assert has_return_history(stocks, date, 12, Frequency.DAILY, sessions)
    assert has_return_history(crypto, date, 12, Frequency.DAILY, days)
    assert not has_return_history(
        crypto.drop(pd.Timestamp("2025-01-05")), date, 12, Frequency.DAILY, days
    )
    # Friday -> Monday and holiday -> next session are single equity returns.
    assert adjacent_observations(sessions, Frequency.DAILY, sessions)[1:].all()
    assert not adjacent_observations(sessions.delete(10), Frequency.DAILY, sessions)[10]
    assert adjacent_observations(days, Frequency.DAILY, days)[1:].all()
    assert not adjacent_observations(days.delete(10), Frequency.DAILY, days)[10]


def test_daily_requires_explicit_schedule_and_enough_calendar_coverage():
    date = datetime(2025, 1, 6)
    with pytest.raises(ValueError, match="explicit session calendar"):
        history_labels(date, 12, Frequency.DAILY)
    with pytest.raises(ValueError, match="explicit session calendar"):
        adjacent_observations(pd.DatetimeIndex(["2025-01-01"]), Frequency.DAILY)
    assert history_labels(
        date, 12, Frequency.DAILY, pd.bdate_range("2024-02-01", "2025-01-06")
    ).empty


def test_weekly_gap_is_not_a_one_week_return():
    labels = pd.date_range("2024-01-01", periods=5, freq="W-MON").delete(2)
    assert adjacent_observations(labels, Frequency.WEEKLY).tolist() == [
        False,
        True,
        False,
        True,
    ]


def test_daily_truncated_schedule_cannot_qualify_stale_prices():
    from datetime import date

    calendar = pd.bdate_range("2024-01-01", "2025-06-30")
    prices = pd.DataFrame({"close": 100.0}, index=calendar)
    with pytest.raises(ValueError, match="does not cover the decision window"):
        has_return_history(
            prices,
            datetime(2025, 10, 23),
            12,
            Frequency.DAILY,
            calendar,
            date(2025, 6, 30),
        )
    # Knowing weekend closure is sufficient: Monday requires Friday's close,
    # not invented Saturday/Sunday observations. An explicitly covered holiday
    # also does not require a bar on that day.
    for decision, end, holiday in [
        (datetime(2025, 1, 6), date(2025, 1, 5), None),
        (datetime(2025, 1, 21), date(2025, 1, 20), "2025-01-20"),
    ]:
        sessions = pd.bdate_range("2023-01-01", end)
        if holiday:
            sessions = sessions.drop(pd.Timestamp(holiday))
        prices = pd.DataFrame({"close": 100.0}, index=sessions)
        assert has_return_history(prices, decision, 12, Frequency.DAILY, sessions, end)
