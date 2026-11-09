"""Calendar labels, UTC instants, complete coverage and observable inputs."""

from datetime import datetime

import pandas as pd
import pytest

from investment_tool.constants import Frequency
from investment_tool.domain.market_time import (
    historical_window,
    normalize_bars,
    observable_history,
    validate_bar_times,
)


def bars(index):
    return pd.DataFrame(
        {"open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0},
        index=index,
    )


@pytest.mark.parametrize(
    "frequency,freq",
    [(Frequency.DAILY, "D"), (Frequency.WEEKLY, "W-MON"), (Frequency.MONTHLY, "MS")],
)
@pytest.mark.parametrize("zone", ["UTC", "America/New_York", "Europe/Madrid"])
def test_start_and_end_labelled_providers_return_identical_contract(
    frequency, freq, zone
):
    starts = pd.date_range("2024-01-01", periods=4, freq=freq, tz=zone)
    first = normalize_bars(bars(starts), frequency, market_timezone=zone)
    second = normalize_bars(
        bars(pd.DatetimeIndex(first.bar_end)),
        frequency,
        market_timezone=zone,
        timestamp_convention="end",
    )
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize("day,hours", [("2024-03-10", 23), ("2024-11-03", 25)])
def test_daily_intervals_follow_daylight_saving_not_fixed_offsets(day, hours):
    result = normalize_bars(
        bars(pd.DatetimeIndex([day], tz="America/New_York")),
        Frequency.DAILY,
        market_timezone="America/New_York",
    )
    row = result.iloc[0]
    assert row.bar_end - row.bar_start == pd.Timedelta(hours=hours)


def test_leap_february_and_week_crossing_year():
    month = normalize_bars(
        bars(pd.DatetimeIndex(["2024-02-01"], tz="UTC")),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )
    assert month.iloc[0].bar_end - month.iloc[0].bar_start == pd.Timedelta(days=29)
    week = normalize_bars(
        bars(pd.DatetimeIndex(["2024-12-30"], tz="UTC")),
        Frequency.WEEKLY,
        market_timezone="UTC",
    )
    assert week.iloc[0].period_end == pd.Timestamp("2025-01-06")


@pytest.mark.parametrize(
    "end,expected",
    [
        ("2024-08-01", ["2024-07-01"]),
        ("2024-08-20", ["2024-07-01"]),
        ("2024-08-31", ["2024-07-01", "2024-08-01"]),
    ],
)
@pytest.mark.parametrize("zone", ["UTC", "America/New_York"])
def test_month_window_uses_full_intervals_in_declared_market_calendar(
    end, expected, zone
):
    frame = normalize_bars(
        bars(pd.date_range("2024-07-01", periods=3, freq="MS", tz=zone)),
        Frequency.MONTHLY,
        market_timezone=zone,
    )
    result = historical_window(frame, Frequency.MONTHLY, "2024-07-01", end)
    assert list(result.index) == list(pd.to_datetime(expected))


def test_partial_request_and_current_month_never_become_complete_later():
    frame = normalize_bars(
        bars(pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")),
        Frequency.MONTHLY,
        market_timezone="UTC",
        coverage_start="2024-01-15T00:00:00Z",
        coverage_end="2024-03-20T00:00:00Z",
        as_of="2024-03-10T00:00:00Z",
    )
    assert frame.is_complete.tolist() == [False, True, False]
    result = historical_window(
        frame,
        Frequency.MONTHLY,
        "2024-01-01",
        "2024-12-31",
        as_of=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    assert result.index.tolist() == [pd.Timestamp("2024-02-01")]


def test_publication_delay_blocks_history_even_with_earlier_label():
    frame = normalize_bars(
        bars(pd.date_range("2024-01-01", periods=3, freq="MS", tz="UTC")),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )
    frame.loc[pd.Timestamp("2024-02-01"), "available_at"] = pd.Timestamp(
        "2024-03-01T09:00:00Z"
    )
    result = observable_history(
        frame, datetime(2024, 3, 1), pd.Timestamp("2024-03-01T08:00:00Z")
    )
    assert result.index.tolist() == [pd.Timestamp("2024-01-01")]
    assert observable_history(
        frame, datetime(2024, 3, 1), pd.Timestamp("2024-03-01T09:00:00Z")
    ).index.tolist() == list(pd.to_datetime(["2024-01-01", "2024-02-01"]))


@pytest.mark.parametrize(
    "mutation",
    ["naive", "duplicate", "frequency", "availability", "boolean", "zone", "missing"],
)
def test_ambiguous_or_inconsistent_contract_is_rejected(mutation):
    frame = normalize_bars(
        bars(pd.date_range("2024-01-01", periods=2, freq="MS", tz="UTC")),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )
    if mutation == "naive":
        frame["available_at"] = frame["available_at"].dt.tz_localize(None)
    elif mutation == "duplicate":
        frame = pd.concat([frame, frame])
    elif mutation == "frequency":
        frame["period_end"] += pd.Timedelta(days=1)
    elif mutation == "availability":
        frame["available_at"] -= pd.Timedelta(days=1)
    elif mutation == "boolean":
        frame["is_complete"] = "true"
    elif mutation == "zone":
        frame["market_timezone"] = None
    else:
        frame = frame.drop(columns=["bar_end"])
    with pytest.raises((ValueError, KeyError)):
        validate_bar_times(frame, Frequency.MONTHLY)


def test_raw_naive_provider_timestamps_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_bars(
            bars(pd.DatetimeIndex(["2024-01-01"])),
            Frequency.MONTHLY,
            market_timezone="UTC",
        )


def test_wrong_source_calendar_boundary_cannot_be_silently_floored():
    with pytest.raises(ValueError, match="declared calendar boundary"):
        normalize_bars(
            bars(pd.DatetimeIndex(["2024-03-31T23:00:00Z"])),
            Frequency.MONTHLY,
            market_timezone="UTC",
        )


def test_actual_session_intervals_can_publish_before_calendar_midnight():
    frame = normalize_bars(
        bars(pd.DatetimeIndex(["2024-03-11"], tz="America/New_York")),
        Frequency.DAILY,
        market_timezone="America/New_York",
    )
    frame["bar_start"] = pd.Timestamp("2024-03-11T13:30:00Z")
    frame["bar_end"] = pd.Timestamp("2024-03-11T20:00:00Z")
    frame["available_at"] = pd.Timestamp("2024-03-11T20:05:00Z")
    validated = validate_bar_times(frame, Frequency.DAILY)
    assert not observable_history(
        validated, datetime(2024, 3, 12), pd.Timestamp("2024-03-11T20:05:00Z")
    ).empty


def test_interval_cannot_claim_a_different_month_than_its_label():
    frame = normalize_bars(
        bars(pd.DatetimeIndex(["2024-03-01"], tz="UTC")),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )
    frame["bar_start"] -= pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="outside its labelled"):
        validate_bar_times(frame, Frequency.MONTHLY)
