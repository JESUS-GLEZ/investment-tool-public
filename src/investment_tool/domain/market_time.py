"""Provider-independent bar intervals and availability rules.

Indexes are calendar labels, never instants. All instant columns are aware UTC.
Adapters declare the source calendar and timestamp convention before returning
bars; the domain never infers either from a timestamp's hour.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd

from investment_tool.constants import Frequency

TIME_CONTRACT_VERSION = 1
# Shared execution policy; adapters and local-data preflight enforce the same bound.
MAX_CRYPTO_EXECUTION_DELAY = pd.Timedelta(minutes=2)
TIME_COLUMNS = (
    "period_end",
    "bar_start",
    "bar_end",
    "available_at",
    "is_complete",
    "market_timezone",
)


def utc_instant(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    """Convert an explicitly zoned instant; naive values are ambiguous."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"Expected a timezone-aware instant, got {value!r}")
    return timestamp.tz_convert("UTC")


def period_code(frequency: Frequency) -> str:
    """Canonical calendar for every configurable bar frequency."""
    codes = {Frequency.DAILY: "D", Frequency.WEEKLY: "W-SUN", Frequency.MONTHLY: "M"}
    if frequency not in codes:
        raise ValueError(f"Unsupported bar frequency: {frequency}")
    return codes[frequency]


def period_bounds(
    labels: pd.DatetimeIndex, frequency: Frequency
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Calendar interval [start, end), independent of provider timestamps."""
    periods = labels.to_period(period_code(frequency))
    return periods.start_time, periods.shift(1).start_time


def normalize_bars(
    data: pd.DataFrame,
    frequency: Frequency,
    *,
    market_timezone: str,
    timestamp_convention: Literal["start", "end"] = "start",
    coverage_start: str | datetime | pd.Timestamp | None = None,
    coverage_end: str | datetime | pd.Timestamp | None = None,
    as_of: str | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Adapter helper for calendar bars labelled at either interval boundary.

    ``end`` means the exclusive right boundary, not an inclusive last-day date.
    Availability at the interval end is a conservative documented proxy, not an
    observed publication time. Providers with publication delays can override
    available_at. Coverage limits prevent truncated requests being called full.
    """
    ZoneInfo(market_timezone)
    if timestamp_convention not in ("start", "end"):
        raise ValueError("Unknown timestamp convention")
    result = data.copy()
    if result.empty:
        return result
    timestamps = pd.DatetimeIndex([utc_instant(v) for v in result.index])
    local = timestamps.tz_convert(market_timezone).tz_localize(None)
    source_labels = local
    if timestamp_convention == "end":
        local = local - pd.Timedelta(1, unit="ns")
    starts, ends = period_bounds(local, frequency)
    if not source_labels.equals(starts if timestamp_convention == "start" else ends):
        raise ValueError("Source labels do not match the declared calendar boundary")
    if starts.has_duplicates:
        raise ValueError("Multiple bars map to the same calendar period")
    result.index = starts
    result["period_end"] = ends
    result["bar_start"] = starts.tz_localize(market_timezone).tz_convert("UTC")
    result["bar_end"] = ends.tz_localize(market_timezone).tz_convert("UTC")
    result["available_at"] = result["bar_end"]
    result["market_timezone"] = market_timezone
    now = utc_instant(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    result["is_complete"] = result["bar_end"] <= now
    if coverage_start is not None:
        result["is_complete"] &= result["bar_start"] >= utc_instant(coverage_start)
    if coverage_end is not None:
        result["is_complete"] &= result["bar_end"] <= utc_instant(coverage_end)
    result = result.sort_index()
    return validate_bar_times(result, frequency)


def validate_bar_times(data: pd.DataFrame, frequency: Frequency) -> pd.DataFrame:
    """Validate and restore serialized timing types without inventing times."""
    result = data.copy()
    if result.empty:
        return result
    missing = set(TIME_COLUMNS) - set(result.columns)
    if missing:
        raise ValueError(
            f"Missing bar time contract fields: {sorted(missing)}. "
            "Use a versioned dataset or explicitly upgrade its known source convention."
        )
    labels = pd.DatetimeIndex(pd.to_datetime(result.index))
    if labels.tz is not None or not labels.equals(labels.normalize()):
        raise ValueError(
            "Bar indexes must be timezone-free calendar labels at midnight"
        )
    if labels.has_duplicates or not labels.is_monotonic_increasing or labels.hasnans:
        raise ValueError("Bar period labels must be unique, sorted and non-null")
    expected_starts, expected_ends = period_bounds(labels, frequency)
    result.index = labels.rename("period")
    result["period_end"] = pd.to_datetime(result["period_end"])
    if not labels.equals(expected_starts) or not pd.DatetimeIndex(
        result["period_end"]
    ).equals(expected_ends):
        raise ValueError("Bar calendar boundaries do not match the declared frequency")
    for column in ("bar_start", "bar_end", "available_at"):
        result[column] = pd.DatetimeIndex([utc_instant(v) for v in result[column]])
    if not result["is_complete"].map(lambda v: isinstance(v, bool)).all():
        raise ValueError("Bar completeness must be boolean")
    for zone in result["market_timezone"].unique():
        if not isinstance(zone, str):
            raise ValueError("Missing market timezone")
        ZoneInfo(zone)
        rows = result["market_timezone"] == zone
        calendar_start = labels[rows].tz_localize(zone).tz_convert("UTC")
        calendar_end = expected_ends[rows].tz_localize(zone).tz_convert("UTC")
        if (pd.DatetimeIndex(result.loc[rows, "bar_start"]) < calendar_start).any() or (
            pd.DatetimeIndex(result.loc[rows, "bar_end"]) > calendar_end
        ).any():
            raise ValueError("Bar interval falls outside its labelled calendar period")
    if (result["bar_start"] >= result["bar_end"]).any() or (
        result["available_at"] < result["bar_end"]
    ).any():
        raise ValueError(
            "Bar intervals must be positive and availability cannot precede completion"
        )
    return result[
        [c for c in result.columns if c not in TIME_COLUMNS] + list(TIME_COLUMNS)
    ]


def historical_window(
    data: pd.DataFrame,
    frequency: Frequency,
    start: str | date,
    end: str | date,
    *,
    as_of: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Complete calendar intervals within inclusive market-local date controls."""
    result = validate_bar_times(data, frequency)
    if result.empty:
        return result
    lower = pd.Timestamp(start)
    upper = pd.Timestamp(end) + pd.DateOffset(days=1)
    if lower > pd.Timestamp(end):
        raise ValueError("Historical start date is after end date")
    now = utc_instant(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    eligible = (result.index >= lower) & (result["period_end"] <= upper)
    eligible &= result["is_complete"] & (result["available_at"] <= now)
    for zone in result["market_timezone"].unique():
        cutoff = upper.tz_localize(zone).tz_convert("UTC")
        eligible &= (result["market_timezone"] != zone) | (
            result["available_at"] <= cutoff
        )
    return pd.DataFrame(result.loc[eligible])


def observable_history(
    data: pd.DataFrame, period_label: datetime, cutoff: datetime
) -> pd.DataFrame:
    """Completed earlier periods whose aggregate was available by the decision."""
    if "available_at" not in data.columns:
        raise ValueError("Strategy inputs lack bar availability information")
    eligible = (data.index < pd.Timestamp(period_label)) & data["is_complete"]
    eligible &= data["available_at"] <= utc_instant(cutoff)
    return pd.DataFrame(data.loc[eligible].sort_index())
