"""Keep declared security lifetimes separate before computing indicators.

Calendar dates are interpreted at midnight in each bar's declared market zone.
An identity end is exclusive. A bar spanning a boundary is not one security's
completed observation, even when all of its prices look numerically valid.
"""

from datetime import date

import numpy as np
import pandas as pd

from investment_tool.data.universes.models import AssetUniverse
from investment_tool.domain.model_data.entities.period import Period

from .validation_report import IssueSeverity, ValidationIssue, ValidationReport


def identity_history(
    frame: pd.DataFrame,
    *,
    symbol: str,
    start: date | None,
    end: date | None,
    source: str | None,
    report: ValidationReport,
) -> pd.DataFrame:
    """Exclude other identities, quarantine mixed bars, preserve valid opens.

    Requires the validated provider-independent bar time contract. Raw input
    snapshots are never modified. Missing closing observations deliberately
    reach lifecycle readiness; they cannot become invented sales or recoveries.
    """
    if frame.empty or (start is None and end is None):
        return frame
    overlaps = pd.Series(True, index=frame.index)
    complete = overlaps.copy()
    valid_open = overlaps.copy()
    for zone in frame["market_timezone"].unique():
        rows = frame["market_timezone"] == zone
        if start is not None:
            lower = pd.Timestamp(start).tz_localize(zone).tz_convert("UTC")
            overlaps.loc[rows] &= frame.loc[rows, "bar_end"] > lower
            complete.loc[rows] &= frame.loc[rows, "bar_start"] >= lower
            valid_open.loc[rows] &= frame.loc[rows, "bar_start"] >= lower
        if end is not None:
            upper = pd.Timestamp(end).tz_localize(zone).tz_convert("UTC")
            overlaps.loc[rows] &= frame.loc[rows, "bar_start"] < upper
            complete.loc[rows] &= frame.loc[rows, "bar_end"] <= upper
            valid_open.loc[rows] &= frame.loc[rows, "bar_start"] < upper
    cleaned = frame.loc[overlaps].copy()
    mixed = overlaps & ~complete
    cleaned.loc[mixed.loc[overlaps], ["high", "low", "close"]] = np.nan
    cleaned.loc[~valid_open.loc[overlaps], "open"] = np.nan
    excluded = ~overlaps
    if excluded.any() or mixed.any():
        report.add_issue(
            ValidationIssue(
                IssueSeverity.WARNING,
                "SecurityIdentity",
                "Other security identities excluded; boundary-spanning bars cannot supply completed prices",
                symbol=symbol,
                action="restrict_security_identity",
                details={
                    "history_start": start.isoformat() if start else None,
                    "history_end_exclusive": end.isoformat() if end else None,
                    "source": source,
                    "excluded_periods": [d.isoformat() for d in frame.index[excluded]],
                    "quarantined_periods": [d.isoformat() for d in frame.index[mixed]],
                    "resolution": (
                        "Supply same-identity quotes and reviewed cash/share entitlements. "
                        "Cancellation or warrants cannot be represented by replacement "
                        "stock prices, an assumed zero recovery, or removing earlier membership."
                    ),
                },
            )
        )
        report.record_modification(
            symbol, "Applied declared security identity lifetime"
        )
    return cleaned


def restrict_identity_execution(
    period: Period, universe: AssetUniverse, report: ValidationReport
) -> None:
    """Check the actual fill instant, which may follow a bar's calendar start."""
    bounded = universe.price_history_starts.keys() | universe.price_history_ends.keys()
    if not bounded:
        return
    zones = period.market_data["market_timezone"].iloc[0]
    for symbol in bounded:
        if symbol not in zones or pd.isna(zones[symbol]):
            continue
        start = universe.price_history_starts.get(symbol)
        end = universe.price_history_ends.get(symbol)
        if period.execution_time is None:
            # A whole bar inside the lifetime already bounds its opening.
            # A mixed bar needs the real execution time, not its month label.
            opening = period.market_data[("open", symbol)].iloc[0]
            if (
                end is not None
                and pd.notna(opening)
                and period.market_data[("bar_end", symbol)].iloc[0]
                > pd.Timestamp(end).tz_localize(zones[symbol])
            ):
                report.add_issue(
                    ValidationIssue(
                        IssueSeverity.ERROR,
                        "SecurityIdentity",
                        "A boundary-spanning bar requires an explicit execution time",
                        symbol=symbol,
                        action="block_backtest",
                        details={
                            "period": period.trading_date.isoformat(),
                            "source": universe.price_history_sources.get(symbol),
                        },
                    )
                )
            continue
        instant = pd.Timestamp(
            period.crypto_execution_times.get(symbol, period.execution_time)
        )
        before_start = start is not None and instant < pd.Timestamp(start).tz_localize(
            zones[symbol]
        )
        after_end = end is not None and instant >= pd.Timestamp(end).tz_localize(
            zones[symbol]
        )
        if not before_start and not after_end:
            continue
        period.market_data[("open", symbol)] = np.nan
        period.crypto_execution_prices.pop(symbol, None)
        report.add_issue(
            ValidationIssue(
                IssueSeverity.WARNING,
                "SecurityIdentity",
                "Opening quote falls outside the declared security lifetime",
                symbol=symbol,
                action="exclude_identity_execution",
                details={
                    "period": period.trading_date.isoformat(),
                    "execution_time": instant.isoformat(),
                    "history_start": start.isoformat() if start else None,
                    "history_end_exclusive": end.isoformat() if end else None,
                    "source": universe.price_history_sources.get(symbol),
                },
            )
        )
