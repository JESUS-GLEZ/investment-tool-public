"""Provider-independent, backward-looking equity eligibility and liquidity rank.

The caller supplies a dated security roster with reviewed issuer identities and
bars complying with the repository time contract. This function cannot establish
historical listing coverage from a present-day roster.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.domain.market_time import utc_instant, validate_bar_times


def rank_equities(
    roster: pd.DataFrame,
    daily_prices: dict[str, pd.DataFrame],
    monthly_prices: dict[str, pd.DataFrame],
    decision_time: datetime,
    *,
    minimum_return_periods: int = 12,
    liquidity_sessions: int = 60,
    minimum_dollar_volume: float = 5_000_000,
    stock_limit: int = 1000,
) -> pd.DataFrame:
    """Audit every roster member; select one share class per issuer by liquidity.

    ``roster`` contains unique ``symbol`` and nonempty ``issuer_id`` strings.
    Dollar volume is raw daily close times volume, never split-adjusted price
    times raw volume. Monthly closes may be split adjusted consistently. The
    supplied daily bars must have matching adjustment for prices and volumes.
    Missing series fail preflight; insufficient observed history is recorded as
    an exclusion. Ties are resolved by symbol for reproducible snapshots.
    """
    decision = utc_instant(decision_time)
    if (
        minimum_return_periods < 2
        or liquidity_sessions < 1
        or stock_limit < 1
        or not np.isfinite(minimum_dollar_volume)
        or minimum_dollar_volume < 0
    ):
        raise ValueError("Invalid equity eligibility limits")
    if not {"symbol", "issuer_id"} <= set(roster.columns) or roster.empty:
        raise ValueError("Provide a nonempty dated symbol/issuer_id roster")
    if (
        roster["symbol"].duplicated().any()
        or not roster[["symbol", "issuer_id"]]
        .map(lambda v: isinstance(v, str) and bool(v.strip()))
        .all()
        .all()
    ):
        raise ValueError("Roster symbols must be unique and issuer IDs nonempty")
    symbols = set(roster["symbol"])
    missing = symbols - daily_prices.keys() | (symbols - monthly_prices.keys())
    if missing:
        raise ValueError(f"Missing price series for dated roster: {sorted(missing)}")
    # Exchange-local calendar month, not the UTC date near month boundaries.
    month = decision.tz_convert("America/New_York").tz_localize(None).to_period("M")
    months = pd.date_range(
        (month - minimum_return_periods - 1).start_time,
        (month - 1).start_time,
        freq="MS",
    )
    # Use a shared observed session calendar. A halt or missing session cannot
    # be hidden by reaching farther back in that particular security's history.
    observed_daily: dict[str, pd.DataFrame] = {}
    sessions: set[pd.Timestamp] = set()
    for symbol in sorted(symbols):
        daily = validate_bar_times(daily_prices[symbol], Frequency.DAILY)
        if not daily.empty:
            daily = daily.loc[
                daily["is_complete"] & (daily["available_at"] <= decision)
            ]
            sessions.update(daily.index)
        observed_daily[symbol] = daily
    recent_sessions = pd.DatetimeIndex(sorted(sessions)[-liquidity_sessions:])
    rows: list[dict[str, object]] = []
    for record in roster[["symbol", "issuer_id"]].to_dict("records"):
        symbol = record["symbol"]
        daily = observed_daily[symbol]
        monthly = validate_bar_times(monthly_prices[symbol], Frequency.MONTHLY)
        if daily.empty or monthly.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "issuer_id": record["issuer_id"],
                    "median_dollar_volume": float("nan"),
                    "reason": "no_price_observations",
                }
            )
            continue
        daily = daily.loc[daily["is_complete"] & (daily["available_at"] <= decision)]
        monthly = monthly.loc[
            monthly["is_complete"] & (monthly["available_at"] <= decision)
        ]
        closes = monthly["close"].reindex(months)
        recent = daily.reindex(recent_sessions)
        dollar_volume = recent["close"] * recent["volume"]
        liquidity = float(dollar_volume.median()) if len(recent) else float("nan")
        # A halted/delisted series must not qualify using an old trailing window.
        fresh = bool(len(recent)) and (
            decision - recent["available_at"].iloc[-1]
        ) <= pd.Timedelta(days=7)
        if not np.isfinite(closes).all() or not (closes > 0).all():
            reason = "insufficient_completed_monthly_history"
        elif (
            len(recent) < liquidity_sessions
            or not fresh
            or not np.isfinite(dollar_volume).all()
            or not (dollar_volume > 0).all()
        ):
            reason = "insufficient_recent_daily_history"
        elif liquidity < minimum_dollar_volume:
            reason = "below_liquidity_floor"
        else:
            reason = "eligible"
        rows.append(
            {
                "symbol": symbol,
                "issuer_id": record["issuer_id"],
                "median_dollar_volume": liquidity,
                "reason": reason,
            }
        )
    audit = pd.DataFrame(rows).sort_values(
        ["median_dollar_volume", "symbol"],
        ascending=[False, True],
        na_position="last",
        ignore_index=True,
    )
    eligible = audit["reason"].eq("eligible")
    duplicates = audit.loc[eligible, "issuer_id"].duplicated()
    audit.loc[duplicates[duplicates].index, "reason"] = "alternate_share_class"
    winners = audit.index[audit["reason"].eq("eligible")][:stock_limit]
    audit["selected"] = audit.index.isin(winners)
    audit.loc[audit["reason"].eq("eligible") & ~audit["selected"], "reason"] = (
        "outside_stock_limit"
    )
    return audit
