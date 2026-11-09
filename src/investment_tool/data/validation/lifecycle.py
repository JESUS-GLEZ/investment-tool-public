"""Audit complete potential holding paths, independently of optimizer results.

Future quotes are inspected only to decide whether the experiment is ready.
They never change historical membership, expected returns, or selected weights.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.data.universes.models import AssetUniverse
from investment_tool.domain.market_quotes import (
    execution_prices,
    period_prices,
    valuation_time,
)
from investment_tool.domain.market_time import observable_history
from investment_tool.domain.return_history import has_return_history
from investment_tool.domain.model_data.model_data import ModelData

from .validation_report import IssueSeverity, ValidationIssue, ValidationReport


def validate_lifecycle(
    model: ModelData, report: ValidationReport, *, universe: AssetUniverse | None = None
) -> None:
    """Check every potential purchase can be marked and exited throughout the run.

    This is deliberately conservative: any asset that could have been bought
    may still be held, including after it leaves the candidate universe. Only
    an explicit reviewed settlement discharges that obligation. A hole blocks
    readiness rather than retroactively deleting a security from the sample.
    """
    current = model.instance.current_trading_period
    if current is None:
        raise ValueError("Lifecycle audit requires a current trading period")
    periods = model.periods[model.periods.index(current) :]
    frequency = model.instance.frequency
    if frequency in (Frequency.MONTHLY, Frequency.WEEKLY):
        code = "M" if frequency == Frequency.MONTHLY else "W-SUN"
        calendar = pd.DatetimeIndex([p.trading_date for p in periods]).to_period(code)
        missing = pd.period_range(calendar[0], calendar[-1], freq=code).difference(
            calendar
        )
        if (
            len(missing)
            or calendar.has_duplicates
            or not calendar.is_monotonic_increasing
        ):
            report.add_issue(
                ValidationIssue(
                    IssueSeverity.ERROR,
                    "LifecycleCoverage",
                    f"{frequency.value.capitalize()} trading calendar has gaps, duplicates or reversed periods",
                    action="block_backtest",
                    details={
                        "missing_months"
                        if frequency == Frequency.MONTHLY
                        else "missing_weeks": [str(period) for period in missing]
                    },
                )
            )
    possible = set(model.portfolio.positions)
    applied_actions: set[str] = set()

    def advance_actions(cutoff: datetime | None) -> None:
        if cutoff is None:
            return
        for event in model.corporate_action_events:
            if (
                event.key in applied_actions
                or max(event.effective_at, event.published_at) > cutoff
            ):
                continue
            applied_actions.add(event.key)
            if event.symbol in possible:
                if event.unresolved_entitlements:
                    report.add_issue(
                        ValidationIssue(
                            IssueSeverity.ERROR,
                            "LifecycleCoverage",
                            "Potential held asset has an incomplete security-action entitlement",
                            symbol=event.symbol,
                            action="block_backtest",
                            details={
                                "phase": "entitlement",
                                "cutoff": cutoff.isoformat(),
                                "effective_at": event.effective_at.isoformat(),
                                "published_at": event.published_at.isoformat(),
                                "unresolved_entitlements": list(
                                    event.unresolved_entitlements
                                ),
                                "source": event.source,
                                "resolution": "Review every component, quote and share-unit basis before crediting this action; do not omit recovery components or retroactively remove the asset.",
                            },
                        )
                    )
                    continue  # A partial action cannot discharge the old holding.
                if not event.retain_original:
                    possible.discard(event.symbol)
                possible.update(symbol for symbol, _ in event.successor_shares)

    holes: dict[tuple[str, str], list[str]] = {}
    stocks = {stock.symbol: stock for stock in model.stocks}
    for period in periods:
        label = period.trading_date.isoformat()
        fills = execution_prices(
            period, model.crypto_symbols, model.corporate_action_events
        )
        advance_actions(period.execution_time)
        for symbol in possible - fills.keys():
            holes.setdefault((symbol, "execution"), []).append(label)
        membership = period.eligible_symbols
        candidates = set(stocks) if membership is None else set(membership)
        # A no-quote unheld candidate can be omitted at this open. Record the
        # dated action; do not drop its valid history or later eligibility.
        excluded = candidates - fills.keys()
        if excluded:
            report.add_issue(
                ValidationIssue(
                    IssueSeverity.INFO,
                    "LifecycleCoverage",
                    "Unheld candidates without valid opening quotes are not purchased",
                    action="exclude_at_execution",
                    details={"period": label, "symbols": sorted(excluded)},
                )
            )
        insufficient_history: set[str] = set()
        for symbol in candidates & fills.keys():
            stock = stocks.get(symbol)
            if stock is None:
                continue
            history = (
                observable_history(
                    stock.data, period.trading_date, period.decision_time
                )
                if period.decision_time is not None
                else stock.data.loc[stock.data.index < period.trading_date]
            )
            minimum = period.minimum_return_periods
            if minimum is not None:
                if not has_return_history(
                    history,
                    period.trading_date,
                    minimum,
                    frequency,
                    model.instance.history_calendars.get(symbol),
                    model.instance.history_calendar_end,
                ):
                    insufficient_history.add(symbol)
                    continue
            possible.add(symbol)
        if insufficient_history:
            report.add_issue(
                ValidationIssue(
                    IssueSeverity.INFO,
                    "LifecycleCoverage",
                    "Candidates lack the configured completed return window at this decision",
                    action="exclude_insufficient_history",
                    details={"period": label, "symbols": sorted(insufficient_history)},
                )
            )
        advance_actions(valuation_time(period))
        if universe is not None:
            cutoff = valuation_time(period)
            for symbol in possible & universe.price_history_ends.keys():
                stock = stocks.get(symbol)
                if cutoff is None or stock is None or stock.data.empty:
                    continue  # Missing quotes are reported by coverage checks.
                zone = stock.data["market_timezone"].iloc[-1]
                end = pd.Timestamp(universe.price_history_ends[symbol]).tz_localize(
                    zone
                )
                # A close ending exactly at the exclusive boundary belongs to
                # the preceding interval. A later snapshot cannot value a
                # cancelled holding using that earlier partial bar's close.
                if pd.Timestamp(cutoff) > end:
                    holes.setdefault((symbol, "identity"), []).append(label)
        for symbol in possible - period_prices(period).keys():
            holes.setdefault((symbol, "valuation"), []).append(label)
    for (symbol, phase), gap_periods in sorted(holes.items()):
        report.add_issue(
            ValidationIssue(
                IssueSeverity.ERROR,
                "LifecycleCoverage",
                f"Potential held asset has unresolved {phase} coverage",
                symbol=symbol,
                action="block_backtest",
                details={
                    "phase": phase,
                    "first_period": gap_periods[0],
                    "periods": gap_periods,
                    "resolution": (
                        "An ended identity requires a reviewed retiring entitlement; "
                        "other gaps require exact-period compatible quotes or a sourced "
                        "settlement. Never drop past membership or held capital."
                    ),
                    "identity_source": universe.price_history_sources.get(symbol)
                    if universe
                    else None,
                    "identity_end_exclusive": universe.price_history_ends[
                        symbol
                    ].isoformat()
                    if universe and symbol in universe.price_history_ends
                    else None,
                },
            )
        )


def readiness_summary(report: dict[str, Any]) -> dict[str, Any]:
    blockers = [
        i for i in report.get("issues", []) if i.get("action") == "block_backtest"
    ]
    return {
        "ready": not blockers,
        "blocker_count": len(blockers),
        "affected_symbols": sorted({i["symbol"] for i in blockers if i.get("symbol")}),
        "scope": "Declared security lifetimes, quotes and reviewed cash/share entitlements; not complete identity or corporate-action certification",
    }
