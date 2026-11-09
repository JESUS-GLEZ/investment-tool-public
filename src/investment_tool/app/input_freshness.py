"""Describe observable investing inputs without changing eligibility or forecasts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from investment_tool.domain.market_time import observable_history
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.stock import Stock


def input_freshness(
    instance: Instance,
    stocks: Sequence[Stock],
    loaded_stocks: Sequence[Stock],
    validation_report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize actual causal history; never infer availability from a request.

    These are strategy-visible price histories. A model may apply narrower
    estimation/news filters. Calendar labels and zoned availability instants
    remain separate, and lagging means behind peers, not a holiday heuristic.
    """
    period = instance.current_trading_period
    decision = period.decision_time if period is not None else None
    assets: list[dict[str, Any]] = []
    for stock in stocks:
        frame = stock.data
        row: dict[str, Any] = {"symbol": stock.symbol}
        if (
            period is None
            or decision is None
            or not {"available_at", "is_complete", "period_end"}.issubset(frame.columns)
        ):
            row["status"] = "timing_unavailable"
        else:
            history = observable_history(frame, period.trading_date, decision)
            if history.empty:
                row["status"] = "no_observable_history"
            else:
                row.update(
                    status="observed",
                    observations=len(history),
                    last_period_start=history.index[-1].date().isoformat(),
                    last_period_end=(
                        pd.Timestamp(history["period_end"].iloc[-1])
                        - pd.Timedelta(days=1)
                    )
                    .date()
                    .isoformat(),
                    latest_available_at=history["available_at"].max().isoformat(),
                )
        assets.append(row)
    ends = [row["last_period_end"] for row in assets if "last_period_end" in row]
    newest = max(ends) if ends else None
    for row in assets:
        if row.get("status") == "observed" and row["last_period_end"] != newest:
            row["status"] = "behind_other_assets"
    times = [
        row["latest_available_at"] for row in assets if "latest_available_at" in row
    ]
    visible = {stock.symbol for stock in stocks}
    excluded = {
        stock.symbol: "Excluded by strategy eligibility/history rules"
        for stock in loaded_stocks
        if stock.symbol not in visible
    }
    excluded.update(validation_report.get("dropped_symbols", {}))
    for issue in validation_report.get("issues", []):
        if issue.get("action") == "exclude_unavailable_candidates":
            for symbol in issue.get("details", {}).get("symbols", []):
                excluded.setdefault(
                    symbol, "No usable series for the configured candidate"
                )
    return {
        "scope": "strategy_visible_price_history",
        "frequency": instance.frequency.value,
        "decision_time": decision.isoformat() if decision is not None else None,
        "requested_start_date": instance.historical_start_date.isoformat(),
        "requested_end_date": instance.historical_end_date.isoformat(),
        "latest_available_at": max(times) if times else None,
        "oldest_asset_period_end": min(ends) if ends else None,
        "newest_asset_period_end": newest,
        "assets": assets,
        "excluded_assets": excluded,
    }
