"""Causal investing provenance stays distinct from requested calendar cutoffs."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from investment_tool.app.input_freshness import input_freshness
from investment_tool.constants import Frequency
from investment_tool.domain.market_time import normalize_bars
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock


def inputs():
    labels = pd.date_range("2024-01-01", periods=4, freq="MS", tz="America/New_York")
    bars = normalize_bars(
        pd.DataFrame({"close": [10.0, 11.0, 12.0, 13.0]}, index=labels),
        Frequency.MONTHLY,
        market_timezone="America/New_York",
        as_of="2024-06-01T00:00:00Z",
    )
    # March was delivered late; April must never enter this April decision.
    bars.loc[pd.Timestamp("2024-03-01"), "available_at"] = pd.Timestamp(
        "2024-04-15T00:00:00Z"
    )
    period = Period(
        pd.Timestamp("2024-04-01").to_pydatetime(),
        pd.DataFrame(),
        decision_time=pd.Timestamp("2024-04-01T04:00:00Z").to_pydatetime(),
    )
    instance = Instance(
        date(2024, 1, 1), date(2024, 12, 31), Frequency.MONTHLY, period, None, period
    )
    stocks = [Stock("ABC", bars), Stock("OLD", bars.iloc[:1].copy())]
    return instance, stocks


def test_freshness_uses_observable_bars_not_request_or_future_data():
    instance, stocks = inputs()
    report = input_freshness(instance, stocks, stocks, {})
    assert report["requested_end_date"] == "2024-12-31"
    assert report["decision_time"] == "2024-04-01T04:00:00+00:00"
    assert report["latest_available_at"] == "2024-03-01T05:00:00+00:00"
    assert report["newest_asset_period_end"] == "2024-02-29"
    assert report["oldest_asset_period_end"] == "2024-01-31"
    assert report["assets"][1]["status"] == "behind_other_assets"
    assert len(stocks[0].data) == 4  # reporting never mutates the input


def test_exclusions_and_unknown_history_are_reported_without_inventing_times():
    instance, stocks = inputs()
    stocks[0].data = stocks[0].data.drop(columns="available_at")
    report = input_freshness(
        instance,
        stocks[:1],
        stocks,
        {
            "dropped_symbols": {"BAD": "Invalid prices"},
            "issues": [
                {
                    "action": "exclude_unavailable_candidates",
                    "details": {"symbols": ["MISSING"]},
                }
            ],
        },
    )
    assert report["latest_available_at"] is None
    assert report["assets"][0]["status"] == "timing_unavailable"
    assert set(report["excluded_assets"]) == {"OLD", "BAD", "MISSING"}
    instance.current_trading_period = None
    assert input_freshness(instance, stocks, stocks, {})["decision_time"] is None


def test_live_recommendation_preserves_portfolio_and_decision(tmp_path, monkeypatch):
    from investment_tool.app.execution_coordinator import _execute_live_investment
    from investment_tool.output.output_manager import OutputManager
    from investment_tool.domain.model_data.entities.portfolio import Portfolio
    from investment_tool.output.policy import OutputMode
    from api.sanitize import sanitize_output

    instance, stocks = inputs()
    strategy = SimpleNamespace(
        stocks=stocks,
        output_mode=OutputMode.STANDARD,
        last_optimization_diagnostics={},
        bind_portfolio=lambda *args: None,
        run_strategy=lambda **kwargs: pd.Series({"ABC": 1.0}),
    )

    def update(weights):
        instance.current_trading_period = None

    model = SimpleNamespace(
        instance=instance,
        stocks=stocks,
        validation_report={},
        corporate_action_events=(),
        portfolio=Portfolio(instance, 1000),
        update=update,
    )
    snapshots = []

    def output(self, **kwargs):
        snapshots.append(kwargs["portfolio_snapshot"])
        return {"concentration": {}}

    monkeypatch.setattr(OutputManager, "generate_live_output", output)
    result = sanitize_output(
        _execute_live_investment(model, strategy, tmp_path),
        "execution",
        OutputMode.STANDARD,
    )
    assert result["data_freshness"]["decision_time"] == "2024-04-01T04:00:00+00:00"
    assert result["data_asof"] == result["data_freshness"]["latest_available_at"]
    assert snapshots[0]["data_freshness"] == result["data_freshness"]
    assert instance.current_trading_period is not None
    assert model.portfolio.cash == 1000
    assert model.portfolio.positions == {}
    assert model.portfolio.history == []
    assert result["rebalance_plan"]["current_value"] == 1000


@pytest.mark.parametrize(
    "frequency,zone,start",
    [
        (Frequency.WEEKLY, "Pacific/Auckland", "2024-01-01"),
        (Frequency.DAILY, "UTC", "2024-01-01"),
    ],
)
def test_coverage_respects_other_frequencies_and_market_timezones(
    frequency, zone, start
):
    labels = pd.date_range(
        start,
        periods=3,
        freq="W-MON" if frequency == Frequency.WEEKLY else "D",
        tz=zone,
    )
    bars = normalize_bars(
        pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=labels),
        frequency,
        market_timezone=zone,
        as_of="2024-06-01T00:00:00Z",
    )
    decision = bars["available_at"].iloc[1]
    period = Period(
        bars.index[2].to_pydatetime(),
        pd.DataFrame(),
        decision_time=decision.to_pydatetime(),
    )
    instance = Instance(
        date(2024, 1, 1), date(2024, 12, 31), frequency, period, None, period
    )
    report = input_freshness(instance, [Stock("ABC", bars)], [], {})
    assert report["frequency"] == frequency.value
    assert report["latest_available_at"] == decision.isoformat()
    assert report["assets"][0]["observations"] == 2
