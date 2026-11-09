"""Readiness must prevent long-run data failures without hindsight selection."""

from datetime import date, datetime
from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from investment_tool.app.data_readiness import require_ready
from investment_tool.constants import Frequency, StockFields
from investment_tool.data.validation.bar_quality import BarQualityValidator
from investment_tool.data.validation.lifecycle import (
    readiness_summary,
    validate_lifecycle,
)
from investment_tool.data.validation.validation_report import ValidationReport
from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.domain.model_data.entities.cash_settlement import CashSettlement
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.entities.security_action import SecurityAction
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.exceptions import BacktestReadinessError


def model_with_gap():
    dates = pd.date_range("2023-08-01", periods=3, freq="MS")
    columns = pd.MultiIndex.from_tuples(
        [
            (StockFields.OPEN, "RETA"),
            (StockFields.CLOSE, "RETA"),
            (StockFields.OPEN, "LIVE"),
            (StockFields.CLOSE, "LIVE"),
            ("bar_end", "LIVE"),
        ]
    )
    rows = [
        [170, 171, 10, 11, pd.Timestamp("2023-09-01T04:00:00Z")],
        [171, 172, 11, 12, pd.Timestamp("2023-10-01T04:00:00Z")],
        [np.nan, np.nan, 12, 13, pd.Timestamp("2023-11-01T04:00:00Z")],
    ]
    periods = [
        Period(
            dt.to_pydatetime(),
            pd.DataFrame([row], index=[dt], columns=columns),
            execution_time=(
                dt.tz_localize("UTC") + pd.Timedelta(hours=14)
            ).to_pydatetime(),
        )
        for dt, row in zip(dates, rows)
    ]
    model = ModelData()
    model.periods = periods
    model.instance = Instance(
        date(2023, 8, 1),
        date(2023, 10, 31),
        Frequency.MONTHLY,
        periods[0],
        periods[0],
        periods[0],
    )
    model.portfolio = Portfolio(model.instance, 1000)
    history = pd.DataFrame(
        {
            "open": [169, 170, 171],
            "high": [170, 171, 172],
            "low": [168, 169, 170],
            "close": [170, 171, 172],
            "volume": [100, 100, 100],
        },
        index=pd.date_range("2023-07-01", periods=3, freq="MS"),
    )
    model.stocks = [Stock("RETA", history), Stock("LIVE", history)]
    return model


def audit(model):
    report = ValidationReport()
    validate_lifecycle(model, report)
    result = report.to_dict()
    result["readiness"] = readiness_summary(result)
    return result


def test_all_gaps_reported_before_optimizer_and_membership_unchanged(tmp_path):
    model = model_with_gap()
    model.periods[-1].eligible_symbols = frozenset({"LIVE"})
    report = audit(model)
    assert report["readiness"]["affected_symbols"] == ["RETA"]
    assert {
        i["details"]["phase"]
        for i in report["issues"]
        if i["action"] == "block_backtest"
    } == {"execution", "valuation"}
    assert model.periods[0].eligible_symbols is None
    assert model.portfolio.positions == {}
    with pytest.raises(BacktestReadinessError, match="before optimization"):
        require_ready(report, tmp_path)
    assert (tmp_path / "data_validation_report.json").is_file()
    model.validation_report = report
    strategy = MagicMock()
    with pytest.raises(BacktestReadinessError):
        Backtester(model, strategy).run_backtest()
    strategy.run_strategy.assert_not_called()


def test_reviewed_cash_settlement_resolves_future_gaps_without_mutating_portfolio():
    model = model_with_gap()
    model.cash_settlements = (
        CashSettlement(
            "RETA",
            datetime.fromisoformat("2023-09-26T00:00:00+00:00"),
            datetime.fromisoformat("2023-09-26T00:00:00+00:00"),
            172.5,
            "reviewed notice",
        ),
    )
    assert audit(model)["readiness"]["ready"]
    assert model.portfolio.cash == 1000
    assert model.portfolio.corporate_actions == []


def test_identity_quarantine_can_be_resolved_by_a_supported_reviewed_entitlement():
    from investment_tool.data.validation.security_identity import identity_history
    from investment_tool.domain.market_time import normalize_bars

    model = model_with_gap()
    raw = normalize_bars(
        pd.DataFrame(
            {
                "open": 170.0,
                "high": 172.0,
                "low": 169.0,
                "close": 171.0,
                "volume": 100.0,
            },
            index=pd.date_range("2023-08-01", periods=3, freq="MS", tz="UTC"),
        ),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )
    cleaned = identity_history(
        raw,
        symbol="RETA",
        start=None,
        end=date(2023, 9, 26),
        source="synthetic fixture notice",
        report=ValidationReport(),
    ).reindex(raw.index)
    for period in model.periods:
        for field in (StockFields.OPEN, StockFields.CLOSE):
            period.market_data[(field, "RETA")] = cleaned.loc[
                period.trading_date, field
            ]
    assert pd.isna(model.periods[1].market_data[(StockFields.CLOSE, "RETA")].iloc[0])
    assert not audit(model)["readiness"]["ready"]
    model.cash_settlements = (
        CashSettlement(
            "RETA",
            datetime.fromisoformat("2023-09-26T00:00:00+00:00"),
            datetime.fromisoformat("2023-09-26T00:00:00+00:00"),
            172.5,
            "synthetic fixture notice",
        ),
    )
    assert audit(model)["readiness"]["ready"]
    assert model.portfolio.cash == 1000
    assert model.portfolio.positions == {}


@pytest.mark.parametrize(
    "end,blocked", [(date(2023, 9, 26), True), (date(2023, 10, 1), False)]
)
def test_a_partial_close_cannot_value_an_identity_after_its_lifetime(end, blocked):
    from investment_tool.data.universes.models import AssetUniverse

    model = model_with_gap()
    model.periods = model.periods[:2]  # Stop in the event month; no later quote gap.
    for stock in model.stocks:
        stock.data["market_timezone"] = "America/New_York"
    universe = AssetUniverse(
        "fixture.json",
        "Fixture",
        ["RETA", "LIVE"],
        [],
        price_history_ends={"RETA": end},
        price_history_sources={"RETA": "synthetic cancellation notice"},
    )
    assert audit(model)["readiness"]["ready"]  # Numerically valid quotes alone.
    report = ValidationReport()
    validate_lifecycle(model, report, universe=universe)
    assert (not readiness_summary(report.to_dict())["ready"]) == blocked
    if blocked:
        issue = next(i for i in report.issues if i.action == "block_backtest")
        assert issue.details["phase"] == "identity"
        assert issue.details["first_period"] == "2023-09-01T00:00:00"
        assert issue.details["identity_source"] == "synthetic cancellation notice"
        model.cash_settlements = (
            CashSettlement(
                "RETA",
                datetime.fromisoformat("2023-09-26T00:00:00+00:00"),
                datetime.fromisoformat("2023-09-26T00:00:00+00:00"),
                172.5,
                "synthetic fixture notice",
            ),
        )
        settled = ValidationReport()
        validate_lifecycle(model, settled, universe=universe)
        assert readiness_summary(settled.to_dict())["ready"]


def test_late_publication_cannot_resolve_earlier_missing_quote():
    model = model_with_gap()
    model.cash_settlements = (
        CashSettlement(
            "RETA",
            datetime.fromisoformat("2023-09-26T00:00:00+00:00"),
            datetime.fromisoformat("2023-10-15T00:00:00+00:00"),
            172.5,
            "late evidence",
        ),
    )
    report = audit(model)
    assert not report["readiness"]["ready"]
    assert report["readiness"]["blocker_count"] == 1  # open missing before publication


def test_prelisting_missing_quotes_do_not_block_or_remove_later_candidate():
    model = model_with_gap()
    model.periods[0].market_data[(StockFields.OPEN, "RETA")] = np.nan
    model.periods[-1].market_data.loc[:, (StockFields.OPEN, "RETA")] = 172
    model.periods[-1].market_data.loc[:, (StockFields.CLOSE, "RETA")] = 173
    report = audit(model)
    assert report["readiness"]["ready"]
    assert any(i["action"] == "exclude_at_execution" for i in report["issues"])
    assert all(p.execution_eligible_symbols is None for p in model.periods)


def test_settled_security_cannot_be_bought_again_from_residual_vendor_bar():
    from investment_tool.domain.market_quotes import execution_prices

    model = model_with_gap()
    event = CashSettlement(
        "RETA",
        datetime.fromisoformat("2023-08-01T12:00:00+00:00"),
        datetime.fromisoformat("2023-08-01T12:00:00+00:00"),
        172.5,
        "reviewed notice",
    )
    assert "RETA" not in execution_prices(model.periods[0], set(), (event,))


@pytest.mark.parametrize(
    "field,value",
    [("open", -1), ("high", 1), ("low", 20), ("volume", 0), ("close", np.inf)],
)
def test_bad_bar_quarantines_only_affected_date(field, value):
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [12.0, 12.0, 12.0],
            "low": [9.0, 9.0, 9.0],
            "close": [11.0, 11.0, 11.0],
            "volume": [100.0, 100.0, 100.0],
        },
        index=pd.date_range("2023-01-01", periods=3, freq="MS"),
    )
    frame.iloc[1, frame.columns.get_loc(field)] = value
    original = frame.copy()
    data = {"stock_dict": {"X": frame}}
    report = ValidationReport()
    BarQualityValidator().validate(data, report)
    pd.testing.assert_frame_equal(frame, original)  # input frame untouched
    assert data["stock_dict"]["X"].iloc[1][["high", "low", "close"]].isna().all()
    if field not in {"open", "volume"}:
        assert data["stock_dict"]["X"].iloc[1]["open"] == 10
    pd.testing.assert_series_equal(data["stock_dict"]["X"].iloc[0], frame.iloc[0])
    pd.testing.assert_series_equal(data["stock_dict"]["X"].iloc[2], frame.iloc[2])
    assert report.issues[0].action == "quarantine_observations"
    assert not report.dropped_symbols


def test_extreme_but_consistent_return_is_preserved():
    frame = pd.DataFrame(
        {
            "open": [10.0, 100.0],
            "high": [11.0, 110.0],
            "low": [9.0, 90.0],
            "close": [10.0, 100.0],
            "volume": [100.0, 100.0],
        }
    )
    data = {"stock_dict": {"X": frame}}
    report = ValidationReport()
    BarQualityValidator().validate(data, report)
    pd.testing.assert_frame_equal(data["stock_dict"]["X"], frame)
    assert not report.issues


def test_entire_missing_month_blocks_instead_of_compressing_trading_calendar():
    model = model_with_gap()
    model.periods[-1].market_data[(StockFields.OPEN, "RETA")] = 172
    model.periods[-1].market_data[(StockFields.CLOSE, "RETA")] = 173
    model.periods.pop(1)
    report = audit(model)
    assert not report["readiness"]["ready"]
    blocker = next(i for i in report["issues"] if i["action"] == "block_backtest")
    assert blocker["details"]["missing_months"] == ["2023-09"]


def test_entire_missing_week_blocks_instead_of_compressing_trading_calendar():
    model = model_with_gap()
    model.instance.frequency = Frequency.WEEKLY
    for period, label in zip(model.periods, ["2024-01-01", "2024-01-08", "2024-01-15"]):
        period.trading_date = pd.Timestamp(label).to_pydatetime()
        period.market_data.index = [period.trading_date]
    model.periods.pop(1)
    report = audit(model)
    assert not report["readiness"]["ready"]
    blockers = [
        i
        for i in report["issues"]
        if i["validator"] == "LifecycleCoverage" and "missing_weeks" in i["details"]
    ]
    assert len(blockers) == 1
    assert blockers[0]["details"]["missing_weeks"] == ["2024-01-08/2024-01-14"]


def test_short_history_does_not_create_a_potential_position():
    model = model_with_gap()
    for period in model.periods:
        period.minimum_return_periods = 12
    assert audit(model)["readiness"]["ready"]


@pytest.mark.parametrize(
    "frequency,step", [(Frequency.WEEKLY, "W-MON"), (Frequency.DAILY, "B")]
)
def test_readiness_uses_same_year_history_span_as_strategy(frequency, step):
    model = model_with_gap()
    period = model.periods[0]
    model.periods = [period]
    period.trading_date = datetime(2025, 1, 6)
    period.market_data.index = [period.trading_date]
    period.minimum_return_periods = 12
    model.instance.frequency = frequency
    labels = pd.date_range("2023-01-02", "2025-01-03", freq=step)
    frame = pd.DataFrame(
        {"open": 100.0, "close": 101.0, "high": 102.0, "low": 99.0, "volume": 1000.0},
        index=labels,
    )
    model.stocks = [Stock("LIVE", frame), Stock("RETA", frame.iloc[-12:])]
    model.instance.history_calendars = {s.symbol: labels for s in model.stocks}
    model.instance.history_calendar_end = date(2025, 1, 5)
    period.market_data[("close", "RETA")] = np.nan
    report = audit(model)
    assert report["readiness"]["ready"]
    exclusions = [
        i for i in report["issues"] if i["action"] == "exclude_insufficient_history"
    ]
    assert exclusions[0]["details"]["symbols"] == ["RETA"]
    period.market_data[("close", "LIVE")] = np.nan
    assert not audit(model)["readiness"]["ready"]


def test_quarantined_month_does_not_become_a_multi_month_return():
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory

    dates = pd.date_range("2023-01-01", periods=4, freq="MS")
    frame = pd.DataFrame(
        {
            "open": [10.0, 10.0, 15.0, 16.0],
            "high": [11.0, 11.0, 16.0, 17.0],
            "low": [9.0, 9.0, 14.0, 15.0],
            "close": [10.0, 10.0, 15.0, 16.0],
            "volume": [100.0, 0.0, 100.0, 100.0],
        },
        index=dates,
    )
    data = {"stock_dict": {"X": frame}}
    BarQualityValidator().validate(data, ValidationReport())
    cleaned = ModelDataFactory._process_stock_data(data["stock_dict"]["X"])
    stock = Stock("X", cleaned)
    assert len(stock.data) == 4
    assert pd.isna(stock.data.loc[dates[2], StockFields.RETURNS])
    assert stock.data.loc[dates[3], StockFields.RETURNS] == pytest.approx(16 / 15 - 1)


def test_invalid_future_close_never_removes_valid_opening_quote():
    frame = pd.DataFrame(
        {
            "open": [10.0],
            "high": [12.0],
            "low": [9.0],
            "close": [np.nan],
            "volume": [100.0],
        },
        index=pd.DatetimeIndex(["2023-01-01"]),
    )
    data = {"stock_dict": {"X": frame}}
    BarQualityValidator().validate(data, ValidationReport())
    cleaned = data["stock_dict"]["X"]
    assert cleaned.iloc[0]["open"] == 10
    assert pd.isna(cleaned.iloc[0]["close"])
    stock = Stock("X", cleaned)
    assert stock.data.iloc[0]["open"] == 10


def test_validator_crash_is_a_readiness_blocker_even_in_warn_mode():
    from investment_tool.data.validation import ValidationLevel, ValidationPipeline
    from investment_tool.data.validation.validators import BaseValidator

    class CrashingValidator(BaseValidator):
        def validate(self, data, report):
            raise RuntimeError("Cannot verify input")

    _, report = ValidationPipeline(
        validators=[CrashingValidator()], level=ValidationLevel.WARN
    ).validate({"stock_dict": {}})
    assert not readiness_summary(report.to_dict())["ready"]


def test_same_day_open_gap_cannot_be_resolved_by_later_cash_notice():
    model = model_with_gap()
    model.portfolio.positions = {"RETA": 2}
    model.periods[0].market_data[(StockFields.OPEN, "RETA")] = np.nan
    model.cash_settlements = (
        CashSettlement(
            "RETA",
            datetime.fromisoformat("2023-08-01T15:00:00+00:00"),
            datetime.fromisoformat("2023-08-01T15:00:00+00:00"),
            172.5,
            "later notice",
        ),
    )
    report = audit(model)
    assert not report["readiness"]["ready"]
    assert model.portfolio.positions == {"RETA": 2}
    assert model.portfolio.cash == 1000


def test_factory_returns_blocked_report_without_fetching_or_scoring_news(monkeypatch):
    from investment_tool.config import AppConfig
    from investment_tool.constants import DataSource, ExecutionMode
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory

    factory = ModelDataFactory(
        DataSource.ALPACA,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        cfg=AppConfig(),
    )
    factory.model_data = model_with_gap()
    factory._validation_report = ValidationReport()
    for method in [
        "_load_input_data",
        "_build_instance",
        "_build_stocks",
        "_build_periods",
        "_build_portfolio",
    ]:
        monkeypatch.setattr(factory, method, lambda: None)
    fetch = MagicMock()
    score = MagicMock()
    monkeypatch.setattr(factory, "_load_news_and_save_input", fetch)
    monkeypatch.setattr(factory, "_build_news", score)
    model = factory._create_model_data()
    assert not model.validation_report["readiness"]["ready"]
    assert model.news == {}
    fetch.assert_not_called()
    score.assert_not_called()


def test_comparison_preserves_readiness_report_before_any_strategy_is_built(tmp_path):
    from unittest.mock import patch
    from investment_tool.app.strategy_comparison_coordinator import (
        StrategyComparisonParameters,
        compare_strategies,
    )
    from investment_tool.constants import DataSource, StrategyType

    model = model_with_gap()
    model.validation_report = audit(model)
    params = StrategyComparisonParameters(
        strategy_types=[StrategyType.OPTIMIZATION],
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=tmp_path,
    )
    partial = {}
    with patch(
        "investment_tool.app.strategy_comparison_coordinator.ModelDataFactory"
    ) as factory, patch(
        "investment_tool.app.strategy_comparison_coordinator._build_strategy"
    ) as build:
        factory.create_model_data.return_value = model
        with pytest.raises(BacktestReadinessError):
            compare_strategies(params, partial_sink=partial)
        build.assert_not_called()
    assert partial["validation_report"]["readiness"]["affected_symbols"] == ["RETA"]
    assert (tmp_path / "data_validation_report.json").is_file()


def test_empty_configured_asset_does_not_crash_bar_validation():
    frame = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    data = {"crypto_dict": {"BTC/USD": frame}}
    report = ValidationReport()
    BarQualityValidator().validate(data, report)
    assert data["crypto_dict"]["BTC/USD"].empty
    assert not report.issues


def test_nullable_missing_prices_are_quarantined():
    frame = pd.DataFrame(
        {"open": [10], "high": [12], "low": [9], "close": [pd.NA], "volume": [100]},
        dtype="Float64",
    )
    data = {"stock_dict": {"X": frame}}
    report = ValidationReport()
    BarQualityValidator().validate(data, report)
    assert data["stock_dict"]["X"].iloc[0]["open"] == 10
    assert pd.isna(data["stock_dict"]["X"].iloc[0]["close"])
    assert report.issues[0].action == "quarantine_observations"


def test_share_transition_requires_successor_valuation_and_exit_quotes():
    from investment_tool.domain.model_data.entities.security_action import (
        SecurityAction,
    )

    model = model_with_gap()
    model.security_actions = (
        SecurityAction(
            "RETA",
            datetime.fromisoformat("2023-09-26T00:00:00Z"),
            datetime.fromisoformat("2023-09-26T00:00:00Z"),
            0,
            "reviewed notice",
            (("LIVE", 2),),
        ),
    )
    assert audit(model)["readiness"]["ready"]
    model.periods[-1].market_data[(StockFields.OPEN, "LIVE")] = np.nan
    result = audit(model)
    assert result["readiness"]["affected_symbols"] == ["LIVE"]
    assert model.portfolio.positions == {}
    assert model.portfolio.corporate_actions == []


def incomplete_action():
    return SecurityAction(
        "RETA",
        datetime.fromisoformat("2023-09-26T00:00:00Z"),
        datetime.fromisoformat("2023-09-26T00:00:00Z"),
        5,
        "reviewed notice",
        (("LIVE", 2),),
        unresolved_entitlements=(
            "Warrant valuation and exit quotes",
            "Verify share units",
        ),
    )


@pytest.mark.parametrize("retain", [False, True])
def test_incomplete_action_blocks_even_with_complete_numerical_quotes(tmp_path, retain):
    model = model_with_gap()
    model.periods = model.periods[:2]  # No quote holes anywhere in this run.
    model.security_actions = (replace(incomplete_action(), retain_original=retain),)
    result = audit(model)
    assert not result["readiness"]["ready"]
    blockers = [i for i in result["issues"] if i["action"] == "block_backtest"]
    assert len(blockers) == 1
    assert blockers[0]["symbol"] == "RETA"
    assert blockers[0]["details"]["phase"] == "entitlement"
    assert blockers[0]["details"]["unresolved_entitlements"] == [
        "Warrant valuation and exit quotes",
        "Verify share units",
    ]
    assert blockers[0]["details"]["source"] == "reviewed notice"
    assert model.portfolio.positions == {}
    assert model.portfolio.cash == 1000
    with pytest.raises(BacktestReadinessError):
        require_ready(result, tmp_path)
    persisted = (tmp_path / "data_validation_report.json").read_text()
    assert "Warrant valuation and exit quotes" in persisted
    model.validation_report = result
    strategy = MagicMock()
    with pytest.raises(BacktestReadinessError):
        Backtester(model, strategy).run_backtest()
    strategy.run_strategy.assert_not_called()


def test_partial_action_cannot_discharge_missing_old_holding_quotes():
    model = model_with_gap()
    model.security_actions = (incomplete_action(),)
    result = audit(model)
    assert {
        i["details"]["phase"]
        for i in result["issues"]
        if i["action"] == "block_backtest"
    } == {"entitlement", "execution", "valuation"}
    model.security_actions = (replace(incomplete_action(), unresolved_entitlements=()),)
    assert audit(model)["readiness"]["ready"]


def test_partial_cash_action_cannot_discharge_remaining_entitlements():
    model = model_with_gap()
    event = incomplete_action()
    model.cash_settlements = (
        CashSettlement(
            event.symbol,
            event.effective_at,
            event.published_at,
            event.cash_per_share,
            event.source,
            unresolved_entitlements=event.unresolved_entitlements,
        ),
    )
    result = audit(model)
    assert any(i["details"].get("phase") == "entitlement" for i in result["issues"])
    assert not result["readiness"]["ready"]
    model.portfolio.positions = {"RETA": 3}
    with pytest.raises(ValueError, match="Incomplete security action"):
        model.portfolio.settle_cash_actions(model.cash_settlements, event.published_at)
    assert model.portfolio.positions == {"RETA": 3}
    assert model.portfolio.cash == 1000


@pytest.mark.parametrize("retain", [False, True])
def test_incomplete_event_before_first_purchase_does_not_block(retain):
    model = model_with_gap()
    model.periods = model.periods[:2]
    model.security_actions = (
        replace(
            incomplete_action(),
            effective_at=datetime.fromisoformat("2023-08-01T12:00:00Z"),
            published_at=datetime.fromisoformat("2023-08-01T12:00:00Z"),
            retain_original=retain,
        ),
    )
    assert audit(model)["readiness"]["ready"]
    model.portfolio.positions = {"RETA": 1}
    assert not audit(model)["readiness"][
        "ready"
    ]  # Initial holdings still require accounting.


def test_incomplete_entitlement_not_available_until_after_run_does_not_block_early():
    model = model_with_gap()
    model.periods = model.periods[:2]
    model.security_actions = (
        replace(
            incomplete_action(),
            published_at=datetime.fromisoformat("2023-11-01T12:00:00Z"),
        ),
    )
    assert audit(model)["readiness"]["ready"]


def test_two_month_newborn_series_does_not_crash_frequency_validation():
    from investment_tool.data.validation.validators import (
        TimeSeriesConsistencyValidator,
    )

    frame = pd.DataFrame(
        {"close": [10.0, 11.0]}, index=pd.date_range("2020-11-01", periods=2, freq="MS")
    )
    data = {"stock_dict": {"VTRS": frame}}
    report = ValidationReport()
    TimeSeriesConsistencyValidator().validate(data, report)
    pd.testing.assert_frame_equal(data["stock_dict"]["VTRS"], frame)
    assert not report.issues


def test_midmonth_birth_quote_cannot_fill_at_months_first_session():
    from investment_tool.domain.market_quotes import execution_prices, period_prices

    model = model_with_gap()
    period = model.periods[0]
    period.market_data[("bar_start", "RETA")] = pd.Timestamp("2023-08-15T04:00:00Z")
    period.market_data[("bar_start", "LIVE")] = pd.Timestamp("2023-08-01T04:00:00Z")
    assert execution_prices(period, set(), ()) == {"LIVE": 10}
    assert period_prices(period)["RETA"] == 171
