"""A reused ticker cannot manufacture returns or discharge an old holding."""

import json
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from investment_tool.app.data_readiness import require_ready
from investment_tool.config import AppConfig, ModelConfig
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.data.assembly.model_data_factory import ModelDataFactory
from investment_tool.data.repositories.base_repository import DataResult
from investment_tool.data.repositories.repository_factory import RepositoryFactory
from investment_tool.data.universes import loader
from investment_tool.data.universes.models import AssetUniverse
from investment_tool.data.validation import ValidationLevel, ValidationReport
from investment_tool.data.validation.security_identity import (
    identity_history,
    restrict_identity_execution,
)
from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.domain.market_quotes import execution_prices
from investment_tool.domain.market_time import normalize_bars
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.exceptions import BacktestReadinessError


def bars(
    start="2020-05-01", periods=8, zone="America/New_York", frequency=Frequency.MONTHLY
):
    frequency_code = {
        Frequency.MONTHLY: "MS",
        Frequency.DAILY: "D",
        Frequency.WEEKLY: "W-MON",
    }[frequency]
    return normalize_bars(
        pd.DataFrame(
            {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 100.0},
            index=pd.date_range(start, periods=periods, freq=frequency_code, tz=zone),
        ),
        frequency,
        market_timezone=zone,
    )


def restrict(frame, start=None, end=None, report=None):
    return identity_history(
        frame,
        symbol="OLD",
        start=start,
        end=end,
        source="reviewed identity evidence",
        report=report if report is not None else ValidationReport(),
    )


def test_mixed_identity_month_cannot_supply_a_close_but_keeps_original_open():
    frame = bars()
    frame.loc["2020-09-01":, ["high", "close"]] = 2000.0
    original = frame.copy(deep=True)
    report = ValidationReport()
    clean = restrict(frame, end=date(2020, 9, 2), report=report)
    assert clean.index.max() == pd.Timestamp("2020-09-01")
    assert clean.loc["2020-09-01", "open"] == 10
    assert clean.loc["2020-09-01", ["high", "low", "close"]].isna().all()
    stock = Stock("OLD", clean)
    assert pd.isna(stock.data.loc["2020-09-01", "returns"])
    assert (stock.data.loc[:"2020-08-01", "returns"].dropna() == 0).all()
    pd.testing.assert_frame_equal(frame, original)
    issue = report.issues[0]
    assert issue.details["quarantined_periods"] == ["2020-09-01T00:00:00"]
    assert issue.details["source"] == "reviewed identity evidence"


def test_reorganized_equity_starts_fresh_without_cross_identity_returns():
    frame = bars()
    frame.loc[:"2020-11-01", "close"] = 0.05
    clean = restrict(frame, start=date(2020, 11, 20))
    assert list(clean.index) == list(pd.to_datetime(["2020-11-01", "2020-12-01"]))
    assert pd.isna(clean.loc["2020-11-01", "open"])
    assert pd.isna(clean.loc["2020-11-01", "close"])
    stock = Stock("NEW", clean)
    assert stock.data.returns.isna().all()


@pytest.mark.parametrize("zone", ["UTC", "America/New_York", "Pacific/Auckland"])
@pytest.mark.parametrize(
    "frequency", [Frequency.MONTHLY, Frequency.WEEKLY, Frequency.DAILY]
)
def test_exclusive_end_uses_each_bar_calendar_and_preserves_exact_boundary(
    zone, frequency
):
    frame = bars("2020-06-01", periods=2, zone=zone, frequency=frequency)
    boundary = frame.iloc[0].period_end.date()
    clean = restrict(frame, end=boundary)
    assert len(clean) == 1
    pd.testing.assert_frame_equal(clean, frame.iloc[:1])


def test_large_consistent_gains_survive_without_an_identity_boundary():
    frame = bars()
    frame.loc["2020-09-01":, ["open", "high", "low", "close"]] *= 200
    report = ValidationReport()
    clean = restrict(frame, report=report)
    pd.testing.assert_frame_equal(clean, frame)
    assert Stock("LIVE", clean).data.loc["2020-09-01", "returns"] == 199
    assert not report.issues


def test_real_partial_bar_contained_in_new_identity_is_usable():
    frame = bars("2020-11-01", periods=2)
    frame.loc["2020-11-01", "bar_start"] = pd.Timestamp("2020-11-20T14:30:00Z")
    clean = restrict(frame, start=date(2020, 11, 20))
    pd.testing.assert_frame_equal(clean, frame)


@pytest.mark.parametrize(
    "execution,allowed",
    [
        ("2020-11-01T14:30:00Z", True),
        ("2020-11-02T05:00:00Z", False),
        ("2020-11-02T14:30:00Z", False),
    ],
)
def test_actual_execution_not_month_label_determines_identity(execution, allowed):
    frame = restrict(bars("2020-11-01", periods=1), end=date(2020, 11, 2))
    assert frame.iloc[0].open == 10  # Calendar starts before the identity ends.
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["OLD"]])
    period = Period(
        frame.index[0].to_pydatetime(), frame, execution_time=pd.Timestamp(execution)
    )
    universe = AssetUniverse(
        "fixture.json",
        "Fixture",
        ["OLD"],
        [],
        price_history_ends={"OLD": date(2020, 11, 2)},
    )
    report = ValidationReport()
    restrict_identity_execution(period, universe, report)
    assert ("OLD" in execution_prices(period, set(), ())) == allowed
    assert bool(report.issues) == (not allowed)


def test_unknown_execution_time_cannot_certify_opening_in_a_mixed_identity_bar():
    frame = restrict(bars("2020-11-01", periods=1), end=date(2020, 11, 2))
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["OLD"]])
    period = Period(frame.index[0].to_pydatetime(), frame)
    universe = AssetUniverse(
        "fixture.json",
        "Fixture",
        ["OLD"],
        [],
        price_history_ends={"OLD": date(2020, 11, 2)},
    )
    report = ValidationReport()
    restrict_identity_execution(period, universe, report)
    assert report.issues[0].action == "block_backtest"
    assert "explicit execution time" in report.issues[0].message


def test_crypto_uses_actual_quote_time_instead_of_scheduled_open():
    frame = restrict(bars("2020-11-01", periods=1, zone="UTC"), end=date(2020, 11, 2))
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["BTC/USD"]])
    period = Period(
        frame.index[0].to_pydatetime(),
        frame,
        execution_time=pd.Timestamp("2020-11-01T23:59:00Z"),
        crypto_execution_prices={"BTC/USD": 10},
    )
    period.crypto_execution_times = {
        "BTC/USD": pd.Timestamp("2020-11-02T00:00:00Z").to_pydatetime()
    }
    universe = AssetUniverse(
        "fixture.json",
        "Fixture",
        [],
        ["BTC/USD"],
        price_history_ends={"BTC/USD": date(2020, 11, 2)},
    )
    restrict_identity_execution(period, universe, ValidationReport())
    assert "BTC/USD" not in execution_prices(period, {"BTC/USD"}, ())


@pytest.mark.parametrize("provider", [DataSource.LOCAL, DataSource.ALPACA])
@pytest.mark.parametrize("level", list(ValidationLevel))
def test_wll_oas_identity_blockers_precede_news_and_optimization(
    tmp_path, monkeypatch, provider, level
):
    manifest = {
        "name": "Original 2020 identities",
        "groups": {"stocks": ["WLL", "OAS", "LIVE"]},
        "crypto_symbols": [],
        "minimum_return_periods": 2,
        "price_history_ends": {"WLL": "2020-09-02", "OAS": "2020-11-19"},
        "price_history_sources": {"WLL": "Whiting notice", "OAS": "Oasis notice"},
    }
    monkeypatch.setattr(loader, "ASSET_CONFIG_DIR", tmp_path)
    (tmp_path / "identities.json").write_text(json.dumps(manifest))
    wll, oas, live = bars(), bars(), bars()
    wll.loc["2020-09-01":, ["high", "close"]] = 2000
    oas.loc["2020-11-01":, ["high", "close"]] = 2000
    repository = MagicMock()
    repository.fetch_all.return_value = DataResult(
        stock_dict={"WLL": wll, "OAS": oas, "LIVE": live},
        metadata={"time_contract_version": 1},
    )
    repository.fetch_execution_data.return_value = {
        label.date().isoformat(): {
            "timestamp": (
                label.tz_localize("America/New_York")
                + pd.Timedelta(hours=9, minutes=30)
            ).isoformat(),
            "crypto_prices": {},
        }
        for label in live.index
    }
    monkeypatch.setattr(RepositoryFactory, "create", lambda **kwargs: repository)
    factory = ModelDataFactory(
        provider,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=tmp_path / "unused.pkl"
        if provider == DataSource.LOCAL
        else None,
        validation_level=level,
        cfg=AppConfig(
            asset_config_file="identities.json",
            model=ModelConfig(
                historical_start_date="2020-05-01",
                historical_end_date="2020-12-31",
                backtest_start_date="2020-08-01",
                min_backtest_periods=2,
            ),
        ),
    )
    fetch_news = MagicMock()
    score_news = MagicMock()
    monkeypatch.setattr(factory, "_load_news_and_save_input", fetch_news)
    monkeypatch.setattr(factory, "_build_news", score_news)
    model = factory._create_model_data()
    assert model.validation_report["readiness"]["affected_symbols"] == ["OAS", "WLL"]
    assert model.portfolio.cash == 1000 and model.portfolio.positions == {}
    fetch_news.assert_not_called()
    score_news.assert_not_called()
    september = next(
        p for p in model.periods if p.trading_date == pd.Timestamp("2020-09-01")
    )
    assert "WLL" in september.eligible_symbols
    assert execution_prices(september, set(), ())["WLL"] == 10
    october = next(
        p for p in model.periods if p.trading_date == pd.Timestamp("2020-10-01")
    )
    assert "WLL" not in october.eligible_symbols
    stocks = {stock.symbol: stock for stock in model.stocks}
    pd.testing.assert_frame_equal(
        stocks["LIVE"].data[live.columns], live, check_names=False
    )
    assert not np.isfinite(stocks["WLL"].data.loc["2020-09-01", "returns"])
    strategy = MagicMock()
    with pytest.raises(BacktestReadinessError):
        Backtester(model, strategy).run_backtest()
    strategy.run_strategy.assert_not_called()
    with pytest.raises(BacktestReadinessError):
        require_ready(model.validation_report, tmp_path / "output")
    report = json.loads((tmp_path / "output/data_validation_report.json").read_text())
    assert {
        i["details"]["source"]
        for i in report["issues"]
        if i["validator"] == "SecurityIdentity"
    } == {"Whiting notice", "Oasis notice"}
