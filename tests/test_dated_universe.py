"""Dated eligibility cannot read future membership or manufacture history."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from investment_tool.data.validation import ValidationReport

import pytest

from investment_tool.data.universes import loader as asset_universe
from investment_tool.constants import Frequency
from investment_tool.data.validation import ValidationPipeline
from investment_tool.domain.market_time import normalize_bars
from investment_tool.data.universes.selection import rank_equities


def manifest():
    return {
        "name": "Dated",
        "groups": {"equities": ["OLD", "NEW"], "fixed": ["GLD"]},
        "crypto_symbols": ["BTC/USD"],
        "membership": {
            "version": 1,
            "minimum_return_periods": 12,
            "stock_limit": 1000,
            "always_considered_stocks": ["GLD"],
            "snapshots": [
                {
                    "effective_from": "2024-01-01T05:00:00Z",
                    "expires_at": "2024-04-01T04:00:00Z",
                    "published_at": "2023-12-31T23:00:00Z",
                    "symbols": ["OLD"],
                    "source": "dated roster A",
                },
                {
                    "effective_from": "2024-04-01T04:00:00Z",
                    "expires_at": "2024-07-01T04:00:00Z",
                    "published_at": "2024-03-31T23:00:00Z",
                    "symbols": ["NEW"],
                    "source": "dated roster B",
                },
            ],
        },
    }


def load(tmp_path, monkeypatch, data):
    monkeypatch.setattr(asset_universe, "ASSET_CONFIG_DIR", tmp_path)
    (tmp_path / "dated.json").write_text(json.dumps(data))
    return asset_universe.load_asset_universe("dated.json")


def test_membership_uses_half_open_instants_and_keeps_fixed_candidates(
    tmp_path, monkeypatch
):
    universe = load(tmp_path, monkeypatch, manifest())
    membership = universe.membership
    assert membership.symbols_at(datetime(2024, 4, 1, 3, 59, tzinfo=timezone.utc)) == {
        "OLD",
        "GLD",
    }
    assert membership.symbols_at(datetime(2024, 4, 1, 4, tzinfo=timezone.utc)) == {
        "NEW",
        "GLD",
    }
    assert universe.crypto == ["BTC/USD"]
    for instant in ["2023-12-31T05:00:00Z", "2024-07-01T04:00:00Z"]:
        with pytest.raises(ValueError, match="No dated universe"):
            membership.symbols_at(datetime.fromisoformat(instant))


@pytest.mark.parametrize("value", [True, 1, 12.0, "12"])
def test_static_history_rule_requires_an_integer(tmp_path, monkeypatch, value):
    data = manifest()
    del data["membership"]
    data["minimum_return_periods"] = value
    with pytest.raises(ValueError, match="minimum_return_periods"):
        load(tmp_path, monkeypatch, data)


def test_static_history_rule_does_not_claim_historical_membership(
    tmp_path, monkeypatch
):
    data = manifest()
    del data["membership"]
    assert load(tmp_path, monkeypatch, data).minimum_return_periods is None
    data["minimum_return_periods"] = 12
    universe = load(tmp_path, monkeypatch, data)
    assert universe.membership is None
    assert "membership_mode" not in universe.summary()
    assert universe.minimum_return_periods == 12


def test_rejects_conflicting_history_rules(tmp_path, monkeypatch):
    data = manifest()
    data["minimum_return_periods"] = 24
    with pytest.raises(ValueError, match="Conflicting"):
        load(tmp_path, monkeypatch, data)


@pytest.mark.parametrize("value", ["20210101", "2021-01-01T00:00:00Z", True])
def test_symbol_identity_date_requires_a_calendar_date(tmp_path, monkeypatch, value):
    data = manifest()
    data["symbol_identity_date"] = value
    with pytest.raises(ValueError, match="symbol_identity_date"):
        load(tmp_path, monkeypatch, data)


@pytest.mark.parametrize("ended", [False, True])
@pytest.mark.parametrize("frequency", [Frequency.MONTHLY, Frequency.WEEKLY])
def test_static_cohort_preserves_eligibility_and_retains_raw_quotes(
    tmp_path, monkeypatch, ended, frequency
):
    from types import SimpleNamespace
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory

    data = manifest()
    del data["membership"]
    data["minimum_return_periods"] = 12
    if ended:
        data["price_history_ends"] = {"OLD": "2022-01-01"}
    universe = load(tmp_path, monkeypatch, data)
    first = SimpleNamespace(
        decision_time=datetime(2021, 5, 1, tzinfo=timezone.utc),
        trading_date=datetime(2021, 5, 1),
    )
    last = SimpleNamespace(
        decision_time=datetime(2026, 9, 1, tzinfo=timezone.utc),
        trading_date=datetime(2026, 9, 1),
    )
    model = SimpleNamespace(
        periods=[first, last],
        instance=SimpleNamespace(current_trading_period=first),
        stocks=[SimpleNamespace(symbol=s) for s in universe.stocks + universe.crypto],
    )
    factory = SimpleNamespace(
        _get_asset_universe=lambda: universe,
        frequency=frequency,
        model_data=model,
        _validation_report=ValidationReport(),
    )
    ModelDataFactory._prepare_universe_membership(factory)
    for period in [first, last]:
        expected = {"OLD", "NEW", "GLD", "BTC/USD"}
        if ended and period is last:
            expected.remove("OLD")
        assert period.eligible_symbols == expected
        assert period.minimum_return_periods == 12
    assert len(model.stocks) == 4
    model.stocks.pop()
    ModelDataFactory._prepare_universe_membership(factory)
    assert all("BTC/USD" not in p.eligible_symbols for p in model.periods)
    assert factory._validation_report.issues[-1].details["symbols"] == ["BTC/USD"]
    assert (
        factory._validation_report.issues[-1].action == "exclude_unavailable_candidates"
    )


def test_daily_history_calendars_are_explicit_and_distinguish_crypto():
    from types import SimpleNamespace
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory

    model = SimpleNamespace(
        instance=SimpleNamespace(
            historical_start_date="2024-12-24", historical_end_date="2024-12-26"
        ),
        stocks=[SimpleNamespace(symbol=s) for s in ["GLD", "BTC/USD"]],
        crypto_symbols={"BTC/USD"},
    )
    schedule = {
        "2024-12-24": {"timestamp": "2024-12-24T14:30:00Z"},
        "2024-12-25": {"market_closed": True},
        "2024-12-26": {"timestamp": "2024-12-26T14:30:00Z"},
    }
    factory = SimpleNamespace(
        model_data=model, _input_data={"execution_dict": schedule}
    )
    ModelDataFactory._prepare_history_calendars(factory)
    assert model.instance.history_calendars["GLD"].equals(
        pd.DatetimeIndex(["2024-12-24", "2024-12-26"])
    )
    assert len(model.instance.history_calendars["BTC/USD"]) == 3
    schedule["2024-12-25"] = {}
    with pytest.raises(ValueError, match="requires a timestamp"):
        ModelDataFactory._prepare_history_calendars(factory)
    del schedule["2024-12-25"]
    with pytest.raises(ValueError, match="explicit session calendar"):
        ModelDataFactory._prepare_history_calendars(factory)


@pytest.mark.parametrize(
    "field,value",
    [
        ("published_at", "2024-01-02T00:00:00Z"),
        ("effective_from", "2024-01-01"),
        ("expires_at", "2023-12-01T00:00:00Z"),
        ("symbols", ["UNKNOWN"]),
        ("symbols", ["OLD", "OLD"]),
    ],
)
def test_rejects_future_publication_ambiguous_times_and_invalid_rosters(
    tmp_path, monkeypatch, field, value
):
    data = manifest()
    data["membership"]["snapshots"][0][field] = value
    with pytest.raises(ValueError):
        load(tmp_path, monkeypatch, data)


def bars(labels, frequency, price=100.0):
    source = pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 100_000.0,
        },
        index=labels.tz_localize("America/New_York"),
    )
    return normalize_bars(
        source,
        frequency,
        market_timezone="America/New_York",
        as_of="2025-01-01T05:00:00Z",
    )


def test_liquidity_rank_never_uses_future_bars_and_deduplicates_issuers():
    daily = bars(pd.bdate_range("2023-08-01", "2024-01-10"), Frequency.DAILY)
    monthly = bars(
        pd.date_range("2022-12-01", "2024-01-01", freq="MS"), Frequency.MONTHLY
    )
    roster = pd.DataFrame({"symbol": ["AA", "AB", "Z"], "issuer_id": ["A", "A", "Z"]})
    daily_prices = {s: daily.copy() for s in roster.symbol}
    daily_prices["AB"]["volume"] *= 2
    daily_prices["Z"].loc["2024-01-01":, "volume"] = 1e12
    result = rank_equities(
        roster,
        daily_prices,
        {s: monthly for s in roster.symbol},
        datetime(2024, 1, 1, 5, tzinfo=timezone.utc),
        stock_limit=1,
    )
    assert result.loc[result.selected, "symbol"].tolist() == ["AB"]
    reasons = result.set_index("symbol")["reason"].to_dict()
    assert reasons == {
        "AB": "eligible",
        "AA": "alternate_share_class",
        "Z": "outside_stock_limit",
    }
    # Rewriting future data cannot change the result.
    daily_prices["Z"].loc["2024-01-01":, "volume"] = 1
    pd.testing.assert_frame_equal(
        result,
        rank_equities(
            roster,
            daily_prices,
            {s: monthly for s in roster.symbol},
            datetime(2024, 1, 1, 5, tzinfo=timezone.utc),
            stock_limit=1,
        ),
    )


def test_future_history_cannot_admit_ipo_early_and_missing_series_fails():
    daily = bars(pd.bdate_range("2023-08-01", "2024-01-10"), Frequency.DAILY)
    monthly = bars(
        pd.date_range("2023-03-01", "2024-01-01", freq="MS"), Frequency.MONTHLY
    )
    roster = pd.DataFrame({"symbol": ["IPO"], "issuer_id": ["IPO"]})
    result = rank_equities(
        roster,
        {"IPO": daily},
        {"IPO": monthly},
        datetime(2024, 1, 1, 5, tzinfo=timezone.utc),
    )
    assert not result.selected.any()
    assert result.reason.iloc[0] == "insufficient_completed_monthly_history"
    with pytest.raises(ValueError, match="Missing price series"):
        rank_equities(
            roster, {}, {"IPO": monthly}, datetime(2024, 1, 1, 5, tzinfo=timezone.utc)
        )


def test_dated_validation_preserves_young_listings_and_missing_quotes():
    prices = bars(pd.date_range("2023-01-01", periods=13, freq="MS"), Frequency.MONTHLY)
    prices.loc[prices.index[-1], "close"] = np.nan
    validated, _ = ValidationPipeline(point_in_time_history=True).validate(
        {"stock_dict": {"IPO": prices}, "crypto_dict": {}, "news_dict": {}}
    )
    assert "IPO" in validated["stock_dict"]
    assert pd.isna(validated["stock_dict"]["IPO"].iloc[-1]["close"])


def test_factory_preflights_all_decisions_and_keeps_quotes_for_departing_holdings(
    tmp_path, monkeypatch
):
    from datetime import date
    from investment_tool.config import AppConfig
    from investment_tool.constants import DataSource, ExecutionMode
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory
    from investment_tool.domain.model_data.entities.instance import Instance
    from investment_tool.domain.model_data.entities.period import Period
    from investment_tool.domain.model_data.entities.stock import Stock

    universe = load(tmp_path, monkeypatch, manifest())
    factory = ModelDataFactory(
        DataSource.ALPACA,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        cfg=AppConfig(asset_config_file=universe.file),
    )
    prices = bars(pd.date_range("2022-01-01", periods=24, freq="MS"), Frequency.MONTHLY)
    factory.model_data.stocks = [
        Stock(s, prices) for s in ["OLD", "NEW", "GLD", "BTC/USD"]
    ]
    periods = [
        Period(
            datetime(2024, 1, 1),
            pd.DataFrame(),
            decision_time=datetime.fromisoformat("2024-01-01T05:00:00Z"),
        ),
        Period(
            datetime(2024, 4, 1),
            pd.DataFrame(),
            decision_time=datetime.fromisoformat("2024-04-01T04:00:00Z"),
        ),
    ]
    factory.model_data.periods = periods
    factory.model_data.instance = Instance(
        date(2022, 1, 1),
        date(2024, 6, 30),
        Frequency.MONTHLY,
        periods[0],
        None,
        periods[0],
    )
    factory._prepare_universe_membership()
    assert periods[0].eligible_symbols == {"OLD", "GLD", "BTC/USD"}
    assert periods[1].eligible_symbols == {"NEW", "GLD", "BTC/USD"}
    assert {s.symbol for s in factory.model_data.stocks} == {
        "OLD",
        "NEW",
        "GLD",
        "BTC/USD",
    }
    # Validate later coverage too, before news or the first solver starts.
    periods[1].decision_time = datetime.fromisoformat("2024-07-01T04:00:00Z")
    with pytest.raises(ValueError, match="No dated universe"):
        factory._prepare_universe_membership()
    periods[1].decision_time = datetime.fromisoformat("2024-04-01T04:00:00Z")
    factory.model_data.stocks = [
        s for s in factory.model_data.stocks if s.symbol != "NEW"
    ]
    factory._validation_report = ValidationReport()
    factory._prepare_universe_membership()
    assert "NEW" not in periods[1].eligible_symbols
    assert factory._validation_report.issues[-1].details["symbols"] == ["NEW"]
    assert "OLD" in periods[0].eligible_symbols


def test_shared_session_calendar_does_not_hide_missing_daily_observations():
    daily = bars(pd.bdate_range("2023-08-01", "2023-12-29"), Frequency.DAILY)
    monthly = bars(
        pd.date_range("2022-12-01", "2023-12-01", freq="MS"), Frequency.MONTHLY
    )
    missing = daily.drop(daily.index[-10])
    roster = pd.DataFrame({"symbol": ["OK", "GAP"], "issuer_id": ["OK", "GAP"]})
    result = rank_equities(
        roster,
        {"OK": daily, "GAP": missing},
        {"OK": monthly, "GAP": monthly},
        datetime(2024, 1, 1, 5, tzinfo=timezone.utc),
    )
    assert result.loc[result.selected, "symbol"].tolist() == ["OK"]
    assert (
        result.set_index("symbol").loc["GAP", "reason"]
        == "insufficient_recent_daily_history"
    )


@pytest.mark.parametrize(
    "history",
    [{"UNKNOWN": "2023-01-01"}, {"OLD": "20230101"}, {"OLD": "bad"}, {"OLD": None}],
)
@pytest.mark.parametrize("field", ["price_history_starts", "price_history_ends"])
def test_history_boundaries_require_explicit_configured_calendar_dates(
    tmp_path, monkeypatch, history, field
):
    data = manifest()
    data[field] = history
    with pytest.raises(ValueError, match="history"):
        load(tmp_path, monkeypatch, data)


def test_factory_identity_boundary_precedes_indicators_for_every_provider(
    tmp_path, monkeypatch
):
    from datetime import date
    from investment_tool.config import AppConfig
    from investment_tool.constants import DataSource, ExecutionMode
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory
    from investment_tool.domain.model_data.entities.instance import Instance

    data = manifest()
    data["price_history_starts"] = {"OLD": "2023-01-01"}
    universe = load(tmp_path, monkeypatch, data)
    prices = bars(pd.date_range("2022-01-01", periods=24, freq="MS"), Frequency.MONTHLY)
    prices.loc[:"2022-12-01", "close"] = 1000
    for provider in [DataSource.LOCAL, DataSource.ALPACA]:
        factory = ModelDataFactory(
            provider,
            ExecutionMode.INVESTING,
            Frequency.MONTHLY,
            0,
            input_data_path=tmp_path / "unused.pkl"
            if provider == DataSource.LOCAL
            else None,
            cfg=AppConfig(asset_config_file=universe.file),
        )
        factory.model_data.instance = Instance(
            date(2022, 1, 1), date(2023, 12, 31), Frequency.MONTHLY, None, None, None
        )
        factory._input_data = {"stock_dict": {"OLD": prices}, "crypto_dict": {}}
        factory._build_stocks()
        stock = factory.model_data.stocks[0]
        assert stock.data.index.min() == pd.Timestamp("2023-01-01")
        assert pd.isna(stock.data.iloc[0]["returns"])
        assert stock.data.iloc[1]["returns"] == 0


def test_identity_end_excludes_reused_ticker_without_dropping_earlier_history(
    tmp_path, monkeypatch
):
    from datetime import date
    from investment_tool.config import AppConfig
    from investment_tool.constants import DataSource, ExecutionMode
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory
    from investment_tool.domain.model_data.entities.instance import Instance

    data = manifest()
    data["price_history_ends"] = {"OLD": "2023-01-01"}
    universe = load(tmp_path, monkeypatch, data)
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.INVESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=tmp_path / "unused.pkl",
        cfg=AppConfig(asset_config_file=universe.file),
    )
    factory.model_data.instance = Instance(
        date(2022, 1, 1), date(2023, 12, 31), Frequency.MONTHLY, None, None, None
    )
    prices = bars(pd.date_range("2022-01-01", periods=24, freq="MS"), Frequency.MONTHLY)
    prices.loc["2023-01-01":, "close"] = 1000
    factory._input_data = {"stock_dict": {"OLD": prices}, "crypto_dict": {}}
    factory._build_stocks()
    stock = factory.model_data.stocks[0]
    assert stock.data.index.min() == pd.Timestamp("2022-01-01")
    assert stock.data.index.max() == pd.Timestamp("2022-12-01")
    assert stock.data.close.max() < 1000


def test_identity_start_must_precede_exclusive_end(tmp_path, monkeypatch):
    data = manifest()
    data["price_history_starts"] = {"OLD": "2023-01-01"}
    data["price_history_ends"] = {"OLD": "2023-01-01"}
    with pytest.raises(ValueError, match="precede"):
        load(tmp_path, monkeypatch, data)


@pytest.mark.parametrize(
    "sources",
    [None, [], {"UNKNOWN": "notice"}, {"NEW": "notice"}, {"OLD": ""}, {"OLD": 5}],
)
def test_identity_evidence_requires_a_bounded_asset_and_nonempty_source(
    tmp_path, monkeypatch, sources
):
    data = manifest()
    data["price_history_ends"] = {"OLD": "2024-02-15"}
    data["price_history_sources"] = sources
    with pytest.raises(ValueError, match="history sources"):
        load(tmp_path, monkeypatch, data)


def test_identity_evidence_is_optional_for_legacy_manifests(tmp_path, monkeypatch):
    data = manifest()
    data["price_history_ends"] = {"OLD": "2024-02-15"}
    assert load(tmp_path, monkeypatch, data).price_history_sources == {}
    data["price_history_sources"] = {"OLD": "reviewed notice"}
    assert load(tmp_path, monkeypatch, data).price_history_sources == {
        "OLD": "reviewed notice"
    }


def cash_action():
    return {
        "symbol": "OLD",
        "effective_at": "2024-02-15T04:00:00Z",
        "published_at": "2024-02-16T04:00:00Z",
        "cash_per_share": 52,
        "source": "Reviewed acquisition notice",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"symbol": "UNKNOWN"},
        {"cash_per_share": True},
        {"cash_per_share": -1},
        {"cash_per_share": float("inf")},
        {"effective_at": "2024-02-15"},
        {"published_at": None},
        {"source": ""},
    ],
)
def test_cash_actions_require_audited_values(tmp_path, monkeypatch, change):
    data = manifest()
    data["cash_settlements"] = [{**cash_action(), **change}]
    with pytest.raises(ValueError):
        load(tmp_path, monkeypatch, data)


def test_cash_actions_do_not_remove_candidates_before_the_event(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from investment_tool.data.assembly.model_data_factory import ModelDataFactory

    data = manifest()
    del data["membership"]
    data["cash_settlements"] = [cash_action()]
    universe = load(tmp_path, monkeypatch, data)
    periods = [
        SimpleNamespace(
            decision_time=datetime.fromisoformat(t), trading_date=datetime(2024, 2, 1)
        )
        for t in ["2024-02-15T04:00:00+00:00", "2024-02-16T04:00:00+00:00"]
    ]
    model = SimpleNamespace(
        periods=periods,
        instance=SimpleNamespace(current_trading_period=periods[0]),
        stocks=[SimpleNamespace(symbol=s) for s in universe.stocks + universe.crypto],
    )
    factory = SimpleNamespace(
        _get_asset_universe=lambda: universe,
        frequency=Frequency.MONTHLY,
        model_data=model,
    )
    ModelDataFactory._prepare_universe_membership(factory)
    assert "OLD" in periods[0].eligible_symbols
    assert "OLD" not in periods[1].eligible_symbols
    assert len(model.cash_settlements) == 1
    data["cash_settlements"] *= 2
    with pytest.raises(ValueError, match="unique"):
        load(tmp_path, monkeypatch, data)


@pytest.mark.parametrize("shares", [None, {}, {"NEW": 2}])
def test_incomplete_security_actions_load_without_inventing_shares(
    tmp_path, monkeypatch, shares
):
    data = manifest()
    event = {**cash_action(), "unresolved_entitlements": ["  Verify warrant quotes  "]}
    if shares is not None:
        event["successor_shares"] = shares
    data["security_actions"] = [event]
    loaded = load(tmp_path, monkeypatch, data).security_actions[0]
    assert loaded.unresolved_entitlements == ("Verify warrant quotes",)
    assert loaded.successor_shares == ((("NEW", 2),) if shares else ())
    assert loaded.cash_per_share == 52  # Known terms retained, not declared complete.


@pytest.mark.parametrize(
    "reasons",
    [None, "Warrants", {}, [False], [1], [""], ["  "], ["Warrants", " Warrants "]],
)
@pytest.mark.parametrize("collection", ["cash_settlements", "security_actions"])
def test_invalid_incomplete_entitlement_declarations_are_rejected(
    tmp_path, monkeypatch, reasons, collection
):
    data = manifest()
    data[collection] = [
        {
            **cash_action(),
            "successor_shares": {"NEW": 2},
            "unresolved_entitlements": reasons,
        }
    ]
    with pytest.raises(ValueError, match="unresolved_entitlements"):
        load(tmp_path, monkeypatch, data)


def test_no_share_terms_and_no_unresolved_reasons_is_not_a_complete_action(
    tmp_path, monkeypatch
):
    data = manifest()
    data["security_actions"] = [{**cash_action(), "unresolved_entitlements": []}]
    with pytest.raises(ValueError, match="requires successor shares"):
        load(tmp_path, monkeypatch, data)


def test_partial_cash_settlement_keeps_its_unresolved_components(tmp_path, monkeypatch):
    data = manifest()
    data["cash_settlements"] = [
        {**cash_action(), "unresolved_entitlements": ["Warrant quotes"]}
    ]
    event = load(tmp_path, monkeypatch, data).cash_settlements[0]
    assert event.unresolved_entitlements == ("Warrant quotes",)
    assert event.cash_per_share == 52
