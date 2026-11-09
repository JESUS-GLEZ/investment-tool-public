"""Synchronize monthly crypto execution with actual US session opens."""

import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from alpaca.data.models import BarSet
from alpaca.trading.models import Calendar
import pandas as pd
import pytest

from investment_tool.config import AppConfig, ModelConfig, PathsConfig
from investment_tool.domain.market_time import normalize_bars
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.data.assembly.model_data_factory import ModelDataFactory
from investment_tool.data.enrich_execution_data import enrich_file
from investment_tool.data.repositories._alpaca_client import AlpacaDataFetcher
from investment_tool.data.repositories.alpaca_repository import AlpacaRepository
from investment_tool.data.repositories.base_repository import DataResult
from investment_tool.data.repositories.cached_repository import CachedRepository
from investment_tool.data.repositories.local_data_saver import LocalDataSaver
from investment_tool.data.repositories.local_repository import LocalRepository


def sdk_bars(rows):
    return BarSet(
        {
            "BTC/USD": [
                {"h": 1000, "l": 1, "c": 100, "v": 1, "n": 1, "vw": 100, **row}
                for row in rows
            ]
        }
    )


def fetcher():
    client = AlpacaDataFetcher.__new__(AlpacaDataFetcher)
    client.calendar_client = MagicMock()
    client.crypto_client = MagicMock()
    client.rate_limit_sleep = 0
    return client


@pytest.mark.parametrize("sdk_format", [False, True])
def test_calendar_holidays_weekends_dst_and_exact_crypto_minute(sdk_format):
    client = fetcher()
    client.calendar_client.get_calendar.return_value = [
        Calendar(date=day, open="09:30", close="16:00")
        for day in ("2024-01-03", "2024-01-02", "2024-07-01", "2024-06-03")
    ]

    def crypto_bars(**kwargs):
        start = pd.Timestamp(kwargs["request_params"].start).tz_localize("UTC")
        # Prefer the exact open even when later minutes are also returned.
        frame = pd.DataFrame(
            {"open": [100, 999]},
            index=pd.MultiIndex.from_tuples(
                [("BTC/USD", start), ("BTC/USD", start + pd.Timedelta(minutes=1))],
                names=["symbol", "timestamp"],
            ),
        )
        if sdk_format:
            return sdk_bars(
                [
                    {"S": "BTC/USD", "t": start.isoformat(), "o": 100},
                    {
                        "S": "BTC/USD",
                        "t": (start + pd.Timedelta(minutes=1)).isoformat(),
                        "o": 999,
                    },
                ]
            )
        return SimpleNamespace(df=frame)

    client.crypto_client.get_crypto_bars.side_effect = crypto_bars
    result = client.get_execution_data(["BTC/USD"], "2024-01-15", "2024-07-01")
    assert result == {
        "2024-01-01": {
            "timestamp": "2024-01-02T14:30:00+00:00",
            "crypto_prices": {"BTC/USD": 100},
            "crypto_timestamps": {"BTC/USD": "2024-01-02T14:30:00+00:00"},
        },
        "2024-06-01": {
            "timestamp": "2024-06-03T13:30:00+00:00",
            "crypto_prices": {"BTC/USD": 100},
            "crypto_timestamps": {"BTC/USD": "2024-06-03T13:30:00+00:00"},
        },
        "2024-07-01": {
            "timestamp": "2024-07-01T13:30:00+00:00",
            "crypto_prices": {"BTC/USD": 100},
            "crypto_timestamps": {"BTC/USD": "2024-07-01T13:30:00+00:00"},
        },
    }
    assert client.crypto_client.get_crypto_bars.call_count == 3
    filters = client.calendar_client.get_calendar.call_args.kwargs["filters"]
    assert filters.start.isoformat() == "2024-01-01"
    assert filters.end.isoformat() == "2024-07-31"


def test_quote_outside_execution_delay_is_not_used():
    client = fetcher()
    client.calendar_client.get_calendar.return_value = [
        Calendar(date="2024-06-03", open="09:30", close="16:00")
    ]
    client.crypto_client.get_crypto_bars.return_value = SimpleNamespace(
        df=pd.DataFrame(
            {"open": [99]},
            index=pd.MultiIndex.from_tuples(
                [("BTC/USD", pd.Timestamp("2024-06-03T13:33:00Z"))],
                names=["symbol", "timestamp"],
            ),
        )
    )
    assert (
        client.get_execution_data(["BTC/USD"], "2024-06-01", "2024-06-30")[
            "2024-06-01"
        ]["crypto_prices"]
        == {}
    )


def test_weekly_calendar_keeps_distinct_weeks_holidays_and_dst():
    client = fetcher()
    client.calendar_client.get_calendar.return_value = [
        Calendar(date=day, open="09:30", close="16:00")
        for day in (
            "2024-03-12",
            "2024-01-03",
            "2024-01-02",
            "2024-01-08",
            "2024-03-04",
            "2024-03-11",
        )
    ]

    def bars(**kwargs):
        opening = pd.Timestamp(kwargs["request_params"].start).tz_localize("UTC")
        return sdk_bars(
            [
                {"t": opening.isoformat(), "o": -1},
                {"t": (opening + pd.Timedelta(minutes=1)).isoformat(), "o": 100},
            ]
        )

    client.crypto_client.get_crypto_bars.side_effect = bars
    result = client.get_execution_data(
        ["BTC/USD"], "2024-01-03", "2024-03-12", Frequency.WEEKLY
    )
    assert list(result) == ["2024-01-01", "2024-01-08", "2024-03-04", "2024-03-11"]
    assert result["2024-01-01"]["timestamp"] == "2024-01-02T14:30:00+00:00"
    assert result["2024-03-04"]["timestamp"] == "2024-03-04T14:30:00+00:00"
    assert result["2024-03-11"]["timestamp"] == "2024-03-11T13:30:00+00:00"
    assert (
        result["2024-03-11"]["crypto_timestamps"]["BTC/USD"]
        == "2024-03-11T13:31:00+00:00"
    )
    assert client.crypto_client.get_crypto_bars.call_count == 4
    filters = client.calendar_client.get_calendar.call_args.kwargs["filters"]
    assert filters.start.isoformat() == "2024-01-01"
    assert filters.end.isoformat() == "2024-03-17"


def test_weekly_stock_only_schedule_crosses_year_without_requesting_crypto():
    client = fetcher()
    client.calendar_client.get_calendar.return_value = [
        Calendar(date=day, open="09:30", close="16:00")
        for day in ("2024-12-30", "2025-01-02", "2025-01-06")
    ]
    result = client.get_execution_data([], "2024-12-30", "2025-01-12", Frequency.WEEKLY)
    assert list(result) == ["2024-12-30", "2025-01-06"]
    client.crypto_client.get_crypto_bars.assert_not_called()


def test_daily_calendar_declares_weekends_and_holidays_closed():
    client = fetcher()
    client.calendar_client.get_calendar.return_value = [
        Calendar(date=day, open="09:30", close="16:00")
        for day in ("2024-01-12", "2024-01-16")
    ]
    result = client.get_execution_data([], "2024-01-12", "2024-01-16", Frequency.DAILY)
    assert result["2024-01-12"]["timestamp"] == "2024-01-12T14:30:00+00:00"
    for day in ("2024-01-13", "2024-01-14", "2024-01-15"):
        assert result[day] == {"market_closed": True, "crypto_prices": {}}
    assert "market_closed" not in result["2024-01-16"]
    client.crypto_client.get_crypto_bars.assert_not_called()


@pytest.mark.parametrize(
    "start,frequency,expected",
    [
        ("2024-01-01", Frequency.WEEKLY, "2024-01-08"),
        ("2024-01-03", Frequency.WEEKLY, "2024-01-08"),
        ("2024-01-31", Frequency.MONTHLY, "2024-02-01"),
        ("2024-01-05", Frequency.DAILY, "2024-01-06"),
    ],
)
def test_news_buckets_end_at_the_next_calendar_boundary(start, frequency, expected):
    from datetime import date

    assert (
        AlpacaDataFetcher._get_period_end(
            date.fromisoformat(start), frequency, date(2024, 2, 29)
        ).isoformat()
        == expected
    )


@pytest.mark.parametrize("format", ["pickle", "json"])
def test_quotes_survive_dataresult_and_local_save_load(tmp_path, format):
    quotes = {
        "2024-06-01": {
            "timestamp": "2024-06-03T13:30:00+00:00",
            "crypto_prices": {"BTC/USD": 100},
            "crypto_timestamps": {"BTC/USD": "2024-06-03T13:31:00+00:00"},
        }
    }
    data = DataResult(execution_dict=quotes)
    assert DataResult.from_dict(data.to_dict()).execution_dict == quotes
    cfg = AppConfig(paths=PathsConfig(input_data_dir=str(tmp_path)))
    path = LocalDataSaver.save_data(data.to_dict(), "monthly", format, cfg=cfg)
    loaded = LocalRepository(Frequency.MONTHLY, path).fetch_all(
        [], [], "2024-01-01", "2024-12-31"
    )
    assert loaded.execution_dict == quotes


def test_cache_delegates_execution_quotes_without_ohlcv_cache():
    inner = MagicMock()
    inner.frequency = Frequency.MONTHLY
    session_factory = MagicMock()
    repository = CachedRepository(inner, session_factory)
    repository.fetch_execution_data(["BTC/USD"], "2024-01-01", "2024-12-31")
    inner.fetch_execution_data.assert_called_once_with(
        ["BTC/USD"], "2024-01-01", "2024-12-31"
    )
    session_factory.assert_not_called()


def factory_with_crypto():
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=Path("unused.pkl"),
        cfg=AppConfig(model=ModelConfig(min_backtest_periods=3)),
    )
    factory.model_data.instance = SimpleNamespace(
        historical_start_date="2024-01-01", historical_end_date="2024-03-31"
    )
    frame = pd.DataFrame(
        {
            "open": [10, 20, 30],
            "high": [11, 21, 31],
            "low": [9, 19, 29],
            "volume": [1, 1, 1],
            "close": [11, 21, 31],
        },
        index=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
    )
    frame = normalize_bars(
        frame.set_axis(frame.index.tz_localize("UTC")),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )
    factory._input_data = {
        "crypto_dict": {"BTC/USD": frame},
        "stock_dict": {},
        "metadata": {"time_contract_version": 1, "frequency": "monthly"},
    }
    factory._build_stocks()
    return factory


def test_legacy_mixed_dataset_fails_preflight_before_any_strategy():
    factory = factory_with_crypto()
    with pytest.raises(ValueError, match="enrich_execution_data"):
        factory._build_periods()


def test_factory_attaches_execution_data_without_changing_monthly_features():
    factory = factory_with_crypto()
    factory._input_data["execution_dict"] = {
        f"2024-{month:02d}-01": {
            "timestamp": f"2024-{month:02d}-02T14:30:00+00:00",
            "crypto_prices": {"BTC/USD": 999},
        }
        for month in (1, 2, 3)
    }
    factory._build_periods()
    current = factory.model_data.instance.current_trading_period
    assert current.crypto_execution_prices == {"BTC/USD": 999}
    assert factory.model_data.stocks[0].data["open"].tolist() == [10, 20, 30]


def test_merge_does_not_fabricate_missing_months_open_or_close():
    factory = factory_with_crypto()
    original = factory.model_data.stocks[0].data
    factory.model_data.stocks.append(SimpleNamespace(symbol="ABC", data=original))
    factory.model_data.stocks[0].data = original.iloc[:2]
    merged = factory._merge_stock_data()
    assert pd.isna(merged.loc[pd.Timestamp("2024-03-01"), ("open", "BTC/USD")])
    assert pd.isna(merged.loc[pd.Timestamp("2024-03-01"), ("close", "BTC/USD")])


@pytest.mark.parametrize("frequency", list(Frequency))
def test_enrichment_preserves_source_and_existing_history(
    tmp_path, monkeypatch, frequency
):
    factory = factory_with_crypto()
    source = tmp_path / "original.pkl"
    output = tmp_path / "enriched.pkl"
    data = {
        **factory._input_data,
        "news_dict": {"ABC": {("2024-01-01", "2024-01-31"): [{"headline": "news"}]}},
    }
    if frequency != Frequency.MONTHLY:
        code = "W-MON" if frequency == Frequency.WEEKLY else "D"
        frame = data["crypto_dict"]["BTC/USD"]
        data["crypto_dict"]["BTC/USD"] = normalize_bars(
            frame[["open", "high", "low", "close", "volume"]].set_axis(
                pd.date_range("2024-01-01", periods=3, freq=code, tz="UTC")
            ),
            frequency,
            market_timezone="UTC",
        )
        data["metadata"]["frequency"] = frequency.value
    source.write_bytes(pickle.dumps(data))
    original_bytes = source.read_bytes()
    quotes = {
        "2024-02-01": {
            "timestamp": "2024-02-01T14:30:00+00:00",
            "crypto_prices": {"BTC/USD": 30},
        }
    }
    repo = MagicMock()
    repo.fetch_execution_data.return_value = quotes
    monkeypatch.setattr(AlpacaRepository, "__init__", lambda self, frequency: None)
    monkeypatch.setattr(
        AlpacaRepository, "fetch_execution_data", repo.fetch_execution_data
    )
    enrich_file(source, output)
    enriched = pickle.loads(output.read_bytes())
    pd.testing.assert_frame_equal(
        enriched["crypto_dict"]["BTC/USD"], data["crypto_dict"]["BTC/USD"]
    )
    assert enriched["news_dict"] == data["news_dict"]
    assert enriched["execution_dict"] == quotes
    assert source.read_bytes() == original_bytes
    expected_end = {
        Frequency.MONTHLY: "2024-03-31",
        Frequency.WEEKLY: "2024-01-21",
        Frequency.DAILY: "2024-01-03",
    }[frequency]
    repo.fetch_execution_data.assert_called_once_with(
        ["BTC/USD"], "2024-01-01", expected_end
    )
    with pytest.raises(ValueError, match="Output already exists"):
        enrich_file(source, output)


@pytest.mark.parametrize("delay", [0, 1, 2])
@pytest.mark.parametrize("sdk_format", [False, True])
def test_first_valid_crypto_fill_within_bound_has_actual_timestamp(delay, sdk_format):
    client = fetcher()
    client.calendar_client.get_calendar.return_value = [
        Calendar(date="2024-08-01", open="09:30", close="16:00")
    ]
    opening = pd.Timestamp("2024-08-01T13:30:00Z")
    actual = opening + pd.Timedelta(minutes=delay)
    # Unsorted input includes a stale price and a price outside the limit.
    raw = [
        {
            "S": "BTC/USD",
            "t": (opening + pd.Timedelta(minutes=3)).isoformat(),
            "o": 900,
        },
        {"S": "BTC/USD", "t": actual.isoformat(), "o": 100},
        {"S": "BTC/USD", "t": (opening - pd.Timedelta(minutes=1)).isoformat(), "o": 10},
    ]
    frame = pd.DataFrame(
        {"open": [item["o"] for item in raw], "symbol": ["BTC/USD"] * 3},
        index=pd.DatetimeIndex([item["t"] for item in raw], name="timestamp"),
    )
    client.crypto_client.get_crypto_bars.return_value = (
        sdk_bars(raw) if sdk_format else SimpleNamespace(df=frame)
    )
    quote = client.get_execution_data(["BTC/USD"], "2024-08-01", "2024-08-31")[
        "2024-08-01"
    ]
    assert quote == {
        "timestamp": opening.isoformat(),
        "crypto_prices": {"BTC/USD": 100},
        "crypto_timestamps": {"BTC/USD": actual.isoformat()},
    }
    request = client.crypto_client.get_crypto_bars.call_args.kwargs["request_params"]
    assert pd.Timestamp(request.end).tz_localize("UTC") == opening + pd.Timedelta(
        minutes=2
    )


def test_invalid_crypto_open_does_not_hide_a_valid_delayed_fill():
    client = fetcher()
    client.calendar_client.get_calendar.return_value = [
        Calendar(date="2024-08-01", open="09:30", close="16:00")
    ]
    client.crypto_client.get_crypto_bars.return_value = SimpleNamespace(
        df=pd.DataFrame(
            {"symbol": ["BTC/USD"] * 3, "open": [float("nan"), -1, 100]},
            index=pd.date_range("2024-08-01T13:30:00Z", periods=3, freq="min"),
        )
    )
    quote = client.get_execution_data(["BTC/USD"], "2024-08-01", "2024-08-31")[
        "2024-08-01"
    ]
    assert quote["crypto_prices"] == {"BTC/USD": 100}
    assert quote["crypto_timestamps"] == {"BTC/USD": "2024-08-01T13:32:00+00:00"}


@pytest.mark.parametrize("delay", [-1, 3])
def test_local_data_cannot_bypass_execution_delay_policy(delay):
    factory = factory_with_crypto()
    factory._input_data["execution_dict"] = {
        f"2024-{month:02d}-01": {
            "timestamp": f"2024-{month:02d}-02T14:30:00+00:00",
            "crypto_prices": {"BTC/USD": 99},
            "crypto_timestamps": {
                "BTC/USD": (
                    pd.Timestamp(f"2024-{month:02d}-02T14:30:00Z")
                    + pd.Timedelta(minutes=delay)
                ).isoformat()
            },
        }
        for month in (1, 2, 3)
    }
    with pytest.raises(ValueError, match="within two minutes"):
        factory._build_periods()


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_factory_keeps_crypto_fill_times_separate_from_stock_open(delay):
    factory = factory_with_crypto()
    factory._input_data["execution_dict"] = {
        f"2024-{month:02d}-01": {
            "timestamp": f"2024-{month:02d}-02T14:30:00+00:00",
            "crypto_prices": {"BTC/USD": 99},
            "crypto_timestamps": {
                "BTC/USD": (
                    pd.Timestamp(f"2024-{month:02d}-02T14:30:00Z")
                    + pd.Timedelta(minutes=delay)
                ).isoformat()
            },
        }
        for month in (1, 2, 3)
    }
    factory._build_periods()
    current = factory.model_data.instance.current_trading_period
    assert (
        current.crypto_execution_times["BTC/USD"]
        == (
            pd.Timestamp(current.execution_time) + pd.Timedelta(minutes=delay)
        ).to_pydatetime()
    )
    assert current.crypto_execution_prices == {"BTC/USD": 99}
    assert factory.model_data.stocks[0].data["open"].tolist() == [10, 20, 30]


def test_missing_crypto_quotes_fail_before_fetching_news(monkeypatch):
    from investment_tool.data.repositories.repository_factory import RepositoryFactory
    from investment_tool.data.validation import ValidationLevel
    from investment_tool.exceptions import DataRetrievalError

    factory = factory_with_crypto()
    factory.validation_level = ValidationLevel.SKIP
    repository = MagicMock()
    repository.source_name = "Fake"
    repository.fetch_all.return_value = DataResult.from_dict(factory._input_data)
    repository.fetch_execution_data.return_value = {}
    monkeypatch.setattr(RepositoryFactory, "create", lambda **kwargs: repository)
    with pytest.raises(DataRetrievalError, match="Missing synchronized crypto"):
        factory._create_model_data()
    assert repository.fetch_all.call_args.kwargs["include_news"] is False
    repository.fetch_news_data.assert_not_called()


def test_valid_execution_preflight_fetches_news_once(monkeypatch):
    from investment_tool.data.repositories.repository_factory import RepositoryFactory
    from investment_tool.data.validation import ValidationLevel

    factory = factory_with_crypto()
    factory.validation_level = ValidationLevel.SKIP
    repository = MagicMock()
    repository.source_name = "Fake"
    repository.fetch_all.return_value = DataResult.from_dict(factory._input_data)
    repository.fetch_execution_data.return_value = {
        f"2024-{month:02d}-01": {
            "timestamp": f"2024-{month:02d}-02T14:30:00+00:00",
            "crypto_prices": {"BTC/USD": 99},
        }
        for month in (1, 2, 3)
    }
    repository.fetch_news_data.return_value = {}
    monkeypatch.setattr(RepositoryFactory, "create", lambda **kwargs: repository)
    factory._create_model_data()
    repository.fetch_news_data.assert_called_once()
