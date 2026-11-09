"""Offline HTTP fixtures exercise the real SDK, pagination and repository output."""

import copy
import json
import logging
from datetime import date

import pandas as pd
import pytest
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.models import Base
from investment_tool.config import AlpacaConfig
from investment_tool.constants import Frequency
from investment_tool.data.repositories._alpaca_client import AlpacaDataFetcher
from investment_tool.data.repositories.alpaca_repository import AlpacaRepository
from investment_tool.data.repositories.cached_repository import CachedRepository


@pytest.fixture
def http(monkeypatch):
    """Replace only the HTTP boundary; unexpected requests fail the test."""
    responses = []
    calls = []

    def request(session, method, url, **kwargs):
        assert method == "GET"
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"]["APCA-API-KEY-ID"] == "fixture-key"
        assert kwargs["headers"]["APCA-API-SECRET-KEY"] == "fixture-secret"
        calls.append((url, copy.deepcopy(kwargs["params"])))
        assert responses, f"Unexpected request to {url}"
        payload, status = responses.pop(0)
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        response.url = url
        return response

    monkeypatch.setattr(requests.Session, "request", request)
    yield responses, calls
    assert not responses, "Not all expected response pages were requested"


@pytest.fixture
def client(http):
    return AlpacaDataFetcher(
        rate_limit_sleep=0,
        alpaca_config=AlpacaConfig(
            api_key="fixture-key",
            api_secret="fixture-secret",
            base_url="https://calendar.example/v2/",
        ),
    )


def bar(timestamp, opening=100):
    return {
        "t": timestamp,
        "o": opening,
        "h": 103,
        "l": 99,
        "c": 102,
        "v": 1000,
        "n": 50,
        "vw": 101,
    }


@pytest.mark.parametrize("crypto", [False, True])
@pytest.mark.parametrize(
    "frequency,dates,timeframe",
    [
        (Frequency.DAILY, ["2024-01-02", "2024-01-03"], "1Day"),
        (Frequency.WEEKLY, ["2024-01-01", "2024-01-08"], "1Week"),
        (Frequency.MONTHLY, ["2024-01-01", "2024-02-01"], "1Month"),
    ],
)
def test_bar_pages_preserve_values_timing_and_stock_adjustment(
    client, http, crypto, frequency, dates, timeframe
):
    responses, calls = http
    symbol = "BTC/USD" if crypto else "ABC"
    zone = "UTC" if crypto else "America/New_York"
    timestamps = pd.DatetimeIndex(dates).tz_localize(zone).tz_convert("UTC")
    for i, timestamp in enumerate(timestamps):
        responses.append(
            (
                {
                    "bars": {symbol: [bar(timestamp.isoformat(), 100 + i)]},
                    "next_page_token": "second-page" if i == 0 else None,
                },
                200,
            )
        )
    if crypto:
        result = client.get_historical_crypto_data(
            [symbol], "2024-01-01", "2024-03-31", frequency
        )[symbol]
    else:
        result = client.get_historical_data(
            [symbol],
            "2024-01-01",
            "2024-03-31",
            frequency,
            symbol_identity_date=date(2021, 1, 1),
        )[symbol]
    assert result.index.equals(pd.DatetimeIndex(dates, name="period"))
    assert result.open.tolist() == [100, 101]
    assert result.close.tolist() == [102, 102]
    assert result.trade_count.tolist() == [50, 50]
    assert result.vwap.tolist() == [101, 101]
    assert result.is_complete.all()
    assert result.market_timezone.eq(zone).all()
    assert result.bar_start.tolist() == list(timestamps)
    assert set(result.columns) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
        "period_end",
        "bar_start",
        "bar_end",
        "available_at",
        "is_complete",
        "market_timezone",
    }
    expected_path = "/v1beta3/crypto/us/bars" if crypto else "/v2/stocks/bars"
    assert [url for url, _ in calls] == [
        "https://data.alpaca.markets" + expected_path
    ] * 2
    params = calls[0][1]
    assert params["symbols"] == symbol
    assert str(params["timeframe"]) == timeframe
    assert params["start"] == (
        "2024-01-01T00:00:00+00:00" if crypto else "2024-01-01T05:00:00+00:00"
    )
    assert params["end"] == (
        "2024-04-01T00:00:00+00:00" if crypto else "2024-04-01T04:00:00+00:00"
    )
    assert calls[1][1]["page_token"] == "second-page"
    if crypto:
        assert "asof" not in params and "adjustment" not in params
    else:
        assert params["asof"] == "2021-01-01"
        assert params["adjustment"] == "split"


def test_real_calendar_and_paginated_minute_bars_keep_dst_and_delayed_fill(
    client, http
):
    responses, calls = http
    responses.append(([{"date": "2024-06-03", "open": "09:30", "close": "16:00"}], 200))
    responses.extend(
        [
            (
                {
                    "bars": {"BTC/USD": [bar("2024-06-03T13:30:00Z", -1)]},
                    "next_page_token": "more",
                },
                200,
            ),
            (
                {
                    "bars": {"BTC/USD": [bar("2024-06-03T13:31:00Z", 101)]},
                    "next_page_token": None,
                },
                200,
            ),
        ]
    )
    result = client.get_execution_data(["BTC/USD"], "2024-06-01", "2024-06-30")
    assert result == {
        "2024-06-01": {
            "timestamp": "2024-06-03T13:30:00+00:00",
            "crypto_prices": {"BTC/USD": 101},
            "crypto_timestamps": {"BTC/USD": "2024-06-03T13:31:00+00:00"},
        }
    }
    assert calls[0] == (
        "https://calendar.example/v2/calendar",
        {"start": "2024-06-01", "end": "2024-06-30"},
    )
    assert calls[1][1]["start"] == "2024-06-03T13:30:00+00:00"
    assert calls[1][1]["end"] == "2024-06-03T13:32:00+00:00"
    assert str(calls[1][1]["timeframe"]) == "1Min"


def article(id):
    return {
        "id": id,
        "headline": "Fixture headline",
        "summary": "Fixture summary",
        "author": "Fixture author",
        "created_at": "2024-01-02T12:00:00Z",
        "updated_at": "2024-01-02T13:00:00Z",
        "source": "Fixture source",
        "content": "",
        "url": "https://news.example/article",
        "symbols": ["ABC"],
        "images": [{"size": "small", "url": "https://news.example/image.png"}],
    }


def test_news_pages_respect_total_limit_and_keep_serializable_output(client, http):
    responses, calls = http
    responses.extend(
        [
            ({"news": [article(1), article(2)], "next_page_token": "more"}, 200),
            ({"news": [article(3)], "next_page_token": "unused"}, 200),
        ]
    )
    result = client.get_news_data(
        ["ABC"], "2024-01-02", "2024-01-02", Frequency.DAILY, limit=3
    )
    items = result["ABC"][("2024-01-02", "2024-01-03")]
    assert [item["id"] for item in items] == [1, 2, 3]
    assert items[0] == {
        "id": 1,
        "symbol": "ABC",
        "headline": "Fixture headline",
        "summary": "Fixture summary",
        "author": "Fixture author",
        "published_at": "2024-01-02T12:00:00+00:00",
        "updated_at": "2024-01-02T13:00:00+00:00",
        "source": "Fixture source",
        "content": "",
        "url": "https://news.example/article",
        "images": [{"size": "small", "url": "https://news.example/image.png"}],
    }
    json.dumps(items)
    assert len(calls) == 2
    assert all(url == "https://data.alpaca.markets/v1beta1/news" for url, _ in calls)
    assert calls[0][1]["limit"] == 3 and calls[1][1]["limit"] == 1
    assert calls[1][1]["page_token"] == "more"
    assert calls[0][1]["symbols"] == "ABC" and calls[0][1]["sort"] == "desc"
    assert calls[0][1]["start"] == "2024-01-02T00:00:00+00:00"
    assert calls[0][1]["end"] == "2024-01-03T00:00:00+00:00"


@pytest.mark.parametrize("status", [200, 403])
def test_empty_and_failed_bars_remain_empty(client, http, status):
    responses, _ = http
    responses.append(
        (
            {"bars": {}, "next_page_token": None}
            if status == 200
            else {"message": "Forbidden"},
            status,
        )
    )
    result = client.get_historical_data(
        ["ABC"], "2024-01-01", "2024-01-31", Frequency.DAILY
    )
    assert result["ABC"].empty


def test_sdk_bars_round_trip_through_sqlite_cache(client, http):
    responses, calls = http
    responses.append(
        ({"bars": {"ABC": [bar("2024-01-02T05:00:00Z")]}, "next_page_token": None}, 200)
    )
    repository = AlpacaRepository.__new__(AlpacaRepository)
    repository.frequency = Frequency.DAILY
    repository.symbol_identity_date = None
    repository._fetcher = client
    repository._logger = logging.getLogger(__name__)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    cached = CachedRepository(repository, sessionmaker(bind=engine))
    first = cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-02")["ABC"]
    second = cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-02")["ABC"]
    pd.testing.assert_frame_equal(first, second, check_dtype=False, check_like=True)
    assert len(calls) == 1
    engine.dispose()
