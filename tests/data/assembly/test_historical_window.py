"""Local and remote sources must use the same inclusive historical window."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from investment_tool.config import AppConfig, ModelConfig, PathsConfig
from investment_tool.domain.market_time import normalize_bars
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.data.assembly.model_data_factory import ModelDataFactory
from investment_tool.data.repositories.base_repository import DataResult
from investment_tool.data.repositories.repository_factory import RepositoryFactory
from investment_tool.data.validation import ValidationLevel


@pytest.mark.parametrize("source", [DataSource.LOCAL, DataSource.ALPACA])
def test_stocks_are_filtered_before_indicators(source):
    factory = ModelDataFactory(
        source,
        ExecutionMode.BACKTESTING,
        Frequency.DAILY,
        0,
        input_data_path=Path("unused.pkl") if source == DataSource.LOCAL else None,
    )
    factory.model_data.instance = SimpleNamespace(
        historical_start_date="2024-01-02", historical_end_date="2024-01-03"
    )
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0],
            "high": [1.0, 2.0, 3.0, 4.0],
            "low": [1.0, 2.0, 3.0, 4.0],
            "close": [1.0, 2.0, 3.0, 4.0],
            "volume": [100.0] * 4,
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    canonical = normalize_bars(
        frame.set_axis(frame.index.tz_localize("UTC")),
        Frequency.DAILY,
        market_timezone="UTC",
    )
    factory._input_data = {
        "stock_dict": {"ABC": canonical, "OUTSIDE": canonical.iloc[:1]}
    }
    factory._build_stocks()
    assert [stock.symbol for stock in factory.model_data.stocks] == ["ABC"]
    assert list(factory.model_data.stocks[0].data.index) == list(
        pd.date_range("2024-01-02", periods=2)
    )
    assert len(frame) == 4  # The loaded file is not modified.


def test_end_date_excludes_the_month_labelled_on_that_day():
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=Path("unused.pkl"),
    )
    factory.model_data.instance = SimpleNamespace(
        historical_start_date="2026-06-01", historical_end_date="2026-08-01"
    )
    dates = pd.date_range("2026-06-01", periods=3, freq="MS", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [100.0] * 3,
        },
        index=dates,
    )
    frame = normalize_bars(frame, Frequency.MONTHLY, market_timezone="America/New_York")
    factory._input_data = {"stock_dict": {"ABC": frame}}

    factory._build_stocks()

    assert list(factory.model_data.stocks[0].data.index) == list(
        pd.date_range("2026-06-01", periods=2, freq="MS")
    )


def test_monthly_stock_and_crypto_timestamps_form_one_period():
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

    def bars(dates):
        return normalize_bars(
            pd.DataFrame(
                {
                    "open": [1.0, 2.0, 3.0],
                    "high": [1.0, 2.0, 3.0],
                    "low": [1.0, 2.0, 3.0],
                    "close": [1.0, 2.0, 3.0],
                    "volume": [100.0] * 3,
                },
                index=pd.to_datetime(dates, utc=True),
            ),
            Frequency.MONTHLY,
            market_timezone="UTC",
        )

    factory._input_data = {
        "stock_dict": {"AAPL": bars(["2024-01-01", "2024-02-01", "2024-03-01"])},
        "crypto_dict": {"BTC/USD": bars(["2024-01-01", "2024-02-01", "2024-03-01"])},
    }
    factory._input_data["execution_dict"] = {
        f"2024-{month:02d}-01": {
            "timestamp": f"2024-{month:02d}-02T14:30:00+00:00",
            "crypto_prices": {"BTC/USD": float(month)},
        }
        for month in (1, 2, 3)
    }
    factory._build_stocks()
    factory._build_periods()

    expected = list(pd.date_range("2024-01-01", periods=3, freq="MS"))
    assert [period.trading_date for period in factory.model_data.periods] == expected
    assert all(
        list(stock.data.index) == expected for stock in factory.model_data.stocks
    )


def test_crypto_pair_is_built_as_portfolio_asset():
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=Path("unused.pkl"),
    )
    factory.model_data.instance = SimpleNamespace(
        historical_start_date="2024-01-01", historical_end_date="2024-03-31"
    )
    prices = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.0, 11.0],
            "low": [10.0, 11.0],
            "close": [10.0, 11.0],
            "volume": [1.0, 1.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-02-01"]),
    )
    prices = normalize_bars(
        prices.set_axis(prices.index.tz_localize("UTC")),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )
    factory._input_data = {"stock_dict": {}, "crypto_dict": {"BTC/USD": prices}}

    factory._build_stocks()

    assert [asset.symbol for asset in factory.model_data.stocks] == ["BTC/USD"]


def test_saved_input_filename_uses_run_dates(tmp_path, monkeypatch):
    cfg = AppConfig(
        model=ModelConfig(
            historical_start_date="2016-01-01",
            historical_end_date="2026-08-01",
        ),
        paths=PathsConfig(input_data_dir=str(tmp_path)),
    )
    factory = ModelDataFactory(
        DataSource.ALPACA,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        save_input_data=True,
        validation_level=ValidationLevel.SKIP,
        cfg=cfg,
    )
    data = DataResult(
        stock_dict={
            "AAPL": normalize_bars(
                pd.DataFrame(
                    {"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC")
                ),
                Frequency.MONTHLY,
                market_timezone="UTC",
            )
        },
        metadata={"time_contract_version": 1},
    )
    repository = SimpleNamespace(
        source_name="Fake",
        fetch_all=lambda **kwargs: data,
        fetch_execution_data=lambda *args: {},
        fetch_news_data=lambda *args: {},
    )
    monkeypatch.setattr(RepositoryFactory, "create", lambda **kwargs: repository)

    factory._load_input_data()
    factory._load_news_and_save_input()

    saved = list(tmp_path.glob("*.pkl"))
    assert len(saved) == 1
    assert "2016-01-01_to_2026-08-01_1stocks" in saved[0].name


@pytest.mark.parametrize("start,expected", [("2020-01-01", 14), ("2019-12-01", 13)])
def test_explicit_backtest_start_keeps_warmup_and_ignores_percentage(start, expected):
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=Path("unused.pkl"),
        cfg=AppConfig(
            model=ModelConfig(
                backtest_start_date=start,
                min_backtest_periods=12,
                start_index_percentage_backtesting=0.9,
            )
        ),
    )
    periods = [
        SimpleNamespace(trading_date=d.to_pydatetime())
        for d in pd.date_range("2018-11-01", periods=26, freq="MS")
    ]
    factory.model_data.periods = periods
    factory.model_data.instance = SimpleNamespace()
    factory._set_trading_periods()
    assert factory.model_data.instance.current_trading_period is periods[expected]
    assert factory.model_data.instance.previous_trading_period is periods[expected - 1]
    assert len(factory.model_data.periods) == 26


@pytest.mark.parametrize(
    "start,message", [("2020-01-02", "period boundary"), ("2018-12-01", "warm-up")]
)
def test_explicit_backtest_start_rejects_missing_boundary_or_short_warmup(
    start, message
):
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=Path("unused.pkl"),
        cfg=AppConfig(
            model=ModelConfig(backtest_start_date=start, min_backtest_periods=12)
        ),
    )
    factory.model_data.periods = [
        SimpleNamespace(trading_date=d.to_pydatetime())
        for d in pd.date_range("2018-11-01", periods=26, freq="MS")
    ]
    factory.model_data.instance = SimpleNamespace()
    with pytest.raises(ValueError, match=message):
        factory._set_trading_periods()
