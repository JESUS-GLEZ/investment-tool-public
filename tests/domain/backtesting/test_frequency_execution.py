"""Weekly fills follow completed history and the first actual US session."""

from pathlib import Path

import pandas as pd
import pytest

from investment_tool.config import AppConfig, ModelConfig
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.data.assembly.model_data_factory import ModelDataFactory
from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.domain.market_time import normalize_bars
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)


def weekly_factory():
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.WEEKLY,
        0,
        input_data_path=Path("unused.pkl"),
        cfg=AppConfig(
            model=ModelConfig(
                historical_start_date="2024-01-01",
                historical_end_date="2024-02-04",
                backtest_start_date="2024-01-15",
                min_backtest_periods=2,
                initial_cash=1000,
            )
        ),
    )

    def frame(zone, opening, close):
        dates = pd.date_range("2024-01-01", periods=5, freq="W-MON", tz=zone)
        return normalize_bars(
            pd.DataFrame(
                {
                    "open": [opening] * 5,
                    "close": [close] * 5,
                    "high": [close] * 5,
                    "low": [opening] * 5,
                    "volume": [100] * 5,
                },
                index=dates,
            ),
            Frequency.WEEKLY,
            market_timezone=zone,
        )

    factory._input_data = {
        "stock_dict": {"AAPL": frame("America/New_York", 100, 110)},
        "crypto_dict": {"BTC/USD": frame("UTC", 10, 1100)},
        "execution_dict": {
            label: {
                "timestamp": opening,
                "crypto_prices": {"BTC/USD": 1000},
                "crypto_timestamps": {"BTC/USD": opening},
            }
            for label, opening in [
                ("2024-01-15", "2024-01-16T14:30:00Z"),  # Monday holiday
                ("2024-01-22", "2024-01-22T14:30:00Z"),
                ("2024-01-29", "2024-01-29T14:30:00Z"),
            ]
        },
    }
    factory._build_instance()
    factory._build_stocks()
    return factory


def test_weekly_mixed_fills_are_synchronized_and_forecasts_never_observe_current_bar():
    factory = weekly_factory()
    factory._build_periods()
    factory._build_portfolio()

    class HistoricalStrategy(BaseInvestmentStrategy):
        def run_strategy(self, progress_callback=None):
            current = self._instance_ref.current_trading_period
            for stock in self.stocks:
                assert (stock.data.index < current.trading_date).all()
                assert (stock.data.available_at <= current.decision_time).all()
            return pd.Series({"AAPL": 0.5, "BTC/USD": 0.5})

    model = factory.model_data
    strategy = HistoricalStrategy(model.instance, model.stocks)
    backtest = Backtester(model, strategy)
    backtest.run_backtest()
    assert len(backtest.results) == 3
    first = backtest.results[0]
    assert first["execution_time"] == "2024-01-16T14:30:00+00:00"
    assert first["total_value"] == pytest.approx(1100)
    assert first["execution_times"]["BTC/USD"] == first["execution_time"]
    assert first["period_start"] == "2024-01-15"
    assert first["period_end"] == "2024-01-21"
    assert (
        model.stocks[1].data.open.iloc[0] == 10
    )  # Original midnight feature untouched.
    assert model.portfolio.history[0]["trades"]["BTC/USD"] == 0.5


def test_weekly_missing_schedule_fails_before_any_backtest():
    factory = weekly_factory()
    del factory._input_data["execution_dict"]["2024-01-22"]
    with pytest.raises(ValueError, match="execution schedule for 2024-01-22"):
        factory._build_periods()


def test_closed_session_cannot_hide_real_stock_opening_data():
    factory = weekly_factory()
    factory._input_data["execution_dict"]["2024-01-15"] = {
        "market_closed": True,
        "crypto_prices": {},
    }
    with pytest.raises(ValueError, match="contradicts observed stock opening"):
        factory._build_periods()


def test_scheduled_trading_period_cannot_disappear_when_all_asset_bars_are_missing():
    factory = weekly_factory()
    for key in ("stock_dict", "crypto_dict"):
        for symbol, frame in factory._input_data[key].items():
            factory._input_data[key][symbol] = frame.drop(pd.Timestamp("2024-01-22"))
    factory._build_stocks()
    with pytest.raises(
        ValueError, match="Missing bar data for scheduled trading periods"
    ):
        factory._build_periods()


@pytest.mark.parametrize("timestamp", ["2024-01-15T04:59:00Z", "2024-01-22T00:00:00Z"])
def test_weekly_schedule_must_follow_cutoff_and_precede_period_end(timestamp):
    factory = weekly_factory()
    factory._input_data["execution_dict"]["2024-01-15"]["timestamp"] = timestamp
    with pytest.raises(ValueError, match="Execution must follow|Invalid execution"):
        factory._build_periods()


def test_weekly_crypto_cannot_use_a_stale_or_late_quote():
    factory = weekly_factory()
    factory._input_data["execution_dict"]["2024-01-15"]["crypto_timestamps"][
        "BTC/USD"
    ] = "2024-01-16T14:33:00Z"
    with pytest.raises(ValueError, match="within two minutes"):
        factory._build_periods()


def test_daily_mixed_backtest_skips_explicitly_closed_sessions_but_keeps_history():
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.DAILY,
        0,
        input_data_path=Path("unused.pkl"),
        cfg=AppConfig(
            model=ModelConfig(
                historical_start_date="2024-01-08",
                historical_end_date="2024-01-21",
                backtest_start_date="2024-01-12",
                min_backtest_periods=2,
                initial_cash=1000,
            )
        ),
    )

    def frame(dates, zone, opening, close):
        return normalize_bars(
            pd.DataFrame(
                {
                    "open": opening,
                    "close": close,
                    "high": close,
                    "low": opening,
                    "volume": 100,
                },
                index=pd.DatetimeIndex(dates).tz_localize(zone),
            ),
            Frequency.DAILY,
            market_timezone=zone,
        )

    dates = pd.date_range("2024-01-08", "2024-01-21", freq="D")
    closed = {"2024-01-13", "2024-01-14", "2024-01-15", "2024-01-20", "2024-01-21"}
    trading = [day for day in dates if day.date().isoformat() not in closed]
    factory._input_data = {
        "stock_dict": {"AAPL": frame(trading, "America/New_York", 100, 110)},
        "crypto_dict": {"BTC/USD": frame(dates, "UTC", 10, 1100)},
        "execution_dict": {
            day.date().isoformat(): {"market_closed": True, "crypto_prices": {}}
            if day.date().isoformat() in closed
            else {
                "timestamp": f"{day.date()}T14:30:00Z",
                "crypto_prices": {"BTC/USD": 1000},
            }
            for day in dates
        },
    }
    factory._build_instance()
    factory._build_stocks()
    factory._build_periods()
    factory._build_portfolio()

    class HistoricalStrategy(BaseInvestmentStrategy):
        def run_strategy(self, progress_callback=None):
            current = self._instance_ref.current_trading_period
            for stock in self.stocks:
                assert (stock.data.index < current.trading_date).all()
                assert (stock.data.available_at <= current.decision_time).all()
            return pd.Series({"AAPL": 0.5, "BTC/USD": 0.5})

    model = factory.model_data
    backtest = Backtester(model, HistoricalStrategy(model.instance, model.stocks))
    backtest.run_backtest()
    assert len(backtest.results) == 5
    assert all(row["period_start"] not in closed for row in backtest.results)
    crypto = next(stock for stock in model.stocks if stock.symbol == "BTC/USD")
    assert pd.Timestamp("2024-01-13") in crypto.data.index
    assert pd.Timestamp("2024-01-15") in crypto.data.index
    assert backtest.results[0]["total_value"] == pytest.approx(1100)
