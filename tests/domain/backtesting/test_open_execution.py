"""Real portfolio accounting at the next period's open."""

from datetime import date, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency
from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.exceptions import BacktestingError
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)


def model(opens=None, closes=None):
    dt = datetime(2024, 6, 1)
    opens = {"AAPL": 125.0} if opens is None else opens
    closes = {"AAPL": 150.0} if closes is None else closes
    row = {
        **{("open", k): v for k, v in opens.items()},
        **{("close", k): v for k, v in closes.items()},
    }
    frame = pd.DataFrame([row], index=[dt])
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    previous = Period(datetime(2024, 5, 1), pd.DataFrame())
    current = Period(dt, frame)
    data = ModelData()
    data.periods = [previous, current]
    data.instance = Instance(
        date(2024, 5, 1),
        date(2024, 6, 30),
        Frequency.MONTHLY,
        current,
        previous,
        current,
    )
    data.portfolio = Portfolio(data.instance, 1000.0)
    return data


@pytest.mark.parametrize("price", [None, 0, -1, np.nan, np.inf])
@pytest.mark.parametrize("held", [False, True])
def test_bad_open_rejected_without_portfolio_mutation(price, held):
    data = model({} if price is None else {"AAPL": price})
    if held:
        data.portfolio.positions = {"AAPL": 2.0}
    positions = data.portfolio.positions.copy()
    strategy = MagicMock()
    strategy.run_strategy.return_value = pd.Series({"AAPL": 1.0})
    backtester = Backtester(data, strategy)
    with pytest.raises(BacktestingError, match="execution prices"):
        backtester.run_backtest()
    assert data.portfolio.positions == positions
    assert data.portfolio.cash == 1000
    assert not data.portfolio.history and not backtester.results
    assert data.instance.current_trading_period == data.periods[1]


def test_liquidates_asset_removed_from_target_universe_at_its_open():
    data = model({"AAPL": 125, "MSFT": 50})
    data.portfolio.cash = 0
    data.portfolio.positions = {"MSFT": 20}
    strategy = MagicMock()
    strategy.run_strategy.return_value = pd.Series({"AAPL": 1.0})
    backtester = Backtester(data, strategy)
    backtester.run_backtest()
    assert data.portfolio.positions == {"AAPL": 8}
    assert data.portfolio.cash == 0
    assert data.portfolio.history[0]["trades"] == {"AAPL": 8, "MSFT": -20}
    assert backtester.results[0]["total_value"] == 1200


def test_backtest_return_includes_funded_initial_purchase_fee():
    data = model({"AAPL": 100}, {"AAPL": 110})
    data.portfolio.transaction_cost_rate = 0.01
    strategy = MagicMock()
    strategy.run_strategy.return_value = pd.Series({"AAPL": 1.0})
    tester = Backtester(data, strategy)
    tester.run_backtest()
    assert data.portfolio.cash >= 0
    assert data.portfolio.positions["AAPL"] == pytest.approx(1000 / 1.01 / 100)
    assert data.portfolio.history[0]["transaction_costs"] == pytest.approx(
        1000 * 0.01 / 1.01
    )
    assert tester.results[0]["total_value"] == pytest.approx(1000 / 1.01 * 1.1)
    assert tester.results[0]["return"] == pytest.approx(1.1 / 1.01 - 1)


def test_crypto_uses_stock_session_quote_instead_of_midnight_open():
    data = model({"AAPL": 125, "BTC/USD": 10}, {"AAPL": 150, "BTC/USD": 60})
    data.crypto_symbols = {"BTC/USD"}
    current = data.periods[1]
    current.execution_time = pd.Timestamp("2024-06-03T13:30:00Z").to_pydatetime()
    current.crypto_execution_prices = {"BTC/USD": 50}
    strategy = MagicMock()
    strategy.run_strategy.return_value = pd.Series({"AAPL": 0.5, "BTC/USD": 0.5})
    backtester = Backtester(data, strategy)
    backtester.run_backtest()
    assert data.portfolio.positions == {"AAPL": 4, "BTC/USD": 10}
    assert backtester.results[0]["total_value"] == 1200
    assert backtester.results[0]["execution_time"] == "2024-06-03T13:30:00+00:00"


def test_missing_synchronized_crypto_quote_does_not_fall_back():
    data = model({"BTC/USD": 10}, {"BTC/USD": 60})
    data.crypto_symbols = {"BTC/USD"}
    strategy = MagicMock()
    strategy.run_strategy.return_value = pd.Series({"BTC/USD": 1.0})
    with pytest.raises(BacktestingError, match="BTC/USD"):
        Backtester(data, strategy).run_backtest()


def test_current_month_prices_are_fills_and_marks_but_not_signal_inputs():
    class HistoricalStrategy(BaseInvestmentStrategy):
        def run_strategy(self, progress_callback=None):
            assert self.stocks[0].data.index.max() == pd.Timestamp("2024-05-01")
            assert self.stocks[0].data["close"].iloc[-1] == 100
            return pd.Series({"AAPL": 1.0})

    data = model()
    history = pd.DataFrame(
        {"close": [90, 100, 999999], "open": [80, 90, 999999]},
        index=pd.date_range("2024-04-01", periods=3, freq="MS"),
    )
    strategy = HistoricalStrategy(data.instance, [Stock("AAPL", history)])
    backtester = Backtester(data, strategy)
    backtester.run_backtest()
    assert data.portfolio.positions == {"AAPL": 8}
    assert backtester.results[0]["total_value"] == 1200


def test_price_from_another_period_cannot_be_used_as_a_fill():
    data = model()
    period = data.periods[1]
    period.market_data.index = [datetime(2024, 5, 1)]
    assert Backtester._extract_period_prices(period) == {}


def test_results_and_trade_history_report_each_assets_actual_fill_time():
    data = model({"AAPL": 125, "BTC/USD": 10}, {"AAPL": 150, "BTC/USD": 60})
    data.crypto_symbols = {"BTC/USD"}
    period = data.periods[1]
    period.execution_time = pd.Timestamp("2024-06-03T13:30:00Z").to_pydatetime()
    period.crypto_execution_prices = {"BTC/USD": 50}
    period.crypto_execution_times = {
        "BTC/USD": pd.Timestamp("2024-06-03T13:31:00Z").to_pydatetime()
    }
    strategy = MagicMock()
    strategy.run_strategy.return_value = pd.Series({"AAPL": 0.5, "BTC/USD": 0.5})
    backtester = Backtester(data, strategy)
    backtester.run_backtest()
    expected = {
        "AAPL": "2024-06-03T13:30:00+00:00",
        "BTC/USD": "2024-06-03T13:31:00+00:00",
    }
    assert backtester.results[0]["execution_times"] == expected
    assert data.portfolio.history[0]["execution_times"] == expected
    assert data.portfolio.positions == {"AAPL": 4, "BTC/USD": 10}
    assert backtester.results[0]["total_value"] == 1200
