"""Unit tests for Portfolio entity."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import numpy as np
import pytest

from investment_tool.domain.model_data.entities.portfolio import Portfolio


def _create_mock_instance(prices: dict[str, float] | None = None) -> MagicMock:
    """Create a mock Instance for testing."""
    mock_instance = MagicMock()
    mock_current_period = MagicMock()
    mock_current_period.trading_date = datetime(2024, 1, 15)
    mock_previous_period = MagicMock()
    mock_previous_period.trading_date = datetime(2024, 1, 14)

    if prices:
        price_series = pd.Series(prices)
        mock_market_data = MagicMock()
        mock_xs_result = MagicMock()
        mock_xs_result.loc.__getitem__ = MagicMock(return_value=price_series)
        mock_market_data.xs.return_value = mock_xs_result
        mock_previous_period.market_data = mock_market_data

    mock_instance.current_trading_period = mock_current_period
    mock_instance.previous_trading_period = mock_previous_period
    return mock_instance


class TestPortfolioInitialization:
    """Tests for Portfolio initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic portfolio creation."""
        portfolio = Portfolio(_create_mock_instance(), initial_cash=100000.0)
        assert portfolio.cash == 100000.0
        assert portfolio.positions == {}
        assert portfolio.history == []

    def test_with_transaction_costs(self) -> None:
        """Test initialization with transaction costs."""
        portfolio = Portfolio(
            _create_mock_instance(), initial_cash=50000.0, transaction_cost_rate=0.001
        )
        assert portfolio.transaction_cost_rate == 0.001


class TestPortfolioValueCalculation:
    """Tests for portfolio value calculation."""

    def test_cash_only(self) -> None:
        """Test value with cash only."""
        portfolio = Portfolio(_create_mock_instance(), initial_cash=100000.0)
        prices = pd.Series({"AAPL": 150.0})
        assert portfolio.calculate_portfolio_value(prices) == 100000.0

    def test_with_positions(self) -> None:
        """Test value with positions."""
        portfolio = Portfolio(_create_mock_instance(), initial_cash=10000.0)
        portfolio.positions = {"AAPL": 100, "MSFT": 50}
        prices = pd.Series({"AAPL": 150.0, "MSFT": 300.0})
        expected = 10000.0 + (100 * 150.0) + (50 * 300.0)
        assert portfolio.calculate_portfolio_value(prices) == expected


class TestTradeCalculations:
    """Tests for trade calculation logic."""

    def test_trades_from_zero(self) -> None:
        """Test trades when starting with no positions."""
        portfolio = Portfolio(_create_mock_instance(), initial_cash=100000.0)
        current = pd.Series(dtype=float)
        target = pd.Series({"AAPL": 100.0, "MSFT": 50.0})
        trades = portfolio._calculate_trades(current, target)
        assert trades["AAPL"] == 100.0
        assert trades["MSFT"] == 50.0

    def test_trades_rebalance(self) -> None:
        """Test trades when rebalancing."""
        portfolio = Portfolio(_create_mock_instance(), initial_cash=100000.0)
        current = pd.Series({"AAPL": 100.0, "MSFT": 50.0})
        target = pd.Series({"AAPL": 80.0, "MSFT": 70.0})
        trades = portfolio._calculate_trades(current, target)
        assert trades["AAPL"] == -20.0
        assert trades["MSFT"] == 20.0


class TestTradeExecution:
    """Tests for trade execution."""

    def test_buy_trades(self) -> None:
        """Test executing buy trades."""
        portfolio = Portfolio(_create_mock_instance(), initial_cash=100000.0)
        trades = pd.Series({"AAPL": 100.0})
        prices = pd.Series({"AAPL": 150.0})
        portfolio._execute_trades(trades, prices)
        assert portfolio.cash == 100000.0 - (100 * 150.0)
        assert portfolio.positions["AAPL"] == 100.0

    def test_with_transaction_costs(self) -> None:
        """Test transaction costs are applied."""
        portfolio = Portfolio(
            _create_mock_instance(), initial_cash=100000.0, transaction_cost_rate=0.01
        )
        trades = pd.Series({"AAPL": 100.0})
        prices = pd.Series({"AAPL": 100.0})
        portfolio._execute_trades(trades, prices)
        trade_value = 10000
        cost = trade_value * 0.01
        assert portfolio.cash == 100000.0 - trade_value - cost


class TestPortfolioUpdate:
    """Tests for portfolio update."""

    def test_requires_series(self) -> None:
        """Test update raises TypeError for non-Series."""
        portfolio = Portfolio(
            _create_mock_instance(prices={"AAPL": 150.0}), initial_cash=100000.0
        )
        with pytest.raises(TypeError, match="pandas Series"):
            portfolio.update({"AAPL": 1.0})  # type: ignore

    def test_raises_without_prices(self) -> None:
        """Test raises when prices unavailable."""
        mock = MagicMock()
        mock.previous_trading_period = None
        portfolio = Portfolio(mock, initial_cash=100000.0)
        with pytest.raises(ValueError, match="prices"):
            portfolio.update(pd.Series({"AAPL": 1.0}))


class TestFundedTransactionCosts:
    """Actual trade quantities, fees and cash must reconcile together."""

    @pytest.mark.parametrize("rate", [0.0, 0.001, 0.01, 0.1])
    def test_initial_purchase_is_funded_and_proportional(self, rate):
        portfolio = Portfolio(_create_mock_instance(), 1000.0, rate)
        prices = pd.Series({"AAPL": 100.0, "MSFT": 50.0})
        portfolio.update(pd.Series({"AAPL": 0.6, "MSFT": 0.4}), execution_prices=prices)
        assert portfolio.cash >= 0
        assert portfolio.positions == pytest.approx(
            {"AAPL": 6 / (1 + rate), "MSFT": 8 / (1 + rate)}
        )
        history = portfolio.history[-1]
        expected_fee = 1000 * rate / (1 + rate)
        assert history["transaction_costs"] == pytest.approx(expected_fee)
        assert history["trades"] == portfolio.positions
        assert history["post_rebalance_value"] == pytest.approx(1000 - expected_fee)

    def test_rotation_pays_both_sale_and_purchase_costs(self):
        portfolio = Portfolio(_create_mock_instance(), 0.0, 0.001)
        portfolio.positions = {"OLD": 10.0}
        prices = pd.Series({"OLD": 100.0, "NEW": 250.0})
        portfolio.update(pd.Series({"NEW": 1.0}), execution_prices=prices)
        bought = 1000 * (1 - 0.001) / (1 + 0.001)
        fee = (1000 + bought) * 0.001
        assert portfolio.positions == pytest.approx({"NEW": bought / 250})
        assert portfolio.history[-1]["trades"] == pytest.approx(
            {"OLD": -10, "NEW": bought / 250}
        )
        assert portfolio.cash >= 0
        assert portfolio.history[-1]["transaction_costs"] == pytest.approx(fee)
        assert portfolio.calculate_portfolio_value(prices) == pytest.approx(1000 - fee)

    def test_partial_rebalance_keeps_sales_and_scales_only_buys(self):
        portfolio = Portfolio(_create_mock_instance(), 100.0, 0.001)
        portfolio.positions = {"AAPL": 9.0}
        prices = pd.Series({"AAPL": 100.0, "MSFT": 100.0})
        portfolio.update(pd.Series({"AAPL": 0.6, "MSFT": 0.4}), execution_prices=prices)
        actual_bought = (100 + 300 * (1 - 0.001)) / (1 + 0.001)
        assert portfolio.history[-1]["trades"]["AAPL"] == -3
        assert portfolio.positions == pytest.approx(
            {"AAPL": 6, "MSFT": actual_bought / 100}
        )
        assert portfolio.cash >= 0
        assert portfolio.history[-1]["pre_rebalance_value"] - portfolio.history[-1][
            "post_rebalance_value"
        ] == pytest.approx(portfolio.history[-1]["transaction_costs"])

    def test_sufficient_cash_preserves_purchase_and_liquidation_has_no_buy_fee(self):
        portfolio = Portfolio(_create_mock_instance(), 1000.0, 0.001)
        prices = pd.Series({"AAPL": 100.0})
        portfolio.update(pd.Series({"AAPL": 0.5}), execution_prices=prices)
        assert portfolio.positions == {"AAPL": 5}
        assert portfolio.cash == 499.5
        portfolio.update(pd.Series({"AAPL": 0.0}), execution_prices=prices)
        assert portfolio.positions == {}
        assert portfolio.cash == 999
        assert portfolio.history[-1]["transaction_costs"] == 0.5

    def test_no_trade_has_no_fee(self):
        portfolio = Portfolio(_create_mock_instance(), 0.0, 0.01)
        portfolio.positions = {"AAPL": 10.0}
        portfolio.update(
            pd.Series({"AAPL": 1.0}), execution_prices=pd.Series({"AAPL": 100.0})
        )
        assert portfolio.cash == 0
        assert portfolio.positions == {"AAPL": 10}
        assert portfolio.history[-1]["transaction_costs"] == 0

    @pytest.mark.parametrize("rate", [np.nan, np.inf, -np.inf, -0.001])
    def test_invalid_cost_rates_are_rejected(self, rate):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            Portfolio(_create_mock_instance(), 1000, rate)

    def test_unfundable_selling_fee_does_not_mutate_portfolio(self):
        portfolio = Portfolio(_create_mock_instance(), 0.0, 2.0)
        portfolio.positions = {"AAPL": 10.0}
        with pytest.raises(ValueError, match="pay transaction costs on sales"):
            portfolio.update(
                pd.Series({"AAPL": 0.0}), execution_prices=pd.Series({"AAPL": 100.0})
            )
        assert portfolio.positions == {"AAPL": 10}
        assert portfolio.cash == 0
        assert portfolio.history == []


def test_cash_settlement_waits_for_publication_and_is_credited_once() -> None:
    from investment_tool.domain.model_data.entities.cash_settlement import (
        CashSettlement,
    )

    event = CashSettlement(
        "GBT",
        datetime.fromisoformat("2022-10-06T04:00:00+00:00"),
        datetime.fromisoformat("2022-10-07T04:00:00+00:00"),
        68.5,
        "reviewed source",
    )
    portfolio = Portfolio(
        _create_mock_instance(), initial_cash=10, transaction_cost_rate=0.1
    )
    portfolio.positions = {"GBT": 2, "OTHER": 3}
    portfolio.settle_cash_actions(
        [event], datetime.fromisoformat("2022-10-06T04:00:00+00:00")
    )
    assert portfolio.cash == 10 and portfolio.positions["GBT"] == 2
    portfolio.settle_cash_actions([event], event.published_at)
    portfolio.settle_cash_actions([event], event.published_at)
    assert portfolio.cash == 147
    assert portfolio.positions == {"OTHER": 3}
    assert len(portfolio.corporate_actions) == 1
    assert portfolio.corporate_actions[0]["proceeds"] == 137
    # Ordinary transaction costs do not apply to a contractual redemption.
    with pytest.raises(ValueError, match="timezone-aware"):
        portfolio.settle_cash_actions([event], datetime(2022, 11, 1))
