"""Unit tests for InvestmentStrategyBuyAndHold."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency, StockFields
from investment_tool.strategies.orchestrators.investment_strategy_buy_and_hold import (
    InvestmentStrategyBuyAndHold,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_stock(symbol: str, n_periods: int = 24) -> Stock:
    """Create a Stock with *n_periods* months of data."""
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    close = 100.0 * np.cumprod(1 + np.full(n_periods, 0.01))
    data = pd.DataFrame(
        {
            StockFields.OPEN: close * 0.99,
            StockFields.HIGH: close * 1.02,
            StockFields.LOW: close * 0.98,
            StockFields.CLOSE: close,
            StockFields.VOLUME: np.full(n_periods, 1_000_000),
        },
        index=dates,
    )
    return Stock(symbol, data)


def _make_instance(trading_date: datetime | None = None) -> Instance:
    td = trading_date or datetime(2024, 1, 15)
    period = Period(trading_date=td, market_data=pd.DataFrame())
    return Instance(
        historical_start_date=date(2022, 1, 1),
        historical_end_date=date(2024, 1, 1),
        frequency=Frequency.MONTHLY,
        initial_trading_period=period,
        previous_trading_period=None,
        current_trading_period=period,
    )


SYMBOLS = ["SPY", "AAPL", "GOOG"]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBuyAndHoldInit:
    """Test constructor validation."""

    def test_creates_successfully(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
        )
        assert strategy._asset_symbol == "SPY"

    def test_default_asset_is_spy(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks
        )
        assert strategy._asset_symbol == "SPY"

    def test_raises_if_target_not_in_stocks(self) -> None:
        stocks = [_make_stock("AAPL"), _make_stock("GOOG")]
        with pytest.raises(ValueError, match="Target asset 'SPY' not found"):
            InvestmentStrategyBuyAndHold(
                instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
            )


class TestBuyAndHoldRunStrategy:
    """Test run_strategy() behaviour."""

    def test_returns_series_with_correct_index(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
        )
        weights = strategy.run_strategy()
        assert isinstance(weights, pd.Series)
        assert set(weights.index) == set(SYMBOLS)

    def test_target_has_weight_one(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
        )
        weights = strategy.run_strategy()
        assert weights["SPY"] == 1.0

    def test_non_target_assets_have_weight_zero(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
        )
        weights = strategy.run_strategy()
        assert weights["AAPL"] == 0.0
        assert weights["GOOG"] == 0.0

    def test_weights_sum_to_one(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
        )
        weights = strategy.run_strategy()
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_stores_optimal_weights_for_warm_start(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
        )
        assert not strategy.is_warm_start
        strategy.run_strategy()
        assert strategy.is_warm_start

    def test_custom_asset_symbol(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="GOOG"
        )
        weights = strategy.run_strategy()
        assert weights["GOOG"] == 1.0
        assert weights["SPY"] == 0.0
        assert weights["AAPL"] == 0.0

    def test_idempotent_across_calls(self) -> None:
        """Multiple calls produce the same weights."""
        stocks = [_make_stock(s) for s in SYMBOLS]
        strategy = InvestmentStrategyBuyAndHold(
            instance=_make_instance(), stocks=stocks, asset_symbol="SPY"
        )
        w1 = strategy.run_strategy()
        w2 = strategy.run_strategy()
        pd.testing.assert_series_equal(w1, w2)
