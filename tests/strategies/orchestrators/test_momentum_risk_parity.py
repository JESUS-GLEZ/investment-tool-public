"""Unit tests for InvestmentStrategyMomentumRiskParity."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency, StockFields
from investment_tool.strategies.orchestrators.investment_strategy_momentum_risk_parity import (
    InvestmentStrategyMomentumRiskParity,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_stock(symbol: str, n_periods: int = 24, trend: float = 0.01) -> Stock:
    """Create a Stock with *n_periods* months of trending close prices."""
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    close = 100.0 * np.cumprod(1 + np.full(n_periods, trend))
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
    """Create a minimal Instance with a current trading period."""
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


# ── Tests ───────────────────────────────────────────────────────────────────


class TestMomentumRiskParityInit:
    """Test strategy construction and parameter storage."""

    def test_basic_init(self) -> None:
        stocks = [_make_stock("AAPL"), _make_stock("GOOG")]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=stocks
        )
        assert strategy.top_n == 10
        assert strategy.lookback == 6
        assert strategy.vol_window == 6
        assert strategy.trend_window == 10

    def test_custom_params(self) -> None:
        stocks = [_make_stock("AAPL")]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance,
            stocks=stocks,
            top_n=3,
            lookback=3,
            vol_window=12,
            trend_window=6,
            min_weight=0.05,
        )
        assert strategy.top_n == 3
        assert strategy.lookback == 3
        assert strategy.vol_window == 12
        assert strategy.trend_window == 6
        assert strategy.min_weight == 0.05


class TestMomentumRiskParityRunStrategy:
    """Test the full run_strategy pipeline."""

    def test_weights_sum_to_one(self) -> None:
        stocks = [
            _make_stock(s, trend=0.01 * (i + 1))
            for i, s in enumerate(["AAPL", "GOOG", "MSFT", "AMZN", "META"])
        ]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=stocks, top_n=3
        )
        weights = strategy.run_strategy()
        assert abs(weights.sum() - 1.0) < 1e-9

    def test_no_negative_weights(self) -> None:
        stocks = [
            _make_stock(s, trend=0.005 * (i + 1))
            for i, s in enumerate(["A", "B", "C", "D", "E"])
        ]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=stocks, top_n=3
        )
        weights = strategy.run_strategy()
        assert (weights >= 0).all()

    def test_respects_top_n(self) -> None:
        stocks = [
            _make_stock(s, trend=0.01 * (i + 1))
            for i, s in enumerate(["A", "B", "C", "D", "E", "F", "G", "H"])
        ]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=stocks, top_n=3
        )
        weights = strategy.run_strategy()
        active = (weights > 0).sum()
        assert active <= 3

    def test_returns_series_indexed_by_symbol(self) -> None:
        stocks = [_make_stock("AAPL"), _make_stock("GOOG")]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=stocks, top_n=5
        )
        weights = strategy.run_strategy()
        assert isinstance(weights, pd.Series)
        # All stocks should appear in the index (even if weight=0)
        for s in ["AAPL", "GOOG"]:
            assert s in weights.index

    def test_prefers_higher_momentum(self) -> None:
        """The stock with higher trend should get non-zero weight."""
        slow = _make_stock("SLOW", trend=0.001)
        fast = _make_stock("FAST", trend=0.05)
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=[slow, fast], top_n=1
        )
        weights = strategy.run_strategy()
        assert weights["FAST"] > weights["SLOW"]

    def test_min_weight_filter(self) -> None:
        """Weights below min_weight should be zeroed out."""
        stocks = [
            _make_stock(s, trend=0.01 * (i + 1))
            for i, s in enumerate(["A", "B", "C", "D", "E"])
        ]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=stocks, top_n=5, min_weight=0.10
        )
        weights = strategy.run_strategy()
        for w in weights:
            assert w == 0.0 or w >= 0.10

    def test_single_stock(self) -> None:
        """With one stock, it should get 100% weight."""
        stock = _make_stock("ONLY", trend=0.02)
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=[stock], top_n=5
        )
        weights = strategy.run_strategy()
        assert abs(weights["ONLY"] - 1.0) < 1e-9


class TestMomentumRiskParityEdgeCases:
    """Test edge cases and robustness."""

    def test_raises_on_insufficient_history(self) -> None:
        """Very short stocks should cause RuntimeError."""
        dates = pd.date_range("2024-01-01", periods=2, freq="MS")
        data = pd.DataFrame(
            {
                StockFields.OPEN: [100.0, 101.0],
                StockFields.HIGH: [102.0, 103.0],
                StockFields.LOW: [98.0, 99.0],
                StockFields.CLOSE: [100.0, 101.0],
                StockFields.VOLUME: [1e6, 1e6],
            },
            index=dates,
        )
        stock = Stock("SHORT", data)
        instance = _make_instance(datetime(2024, 3, 15))
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=[stock], lookback=6
        )
        with pytest.raises(RuntimeError, match="sufficient history"):
            strategy.run_strategy()

    def test_refresh_data(self) -> None:
        """After refresh_data, strategy should use updated period."""
        stocks = [_make_stock("AAPL")]
        instance = _make_instance()
        strategy = InvestmentStrategyMomentumRiskParity(
            instance=instance, stocks=stocks
        )
        # Advance the instance
        new_period = Period(
            trading_date=datetime(2024, 2, 15), market_data=pd.DataFrame()
        )
        instance.current_trading_period = new_period
        strategy.refresh_data()
        # Should not raise
        weights = strategy.run_strategy()
        assert abs(weights.sum() - 1.0) < 1e-9
