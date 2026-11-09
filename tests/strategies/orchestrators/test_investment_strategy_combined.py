"""Unit tests for InvestmentStrategyCombined.

Heavy dependencies (sentiment analysis, Ridge VARX, Platypus optimizer)
are mocked so tests run in milliseconds and focus on orchestration logic.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency, StockFields
from investment_tool.strategies.orchestrators.investment_strategy_combined import (
    InvestmentStrategyCombined,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_stock(symbol: str, n_periods: int = 24, trend: float = 0.01) -> Stock:
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


def _make_news(symbol: str, news_date: date) -> tuple[tuple[str, date], News]:
    news = News(
        symbol=symbol,
        period_start_date=news_date,
        period_end_date=news_date,
        data=[
            {
                "headline": f"{symbol} headline",
                "summary": f"{symbol} summary",
                "published_at": news_date.isoformat() + "T12:00:00Z",
            }
        ],
    )
    return (symbol, news_date), news


SYMBOLS = ["AAPL", "GOOG", "MSFT"]

# Shared mock return values
_MOCK_SENTIMENT = {s: {date(2023, 12, 1): 0.5} for s in SYMBOLS}
_MOCK_EXPECTED_RETURNS = {s: 0.01 for s in SYMBOLS}
_MOCK_WEIGHTS = pd.Series([0.5, 0.3, 0.2], index=SYMBOLS)


# ── Init Tests ───────────────────────────────────────────────────────────────


class TestCombinedInit:
    """Test construction and parameter storage."""

    def test_default_params(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        key, news = _make_news("AAPL", date(2023, 5, 1))
        strategy = InvestmentStrategyCombined(
            instance=instance, stocks=stocks, news={key: news}
        )
        assert strategy.min_assets == 2
        assert strategy.max_assets == 5
        assert strategy.risk_aversion == 0.1
        assert strategy.output_directory is None

    def test_custom_params(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        out_dir = Path("/tmp/test_output")
        strategy = InvestmentStrategyCombined(
            instance=instance,
            stocks=stocks,
            news=None,
            min_assets=3,
            max_assets=8,
            risk_aversion=0.3,
            output_directory=out_dir,
        )
        assert strategy.min_assets == 3
        assert strategy.max_assets == 8
        assert strategy.risk_aversion == 0.3
        assert strategy.output_directory == out_dir

    def test_legacy_num_assets_to_hold(self) -> None:
        """Test backward-compat: num_assets_to_hold overrides max_assets."""
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyCombined(
            instance=instance,
            stocks=stocks,
            news=None,
            num_assets_to_hold=7,
        )
        assert strategy.max_assets == 7


# ── Pipeline Step Tests ──────────────────────────────────────────────────────


class TestCombinedPipelineSteps:
    """Test each pipeline step in isolation with mocks."""

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.compute_sentiment_scores"
    )
    def test_compute_sentiment_calls_transformer(
        self, mock_sentiment: MagicMock
    ) -> None:
        mock_sentiment.return_value = _MOCK_SENTIMENT

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        key, news = _make_news("AAPL", date(2023, 5, 1))
        strategy = InvestmentStrategyCombined(
            instance=instance, stocks=stocks, news={key: news}
        )

        result = strategy._compute_sentiment()

        mock_sentiment.assert_called_once_with(strategy.news)
        assert result == _MOCK_SENTIMENT

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.RidgeVARXPredictionStrategy"
    )
    def test_predict_returns_calls_ridge_varx(self, mock_ridge: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_ridge.return_value = mock_instance

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyCombined(
            instance=instance, stocks=stocks, news=None
        )

        result = strategy._predict_returns(_MOCK_SENTIMENT)

        assert result == _MOCK_EXPECTED_RETURNS

    def test_forecast_is_unchanged_by_current_or_future_prices(self) -> None:
        """May's forecast uses completed April history, even with future bars."""
        stocks = [_make_stock(s, n_periods=36) for s in SYMBOLS]
        cutoff = datetime(2024, 5, 1)
        baseline = InvestmentStrategyCombined(
            instance=_make_instance(cutoff), stocks=stocks, news=None
        )
        altered_stocks = []
        for stock in stocks:
            data = stock.data.copy()
            data.loc[data.index >= cutoff, StockFields.CLOSE] *= 100
            altered_stocks.append(Stock(stock.symbol, data))
        altered = InvestmentStrategyCombined(
            instance=_make_instance(cutoff), stocks=altered_stocks, news=None
        )
        # Includes future sentiment as well: it cannot enter April features.
        sentiment = {
            s: {date(2024, 4, 1): 0.2, date(2024, 5, 1): -0.9} for s in SYMBOLS
        }
        historical_sentiment = {s: {date(2024, 4, 1): 0.2} for s in SYMBOLS}

        result = baseline._predict_returns(historical_sentiment)
        assert altered._predict_returns(sentiment) == pytest.approx(result)
        assert all(
            stock.data.index[-1] == pd.Timestamp("2024-04-01")
            for stock in altered.stocks
        )

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.RidgeVARXPredictionStrategy"
    )
    def test_predict_returns_raises_on_empty(self, mock_ridge: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.predict_returns.return_value = {}
        mock_ridge.return_value = mock_instance

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyCombined(
            instance=instance, stocks=stocks, news=None
        )

        with pytest.raises(ValueError, match="No expected returns"):
            strategy._predict_returns(_MOCK_SENTIMENT)

    def test_build_optimizer_returns_platypus_strategy(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyCombined(
            instance=instance, stocks=stocks, news=None
        )

        cov = strategy.build_covariance_matrix()
        optimizer = strategy._build_optimizer(_MOCK_EXPECTED_RETURNS, cov)

        from investment_tool.strategies.optimization_strategies.platypus_optimization import (
            PlatypusOptimizationStrategy,
        )

        assert isinstance(optimizer, PlatypusOptimizationStrategy)
        assert optimizer.risk_aversion == 0.1
        assert optimizer.population_size == 300
        assert optimizer.n_generations == 6_000

    def test_build_optimizer_uses_fresh_current_window_seeds(self) -> None:
        """After storing weights with front, _build_optimizer should pass them."""
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyCombined(
            instance=instance, stocks=stocks, news=None
        )

        # Simulate warm-start with a front of 2 solutions
        front_weights = [
            np.array(_MOCK_WEIGHTS.values),
            np.array([0.4, 0.3, 0.3]),
        ]
        strategy.store_optimal_weights(_MOCK_WEIGHTS, front_weights=front_weights)

        cov = strategy.build_covariance_matrix()
        optimizer = strategy._build_optimizer(_MOCK_EXPECTED_RETURNS, cov)

        assert optimizer.seed_weights is None
        assert optimizer.seed_max_subsets == 360_000
        assert optimizer.search_seconds is None


# ── run_strategy Orchestration Tests ─────────────────────────────────────────


class TestCombinedRunStrategy:
    """Test the full orchestration with all heavy dependencies mocked."""

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.compute_sentiment_scores"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.RidgeVARXPredictionStrategy"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.MultiGreedyOptimizationStrategy"
    )
    def test_full_pipeline_returns_weights(
        self,
        mock_optimizer_cls: MagicMock,
        mock_ridge_cls: MagicMock,
        mock_sentiment: MagicMock,
    ) -> None:
        # Wire mocks
        mock_sentiment.return_value = _MOCK_SENTIMENT
        mock_ridge_inst = MagicMock()
        mock_ridge_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_ridge_cls.return_value = mock_ridge_inst
        mock_opt_inst = MagicMock()
        mock_opt_result = MagicMock()
        mock_opt_result.front = [MagicMock()]
        mock_opt_inst.optimize.return_value = mock_opt_result
        mock_optimizer_cls.return_value = mock_opt_inst

        mock_selector = MagicMock()
        mock_selector.select.return_value = _MOCK_WEIGHTS

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        key, news = _make_news("AAPL", date(2023, 5, 1))
        strategy = InvestmentStrategyCombined(
            instance=instance,
            stocks=stocks,
            news={key: news},
            selection_strategy=mock_selector,
        )

        result = strategy.run_strategy()

        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, _MOCK_WEIGHTS)
        mock_selector.select.assert_called_once()

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.compute_sentiment_scores"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.RidgeVARXPredictionStrategy"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.MultiGreedyOptimizationStrategy"
    )
    def test_run_strategy_stores_weights_for_warm_start(
        self,
        mock_optimizer_cls: MagicMock,
        mock_ridge_cls: MagicMock,
        mock_sentiment: MagicMock,
    ) -> None:
        mock_sentiment.return_value = _MOCK_SENTIMENT
        mock_ridge_inst = MagicMock()
        mock_ridge_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_ridge_cls.return_value = mock_ridge_inst
        mock_opt_inst = MagicMock()
        mock_opt_result = MagicMock()
        mock_opt_result.front = [MagicMock()]
        mock_opt_inst.optimize.return_value = mock_opt_result
        mock_optimizer_cls.return_value = mock_opt_inst

        mock_selector = MagicMock()
        mock_selector.select.return_value = _MOCK_WEIGHTS

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyCombined(
            instance=instance,
            stocks=stocks,
            news=None,
            selection_strategy=mock_selector,
        )

        assert strategy.is_warm_start is False
        strategy.run_strategy()
        assert strategy.is_warm_start is True

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.compute_sentiment_scores"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.RidgeVARXPredictionStrategy"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_combined.MultiGreedyOptimizationStrategy"
    )
    def test_run_strategy_raises_on_empty_weights(
        self,
        mock_optimizer_cls: MagicMock,
        mock_ridge_cls: MagicMock,
        mock_sentiment: MagicMock,
    ) -> None:
        mock_sentiment.return_value = _MOCK_SENTIMENT
        mock_ridge_inst = MagicMock()
        mock_ridge_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_ridge_cls.return_value = mock_ridge_inst
        mock_opt_inst = MagicMock()
        mock_opt_result = MagicMock()
        mock_opt_result.front = [MagicMock()]
        mock_opt_inst.optimize.return_value = mock_opt_result
        mock_optimizer_cls.return_value = mock_opt_inst

        mock_selector = MagicMock()
        mock_selector.select.return_value = pd.Series(dtype=float)

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyCombined(
            instance=instance,
            stocks=stocks,
            news=None,
            selection_strategy=mock_selector,
        )

        with pytest.raises(RuntimeError, match="empty weights"):
            strategy.run_strategy()
