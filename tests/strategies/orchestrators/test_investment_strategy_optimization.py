"""Unit tests for InvestmentStrategyOptimization.

Heavy dependencies (Platypus optimizer) are mocked so tests run fast
and focus on orchestration logic, error mapping, and warm-start wiring.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency, StockFields
from investment_tool.strategies.orchestrators.investment_strategy_optimization import (
    InvestmentStrategyOptimization,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.exceptions import OptimizationError, PredictionError
from investment_tool.strategies.optimization_strategies.seeded_optimization import (
    MultiGreedyOptimizationStrategy,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


SYMBOLS = ["AAPL", "GOOG", "MSFT"]
_MOCK_EXPECTED_RETURNS = {s: 0.01 for s in SYMBOLS}
_MOCK_WEIGHTS = pd.Series([0.5, 0.3, 0.2], index=SYMBOLS)


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


# ── Init Tests ───────────────────────────────────────────────────────────────


class TestOptimizationInit:
    """Test construction and parameter storage."""

    def test_default_params(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance, stocks=stocks, news=None
        )
        assert strategy.min_assets == 2
        assert strategy.max_assets == 5
        assert strategy.risk_aversion == 0.1

    def test_custom_params(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        out_dir = Path("/tmp/test_out")
        strategy = InvestmentStrategyOptimization(
            instance=instance,
            stocks=stocks,
            news=None,
            min_assets=3,
            max_assets=10,
            risk_aversion=0.5,
            output_directory=out_dir,
        )
        assert strategy.min_assets == 3
        assert strategy.max_assets == 10
        assert strategy.risk_aversion == 0.5
        assert strategy.output_directory == out_dir

    def test_legacy_num_assets_to_hold(self) -> None:
        """Test backward-compat: num_assets_to_hold overrides max_assets."""
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance,
            stocks=stocks,
            news=None,
            num_assets_to_hold=8,
        )
        assert strategy.max_assets == 8


# ── Pipeline Step Tests ──────────────────────────────────────────────────────


class TestOptimizationPipelineSteps:
    """Test each pipeline step in isolation."""

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.AveragePredictionStrategy"
    )
    def test_predict_returns_returns_dict(self, mock_avg_cls: MagicMock) -> None:
        mock_inst = MagicMock()
        mock_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_avg_cls.return_value = mock_inst

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance, stocks=stocks, news=None
        )

        result = strategy._predict_returns()
        assert result == _MOCK_EXPECTED_RETURNS

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.AveragePredictionStrategy"
    )
    def test_predict_returns_raises_prediction_error_on_empty(
        self, mock_avg_cls: MagicMock
    ) -> None:
        mock_inst = MagicMock()
        mock_inst.predict_returns.return_value = {}
        mock_avg_cls.return_value = mock_inst

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance, stocks=stocks, news=None
        )

        with pytest.raises(PredictionError, match="No expected returns"):
            strategy._predict_returns()

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.AveragePredictionStrategy"
    )
    def test_predict_returns_wraps_exception_as_prediction_error(
        self, mock_avg_cls: MagicMock
    ) -> None:
        mock_inst = MagicMock()
        mock_inst.predict_returns.side_effect = RuntimeError("boom")
        mock_avg_cls.return_value = mock_inst

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance, stocks=stocks, news=None
        )

        with pytest.raises(PredictionError, match="boom"):
            strategy._predict_returns()

    def test_build_optimizer_returns_seeded_nsga_strategy(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance, stocks=stocks, news=None
        )

        cov = strategy.build_covariance_matrix()
        optimizer = strategy._build_optimizer(_MOCK_EXPECTED_RETURNS, cov)

        assert isinstance(optimizer, MultiGreedyOptimizationStrategy)

    def test_build_optimizer_does_not_reuse_old_universe_weights(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
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

    def test_build_optimizer_no_seed_initially(self) -> None:
        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance, stocks=stocks, news=None
        )

        cov = strategy.build_covariance_matrix()
        optimizer = strategy._build_optimizer(_MOCK_EXPECTED_RETURNS, cov)

        assert optimizer.seed_weights is None


# ── run_strategy Orchestration Tests ─────────────────────────────────────────


class TestOptimizationRunStrategy:
    """Test full orchestration with mocked dependencies."""

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.AveragePredictionStrategy"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.MultiGreedyOptimizationStrategy"
    )
    def test_full_pipeline_returns_weights(
        self,
        mock_opt_cls: MagicMock,
        mock_avg_cls: MagicMock,
    ) -> None:
        mock_avg_inst = MagicMock()
        mock_avg_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_avg_cls.return_value = mock_avg_inst
        mock_opt_inst = MagicMock()
        mock_opt_result = MagicMock()
        mock_opt_result.front = [MagicMock()]
        mock_opt_inst.optimize.return_value = mock_opt_result
        mock_opt_cls.return_value = mock_opt_inst

        mock_selector = MagicMock()
        mock_selector.select.return_value = _MOCK_WEIGHTS

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance,
            stocks=stocks,
            news=None,
            selection_strategy=mock_selector,
        )

        result = strategy.run_strategy()

        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, _MOCK_WEIGHTS)
        mock_selector.select.assert_called_once()

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.AveragePredictionStrategy"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.MultiGreedyOptimizationStrategy"
    )
    def test_stores_weights_for_warm_start(
        self,
        mock_opt_cls: MagicMock,
        mock_avg_cls: MagicMock,
    ) -> None:
        mock_avg_inst = MagicMock()
        mock_avg_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_avg_cls.return_value = mock_avg_inst
        mock_opt_inst = MagicMock()
        mock_opt_result = MagicMock()
        mock_opt_result.front = [MagicMock()]
        mock_opt_inst.optimize.return_value = mock_opt_result
        mock_opt_cls.return_value = mock_opt_inst

        mock_selector = MagicMock()
        mock_selector.select.return_value = _MOCK_WEIGHTS

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance,
            stocks=stocks,
            news=None,
            selection_strategy=mock_selector,
        )

        assert strategy.is_warm_start is False
        strategy.run_strategy()
        assert strategy.is_warm_start is True

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.AveragePredictionStrategy"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.MultiGreedyOptimizationStrategy"
    )
    def test_raises_optimization_error_on_empty_weights(
        self,
        mock_opt_cls: MagicMock,
        mock_avg_cls: MagicMock,
    ) -> None:
        mock_avg_inst = MagicMock()
        mock_avg_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_avg_cls.return_value = mock_avg_inst
        mock_opt_inst = MagicMock()
        mock_opt_result = MagicMock()
        mock_opt_result.front = [MagicMock()]
        mock_opt_inst.optimize.return_value = mock_opt_result
        mock_opt_cls.return_value = mock_opt_inst

        mock_selector = MagicMock()
        mock_selector.select.return_value = pd.Series(dtype=float)

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance,
            stocks=stocks,
            news=None,
            selection_strategy=mock_selector,
        )

        with pytest.raises(OptimizationError, match="empty weights"):
            strategy.run_strategy()

    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.AveragePredictionStrategy"
    )
    @patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.MultiGreedyOptimizationStrategy"
    )
    def test_raises_optimization_error_on_optimizer_failure(
        self,
        mock_opt_cls: MagicMock,
        mock_avg_cls: MagicMock,
    ) -> None:
        mock_avg_inst = MagicMock()
        mock_avg_inst.predict_returns.return_value = _MOCK_EXPECTED_RETURNS
        mock_avg_cls.return_value = mock_avg_inst
        mock_opt_inst = MagicMock()
        mock_opt_inst.optimize.side_effect = RuntimeError("solver failed")
        mock_opt_cls.return_value = mock_opt_inst

        stocks = [_make_stock(s) for s in SYMBOLS]
        instance = _make_instance()
        strategy = InvestmentStrategyOptimization(
            instance=instance, stocks=stocks, news=None
        )

        with pytest.raises(OptimizationError, match="solver failed"):
            strategy.run_strategy()


@pytest.mark.parametrize("estimator", ["ewm_6", "ewm_12", "ewm_12_mean_50"])
def test_configured_forecast_matches_shared_predictor_and_ignores_future_prices(
    estimator,
):
    from investment_tool.constants import OptimizationReturnEstimator
    from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
        AveragePredictionStrategy,
    )
    from investment_tool.strategies.prediction_strategies.momentum_prediction_strategy import (
        MeanBlendPredictionStrategy,
    )
    from tests.strategies.prediction_strategies.test_fuzzy_prediction_strategy import (
        stock,
    )

    asset = stock(returns=np.random.default_rng(42).normal(0.01, 0.08, 70))
    settings = dict(
        instance=_make_instance(datetime(2024, 1, 1)),
        stocks=[asset],
        news=None,
        return_estimator=OptimizationReturnEstimator(estimator),
    )
    strategy = InvestmentStrategyOptimization(**settings)
    prefix = strategy.stocks_with_recent_history()
    shared = (
        MeanBlendPredictionStrategy(prefix, span=12, momentum_weight=0.5)
        if estimator == "ewm_12_mean_50"
        else AveragePredictionStrategy(prefix, span=12 if estimator == "ewm_12" else 6)
    )
    forecast = strategy._predict_returns()
    assert forecast == shared.predict_returns()
    asset.data.loc[
        asset.data.index >= "2024-01-01", ["open", "high", "low", "close"]
    ] *= 20
    assert InvestmentStrategyOptimization(**settings)._predict_returns() == forecast


def test_unknown_return_estimator_fails_at_construction():
    with pytest.raises(ValueError):
        InvestmentStrategyOptimization(
            instance=_make_instance(),
            stocks=[_make_stock("A")],
            news=None,
            return_estimator="unknown",
        )


@pytest.mark.parametrize("legacy_maximum", [None, 5])
def test_small_news_shortlist_fails_at_construction(legacy_maximum):
    with pytest.raises(ValueError, match="at least maximum assets"):
        InvestmentStrategyOptimization(
            instance=_make_instance(),
            stocks=[_make_stock("A")],
            news=None,
            news_shortlist=True,
            news_top_assets=2,
            max_assets=3,
            num_assets_to_hold=legacy_maximum,
        )


def test_news_candidates_are_shared_by_forecasts_and_covariance_without_removing_holdings():
    stocks = [_make_stock(s) for s in SYMBOLS]
    instance = _make_instance()
    strategy = InvestmentStrategyOptimization(
        instance, stocks, None, max_assets=2, news_shortlist=True
    )
    with patch(
        "investment_tool.strategies.orchestrators.investment_strategy_optimization.news_candidates",
        return_value=({"AAPL", "MSFT"}, {"enabled": True}),
    ) as shortlist:
        forecast = strategy._predict_returns()
        covariance = strategy.build_covariance_matrix()
        assert list(forecast) == list(covariance.index) == ["AAPL", "MSFT"]
        assert shortlist.call_count == 1
        assert {s.symbol for s in strategy.stocks} == set(SYMBOLS)
        assert {s.symbol for s in strategy._raw_stocks} == set(SYMBOLS)
        instance.current_trading_period.trading_date = datetime(2024, 2, 1)
        strategy.stocks_with_recent_history()
        assert shortlist.call_count == 2


@pytest.mark.parametrize("balance", ["capital_weights", "hybrid"])
def test_capital_balance_rejects_unsupported_backend_instead_of_silent_risk_model(
    balance,
):
    with pytest.raises(ValueError, match="evolutionary backend"):
        InvestmentStrategyOptimization(
            _make_instance(),
            [_make_stock(s) for s in SYMBOLS],
            None,
            balance=balance,
            optimizer_type="decomposition",
        )
