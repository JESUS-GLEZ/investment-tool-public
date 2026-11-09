"""Unit tests for BaseInvestmentStrategy.

Tests data filtering, look-ahead bias prevention, iteration tracking,
warm-start state management, and the shared covariance matrix builder.
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency, StockFields
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    BaseOptimizationStrategy,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)
from investment_tool.strategies.optimization_strategies.front_cache import (
    CandidateFrontCache,
)
from investment_tool.strategies.optimization_strategies.seeded_optimization import (
    MultiGreedyOptimizationStrategy,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


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


def _make_news(symbol: str, news_date: date) -> tuple[tuple[str, date], News]:
    """Create a single news entry for testing."""
    news = News(
        symbol=symbol,
        period_start_date=news_date,
        period_end_date=news_date,
        data=[
            {
                "headline": "Test headline",
                "summary": "Test summary",
                "published_at": news_date.isoformat() + "T12:00:00Z",
            }
        ],
    )
    return (symbol, news_date), news


class ConcreteStrategy(BaseInvestmentStrategy):
    """Minimal concrete subclass for testing the abstract base class."""

    def run_strategy(self) -> pd.Series:
        """Return equal weights for all stocks."""
        n = len(self.stocks)
        weights = pd.Series(
            1.0 / n if n > 0 else 0.0,
            index=[s.symbol for s in self.stocks],
        )
        return weights


# ── Init Tests ───────────────────────────────────────────────────────────────


class TestBaseStrategyInit:
    """Test construction and validation."""

    def test_basic_construction(self) -> None:
        """Strategy initialises with valid inputs."""
        stocks = [_make_stock("AAPL"), _make_stock("GOOG")]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        assert len(strategy.stocks) == 2
        assert strategy._iteration == 0
        assert strategy._previous_optimal_weights is None

    def test_new_asset_cannot_collapse_covariance(self) -> None:
        old_data = _make_stock("OLD", n_periods=24, trend=0.02).data.copy()
        old_data[StockFields.CLOSE] *= 1 + 0.05 * np.sin(np.arange(24))
        old = Stock("OLD", old_data)
        recent_data = _make_stock("NEW", n_periods=24).data.iloc[-3:]
        recent = Stock("NEW", recent_data)
        strategy = ConcreteStrategy(
            instance=_make_instance(), stocks=[old, recent], news=None
        )

        assert [s.symbol for s in strategy.stocks_with_recent_history()] == ["OLD"]
        covariance = strategy.build_covariance_matrix()
        assert list(covariance.index) == ["OLD"]
        assert covariance.loc["OLD", "OLD"] > 1e-8

    def test_raises_on_none_instance(self) -> None:
        stocks = [_make_stock("AAPL")]
        with pytest.raises(ValueError, match="current_trading_period"):
            ConcreteStrategy(instance=None, stocks=stocks)  # type: ignore[arg-type]

    def test_raises_on_none_trading_period(self) -> None:
        instance = Instance(
            historical_start_date=date(2022, 1, 1),
            historical_end_date=date(2024, 1, 1),
            frequency=Frequency.MONTHLY,
            initial_trading_period=None,
            previous_trading_period=None,
            current_trading_period=None,
        )
        stocks = [_make_stock("AAPL")]
        with pytest.raises(ValueError, match="current_trading_period"):
            ConcreteStrategy(instance=instance, stocks=stocks)

    def test_raises_on_empty_stocks(self) -> None:
        instance = _make_instance()
        with pytest.raises(ValueError, match="At least one stock"):
            ConcreteStrategy(instance=instance, stocks=[])

    def test_news_default_empty(self) -> None:
        """No news → empty dict."""
        stocks = [_make_stock("AAPL")]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)
        assert strategy.news == {}


# ── Data Filtering Tests ─────────────────────────────────────────────────────


class TestDataFiltering:
    """Test look-ahead bias prevention via data filtering."""

    def test_stock_data_filtered_before_trading_date(self) -> None:
        """Only data strictly before the trading date should be kept."""
        stock = _make_stock("AAPL", n_periods=24)
        # Trading date is 2023-07-15 → only data before 2023-07 kept
        instance = _make_instance(datetime(2023, 7, 15))
        strategy = ConcreteStrategy(instance=instance, stocks=[stock])

        for s in strategy.stocks:
            last_date = s.data.index.max()
            assert last_date < pd.Timestamp("2023-07-15")

    def test_stock_data_sorted_ascending(self) -> None:
        stock = _make_stock("AAPL")
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=[stock])

        dates = strategy.stocks[0].data.index
        assert dates.is_monotonic_increasing

    def test_news_filtered_before_trading_date(self) -> None:
        """News on or after trading date should be excluded."""
        instance = _make_instance(datetime(2023, 6, 15))
        stocks = [_make_stock("AAPL")]

        key_before, news_before = _make_news("AAPL", date(2023, 5, 1))
        key_after, news_after = _make_news("AAPL", date(2023, 7, 1))
        key_on, news_on = _make_news("AAPL", date(2023, 6, 15))

        all_news = {key_before: news_before, key_after: news_after, key_on: news_on}
        strategy = ConcreteStrategy(instance=instance, stocks=stocks, news=all_news)

        # Only the news before 2023-06-15 should survive
        assert len(strategy.news) == 1
        assert key_before in strategy.news
        assert key_after not in strategy.news
        assert key_on not in strategy.news


# ── Iteration & Warm-Start Tests ─────────────────────────────────────────────


class TestIterationTracking:
    """Test iteration counter and warm-start state."""

    def test_iteration_starts_at_zero(self) -> None:
        stocks = [_make_stock("AAPL")]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)
        assert strategy._iteration == 0

    def test_refresh_data_increments_iteration(self) -> None:
        stocks = [_make_stock("AAPL")]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        strategy.refresh_data()
        assert strategy._iteration == 1

        strategy.refresh_data()
        assert strategy._iteration == 2

    def test_is_warm_start_false_initially(self) -> None:
        stocks = [_make_stock("AAPL")]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)
        assert strategy.is_warm_start is False

    def test_store_optimal_weights_enables_warm_start(self) -> None:
        stocks = [_make_stock("AAPL"), _make_stock("GOOG")]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        weights = pd.Series([0.6, 0.4], index=["AAPL", "GOOG"])
        strategy.store_optimal_weights(weights)

        assert strategy.is_warm_start is True
        assert strategy._previous_optimal_weights is not None
        np.testing.assert_array_almost_equal(
            strategy._previous_optimal_weights, [0.6, 0.4]
        )

    def test_stored_weights_are_numpy_array(self) -> None:
        stocks = [_make_stock("AAPL")]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        weights = pd.Series([1.0], index=["AAPL"])
        strategy.store_optimal_weights(weights)

        assert isinstance(strategy._previous_optimal_weights, np.ndarray)

    def test_selected_risk_aligns_weights_to_the_optimizer_asset_order(self) -> None:
        strategy = ConcreteStrategy(
            _make_instance(), [_make_stock("A"), _make_stock("B")]
        )
        strategy._cost_returns = pd.Series([0.04, 0.01], index=["B", "A"])
        strategy._optimization_covariance = np.array([[0.09, 0.012], [0.012, 0.04]])
        strategy.store_optimal_weights(pd.Series({"A": 0.75, "B": 0.25}))
        assert strategy.last_optimization_diagnostics[
            "selected_variance"
        ] == pytest.approx(0.032625)
        assert strategy.last_optimization_diagnostics[
            "selected_volatility"
        ] == pytest.approx(np.sqrt(0.032625))

    @pytest.mark.parametrize("variance", [0.0, -0.01, np.nan])
    def test_selected_risk_does_not_invent_validity_for_invalid_variance(
        self, variance: float
    ) -> None:
        strategy = ConcreteStrategy(_make_instance(), [_make_stock("A")])
        strategy._cost_returns = pd.Series({"A": 0.04})
        strategy._optimization_covariance = np.array([[variance]])
        strategy.store_optimal_weights(pd.Series({"A": 1.0}))
        if variance == 0:
            assert strategy.last_optimization_diagnostics["selected_volatility"] == 0
        else:
            assert "selected_volatility" not in strategy.last_optimization_diagnostics


# ── Covariance Matrix Tests ──────────────────────────────────────────────────


class TestCovarianceMatrix:
    """Test the shared covariance matrix builder."""

    def test_returns_dataframe(self) -> None:
        stocks = [_make_stock("AAPL", n_periods=24), _make_stock("GOOG", n_periods=24)]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        cov = strategy.build_covariance_matrix()

        assert isinstance(cov, pd.DataFrame)
        assert cov.shape[0] == cov.shape[1]

    def test_covariance_symmetric(self) -> None:
        stocks = [_make_stock("AAPL", n_periods=24), _make_stock("GOOG", n_periods=24)]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        cov = strategy.build_covariance_matrix()

        np.testing.assert_array_almost_equal(cov.values, cov.values.T)

    def test_covariance_positive_semidefinite(self) -> None:
        stocks = [
            _make_stock("AAPL", n_periods=24),
            _make_stock("GOOG", n_periods=24),
            _make_stock("MSFT", n_periods=24),
        ]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        cov = strategy.build_covariance_matrix()
        eigenvalues = np.linalg.eigvalsh(cov.values)

        # All eigenvalues ≥ 0 (PSD)
        assert np.all(eigenvalues >= -1e-8)

    def test_covariance_indexed_by_symbol(self) -> None:
        stocks = [_make_stock("AAPL", n_periods=24), _make_stock("GOOG", n_periods=24)]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        cov = strategy.build_covariance_matrix()

        assert list(cov.index) == ["AAPL", "GOOG"]
        assert list(cov.columns) == ["AAPL", "GOOG"]

    def test_diagonal_positive(self) -> None:
        """Diagonal entries (variances) must be positive."""
        stocks = [_make_stock("AAPL", n_periods=24), _make_stock("GOOG", n_periods=24)]
        instance = _make_instance()
        strategy = ConcreteStrategy(instance=instance, stocks=stocks)

        cov = strategy.build_covariance_matrix()

        for i in range(cov.shape[0]):
            assert cov.iloc[i, i] > 0

    def test_eligible_ipo_does_not_discard_old_variance_history(self) -> None:
        data = _make_stock("OLD", n_periods=60).data.copy()
        data[StockFields.CLOSE] *= 1 + 0.03 * np.sin(np.arange(60))
        old = Stock("OLD", data)
        new = Stock("NEW", data.iloc[-13:])  # now has the 12 eligible returns
        instance = _make_instance(datetime(2027, 1, 1))
        first = ConcreteStrategy(instance, [old])
        second = ConcreteStrategy(instance, [old, new])
        before = first.build_covariance_matrix()
        after = second.build_covariance_matrix()
        assert list(after.index) == ["OLD", "NEW"]
        assert after.loc["OLD", "OLD"] == pytest.approx(
            before.loc["OLD", "OLD"], rel=1e-12
        )
        assert second.last_covariance_diagnostics["observed_returns"] == {
            "OLD": 59,
            "NEW": 12,
        }

    def test_future_prices_cannot_change_covariance(self) -> None:
        data = _make_stock("A", n_periods=36).data.copy()
        data[StockFields.CLOSE] *= 1 + 0.05 * np.sin(np.arange(36))
        modified = data.copy()
        modified.loc[modified.index >= "2024-01-01", StockFields.CLOSE] *= 1000
        instance = _make_instance(datetime(2024, 1, 1))
        before = ConcreteStrategy(instance, [Stock("A", data)])
        after = ConcreteStrategy(instance, [Stock("A", modified)])
        pd.testing.assert_frame_equal(
            before.build_covariance_matrix(), after.build_covariance_matrix()
        )

    @pytest.mark.parametrize("cached", [False, True])
    def test_covariance_diagnostics_reach_optimization_result(
        self, cached: bool
    ) -> None:
        strategy = ConcreteStrategy(_make_instance(), [_make_stock("A")])
        strategy.build_covariance_matrix()
        result = OptimizationResult(front=[])
        if cached:
            optimizer = Mock(spec=MultiGreedyOptimizationStrategy)
            strategy.candidate_front_cache = Mock(spec=CandidateFrontCache)
            strategy.candidate_front_cache.optimize.return_value = result
        else:
            optimizer = Mock(spec=BaseOptimizationStrategy)
            optimizer.optimize.return_value = result
        optimizer.trading_costs = None
        actual = strategy.optimize_portfolios(optimizer)
        assert actual.search_diagnostics["covariance"] == (
            strategy.last_covariance_diagnostics
        )
        assert actual.search_diagnostics["covariance"]["shrinkage"] == 0.1


def test_dated_eligibility_preserves_quotes_and_checks_adjacent_months() -> None:
    stocks = [_make_stock("OLD", 24), _make_stock("NEW", 24), _make_stock("OTHER", 24)]
    stocks[1] = Stock("NEW", stocks[1].data.iloc[12:])
    instance = _make_instance(datetime(2023, 12, 1))
    period = instance.current_trading_period
    period.eligible_symbols = frozenset({"OLD", "NEW"})
    period.minimum_return_periods = 12
    strategy = ConcreteStrategy(instance, stocks)
    assert [s.symbol for s in strategy.stocks] == ["OLD"]
    # NEW enters when 13 closes / 12 genuinely adjacent monthly returns exist.
    period.trading_date = datetime(2024, 2, 1)
    # Supply another completed close, but membership still excludes OTHER.
    new = _make_stock("NEW", 25)
    strategy._raw_stocks = [stocks[0], Stock("NEW", new.data.iloc[12:]), stocks[2]]
    strategy.refresh_data()
    assert [s.symbol for s in strategy.stocks] == ["NEW"]
    assert "OTHER" in {s.symbol for s in strategy._raw_stocks}


def test_dated_history_rejects_a_missing_month_and_stale_series() -> None:
    old = _make_stock("OLD", 24)
    gap = Stock("GAP", old.data.drop(old.data.index[11]))
    instance = _make_instance(datetime(2024, 1, 1))
    instance.current_trading_period.eligible_symbols = frozenset({"GAP"})
    instance.current_trading_period.minimum_return_periods = 12
    with pytest.raises(ValueError, match="required completed monthly history"):
        ConcreteStrategy(instance, [gap])
    instance.current_trading_period.trading_date = datetime(2024, 3, 1)
    with pytest.raises(ValueError, match="required completed monthly history"):
        ConcreteStrategy(instance, [old])


def test_dated_estimator_does_not_bridge_old_missing_months() -> None:
    stock = _make_stock("OLD", 36)
    stock = Stock("OLD", stock.data.drop(stock.data.index[4]))
    instance = _make_instance(datetime(2025, 1, 1))
    instance.current_trading_period.eligible_symbols = frozenset({"OLD"})
    instance.current_trading_period.minimum_return_periods = 12
    strategy = ConcreteStrategy(instance, [stock])
    assert len(strategy.stocks) == 1
    assert pd.isna(strategy.stocks[0].data.loc["2022-06-01", StockFields.RETURNS])
    assert strategy.stocks[0].data.loc[
        "2022-07-01", StockFields.RETURNS
    ] == pytest.approx(0.01)


def test_execution_candidates_intersect_membership_without_changing_history() -> None:
    stocks = [_make_stock(s, 24) for s in ["OLD", "LIVE", "OUTSIDE"]]
    instance = _make_instance(datetime(2024, 1, 1))
    period = instance.current_trading_period
    period.eligible_symbols = frozenset({"OLD", "LIVE"})
    period.execution_eligible_symbols = frozenset({"LIVE", "OUTSIDE"})
    strategy = ConcreteStrategy(instance, stocks)
    assert [s.symbol for s in strategy.stocks] == ["LIVE"]
    assert {s.symbol for s in strategy._raw_stocks} == {"OLD", "LIVE", "OUTSIDE"}
    assert strategy.stocks[0].data.index.max() < period.trading_date
    period.execution_eligible_symbols = None
    strategy.refresh_data()
    assert {s.symbol for s in strategy.stocks} == {"OLD", "LIVE"}


@pytest.mark.parametrize(
    "frequency,step", [(Frequency.WEEKLY, "W-MON"), (Frequency.DAILY, "B")]
)
def test_dated_history_preserves_year_span_and_cutoff_with_other_frequencies(
    frequency, step
):
    from investment_tool.domain.market_time import normalize_bars

    labels = pd.date_range("2023-01-02", "2025-01-13", freq=step)
    frame = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000.0},
        index=labels.tz_localize("America/New_York"),
    )
    frame = normalize_bars(
        frame,
        frequency,
        market_timezone="America/New_York",
        as_of="2026-01-01T00:00:00Z",
    )
    late = frame.copy()
    late.loc[labels[labels < "2025-01-06"][-1], "available_at"] = pd.Timestamp(
        "2025-01-07T05:00:00Z"
    )
    gap = frame.drop(labels[10])
    recent_gap = frame.drop(labels[labels < "2025-01-06"][-5])
    stocks = [
        Stock("FULL", frame),
        Stock("OLD_GAP", gap),
        Stock("NEW", frame.iloc[-12:]),
        Stock("LATE", late),
        Stock("RECENT_GAP", recent_gap),
    ]
    instance = _make_instance(datetime(2025, 1, 6))
    instance.frequency = frequency
    instance.history_calendars = {s.symbol: labels for s in stocks}
    instance.current_trading_period.minimum_return_periods = 12
    instance.current_trading_period.decision_time = pd.Timestamp(
        "2025-01-06T05:00:00Z"
    ).to_pydatetime()
    strategy = ConcreteStrategy(instance, stocks)
    assert {s.symbol for s in strategy.stocks} == {"FULL", "OLD_GAP"}
    old_gap = next(s for s in strategy.stocks if s.symbol == "OLD_GAP")
    assert pd.isna(old_gap.data.loc[labels[11], "returns"])
    assert pd.notna(old_gap.data.loc[labels[12], "returns"])
    assert all(s.data.index.max() < pd.Timestamp("2025-01-06") for s in strategy.stocks)
    for stock in stocks:
        stock.data.loc[stock.data.index >= "2025-01-06", "close"] = 1e12
    strategy.refresh_data()
    assert {s.symbol for s in strategy.stocks} == {"FULL", "OLD_GAP"}
