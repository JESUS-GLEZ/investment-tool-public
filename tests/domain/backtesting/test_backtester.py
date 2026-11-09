"""Unit tests for Backtester metrics calculation."""

from __future__ import annotations

from datetime import date, datetime
from dataclasses import replace
from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency, StockFields
from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.exceptions import BacktestingError


def _create_mock_model_data(frequency: Frequency = Frequency.MONTHLY) -> MagicMock:
    """Create mock ModelData for testing."""
    mock = MagicMock()
    mock.instance.frequency = frequency
    return mock


def _create_mock_strategy() -> MagicMock:
    """Create mock investment strategy."""
    mock = MagicMock()
    mock.run_strategy.return_value = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
    return mock


class TestBacktesterInitialization:
    """Tests for Backtester initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic backtester creation."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()

        backtester = Backtester(model_data, strategy)

        assert backtester.model_data is model_data
        assert backtester.investment_strategy is strategy
        assert backtester.results == []
        assert backtester.risk_free_rate == 0.0

    def test_with_risk_free_rate(self) -> None:
        """Test initialization with risk-free rate."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()

        backtester = Backtester(model_data, strategy, risk_free_rate=0.02)

        assert backtester.risk_free_rate == 0.02

    def test_frequency_from_model_data(self) -> None:
        """Test that frequency is taken from model data."""
        model_data = _create_mock_model_data(Frequency.WEEKLY)
        strategy = _create_mock_strategy()

        backtester = Backtester(model_data, strategy)

        assert backtester.frequency == Frequency.WEEKLY


def test_period_returns_use_that_periods_selected_holdings_and_final_price() -> None:
    dates = [datetime(2024, month, 1) for month in (1, 2, 3)]
    prices = [(100.0, 100.0), (110.0, 90.0), (121.0, 120.0)]
    opens = [(100.0, 100.0), (125.0, 100.0), (132.0, 100.0)]
    periods = []
    for trading_date, (aapl, msft), (aapl_open, msft_open) in zip(dates, prices, opens):
        market_data = pd.DataFrame(
            [[aapl, msft, aapl_open, msft_open]],
            index=[trading_date],
            columns=pd.MultiIndex.from_tuples(
                [
                    (StockFields.CLOSE, "AAPL"),
                    (StockFields.CLOSE, "MSFT"),
                    (StockFields.OPEN, "AAPL"),
                    (StockFields.OPEN, "MSFT"),
                ]
            ),
        )
        periods.append(Period(trading_date, market_data))

    model_data = ModelData()
    model_data.periods = periods
    model_data.instance = Instance(
        historical_start_date=date(2024, 1, 1),
        historical_end_date=date(2024, 3, 1),
        frequency=Frequency.MONTHLY,
        initial_trading_period=periods[1],
        previous_trading_period=periods[0],
        current_trading_period=periods[1],
    )
    model_data.portfolio = Portfolio(model_data.instance, initial_cash=1000.0)
    strategy = MagicMock()
    strategy.last_optimization_diagnostics = {"restarts": [{"seed": 17, "nfe": 60}]}
    strategy.run_strategy.side_effect = [
        pd.Series({"AAPL": 1.0, "MSFT": 0.0}),
        pd.Series({"AAPL": 0.0, "MSFT": 1.0}),
    ]

    backtester = Backtester(model_data, strategy)
    checkpoints = []
    backtester.run_backtest(
        period_complete_callback=lambda current: checkpoints.append(
            list(current.results)
        )
    )

    assert [len(rows) for rows in checkpoints] == [1, 2]
    assert checkpoints[0][0]["optimization_diagnostics"] == {
        "restarts": [{"seed": 17, "nfe": 60}]
    }
    assert model_data.portfolio.history[0]["prices"] == {"AAPL": 125.0}
    assert model_data.portfolio.history[1]["pre_rebalance_value"] == pytest.approx(
        1056.0
    )
    assert model_data.portfolio.history[1]["post_rebalance_value"] == pytest.approx(
        1056.0
    )
    assert model_data.portfolio.history[1]["positions"] == {"MSFT": 10.56}

    assert [row["date"] for row in backtester.results] == dates[1:]
    assert [row["selected_assets"] for row in backtester.results] == [
        {"AAPL": 1.0},
        {"MSFT": 1.0},
    ]
    assert backtester.results[0]["total_value"] == pytest.approx(880.0)
    assert backtester.results[0]["return"] == pytest.approx(-0.12)
    assert backtester.results[1]["total_value"] == pytest.approx(8 * 132 * 120 / 100)
    assert backtester.results[1]["return"] == pytest.approx(
        (132 / 110) * (120 / 100) - 1
    )


class TestPeriodsPerYear:
    """Tests for _get_periods_per_year method."""

    def test_daily_frequency(self) -> None:
        """Test periods per year for daily frequency."""
        model_data = _create_mock_model_data(Frequency.DAILY)
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        assert backtester._get_periods_per_year() == 252

    def test_weekly_frequency(self) -> None:
        """Test periods per year for weekly frequency."""
        model_data = _create_mock_model_data(Frequency.WEEKLY)
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        assert backtester._get_periods_per_year() == 52

    def test_monthly_frequency(self) -> None:
        """Test periods per year for monthly frequency."""
        model_data = _create_mock_model_data(Frequency.MONTHLY)
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        assert backtester._get_periods_per_year() == 12


class TestDailyReturnCalculation:
    """Tests for _calculate_daily_return method."""

    def test_first_period_return_is_zero(self) -> None:
        """Test that first period return is 0."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        # No previous results
        ret = backtester._calculate_daily_return(100000.0)

        assert ret == 0.0

    def test_positive_return(self) -> None:
        """Test positive return calculation."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        # Add a previous result
        backtester.results = [{"total_value": 100000.0}]

        # 10% gain
        ret = backtester._calculate_daily_return(110000.0)

        assert abs(ret - 0.10) < 0.0001

    def test_negative_return(self) -> None:
        """Test negative return calculation."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        backtester.results = [{"total_value": 100000.0}]

        # 5% loss
        ret = backtester._calculate_daily_return(95000.0)

        assert abs(ret - (-0.05)) < 0.0001

    def test_no_change(self) -> None:
        """Test zero return when value unchanged."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        backtester.results = [{"total_value": 100000.0}]

        ret = backtester._calculate_daily_return(100000.0)

        assert ret == 0.0


class TestGetResults:
    """Tests for get_results method."""

    def test_raises_when_no_results(self) -> None:
        """Test that error raised when no results."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        with pytest.raises(BacktestingError, match="No results"):
            backtester.get_results()

    def test_returns_dataframe(self) -> None:
        """Test that results are returned as DataFrame."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        # Add some results
        backtester.results = [
            {"date": datetime(2024, 1, 1), "total_value": 100000, "return": 0.0},
            {"date": datetime(2024, 2, 1), "total_value": 105000, "return": 0.05},
            {"date": datetime(2024, 3, 1), "total_value": 103000, "return": -0.019},
        ]

        df = backtester.get_results()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_cumulative_return_calculated(self) -> None:
        """Test that cumulative return column is added."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        backtester.results = [
            {"date": datetime(2024, 1, 1), "total_value": 100000, "return": 0.0},
            {"date": datetime(2024, 2, 1), "total_value": 110000, "return": 0.10},
            {"date": datetime(2024, 3, 1), "total_value": 121000, "return": 0.10},
        ]

        df = backtester.get_results()

        assert "cumulative_return" in df.columns
        # Cumulative: (1+0)*(1+0.1)*(1+0.1) - 1 = 0.21
        assert abs(df["cumulative_return"].iloc[-1] - 0.21) < 0.001

    def test_date_is_index(self) -> None:
        """Test that date is set as index."""
        model_data = _create_mock_model_data()
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        backtester.results = [
            {"date": datetime(2024, 1, 1), "total_value": 100000, "return": 0.0},
        ]

        df = backtester.get_results()

        assert df.index.name == "date"


class TestComputeAdditionalMetrics:
    """Tests for compute_additional_metrics method."""

    def test_basic_metrics_computed(self) -> None:
        """Test that basic metrics are computed."""
        model_data = _create_mock_model_data(Frequency.MONTHLY)
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        # Add sample results with known returns
        np.random.seed(42)
        returns = [
            0.01,
            0.02,
            -0.01,
            0.03,
            0.01,
            -0.02,
            0.02,
            0.01,
            0.03,
            -0.01,
            0.02,
            0.01,
        ]
        backtester.results = [
            {
                "date": datetime(2024, i + 1, 1),
                "total_value": 100000 * float(np.prod(1 + np.array(returns[: i + 1]))),
                "return": r,
            }
            for i, r in enumerate(returns)
        ]

        backtester.compute_additional_metrics()

        assert "average_return" in backtester.metrics
        assert "std_return" in backtester.metrics
        assert "annualized_return" in backtester.metrics
        assert "annualized_volatility" in backtester.metrics
        assert "sharpe_ratio" in backtester.metrics

    def test_average_return_calculation(self) -> None:
        """Test average return calculation."""
        model_data = _create_mock_model_data(Frequency.MONTHLY)
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        # Known returns: avg = 0.02
        backtester.results = [
            {"date": datetime(2024, 1, 1), "total_value": 101000, "return": 0.01},
            {
                "date": datetime(2024, 2, 1),
                "total_value": 101000 * 1.02,
                "return": 0.02,
            },
            {
                "date": datetime(2024, 3, 1),
                "total_value": 101000 * 1.02 * 1.03,
                "return": 0.03,
            },
        ]

        backtester.compute_additional_metrics()

        expected_avg = (0.01 + 0.02 + 0.03) / 3
        assert abs(backtester.metrics["average_return"] - expected_avg) < 0.0001

    def test_sharpe_ratio_with_risk_free_rate(self) -> None:
        """Test Sharpe ratio accounts for risk-free rate."""
        model_data = _create_mock_model_data(Frequency.MONTHLY)
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy, risk_free_rate=0.02)

        returns = np.array([0.01, 0.02] * 6)
        values = 100000 * np.cumprod(1 + returns)
        backtester.results = [
            {"date": datetime(2024, i + 1, 1), "total_value": value, "return": ret}
            for i, (value, ret) in enumerate(zip(values, returns))
        ]
        backtester.compute_additional_metrics()
        rf_monthly = 1.02 ** (1 / 12) - 1
        expected = (returns.mean() - rf_monthly) / returns.std(ddof=1) * np.sqrt(12)
        assert backtester.metrics["sharpe_ratio"] == pytest.approx(expected)

    def test_annualization_for_monthly(self) -> None:
        """Test that annualization uses correct periods."""
        model_data = _create_mock_model_data(Frequency.MONTHLY)
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        # 1% monthly return
        backtester.results = [
            {
                "date": datetime(2024, i + 1, 1),
                "total_value": 100000 * 1.01 ** (i + 1),
                "return": 0.01,
            }
            for i in range(12)
        ]

        backtester.compute_additional_metrics()

        # Annualized return: (1 + 0.01)^12 - 1 ≈ 0.1268
        expected_annualized = (1 + 0.01) ** 12 - 1
        assert abs(backtester.metrics["annualized_return"] - expected_annualized) < 0.01


class TestRecordSnapshot:
    """Tests for _record_snapshot method."""

    def test_snapshot_structure(self) -> None:
        """Test that snapshot has correct structure."""
        model_data = _create_mock_model_data()
        model_data.portfolio.calculate_portfolio_value.return_value = 100000.0
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        prices = pd.Series({"AAPL": 150.0})
        weights = pd.Series({"AAPL": 1.0})
        asset_prices = {"AAPL": 150.0}
        snapshot = backtester._record_snapshot(
            datetime(2024, 1, 1), prices, weights, asset_prices
        )

        assert "date" in snapshot
        assert "total_value" in snapshot
        assert "return" in snapshot
        assert "selected_assets" in snapshot
        assert "asset_prices" in snapshot
        assert snapshot["date"] == datetime(2024, 1, 1)
        assert snapshot["selected_assets"] == {"AAPL": 1.0}
        assert snapshot["asset_prices"] == {"AAPL": 150.0}

    def test_selected_assets_excludes_dust_positions(self) -> None:
        """Test that weights below 1% are excluded from selected_assets."""
        model_data = _create_mock_model_data()
        model_data.portfolio.calculate_portfolio_value.return_value = 100000.0
        strategy = _create_mock_strategy()
        backtester = Backtester(model_data, strategy)

        prices = pd.Series({"AAPL": 150.0, "GOOG": 200.0, "MSFT": 300.0})
        weights = pd.Series({"AAPL": 0.60, "GOOG": 0.005, "MSFT": 0.395})
        asset_prices = {"AAPL": 150.0, "GOOG": 200.0, "MSFT": 300.0}
        snapshot = backtester._record_snapshot(
            datetime(2024, 1, 1), prices, weights, asset_prices
        )

        selected = snapshot["selected_assets"]
        assert "AAPL" in selected
        assert "MSFT" in selected
        assert "GOOG" not in selected  # 0.5% is below the 1% threshold


class TestExtractPeriodPrices:
    """Tests for _extract_period_prices static method."""

    def test_extracts_close_prices(self) -> None:
        """Test that close prices are extracted from market data."""
        from investment_tool.constants import StockFields
        from investment_tool.domain.model_data.entities.period import Period

        dt = datetime(2024, 1, 15)
        columns = pd.MultiIndex.from_tuples(
            [(StockFields.CLOSE, "AAPL"), (StockFields.CLOSE, "GOOG")],
        )
        data = pd.DataFrame([[150.0, 200.0]], index=[dt], columns=columns)
        period = Period(trading_date=dt, market_data=data)

        result = Backtester._extract_period_prices(period)

        assert result == {"AAPL": 150.0, "GOOG": 200.0}

    def test_returns_empty_dict_on_empty_market_data(self) -> None:
        """Test graceful handling of empty market data."""
        from investment_tool.domain.model_data.entities.period import Period

        period = Period(trading_date=datetime(2024, 1, 15), market_data=pd.DataFrame())

        result = Backtester._extract_period_prices(period)

        assert result == {}


def _corporate_action_backtest():
    from investment_tool.domain.model_data.entities.cash_settlement import (
        CashSettlement,
    )

    dates = [datetime(2022, month, 1) for month in (9, 10, 11)]
    columns = pd.MultiIndex.from_tuples(
        [
            (StockFields.OPEN, "GBT"),
            (StockFields.CLOSE, "GBT"),
            (StockFields.OPEN, "LIVE"),
            (StockFields.CLOSE, "LIVE"),
            ("bar_end", "LIVE"),
        ]
    )
    rows = [
        [68, 68, 10, 10, pd.Timestamp("2022-10-01T04:00:00Z")],
        [68, np.nan, 10, 12, pd.Timestamp("2022-11-01T04:00:00Z")],
        [np.nan, np.nan, 12, 15, pd.Timestamp("2022-12-01T05:00:00Z")],
    ]
    periods = [
        Period(
            d,
            pd.DataFrame([row], index=[d], columns=columns),
            execution_time=pd.Timestamp(d, tz="UTC").to_pydatetime(),
        )
        for d, row in zip(dates, rows)
    ]
    model = ModelData()
    model.periods = periods
    model.instance = Instance(
        date(2022, 9, 1),
        date(2022, 11, 30),
        Frequency.MONTHLY,
        periods[1],
        periods[0],
        periods[1],
    )
    model.portfolio = Portfolio(model.instance, 1000)
    model.cash_settlements = (
        CashSettlement(
            "GBT",
            datetime.fromisoformat("2022-10-06T04:00:00+00:00"),
            datetime.fromisoformat("2022-10-06T04:00:00+00:00"),
            68.5,
            "reviewed notice",
        ),
    )
    strategy = _create_mock_strategy()
    strategy.run_strategy.side_effect = [
        pd.Series({"GBT": 1.0}),
        pd.Series({"LIVE": 1.0}),
    ]
    return Backtester(model, strategy)


def test_redeemed_portfolio_becomes_cash_and_is_reinvested_next_month() -> None:
    tester = _corporate_action_backtest()
    tester.run_backtest()
    october, november = tester.results
    assert october["total_value"] == pytest.approx(1000 * 68.5 / 68)
    assert october["positions"] == {}
    assert october["cash"] == pytest.approx(october["total_value"])
    assert october["corporate_actions"][0]["symbol"] == "GBT"
    assert october["corporate_actions"][0]["proceeds"] == pytest.approx(
        october["total_value"]
    )
    assert november["total_value"] == pytest.approx(october["total_value"] * 15 / 12)
    assert november["corporate_actions"] == []
    assert tester.model_data.portfolio.positions.keys() == {"LIVE"}
    assert tester.model_data.periods[2].execution_eligible_symbols == {"LIVE"}


def test_missing_future_close_is_not_used_to_filter_execution_candidates() -> None:
    tester = _corporate_action_backtest()
    tester.model_data.cash_settlements = ()
    with pytest.raises(BacktestingError, match="Missing current-period close"):
        tester.run_backtest()
    # GBT has a real opening quote. The missing end-of-month price cannot
    # remove it from the search in advance and introduce hindsight selection.
    assert "GBT" in tester.model_data.periods[1].execution_eligible_symbols
    tester.investment_strategy.run_strategy.assert_called_once()


def test_unreviewed_missing_held_quote_fails_before_expensive_search() -> None:
    tester = _corporate_action_backtest()
    tester.model_data.instance.current_trading_period = tester.model_data.periods[2]
    tester.model_data.cash_settlements = ()
    tester.model_data.portfolio.positions = {"GBT": 2}
    with pytest.raises(BacktestingError, match="held assets.*GBT"):
        tester.run_backtest()
    tester.investment_strategy.run_strategy.assert_not_called()
    assert tester.model_data.portfolio.positions == {"GBT": 2}


def test_mixed_merger_received_holdings_are_valued_and_sold_next_month():
    from investment_tool.domain.model_data.entities.security_action import (
        SecurityAction,
    )

    tester = _corporate_action_backtest()
    model = tester.model_data
    model.cash_settlements = ()
    model.security_actions = (
        SecurityAction(
            "GBT",
            datetime.fromisoformat("2022-10-06T04:00:00Z"),
            datetime.fromisoformat("2022-10-06T04:00:00Z"),
            5,
            "reviewed notice",
            (("LIVE", 2),),
        ),
    )
    tester.run_backtest()
    october, november = tester.results
    quantity = 1000 / 68
    assert october["positions"] == {"LIVE": pytest.approx(quantity * 2)}
    assert october["total_value"] == pytest.approx(quantity * (5 + 2 * 12))
    assert october["cash"] == pytest.approx(quantity * 5)
    assert october["corporate_actions"][0]["successor_shares"] == {
        "LIVE": pytest.approx(quantity * 2)
    }
    assert november["total_value"] == pytest.approx(october["total_value"] * 15 / 12)
    assert november["corporate_actions"] == []


def _accounting_state(tester):
    data = tester.model_data
    return {
        "positions": dict(data.portfolio.positions),
        "cash": data.portfolio.cash,
        "history": deepcopy(data.portfolio.history),
        "actions": deepcopy(data.portfolio.corporate_actions),
        "applied": set(data.portfolio._applied_actions),
        "current": data.instance.current_trading_period,
        "previous": data.instance.previous_trading_period,
        "results": deepcopy(tester.results),
        "starting_value": tester._starting_value,
    }


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("later_bound", ["effective", "publication"])
def test_between_cutoff_incomplete_action_fails_before_rebalance(
    held, retain, later_bound
):
    tester = _corporate_action_backtest()
    data = tester.model_data
    data.periods[1].market_data[(StockFields.CLOSE, "GBT")] = 68
    if held:
        data.portfolio.positions = {"GBT": 2, "LIVE": 3}
    data.cash_settlements = (
        replace(
            data.cash_settlements[0],
            **(
                {"published_at": datetime.fromisoformat("2022-09-30T04:00:00Z")}
                if later_bound == "effective"
                else {"effective_at": datetime.fromisoformat("2022-09-30T04:00:00Z")}
            ),
            retain_original=retain,
            unresolved_entitlements=("Warrant quotes",),
        ),
    )
    before = _accounting_state(tester)
    with pytest.raises(BacktestingError, match="Incomplete security action for GBT"):
        tester.run_backtest()
    assert _accounting_state(tester) == before
    tester.investment_strategy.run_strategy.assert_called_once()
    assert "GBT" in data.periods[1].execution_eligible_symbols


def test_between_cutoff_failure_preserves_last_completed_checkpoint():
    tester = _corporate_action_backtest()
    data = tester.model_data
    data.instance.current_trading_period = data.periods[0]
    data.cash_settlements = (
        replace(data.cash_settlements[0], unresolved_entitlements=("Warrant quotes",)),
    )
    tester.investment_strategy.last_optimization_diagnostics = {}
    tester.investment_strategy.run_strategy.side_effect = [
        pd.Series({"LIVE": 1.0}),
        pd.Series({"GBT": 1.0}),
    ]
    checkpoints = []
    with pytest.raises(BacktestingError, match="Incomplete security action"):
        tester.run_backtest(
            period_complete_callback=lambda current: checkpoints.append(
                _accounting_state(current)
            )
        )
    assert len(checkpoints) == 1
    assert _accounting_state(tester) == checkpoints[0]


def test_future_incomplete_action_does_not_commit_an_earlier_opening_settlement():
    from investment_tool.domain.model_data.entities.security_action import (
        SecurityAction,
    )

    tester = _corporate_action_backtest()
    data = tester.model_data
    data.portfolio.positions = {"GBT": 2}
    open_time = data.periods[1].execution_time
    data.cash_settlements = (
        replace(
            data.cash_settlements[0], effective_at=open_time, published_at=open_time
        ),
    )
    data.security_actions = (
        SecurityAction(
            "LIVE",
            datetime.fromisoformat("2022-10-06T04:00:00Z"),
            datetime.fromisoformat("2022-10-06T04:00:00Z"),
            0,
            "reviewed notice",
            unresolved_entitlements=("Warrant quotes",),
        ),
    )
    tester.investment_strategy.run_strategy.side_effect = None
    tester.investment_strategy.run_strategy.return_value = pd.Series({"LIVE": 1.0})
    before = _accounting_state(tester)
    with pytest.raises(BacktestingError, match="Incomplete security action for LIVE"):
        tester.run_backtest()
    assert _accounting_state(tester) == before


def test_between_cutoff_successor_chain_is_checked_before_any_trades():
    from investment_tool.domain.model_data.entities.security_action import (
        SecurityAction,
    )

    tester = _corporate_action_backtest()
    data = tester.model_data
    data.cash_settlements = ()
    early = datetime.fromisoformat("2022-10-06T04:00:00Z")
    late = datetime.fromisoformat("2022-10-07T04:00:00Z")
    data.security_actions = (
        SecurityAction("GBT", early, early, 5, "reviewed merger", (("LIVE", 2),)),
        SecurityAction(
            "LIVE",
            late,
            late,
            0,
            "reviewed distribution",
            retain_original=True,
            unresolved_entitlements=("Warrant quotes",),
        ),
    )
    before = _accounting_state(tester)
    with pytest.raises(BacktestingError, match="Incomplete security action for LIVE"):
        tester.run_backtest()
    assert _accounting_state(tester) == before


def test_incomplete_action_does_not_block_a_holding_sold_before_the_event():
    tester = _corporate_action_backtest()
    data = tester.model_data
    data.portfolio.positions = {"GBT": 2}
    data.cash_settlements = (
        replace(data.cash_settlements[0], unresolved_entitlements=("Warrant quotes",)),
    )
    tester.investment_strategy.run_strategy.side_effect = [
        pd.Series({"LIVE": 1.0}),
        pd.Series({"LIVE": 1.0}),
    ]
    tester.run_backtest()
    assert len(tester.results) == 2
    assert data.portfolio.positions.keys() == {"LIVE"}
    assert data.portfolio.corporate_actions == []


def test_unheld_opening_distribution_is_not_reapplied_after_purchase():
    tester = _corporate_action_backtest()
    data = tester.model_data
    open_time = data.periods[1].execution_time
    data.cash_settlements = (
        replace(
            data.cash_settlements[0],
            effective_at=open_time,
            published_at=open_time,
            retain_original=True,
            unresolved_entitlements=("Warrant quotes",),
        ),
    )
    data.periods[1].market_data[(StockFields.CLOSE, "GBT")] = 68
    tester._process_trading_period(data.periods[1])
    assert tester.results[0]["total_value"] == pytest.approx(1000)
    assert data.portfolio.corporate_actions == []
