"""Unit tests for strategy_comparison_coordinator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from investment_tool.app.strategy_comparison_coordinator import (
    StrategyComparisonParameters,
    compare_strategies,
)
from investment_tool.constants import DataSource, Frequency, StrategyType
from investment_tool.data.validation import ValidationLevel
from investment_tool.strategies.selection_strategies import (
    ReturnWithRiskBudgetSelection,
)


def test_budget_selector_is_only_applied_to_optimization_and_diagnostics_survive(
    tmp_path,
):
    with (
        patch(
            "investment_tool.app.strategy_comparison_coordinator.ModelDataFactory"
        ) as factory,
        patch(
            "investment_tool.app.strategy_comparison_coordinator._build_strategy"
        ) as build,
        patch(
            "investment_tool.app.strategy_comparison_coordinator.BacktestingService"
        ) as service,
        patch(
            "investment_tool.app.strategy_comparison_coordinator.StrategyComparator"
        ) as comparator,
    ):
        factory.create_model_data.return_value.stocks = [MagicMock(symbol="SPY")]
        factory.create_model_data.return_value.portfolio.history = []
        comparator.return_value.run.return_value = {"metrics_df": {}, "saved_files": []}
        backtester = MagicMock(
            results=[{"period_start": "2024-01-01", "total_value": 1000}]
        )

        def run_backtest(**kwargs):
            selector = build.call_args.kwargs.get("selection_strategy")
            if selector:
                selector.last_diagnostics = {"volatility_budget": 0.125}
            kwargs["period_complete_callback"](backtester)
            return backtester

        service.return_value.run_backtest.side_effect = run_backtest
        params = StrategyComparisonParameters(
            strategy_types=[StrategyType.OPTIMIZATION, StrategyType.COMBINED],
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=tmp_path,
            optimization_selection="return_with_risk_budget",
            optimization_volatility_increase=0.25,
        )
        partial = {}
        result = compare_strategies(params, partial_sink=partial)
        first = build.call_args_list[0].kwargs
        assert isinstance(first["selection_strategy"], ReturnWithRiskBudgetSelection)
        assert first["selection_strategy"].volatility_increase == 0.25
        assert "selection_strategy" not in build.call_args_list[1].kwargs
        assert "selection_strategy" not in build.call_args_list[2].kwargs
        expected = [
            {
                "period_start": "2024-01-01",
                "sharpe_weights": {},
                "volatility_budget": 0.125,
            }
        ]
        assert (
            result["individual_results"]["optimization"]["selection_diagnostics"]
            == expected
        )
        assert (
            partial["individual_results"]["optimization"]["selection_diagnostics"]
            == expected
        )


class TestStrategyComparisonParameters:
    """Test StrategyComparisonParameters container."""

    def test_defaults(self) -> None:
        params = StrategyComparisonParameters(
            strategy_types=[StrategyType.COMBINED],
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=Path("/tmp"),
        )
        assert params.benchmark_asset == "SPY"
        assert params.include_benchmark is True
        assert params.validation_level == ValidationLevel.WARN

    def test_custom_values(self) -> None:
        params = StrategyComparisonParameters(
            strategy_types=[StrategyType.OPTIMIZATION],
            data_source=DataSource.LOCAL,
            frequency=Frequency.WEEKLY,
            output_directory=Path("/tmp"),
            benchmark_asset="QQQ",
            include_benchmark=False,
        )
        assert params.benchmark_asset == "QQQ"
        assert params.include_benchmark is False
        assert params.frequency == Frequency.WEEKLY


class TestCompareStrategies:
    """Test compare_strategies orchestration (with mocked internals)."""

    @patch("investment_tool.app.strategy_comparison_coordinator.ModelDataFactory")
    @patch("investment_tool.app.strategy_comparison_coordinator._build_strategy")
    @patch("investment_tool.app.strategy_comparison_coordinator.BacktestingService")
    @patch("investment_tool.app.strategy_comparison_coordinator.StrategyComparator")
    def test_runs_all_strategies(
        self,
        mock_comparator_cls: MagicMock,
        mock_bt_service_cls: MagicMock,
        mock_build_strategy: MagicMock,
        mock_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Wire mocks — stocks need .symbol for the benchmark pre-check
        mock_stock_spy = MagicMock()
        mock_stock_spy.symbol = "SPY"
        mock_stock_aapl = MagicMock()
        mock_stock_aapl.symbol = "AAPL"
        mock_model_data = MagicMock()
        mock_model_data.stocks = [mock_stock_spy, mock_stock_aapl]
        mock_factory.create_model_data.return_value = mock_model_data
        mock_strategy = MagicMock()
        mock_build_strategy.return_value = mock_strategy

        mock_backtester = MagicMock()
        mock_backtester.results = [
            {"date": "2024-01-15", "total_value": 10000.0, "return": 0.0}
        ]
        mock_bt_service = MagicMock()
        mock_bt_service.run_backtest.return_value = mock_backtester
        mock_bt_service_cls.return_value = mock_bt_service

        mock_comp_instance = MagicMock()
        mock_comp_instance.run.return_value = {
            "metrics_df": MagicMock(),
            "saved_files": [],
            "output_dir": tmp_path,
        }
        mock_comparator_cls.return_value = mock_comp_instance

        params = StrategyComparisonParameters(
            strategy_types=[StrategyType.COMBINED, StrategyType.OPTIMIZATION],
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=tmp_path,
            include_benchmark=True,
        )

        result = compare_strategies(params)

        # Should build 3 strategies: combined, optimization, buy_and_hold
        assert mock_build_strategy.call_count == 3
        assert mock_bt_service_cls.call_count == 3
        assert mock_comparator_cls.call_count == 1
        assert "metrics_df" in result

    @patch("investment_tool.app.strategy_comparison_coordinator.ModelDataFactory")
    @patch("investment_tool.app.strategy_comparison_coordinator._build_strategy")
    @patch("investment_tool.app.strategy_comparison_coordinator.BacktestingService")
    @patch("investment_tool.app.strategy_comparison_coordinator.StrategyComparator")
    def test_no_benchmark_when_disabled(
        self,
        mock_comparator_cls: MagicMock,
        mock_bt_service_cls: MagicMock,
        mock_build_strategy: MagicMock,
        mock_factory: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_stock = MagicMock()
        mock_stock.symbol = "AAPL"
        mock_model_data = MagicMock()
        mock_model_data.stocks = [mock_stock]
        mock_factory.create_model_data.return_value = mock_model_data
        mock_build_strategy.return_value = MagicMock()

        mock_backtester = MagicMock()
        mock_backtester.results = [
            {"date": "2024-01-15", "total_value": 10000.0, "return": 0.0}
        ]
        mock_bt_service = MagicMock()
        mock_bt_service.run_backtest.return_value = mock_backtester
        mock_bt_service_cls.return_value = mock_bt_service

        mock_comp = MagicMock()
        mock_comp.run.return_value = {
            "metrics_df": MagicMock(),
            "saved_files": [],
            "output_dir": tmp_path,
        }
        mock_comparator_cls.return_value = mock_comp

        params = StrategyComparisonParameters(
            strategy_types=[StrategyType.COMBINED],
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=tmp_path,
            include_benchmark=False,
        )

        compare_strategies(params)

        # Only 1 strategy (no benchmark added)
        assert mock_build_strategy.call_count == 1
