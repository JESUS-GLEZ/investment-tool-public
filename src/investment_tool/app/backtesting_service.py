"""
Backtesting service for the Investment Tool.

This module provides a service layer for running backtests, coordinating
between the model data, investment strategy, and backtester components.
"""

from __future__ import annotations

import logging
from typing import Callable

from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)
from investment_tool.domain.model_data.model_data import ModelData

logger = logging.getLogger(__name__)


class BacktestingService:
    """Service for executing backtests on historical data.

    Coordinates the backtesting process by combining model data with
    an investment strategy and running the simulation.

    Parameters
    ----------
    model_data : ModelData
        Historical data including stocks, periods, and portfolio.
    investment_strategy : BaseInvestmentStrategy
        Strategy to apply at each rebalancing period.

    Attributes
    ----------
    model_data : ModelData
        The model data used for backtesting.
    investment_strategy : BaseInvestmentStrategy
        The investment strategy being tested.
    backtester : Backtester | None
        The backtester instance, created after calling run_backtest().
    """

    def __init__(
        self, model_data: ModelData, investment_strategy: BaseInvestmentStrategy
    ) -> None:
        self.model_data = model_data
        self.investment_strategy = investment_strategy
        self.backtester: Backtester | None = None

    def run_backtest(
        self,
        progress_callback: ProgressCallback | None = None,
        period_complete_callback: Callable[[Backtester], None] | None = None,
    ) -> Backtester:
        """Execute the backtest and return the backtester instance.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Called periodically with ``(fraction_complete, message)``.
            Threaded to the backtester.

        Returns
        -------
        Backtester
            Backtester instance with results after running the simulation.

        Raises
        ------
        RuntimeError
            If an error occurs during the backtest execution.
        """
        try:
            self.backtester = Backtester(self.model_data, self.investment_strategy)
            self.backtester.run_backtest(
                progress_callback=progress_callback,
                period_complete_callback=period_complete_callback,
            )
            return self.backtester
        except Exception as e:
            logger.error(f"An error occurred during backtesting: {e}")
            raise
