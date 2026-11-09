"""
Model data container for execution state.

This module defines the ModelData class which extends InputData to provide
portfolio update capabilities and trading period advancement.
"""

from __future__ import annotations

import logging

import pandas as pd

from investment_tool.domain.model_data.input_data import InputData

logger = logging.getLogger(__name__)


class ModelData(InputData):
    """Execution model data with portfolio update capabilities.

    Extends InputData to add portfolio rebalancing and trading period
    management during backtesting and live execution.
    """

    def __init__(self) -> None:
        """Initialize ModelData with empty state."""
        super().__init__()

    def update(
        self, new_weights: pd.Series, *, execution_prices: pd.Series | None = None
    ) -> None:
        """Update portfolio weights and advance to next trading period.

        Parameters
        ----------
        new_weights : pd.Series
            Target portfolio weights indexed by asset symbol.
        execution_prices : pd.Series | None
            Optional explicit rebalance prices supplied by the backtester.

        Notes
        -----
        After updating the portfolio, this method automatically advances
        the current trading period to the next one if available.
        """
        if execution_prices is None:
            self.portfolio.update(new_weights)
        else:
            self.portfolio.update(new_weights, execution_prices=execution_prices)

        current_period = self.instance.current_trading_period
        if current_period is not None:
            logger.info(
                f"Portfolio successfully updated on {current_period.trading_date}."
            )
            if self.portfolio.history:
                logger.info(
                    f"Portfolio value was {self.portfolio.history[-1]['pre_rebalance_value']:.2f}"
                )
                logger.info(
                    f"Portfolio value is {self.portfolio.history[-1]['post_rebalance_value']:.2f}"
                )

            # Advance to next period
            current_period_index = self.periods.index(current_period)
            if current_period_index + 1 < len(self.periods):
                self.instance.previous_trading_period = self.periods[
                    current_period_index
                ]
                self.instance.current_trading_period = self.periods[
                    current_period_index + 1
                ]
