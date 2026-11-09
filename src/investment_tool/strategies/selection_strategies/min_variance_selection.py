"""Minimum-variance portfolio selection strategy.

Selects the portfolio from the Pareto front with the lowest variance.
"""

from __future__ import annotations

import logging

import pandas as pd
from platypus import Solution

from investment_tool.strategies.selection_strategies.base_selection_strategy import (
    BaseSelectionStrategy,
)

logger = logging.getLogger(__name__)


class MinVarianceSelection(BaseSelectionStrategy):
    """Select the minimum-variance portfolio from the Pareto front."""

    def select(
        self,
        front: list[Solution],
        asset_index: pd.Index,
    ) -> pd.Series:
        """Pick the portfolio with the lowest variance.

        Parameters
        ----------
        front : list[Solution]
            Non-dominated solutions.
        asset_index : pd.Index
            Asset symbols.

        Returns
        -------
        pd.Series
            Normalised weights.
        """
        best = min(front, key=lambda s: float(s.objectives[0]))
        logger.info(
            "Selected minimum-variance portfolio (var=%.6f, objectives=%s)",
            float(best.objectives[0]),
            best.objectives,
        )
        result: pd.Series = self._normalise(best, asset_index)
        return result
