"""Sharpe-ratio portfolio selection strategy.

Selects the portfolio from the Pareto front that maximises the
Sharpe-like ratio ``expected_return / sqrt(variance)``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from platypus import Solution

from investment_tool.strategies.selection_strategies.base_selection_strategy import (
    BaseSelectionStrategy,
)

logger = logging.getLogger(__name__)


class SharpeSelection(BaseSelectionStrategy):
    """Select the highest Sharpe-like ratio portfolio.

    Parameters
    ----------
    eps : float, optional
        Floor for variance to avoid division by zero, by default 1e-12.
    """

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def select(
        self,
        front: list[Solution],
        asset_index: pd.Index,
    ) -> pd.Series:
        """Pick the portfolio maximising ``return / sqrt(variance)``.

        Parameters
        ----------
        front : list[Solution]
            Non-dominated solutions with objectives
            ``[variance, -expected_return, ...]``.
        asset_index : pd.Index
            Asset symbols.

        Returns
        -------
        pd.Series
            Normalised weights.
        """
        best = max(front, key=self._sharpe)
        logger.info(
            "Selected portfolio by Sharpe-like ratio: %.4f  (objectives=%s)",
            self._sharpe(best),
            best.objectives,
        )
        result: pd.Series = self._normalise(best, asset_index)
        return result

    def _sharpe(self, sol: Solution) -> float:
        """Compute the Sharpe-like ratio for a solution."""
        var = max(float(sol.objectives[0]), self.eps)
        ret = -float(sol.objectives[1])
        return float(ret / np.sqrt(var))
