"""Risk-aversion utility portfolio selection strategy.

Selects the portfolio from the Pareto front that maximises the
classical mean-variance utility ``U = μ - (λ/2) σ²``.
"""

from __future__ import annotations

import logging

import pandas as pd
from platypus import Solution

from investment_tool.strategies.selection_strategies.base_selection_strategy import (
    BaseSelectionStrategy,
)

logger = logging.getLogger(__name__)


class RiskAversionSelection(BaseSelectionStrategy):
    """Select the portfolio maximising risk-adjusted utility.

    Utility ``U(w) = expected_return - (risk_aversion / 2) * variance``.
    Higher *risk_aversion* biases selection towards lower-variance
    portfolios.

    Parameters
    ----------
    risk_aversion : float, optional
        Risk aversion coefficient (≥ 0).  Default 0.5.
    """

    def __init__(self, risk_aversion: float = 0.5) -> None:
        self.risk_aversion = risk_aversion

    def select(
        self,
        front: list[Solution],
        asset_index: pd.Index,
    ) -> pd.Series:
        """Pick the portfolio maximising mean-variance utility.

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
        best = max(front, key=self._utility)
        logger.info(
            "Selected portfolio by risk-aversion utility (λ=%.2f): U=%.6f  "
            "(objectives=%s)",
            self.risk_aversion,
            self._utility(best),
            best.objectives,
        )
        result: pd.Series = self._normalise(best, asset_index)
        return result

    def _utility(self, sol: Solution) -> float:
        """Compute mean-variance utility for a solution."""
        var = float(sol.objectives[0])
        ret = -float(sol.objectives[1])
        return ret - (self.risk_aversion / 2.0) * var
