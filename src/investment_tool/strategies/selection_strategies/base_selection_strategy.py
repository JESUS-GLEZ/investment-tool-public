"""Abstract base class for portfolio selection strategies.

All concrete selectors must subclass :class:`BaseSelectionStrategy` and
implement :meth:`select`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from platypus import Solution

from investment_tool.constants import PortfolioBalance

logger = logging.getLogger(__name__)


class BaseSelectionStrategy(ABC):
    """Interface for selecting a single portfolio from a Pareto front.

    Subclasses implement :meth:`select`, which receives the non-dominated
    front and an asset index and returns a normalised weight vector.
    """

    @abstractmethod
    def select(
        self,
        front: list[Solution],
        asset_index: pd.Index,
    ) -> pd.Series:
        """Pick one portfolio from the Pareto front.

        Parameters
        ----------
        front : list[Solution]
            Non-dominated solutions with objectives
            ``[variance, -expected_return, ...]``.
        asset_index : pd.Index
            Asset symbols, used to index the returned Series.

        Returns
        -------
        pd.Series
            Normalised weights indexed by *asset_index*, summing to 1.
        """

    @staticmethod
    def _balance_diagnostics(solution: Solution) -> dict[str, float]:
        """Report the configured diversification quantity consistently."""
        balance = getattr(
            solution.problem, "balance", PortfolioBalance.RISK_CONTRIBUTIONS
        )
        diagnostics: dict[str, float] = {}
        if len(solution.objectives) > 2:
            key = (
                "selected_capital_weight_hhi"
                if balance == PortfolioBalance.CAPITAL_WEIGHTS
                else "selected_risk_contribution_hhi"
            )
            diagnostics[key] = float(solution.objectives[2])
        if balance == PortfolioBalance.HYBRID:
            w = np.asarray(solution.variables, dtype=float)
            active = w > solution.problem.activity_tol
            rc = solution.problem.balance_contributions(w)
            diagnostics.update(
                selected_capital_weight_hhi=float(w @ w),
                selected_risk_contribution_hhi=float(rc @ rc),
                selected_capital_weight_min=float(w[active].min()),
                selected_capital_weight_max=float(w[active].max()),
                selected_risk_contribution_min=float(rc[active].min()),
                selected_risk_contribution_max=float(rc[active].max()),
            )
        return diagnostics

    @staticmethod
    def _normalise(solution: Solution, asset_index: pd.Index) -> pd.Series:
        """Extract weights from a solution and normalise to sum = 1.

        Parameters
        ----------
        solution : Solution
            Platypus solution whose ``variables`` are portfolio weights.
        asset_index : pd.Index
            Asset symbols.

        Returns
        -------
        pd.Series
            Normalised weights.
        """
        weights = np.asarray(solution.variables, dtype=float)
        series = pd.Series(weights, index=asset_index)
        total = series.sum()
        if not np.isclose(total, 1.0, rtol=1e-5):
            logger.warning("Portfolio weights sum to %.6f — normalising", total)
            series = pd.Series(series / total)
        return series
