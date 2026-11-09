"""Select expected return within a volatility budget relative to Sharpe."""

from __future__ import annotations

import logging
import math

import pandas as pd
from platypus import Solution

from investment_tool.strategies.selection_strategies.base_selection_strategy import (
    BaseSelectionStrategy,
)
from investment_tool.strategies.selection_strategies.sharpe_selection import (
    SharpeSelection,
)

logger = logging.getLogger(__name__)


class ReturnWithRiskBudgetSelection(BaseSelectionStrategy):
    """Maximise return with volatility at most (1 + increase) times Sharpe's.

    Selection uses the same feasible current-period front for both portfolios.
    All configured balance and cardinality constraints therefore remain intact.
    The reference portfolio is always eligible; tied returns prefer lower risk.
    The budget describes predicted volatility, not a limit on realised losses.
    """

    def __init__(self, volatility_increase: float = 0.25) -> None:
        if not math.isfinite(volatility_increase) or not 0 <= volatility_increase <= 1:
            raise ValueError("Volatility increase must be between 0 and 1")
        self.volatility_increase = volatility_increase
        self.last_diagnostics: dict[str, float] = {}
        self.reference_weights: dict[str, float] = {}

    def select(self, front: list[Solution], asset_index: pd.Index) -> pd.Series:
        if not front:
            raise ValueError("Cannot select a portfolio from an empty front")
        if any(
            not math.isfinite(float(s.objectives[0]))
            or float(s.objectives[0]) < 0
            or not math.isfinite(float(s.objectives[1]))
            for s in front
        ):
            raise ValueError(
                "Portfolio objectives must have finite return and variance"
            )

        sharpe = SharpeSelection()
        reference = max(front, key=sharpe._sharpe)
        reference_series: pd.Series = self._normalise(reference, asset_index)
        self.reference_weights = {
            str(symbol): float(weight)
            for symbol, weight in reference_series.items()
            if weight > 0
        }
        reference_variance = float(reference.objectives[0])
        variance_budget = reference_variance * (1 + self.volatility_increase) ** 2
        eligible = [s for s in front if float(s.objectives[0]) <= variance_budget]
        best = max(
            eligible,
            key=lambda s: (-float(s.objectives[1]), -float(s.objectives[0])),
        )
        self.last_diagnostics = {
            "sharpe_expected_return": -float(reference.objectives[1]),
            "sharpe_volatility": math.sqrt(reference_variance),
            "volatility_budget": math.sqrt(variance_budget),
            "selected_expected_return": -float(best.objectives[1]),
            "selected_volatility": math.sqrt(float(best.objectives[0])),
            "front_min_expected_return": min(-float(s.objectives[1]) for s in front),
            "front_max_expected_return": max(-float(s.objectives[1]) for s in front),
        }
        self.last_diagnostics.update(self._balance_diagnostics(best))
        logger.info("Return within volatility budget: %s", self.last_diagnostics)
        result: pd.Series = self._normalise(best, asset_index)
        return result
