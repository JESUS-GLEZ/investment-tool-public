"""Portfolio selection strategies.

Provides strategy classes that select a single portfolio from a Pareto
front produced by a multi-objective optimiser.

Selection is a distinct concern from optimisation: the optimiser finds
the efficient set, the selector applies an investor-specific decision
policy to pick one allocation from that set.
"""

from .base_selection_strategy import BaseSelectionStrategy
from .min_variance_selection import MinVarianceSelection
from .risk_aversion_selection import RiskAversionSelection
from .sharpe_selection import SharpeSelection
from .return_with_risk_budget_selection import ReturnWithRiskBudgetSelection
from .return_target_selection import ReturnTargetSelection
