"""Abstract base class for portfolio optimization strategies.

All concrete optimization strategies must subclass
:class:`BaseOptimizationStrategy` and implement :meth:`optimize`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from investment_tool.output.policy import OutputMode
from collections.abc import Callable
from investment_tool.domain.trading_costs import TradingCosts

from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)

#: Signature for progress callbacks.
#:
#: ``fraction`` is a float in [0, 1] indicating overall completion.
#: ``message`` is a short human-readable status string.
ProgressCallback = Callable[[float, str], None]


class BaseOptimizationStrategy(ABC):
    """Interface for portfolio optimization.

    Subclasses receive expected returns and a covariance matrix at
    construction time and produce an :class:`OptimizationResult`
    (Pareto front + archives) via :meth:`optimize`.
    """

    output_mode = OutputMode.DIAGNOSTIC
    trading_costs: TradingCosts | None = None

    @property
    def cost_kwargs(self) -> dict[str, TradingCosts]:
        return (
            {"trading_costs": self.trading_costs}
            if self.trading_costs is not None
            else {}
        )

    @abstractmethod
    def optimize(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """Run the optimization and return the result.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Called periodically with ``(fraction_complete, message)``.
            Callers (e.g. a background job worker) can use this to
            report progress to the user.

        Returns
        -------
        OptimizationResult
            Pareto front, archives, and optional Markowitz frontier.
        """
