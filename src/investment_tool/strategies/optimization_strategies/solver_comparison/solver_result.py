"""Solver result container.

Holds the Pareto front, raw archives, timing, quality metrics, and
best portfolio produced by a single solver run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from platypus import Solution

from investment_tool.strategies.optimization_strategies.solver_comparison.solver_config import (
    SolverConfig,
)


@dataclass
class SolverResult:
    """Output of running one solver on one problem instance.

    Parameters
    ----------
    config : SolverConfig
        The solver configuration that produced this result.
    front : list[Solution]
        Pruned non-dominated set.
    all_archives : list[list[Solution]]
        Raw archives from every restart.
    wall_time_seconds : float
        Total wall-clock time in seconds.
    metrics : dict[str, float]
        Quality metrics (hypervolume, spacing, best_sharpe, etc.).
    best_weights : pd.Series
        Best portfolio weights by Sharpe-like ratio.
    """

    config: SolverConfig
    front: list[Solution]
    all_archives: list[list[Solution]]
    wall_time_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)
    best_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
