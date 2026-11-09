"""Solver comparison framework for portfolio optimisation.

Provides tools to benchmark multiple evolutionary algorithms on the
same portfolio problem, compute quality metrics, and generate
comparison reports.
"""

from .solver_benchmark import SolverBenchmark
from .solver_config import SolverConfig
from .solver_result import SolverResult

__all__ = [
    "SolverBenchmark",
    "SolverConfig",
    "SolverResult",
]
