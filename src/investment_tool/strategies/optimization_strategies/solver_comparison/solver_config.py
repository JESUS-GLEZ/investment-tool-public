"""Solver configuration dataclass.

Describes a single solver to benchmark: algorithm class, hyperparameters,
and computational budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from platypus import AbstractGeneticAlgorithm


@dataclass(frozen=True)
class SolverConfig:
    """Immutable description of one solver to benchmark.

    Parameters
    ----------
    label : str
        Human-readable name (e.g. ``"GWASFGA"``, ``"NSGA-II"``).
    algorithm : type[AbstractGeneticAlgorithm]
        Platypus algorithm class.
    algorithm_kwargs : dict[str, Any]
        Extra keyword arguments passed to the algorithm constructor
        (e.g. ``{"variator": PortfolioVariator(...)}``).
    population_size : int
        GA population size.
    n_generations : int
        Number of function evaluations per restart.
    n_restarts : int
        Number of independent multi-start runs.
    """

    label: str
    algorithm: type[AbstractGeneticAlgorithm]
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)
    population_size: int = 100
    n_generations: int = 10_000
    n_restarts: int = 3
    random_seed: int | None = None
