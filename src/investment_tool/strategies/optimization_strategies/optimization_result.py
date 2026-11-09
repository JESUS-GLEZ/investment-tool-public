"""Optimization result container.

Carries the Pareto front, raw archives, and optional Markowitz
benchmark from the optimizer to the investment strategy layer, where
portfolio selection is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from platypus import Solution


@dataclass
class OptimizationResult:
    """Output of a portfolio optimisation run (before selection).

    Parameters
    ----------
    front : list[Solution]
        Pruned non-dominated Pareto front.
    all_archives : list[list[Solution]]
        Raw archives from every restart.
    markowitz_front : list[Solution] | None
        Optional Markowitz efficient-frontier solutions.
    selected_solution : Solution | None
        The solution chosen by the selection strategy.  Set by the
        investment strategy layer after selection; ``None`` until then.
    """

    front: list[Solution]
    all_archives: list[list[Any]] = field(default_factory=list)
    markowitz_front: list[Solution] | None = None
    selected_solution: Solution | None = None
    search_diagnostics: dict[str, Any] = field(default_factory=dict)
    # A cached Standard MV curve has objective coordinates, not portfolio weights.
    reference_objectives: list[list[float]] | None = None


def find_solution_for_weights(
    front: list[Solution],
    weights: pd.Series,
) -> Solution | None:
    """Return the ``Solution`` in *front* whose variables best match *weights*.

    The selector returns a normalised ``pd.Series``; this function finds the
    closest solution in the Pareto front by L2 distance on the weight vector,
    so the selected portfolio can be highlighted on diagnostic plots.

    Parameters
    ----------
    front : list[Solution]
        Pareto-front solutions whose ``variables`` are raw (un-normalised)
        weights as returned by the optimiser.
    weights : pd.Series
        Normalised portfolio weights returned by a selection strategy.

    Returns
    -------
    Solution | None
        The closest matching solution, or ``None`` if *front* is empty.
    """
    if not front:
        return None
    target = weights.to_numpy(dtype=float)
    best_sol = None
    best_dist = float("inf")
    for sol in front:
        try:
            raw = np.asarray(sol.variables, dtype=float)
        except (TypeError, ValueError):
            continue
        if raw.shape != target.shape:
            continue
        total = raw.sum()
        normalised = raw / total if total > 0 else raw
        dist = float(np.linalg.norm(normalised - target))
        if dist < best_dist:
            best_dist = dist
            best_sol = sol
    return best_sol
