"""Quality metrics for multi-objective Pareto front comparison.

All metrics assume minimisation objectives where
``objectives = [variance, -expected_return, ...]``.
"""

from __future__ import annotations

import logging

import numpy as np
from platypus import Solution

from investment_tool.strategies.optimization_strategies.utils import sharpe_like

logger = logging.getLogger(__name__)


def hypervolume(front: list[Solution], ref_point: np.ndarray) -> float:
    """Compute the 2-D hypervolume indicator.

    Uses the first two objectives (variance, -return).  Solutions whose
    objectives exceed the reference point in *any* dimension are excluded.

    Parameters
    ----------
    front : list[Solution]
        Non-dominated solutions.
    ref_point : np.ndarray
        Reference point ``[max_variance, max_neg_return]``.
        Must dominate all solutions for a meaningful result.

    Returns
    -------
    float
        Hypervolume value (higher is better).
    """
    if not front:
        return 0.0

    points = np.array([[s.objectives[0], s.objectives[1]] for s in front])

    # Filter out points that exceed the reference in any dimension
    mask = np.all(points < ref_point, axis=1)
    points = points[mask]
    if len(points) == 0:
        return 0.0

    # Sort by first objective ascending
    order = np.argsort(points[:, 0])
    points = points[order]

    # Sweep-line 2-D hypervolume
    hv = 0.0
    prev_y = ref_point[1]
    for i in range(len(points)):
        x_width = ref_point[0] - points[i, 0]
        y_height = prev_y - points[i, 1]
        if x_width > 0 and y_height > 0:
            hv += x_width * y_height
        prev_y = min(prev_y, points[i, 1])

    return float(hv)


def spacing(front: list[Solution]) -> float:
    """Compute the spacing metric (uniformity of Pareto front).

    Measures the standard deviation of consecutive Euclidean distances
    along the front sorted by the first objective.  Lower is better
    (more uniform distribution).

    Parameters
    ----------
    front : list[Solution]
        Non-dominated solutions.

    Returns
    -------
    float
        Spacing value (lower is better).  Returns 0 if fewer than 2 solutions.
    """
    if len(front) < 2:
        return 0.0

    points = np.array([[s.objectives[0], s.objectives[1]] for s in front])
    order = np.argsort(points[:, 0])
    points = points[order]

    dists = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    return float(np.std(dists)) if len(dists) > 0 else 0.0


def spread(front: list[Solution]) -> float:
    """Compute the spread (extent) of the Pareto front.

    Measures the Euclidean distance between the extreme points of the
    front.  Higher is better (wider range of trade-offs).

    Parameters
    ----------
    front : list[Solution]
        Non-dominated solutions.

    Returns
    -------
    float
        Spread value (higher is better).
    """
    if len(front) < 2:
        return 0.0

    points = np.array([[s.objectives[0], s.objectives[1]] for s in front])
    order = np.argsort(points[:, 0])
    points = points[order]

    return float(np.linalg.norm(points[-1] - points[0]))


def front_cardinality(front: list[Solution]) -> int:
    """Count the number of non-dominated solutions.

    Parameters
    ----------
    front : list[Solution]
        Non-dominated solutions.

    Returns
    -------
    int
        Number of solutions on the front.
    """
    return len(front)


def best_sharpe(front: list[Solution]) -> float:
    """Maximum Sharpe-like ratio on the Pareto front.

    Parameters
    ----------
    front : list[Solution]
        Non-dominated solutions.

    Returns
    -------
    float
        Best Sharpe-like ratio.  Returns ``-inf`` for empty fronts.
    """
    if not front:
        return float("-inf")
    return float(max(sharpe_like(s) for s in front))


def median_sharpe(front: list[Solution]) -> float:
    """Median Sharpe-like ratio on the Pareto front.

    A robustness indicator: a solver whose *median* portfolio is good
    provides more reliable options than one with only one lucky outlier.

    Parameters
    ----------
    front : list[Solution]
        Non-dominated solutions.

    Returns
    -------
    float
        Median Sharpe-like ratio.  Returns 0 for empty fronts.
    """
    if not front:
        return 0.0
    values = [sharpe_like(s) for s in front]
    return float(np.median(values))


def compute_all_metrics(
    front: list[Solution],
    ref_point: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all quality metrics for a Pareto front.

    Parameters
    ----------
    front : list[Solution]
        Non-dominated solutions.
    ref_point : np.ndarray | None, optional
        Reference point for hypervolume.  If ``None``, derived from the
        front extremes with a 10% margin.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``hypervolume``, ``spacing``, ``spread``,
        ``cardinality``, ``best_sharpe``, ``median_sharpe``.
    """
    if ref_point is None and front:
        points = np.array([[s.objectives[0], s.objectives[1]] for s in front])
        margin = 0.1 * (points.max(axis=0) - points.min(axis=0))
        margin = np.maximum(margin, 1e-6)
        ref_point = points.max(axis=0) + margin

    rp = ref_point if ref_point is not None else np.array([1.0, 1.0])

    return {
        "hypervolume": hypervolume(front, rp),
        "spacing": spacing(front),
        "spread": spread(front),
        "cardinality": float(front_cardinality(front)),
        "best_sharpe": best_sharpe(front),
        "median_sharpe": median_sharpe(front),
    }
