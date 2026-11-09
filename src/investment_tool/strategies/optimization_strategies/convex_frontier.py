"""Convex long-only mean–variance reference, without BPRC constraints."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from platypus import Solution
from investment_tool.domain.trading_costs import TradingCosts, expected_net_return

from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.optimization_strategies.utils import merge_nondominated


def convex_markowitz_front(
    mu: np.ndarray,
    cov: np.ndarray,
    n_points: int = 100,
    trading_costs: TradingCosts | None = None,
) -> list[Solution]:
    """Solve minimum variance at target returns from GMV to maximum return.

    Scaling the quadratic objective improves numerical accuracy with monthly or
    daily covariances. Fail explicitly rather than substitute a heuristic curve.
    """
    mu, cov = np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)
    if n_points < 2:
        raise ValueError("At least two reference points are required")
    model = MarkowitzLongOnly(mu, cov, trading_costs=trading_costs)
    w = cp.Variable(len(mu))
    scale = max(float(np.max(np.diag(cov))), 1e-12)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov / scale)))
    base = [w >= 0, cp.sum(w) == 1]
    net_return = mu @ w
    if trading_costs is not None:
        # On the simplex, buying from cash has constant turnover. Avoid a
        # redundant absolute-value epigraph at the maximum-return endpoint.
        turnover = (
            1.0
            if not np.any(trading_costs.current_weights)
            else cp.norm1(w - trading_costs.current_weights)
        )
        net_return = net_return - trading_costs.rate * (
            turnover + trading_costs.outside_weight
        )

    def solve(problem: cp.Problem) -> np.ndarray:
        problem.solve(
            solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10
        )
        if problem.status != cp.OPTIMAL or w.value is None:
            raise RuntimeError(f"MV reference solve failed: {problem.status}")
        weights: np.ndarray = np.maximum(np.asarray(w.value, dtype=float), 0)
        normalized: np.ndarray = weights / weights.sum()
        return normalized

    gmv = solve(cp.Problem(objective, base))
    maximum_return = (
        float(mu.max())
        if trading_costs is None
        else expected_net_return(
            mu, solve(cp.Problem(cp.Maximize(net_return), base)), trading_costs
        )
    )
    target = cp.Parameter()
    problem = cp.Problem(objective, [*base, net_return >= target])
    solutions: list[Solution] = []
    for expected_return in np.linspace(
        expected_net_return(mu, gmv, trading_costs), maximum_return, n_points
    ):
        target.value = expected_return
        weights = solve(problem)
        if expected_net_return(mu, weights, trading_costs) < expected_return - 1e-7:
            raise RuntimeError("MV reference target-return tolerance exceeded")
        solution = Solution(model)
        solution.variables[:] = weights.tolist()
        solution.evaluate()
        solutions.append(solution)
    front: list[Solution] = merge_nondominated([solutions], target_size=n_points)
    return front
