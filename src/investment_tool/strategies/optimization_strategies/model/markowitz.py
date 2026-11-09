"""Markowitz long-only portfolio model for Platypus.

Implements a 2-objective (variance, -return) unconstrained optimisation
where feasibility is enforced by projecting weights onto the probability
simplex inside ``evaluate``.
"""

from __future__ import annotations

import numpy as np
from platypus import Problem, Real, Solution
from investment_tool.domain.trading_costs import TradingCosts, expected_net_return


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Project *v* onto the probability simplex.

    Uses the efficient O(n log n) algorithm from Duchi et al. (2008).

    Parameters
    ----------
    v : np.ndarray
        Arbitrary real vector.

    Returns
    -------
    np.ndarray
        Projected vector satisfying ``w_i >= 0`` and ``sum(w) == 1``.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n == 0:
        return v

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.full(n, 1.0 / n)
    rho_max = rho[-1]
    theta = (cssv[rho_max] - 1) / (rho_max + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    if s <= 0:
        return np.full(n, 1.0 / n)
    result: np.ndarray = w / s
    return result


class MarkowitzLongOnly(Problem):
    """Markowitz portfolio selection (long-only, fully-invested).

    Decision variables
        ``w_i ∈ [0, 1]``, ``i = 1..N`` (asset weights).

    Feasibility
        Weights are projected onto the simplex inside :meth:`evaluate`.

    Objectives (both minimised)
        1. ``variance(w) = w' Σ w``
        2. ``-expected_return(w) = -μ' w``

    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector, shape ``(N,)``.
    cov : np.ndarray
        Covariance matrix, shape ``(N, N)``, symmetric PSD.

    Raises
    ------
    ValueError
        If ``mu`` is not 1-D, ``cov`` is not square or symmetric, or
        dimensions do not match.
    """

    def __init__(
        self, mu: np.ndarray, cov: np.ndarray, trading_costs: TradingCosts | None = None
    ) -> None:
        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if mu.ndim != 1:
            raise ValueError("mu must be a 1-D array of expected returns")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != mu.size:
            raise ValueError("cov must be a square matrix matching len(mu)")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("covariance matrix must be symmetric")

        n_assets = mu.size
        super().__init__(nvars=n_assets, nobjs=2)
        for i in range(n_assets):
            self.types[i] = Real(0.0, 1.0)

        self.mu = mu
        if (
            trading_costs is not None
            and trading_costs.current_weights.shape != mu.shape
        ):
            raise ValueError("Trading-cost weights must align with expected returns")
        self.trading_costs = trading_costs
        self.cov = cov

    def evaluate(self, solution: Solution) -> None:
        """Evaluate objectives: variance and negative expected return."""
        w = _project_to_simplex(np.asarray(solution.variables, dtype=float))
        solution.variables[:] = w.tolist()

        variance = float(w @ self.cov @ w)
        exp_ret = expected_net_return(self.mu, w, self.trading_costs)
        solution.objectives[:] = [variance, -exp_ret]

    def random(self) -> Solution:
        """Sample a feasible portfolio uniformly on the simplex."""
        sol = Solution(self)
        w = np.random.rand(self.nvars)
        w = w / w.sum()
        sol.variables[:] = w.tolist()
        self.evaluate(sol)
        return sol
