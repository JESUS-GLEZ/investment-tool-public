"""Balanced Portfolios in Risk Contributions (BPRC) model for Platypus.

Implements a 3-objective constrained portfolio problem that simultaneously
minimises variance, maximises expected return, and minimises the Herfindahl
index of normalised risk contributions — promoting diversification.

Optionally, the HHI objective can be converted to an epsilon-constraint
(``hhi_as_constraint=True``), reducing the problem to 2 objectives
(variance, -return) plus 3 constraints — which is easier for most
evolutionary algorithms to solve consistently.
"""

from __future__ import annotations

import numpy as np
from platypus import Problem, Real, Solution

from investment_tool.constants import PortfolioBalance
from investment_tool.domain.trading_costs import TradingCosts, expected_net_return
from investment_tool.defaults import CONSTRUCTION, OPTIMIZER
from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    repair_portfolio,
)
from investment_tool.strategies.optimization_strategies.utils import (
    normalized_risk_contributions,
)


class BalancedPortfoliosRiskContributions(Problem):
    """Shared balanced-portfolio model; risk contributions remain the default.

    ``balance=capital_weights`` applies the same bounds and HHI objective to
    capital allocations instead, without requiring equal weights or risk parity.
    ``balance=hybrid`` bounds both quantities and retains risk-contribution HHI
    as the third objective (or HHI constraint in two-objective mode).

    Decision variables
        ``w_i ∈ [0, 1]``, ``i = 1..N``.  Projected onto the simplex
        (long-only, fully-invested) inside :meth:`evaluate`.

    **3-objective mode** (default, ``hhi_as_constraint=False``):

    Objectives (all minimised)
        1. ``variance(w) = w' Σ w``
        2. ``-expected_return(w) = -μ' w``
        3. ``HHI(g(w)) = Σ_i g_i(w)²`` — Herfindahl index of
           normalised risk contributions.

    Constraints (Platypus ``<= 0`` convention)
        For every active asset (``w_i > tol``):
        * ``1/u_assets - slack <= g_i(w)``
        * ``g_i(w) <= 1/l_assets + slack``

    **2-objective mode** (``hhi_as_constraint=True``):

    Objectives (all minimised)
        1. ``variance(w)``
        2. ``-expected_return(w)``

    Constraints
        Same risk-contribution bounds plus ``HHI(g(w)) <= hhi_bound``.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector, shape ``(N,)``.
    cov : np.ndarray
        Covariance matrix, shape ``(N, N)``, symmetric PSD.
    l_assets : int
        Lower bound on the number of active assets (used for the
        upper risk-contribution constraint ``1/l``).
    u_assets : int
        Upper bound on the number of active assets (used for the
        lower risk-contribution constraint ``1/u``).
    activity_tol : float, optional
        Threshold to consider a weight active, by default 1e-4.
    constraint_slack : float, optional
        Fractional relaxation applied to risk-contribution bounds,
        by default 0.1.  The bounds become
        ``[1/u - slack/u, 1/l + slack/l]``.
    hhi_as_constraint : bool, optional
        When ``True``, HHI is moved from an objective to a constraint,
        reducing the problem from 3 to 2 objectives.  Default ``False``.
    hhi_bound : float, optional
        Upper bound for HHI when used as a constraint, by default 0.5.

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes or invalid bounds.
    """

    def __init__(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        l_assets: int,
        u_assets: int,
        activity_tol: float = OPTIMIZER.activity_tol,
        constraint_slack: float = OPTIMIZER.constraint_slack,
        hhi_as_constraint: bool = CONSTRUCTION.hhi_as_constraint,
        hhi_bound: float = CONSTRUCTION.hhi_bound,
        balance: PortfolioBalance = PortfolioBalance.RISK_CONTRIBUTIONS,
        trading_costs: TradingCosts | None = None,
    ):
        mu = np.asarray(mu, dtype=float)
        cov = np.asarray(cov, dtype=float)

        if mu.ndim != 1:
            raise ValueError("mu must be a 1-D array")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] != mu.size:
            raise ValueError("cov must be square with size matching len(mu)")
        if not np.allclose(cov, cov.T, atol=1e-10):
            raise ValueError("covariance matrix must be symmetric")
        if not (1 <= l_assets <= u_assets <= mu.size):
            raise ValueError("Require 1 <= l_assets <= u_assets <= N")

        self.balance = PortfolioBalance(balance)
        n_assets = mu.size
        self._hhi_as_constraint = hhi_as_constraint
        self._hhi_bound = float(hhi_bound)
        self._constraint_slack = float(constraint_slack)

        if hhi_as_constraint:
            # 2 objectives, 3 constraints (rc_lower, rc_upper, hhi)
            super().__init__(n_assets, 2, 3)
        else:
            # 3 objectives, 2 constraints (rc_lower, rc_upper)
            super().__init__(n_assets, 3, 2)

        for i in range(n_assets):
            self.types[i] = Real(0.0, 1.0)
        self.constraints[:] = "<=0"

        self.mu = mu
        if (
            trading_costs is not None
            and trading_costs.current_weights.shape != mu.shape
        ):
            raise ValueError("Trading-cost weights must align with expected returns")
        self.trading_costs = trading_costs
        self.cov = cov
        self.l_assets = int(l_assets)
        self.u_assets = int(u_assets)
        self.activity_tol = float(activity_tol)

    # ---------- math helpers ----------

    def _normalized_risk_contributions(self, w: np.ndarray) -> np.ndarray:
        """Compute normalised risk contributions ``g_i(w)``.

        Delegates to :func:`~..utils.normalized_risk_contributions`.
        """
        result: np.ndarray = normalized_risk_contributions(
            w, self.cov, self.activity_tol
        )
        return result

    def balance_contributions(self, w: np.ndarray) -> np.ndarray:
        """The configured diversification quantity, sharing one evaluator."""
        if self.balance == PortfolioBalance.CAPITAL_WEIGHTS:
            return w.copy()
        return self._normalized_risk_contributions(w)

    @property
    def balance_bounds(self) -> tuple[float, float]:
        """Shared per-active-asset bounds, including the configured slack."""
        return (
            1.0 / self.u_assets - self._constraint_slack / self.u_assets,
            1.0 / self.l_assets + self._constraint_slack / self.l_assets,
        )

    # ---------- Platypus API ----------

    def evaluate(self, solution: Solution) -> None:
        """Evaluate objectives and constraints for a candidate portfolio."""
        # Repair / project to simplex (long-only, sum=1)
        w = np.asarray(solution.variables, dtype=float)
        w = repair_portfolio(w, dust_tol=self.activity_tol)  # aligned with activity_tol
        solution.variables[:] = w.tolist()

        # Objectives
        Sw = self.cov @ w
        variance = float(w @ Sw)
        exp_return = expected_net_return(self.mu, w, self.trading_costs)
        gi = self.balance_contributions(w)
        diversity_hhi = float(np.sum(gi**2))

        if self._hhi_as_constraint:
            # 2-objective mode: HHI becomes a constraint
            solution.objectives[:] = [variance, -exp_return]
        else:
            # 3-objective mode (original)
            solution.objectives[:] = [variance, -exp_return, diversity_hhi]

        # Risk-contribution bounds with slack relaxation
        lower, upper = self.balance_bounds
        tol = 1e-12

        active = w > self.activity_tol
        k = int(active.sum())

        if k == 0:
            rc_lower_violation = 1.0
            rc_upper_violation = 1.0
        else:
            gi_active = gi[active]
            if self.balance == PortfolioBalance.HYBRID:
                # Max violations aggregate both sets of bounds without adding
                # an objective or allowing one quantity to offset the other.
                gi_active = np.concatenate((gi_active, w[active]))
            rc_lower_violation = (
                float(np.max(np.maximum(lower - gi_active - tol, 0.0)))
                if gi_active.size
                else 0.0
            )
            rc_upper_violation = (
                float(np.max(np.maximum(gi_active - upper - tol, 0.0)))
                if gi_active.size
                else 0.0
            )

        rc_lower_violation = max(rc_lower_violation, float(max(0, self.l_assets - k)))
        rc_upper_violation = max(rc_upper_violation, float(max(0, k - self.u_assets)))

        if self._hhi_as_constraint:
            # 3 constraints: rc_lower, rc_upper, hhi_bound
            hhi_violation = max(0.0, diversity_hhi - self._hhi_bound)
            solution.constraints[0] = rc_lower_violation
            solution.constraints[1] = rc_upper_violation
            solution.constraints[2] = hhi_violation
        else:
            # 2 constraints: rc_lower, rc_upper
            solution.constraints[0] = rc_lower_violation
            solution.constraints[1] = rc_upper_violation

    def random(self) -> Solution:
        """Sample a feasible-ish portfolio with random cardinality.

        Chooses ``k ~ Uniform{l_assets, ..., u_assets}`` active assets,
        assigns random weights, projects to the simplex, and evaluates.

        Returns
        -------
        Solution
            Evaluated portfolio solution.
        """
        n = self.nvars
        sol = Solution(self)
        k = np.random.randint(self.l_assets, self.u_assets + 1)
        idx = np.random.choice(n, size=k, replace=False)
        x = np.zeros(n)
        x[idx] = np.random.rand(k)
        x = repair_portfolio(x, dust_tol=self.activity_tol)
        sol.variables[:] = x.tolist()
        self.evaluate(sol)
        return sol
