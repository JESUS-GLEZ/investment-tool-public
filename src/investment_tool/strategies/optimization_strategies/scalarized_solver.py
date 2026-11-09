"""Scalarized subproblem solver for BPRC portfolio optimisation.

Wraps :func:`scipy.optimize.minimize` (SLSQP) to solve a single
weighted-sum or augmented Tchebycheff scalarization of the BPRC
objectives (variance, −return, HHI) subject to simplex and
risk-contribution constraints.

This provides **exact local convergence** on each subproblem — the
key advantage over purely evolutionary approaches that struggle with
constraint handling and fine-grained convergence.
"""

from __future__ import annotations

import logging
from enum import Enum, unique
from typing import Any

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from investment_tool.defaults import CONSTRUCTION, DECOMPOSITION, OPTIMIZER
from investment_tool.domain.trading_costs import TradingCosts, expected_net_return
from investment_tool.strategies.optimization_strategies.utils import (
    normalized_risk_contributions,
)

logger = logging.getLogger(__name__)


@unique
class ScalarizationMethod(str, Enum):
    """Scalarization method for multi-objective decomposition.

    Members
    -------
    WEIGHTED_SUM
        Linear weighted sum of normalised objectives.  Fast and
        effective for convex Pareto fronts but cannot recover solutions
        in non-convex regions.
    AUGMENTED_TCHEBYCHEFF
        Augmented Tchebycheff scalarization with a small ``ρ``-term.
        Can recover non-convex front regions and produces
        well-distributed solutions.
    """

    WEIGHTED_SUM = "weighted_sum"
    AUGMENTED_TCHEBYCHEFF = "augmented_tchebycheff"


class ScalarizedSubproblemSolver:
    """Solve a single scalarized BPRC portfolio subproblem via SLSQP.

    Given a weight vector ``λ`` in the objective space, scalarizes the
    BPRC objectives into a single function and solves the resulting NLP:

    **Decision variables**
        ``w_i ∈ [0, 1]``, ``i = 1..N``, with ``∑ w_i = 1`` (simplex).

    **Objectives** (normalised, then scalarized)
        1. ``variance(w) = w' Σ w``
        2. ``−expected_return(w) = −μ' w``
        3. ``HHI(g(w))`` — Herfindahl index of risk contributions
           (only when ``n_objectives == 3``).

    **Constraints**
        - Simplex: ``∑ w_i = 1``, ``w_i ≥ 0``.
        - Risk-contribution bounds with slack relaxation (same as BPRC).
        - Optional HHI upper bound (when ``hhi_as_constraint=True``).

    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector, shape ``(N,)``.
    cov : np.ndarray
        Covariance matrix, shape ``(N, N)``.
    l_assets : int
        Lower bound on active assets.
    u_assets : int
        Upper bound on active assets.
    activity_tol : float, optional
        Weight threshold for "active" status, by default 1e-4.
    constraint_slack : float, optional
        Fractional relaxation for risk-contribution bounds, by default 0.1.
    hhi_as_constraint : bool, optional
        Move HHI from objective to constraint.  When ``True``, the
        scalarization uses only 2 objectives.  Default ``False``.
    hhi_bound : float, optional
        Upper bound for HHI constraint, by default 0.5.
    method : ScalarizationMethod, optional
        Scalarization approach, by default ``AUGMENTED_TCHEBYCHEFF``.
    rho : float, optional
        Augmentation coefficient for Tchebycheff, by default 0.05.
    ideal : np.ndarray | None, optional
        Ideal point for normalisation.  Estimated if not provided.
    nadir : np.ndarray | None, optional
        Nadir point for normalisation.  Estimated if not provided.
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
        method: ScalarizationMethod = ScalarizationMethod.AUGMENTED_TCHEBYCHEFF,
        rho: float = 0.05,
        ideal: np.ndarray | None = None,
        nadir: np.ndarray | None = None,
        trading_costs: TradingCosts | None = None,
    ) -> None:
        self.mu = np.asarray(mu, dtype=float)
        self.trading_costs = trading_costs
        self.cov = np.asarray(cov, dtype=float)
        self.n_assets = self.mu.size
        self.l_assets = int(l_assets)
        self.u_assets = int(u_assets)
        self.activity_tol = float(activity_tol)
        self.constraint_slack = float(constraint_slack)
        self.hhi_as_constraint = hhi_as_constraint
        self.hhi_bound = float(hhi_bound)
        self.method = ScalarizationMethod(method)
        self.rho = float(rho)

        self.n_objectives = 2 if hhi_as_constraint else 3

        # Normalisation references — estimated lazily if not provided
        self.ideal = np.asarray(ideal, dtype=float) if ideal is not None else None
        self.nadir = np.asarray(nadir, dtype=float) if nadir is not None else None

    # ------------------------------------------------------------------
    # Objective helpers
    # ------------------------------------------------------------------

    def raw_objectives(self, w: np.ndarray) -> np.ndarray:
        """Compute raw (un-normalised) BPRC objectives.

        Returns
        -------
        np.ndarray
            ``[variance, -return]`` (2-obj) or
            ``[variance, -return, HHI]`` (3-obj).
        """
        Sw = self.cov @ w
        variance = float(w @ Sw)
        neg_return = -expected_net_return(self.mu, w, self.trading_costs)
        if self.hhi_as_constraint:
            return np.array([variance, neg_return])
        gi = self._normalized_risk_contributions(w)
        hhi = float(np.sum(gi**2))
        return np.array([variance, neg_return, hhi])

    def _normalized_risk_contributions(self, w: np.ndarray) -> np.ndarray:
        """Normalised risk contributions — delegates to shared utility."""
        result: np.ndarray = normalized_risk_contributions(
            w, self.cov, self.activity_tol
        )
        return result

    # ------------------------------------------------------------------
    # Scalarization
    # ------------------------------------------------------------------

    def _normalise(self, objs: np.ndarray) -> np.ndarray:
        """Normalise objectives to [0, 1] using ideal/nadir bounds."""
        if self.ideal is None or self.nadir is None:
            return objs  # no normalisation available
        rng = self.nadir - self.ideal
        rng = np.where(np.abs(rng) < 1e-12, 1.0, rng)
        result: np.ndarray = (objs - self.ideal) / rng
        return result

    def _scalarize(
        self,
        w: np.ndarray,
        lam: np.ndarray,
    ) -> float:
        """Compute the scalarized objective for weight vector ``lam``.

        Parameters
        ----------
        w : np.ndarray
            Portfolio weight vector on the simplex.
        lam : np.ndarray
            Scalarization weight vector in objective space.

        Returns
        -------
        float
            Scalar objective value (lower is better).
        """
        objs = self.raw_objectives(w)
        objs_n = self._normalise(objs)
        lam_safe = np.maximum(lam, 1e-6)

        if self.method == ScalarizationMethod.WEIGHTED_SUM:
            return float(lam_safe @ objs_n)

        # Augmented Tchebycheff
        tcheb = float(np.max(lam_safe * objs_n))
        aug = self.rho * float(np.sum(lam_safe * objs_n))
        return tcheb + aug

    # ------------------------------------------------------------------
    # Constraints for SLSQP
    # ------------------------------------------------------------------

    def _build_constraints(self) -> list[dict[str, Any]]:
        """Build scipy constraint dicts for the SLSQP solver.

        Returns
        -------
        list[dict[str, Any]]
            Equality (simplex) and inequality (risk-contribution, HHI)
            constraints.
        """
        constraints: list[dict[str, Any]] = [
            # Simplex: sum(w) == 1
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        ]

        rc_lower = 1.0 / self.u_assets - self.constraint_slack / self.u_assets
        rc_upper = 1.0 / self.l_assets + self.constraint_slack / self.l_assets

        def rc_lower_constraint(w: np.ndarray) -> float:
            """g_i >= rc_lower for all active i  ⇔  min(g_i) - rc_lower >= 0."""
            gi = self._normalized_risk_contributions(w)
            active = w > self.activity_tol
            if not active.any():
                return -1.0  # infeasible
            return float(np.min(gi[active]) - rc_lower)

        def rc_upper_constraint(w: np.ndarray) -> float:
            """g_i <= rc_upper for all active i  ⇔  rc_upper - max(g_i) >= 0."""
            gi = self._normalized_risk_contributions(w)
            active = w > self.activity_tol
            if not active.any():
                return -1.0  # infeasible
            return float(rc_upper - np.max(gi[active]))

        constraints.append({"type": "ineq", "fun": rc_lower_constraint})
        constraints.append({"type": "ineq", "fun": rc_upper_constraint})

        if self.hhi_as_constraint:

            def hhi_constraint(w: np.ndarray) -> float:
                """HHI <= hhi_bound  ⇔  hhi_bound - HHI >= 0."""
                gi = self._normalized_risk_contributions(w)
                hhi = float(np.sum(gi**2))
                return self.hhi_bound - hhi

            constraints.append({"type": "ineq", "fun": hhi_constraint})

        return constraints

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(
        self,
        lam: np.ndarray,
        x0: np.ndarray | None = None,
        n_random_starts: int = DECOMPOSITION.n_random_starts,
        maxiter: int = DECOMPOSITION.maxiter,
    ) -> tuple[np.ndarray, float, bool]:
        """Solve a single scalarized subproblem.

        Runs SLSQP from ``x0`` (if provided) plus ``n_random_starts``
        random starting points, and returns the best feasible solution.

        Parameters
        ----------
        lam : np.ndarray
            Scalarization weight vector (same dimension as objectives).
        x0 : np.ndarray | None, optional
            Initial guess (e.g. from warm-start).  When ``None``, only
            random starting points are used.
        n_random_starts : int, optional
            Number of additional random restarts, by default 3.
        maxiter : int, optional
            Maximum SLSQP iterations per start, by default 500.

        Returns
        -------
        tuple[np.ndarray, float, bool]
            ``(weights, objective_value, converged)`` — the best solution
            found, its scalarized value, and whether any run converged.
        """
        bounds = [(0.0, 1.0)] * self.n_assets
        constraints = self._build_constraints()

        starting_points: list[np.ndarray] = []
        if x0 is not None:
            starting_points.append(np.asarray(x0, dtype=float).copy())

        for _ in range(n_random_starts):
            k = np.random.randint(self.l_assets, self.u_assets + 1)
            idx = np.random.choice(self.n_assets, size=k, replace=False)
            w0 = np.zeros(self.n_assets)
            w0[idx] = np.random.dirichlet(np.ones(k))
            starting_points.append(w0)

        best_w: np.ndarray | None = None
        best_val = np.inf
        any_converged = False

        for start in starting_points:
            try:
                result: OptimizeResult = minimize(
                    fun=lambda w: self._scalarize(w, lam),
                    x0=start,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
                )
                if result.success:
                    any_converged = True
                # Accept result even if not converged — it may still be
                # better than other starts
                val = float(result.fun)
                if val < best_val:
                    best_val = val
                    best_w = np.clip(result.x, 0.0, None)
                    s = best_w.sum()
                    if s > 0:
                        best_w /= s
            except Exception:
                logger.debug(
                    "SLSQP failed for lambda=%s, skipping start",
                    lam,
                    exc_info=True,
                )

        if best_w is None:
            # Fallback: uniform portfolio
            best_w = np.full(self.n_assets, 1.0 / self.n_assets)
            best_val = self._scalarize(best_w, lam)
            logger.warning(
                "All SLSQP starts failed for lambda=%s; returning uniform portfolio",
                lam,
            )

        # Zero out dust
        best_w[best_w < self.activity_tol] = 0.0
        s = best_w.sum()
        if s > 0:
            best_w /= s

        return best_w, best_val, any_converged
