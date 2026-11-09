"""Decomposition-based portfolio optimisation strategy.

Decomposes the multi-objective BPRC problem into a set of scalarized
subproblems (one per reference direction) and solves each exactly via
:class:`ScalarizedSubproblemSolver` (SLSQP).  The union of subproblem
solutions is then filtered to the non-dominated front.

This provides **exact local convergence** on each subproblem — addressing
the core weakness of purely evolutionary approaches that struggle with
constraint handling and fine-grained convergence.

The strategy supports:

* **Warm-starting** via ``seed_weights``: each subproblem is initialised
  from the nearest prior-period solution (natural ``x0`` for SLSQP).
* **Adaptive normalisation**: ideal/nadir points are estimated from a
  quick pilot sample, then refined as subproblems are solved.
* **2- and 3-objective modes**: reference directions are generated via
  Das–Dennis (2-obj) or Riesz s-energy (3-obj) from ``pymoo``.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode
from investment_tool.output.front_plot import plot_final_front

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from platypus import Solution, nondominated_prune, nondominated_sort

from investment_tool.defaults import CONSTRUCTION, DECOMPOSITION, OPTIMIZER
from investment_tool.domain.trading_costs import TradingCosts
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    BaseOptimizationStrategy,
    ProgressCallback,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)
from investment_tool.strategies.optimization_strategies.scalarized_solver import (
    ScalarizationMethod,
    ScalarizedSubproblemSolver,
)
from investment_tool.strategies.optimization_strategies.utils import (
    compute_markowitz_front,
    plot_solution,
    plot_solutions_with_markowitz_front,
)

logger = logging.getLogger(__name__)

# Defaults imported from defaults.py (DECOMPOSITION group).


def _generate_reference_directions(
    n_objectives: int,
    n_directions: int,
) -> np.ndarray:
    """Generate uniformly-spaced reference direction vectors.

    Uses Das–Dennis for 2 objectives and Riesz s-energy for 3+
    objectives (via ``pymoo``).

    Parameters
    ----------
    n_objectives : int
        Number of objectives (2 or 3).
    n_directions : int
        Desired number of direction vectors.

    Returns
    -------
    np.ndarray
        Reference directions, shape ``(n_directions, n_objectives)``.
        Each row sums to 1 and has non-negative components.
    """
    from pymoo.util.ref_dirs import get_reference_directions

    if n_objectives == 2:
        # Das-Dennis produces exactly (n_partitions + 1) points in 2D
        n_partitions = max(n_directions - 1, 2)
        ref_dirs: np.ndarray = get_reference_directions(
            "das-dennis", n_objectives, n_partitions=n_partitions
        )
    else:
        # Riesz s-energy for good 3D coverage
        ref_dirs = get_reference_directions(
            "energy", n_objectives, n_points=n_directions
        )
    return ref_dirs


class DecompositionOptimizationStrategy(BaseOptimizationStrategy):
    """Decomposition-based portfolio optimisation via scalarized SLSQP.

    Generates a set of reference directions in objective space, solves
    each scalarized subproblem via SLSQP, and returns the non-dominated
    front.  This approach converges to the true Pareto front much more
    reliably than evolutionary algorithms for the continuous-variable
    BPRC problem.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns per asset, indexed by symbol.
    covariance_matrix : pd.DataFrame
        Covariance matrix of asset returns.
    l_assets : int
        Minimum number of active assets.
    u_assets : int
        Maximum number of active assets.
    n_directions : int, optional
        Number of scalarized subproblems, by default 100.
    n_random_starts : int, optional
        Random restarts per subproblem, by default 3.
    maxiter : int, optional
        Maximum SLSQP iterations per start, by default 500.
    activity_tol : float, optional
        Weight threshold for "active" status, by default 1e-4.
    constraint_slack : float, optional
        Relaxation for risk-contribution bounds, by default 0.1.
    hhi_as_constraint : bool, optional
        Move HHI from objective to constraint, by default ``False``.
    hhi_bound : float, optional
        Upper bound for HHI constraint, by default 0.5.
    scalarization_method : ScalarizationMethod, optional
        Scalarization approach, by default ``AUGMENTED_TCHEBYCHEFF``.
    rho : float, optional
        Augmentation coefficient for Tchebycheff, by default 0.05.
    seed_weights : list[np.ndarray] | None, optional
        Prior-period Pareto front weights for warm-starting.
    output_directory : Path | None, optional
        Output directory for diagnostic plots.
    date_label : str | None, optional
        Date label for plot filenames.
    markowitz_front : list[Solution] | None, optional
        Pre-computed Markowitz frontier for overlay plots.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        l_assets: int,
        u_assets: int,
        n_directions: int = DECOMPOSITION.n_directions,
        n_random_starts: int = DECOMPOSITION.n_random_starts,
        maxiter: int = DECOMPOSITION.maxiter,
        activity_tol: float = OPTIMIZER.activity_tol,
        constraint_slack: float = OPTIMIZER.constraint_slack,
        hhi_as_constraint: bool = CONSTRUCTION.hhi_as_constraint,
        hhi_bound: float = CONSTRUCTION.hhi_bound,
        scalarization_method: ScalarizationMethod = ScalarizationMethod.AUGMENTED_TCHEBYCHEFF,
        rho: float = 0.05,
        seed_weights: list[np.ndarray] | None = None,
        output_directory: Path | None = None,
        date_label: str | None = None,
        markowitz_front: list[Solution] | None = None,
    ) -> None:
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.l_assets = l_assets
        self.u_assets = u_assets
        self.n_directions = n_directions
        self.n_random_starts = n_random_starts
        self.maxiter = maxiter
        self.activity_tol = activity_tol
        self.constraint_slack = constraint_slack
        self.hhi_as_constraint = hhi_as_constraint
        self.hhi_bound = hhi_bound
        self.scalarization_method = ScalarizationMethod(scalarization_method)
        self.rho = rho
        self.seed_weights = seed_weights
        self.output_directory = output_directory
        self.date_label = date_label
        self.markowitz_front = markowitz_front

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_ideal_nadir(
        self,
        solver: ScalarizedSubproblemSolver,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate ideal and nadir points via per-objective optimisation.

        Solves ``n_objectives`` single-objective problems (one per
        objective) to build the payoff table and extract ideal/nadir.

        Parameters
        ----------
        solver : ScalarizedSubproblemSolver
            Configured solver (used for constraint construction).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(ideal, nadir)`` — component-wise min and max of the
            payoff table.
        """
        n_obj = solver.n_objectives
        payoff = np.zeros((n_obj, n_obj))

        for i in range(n_obj):
            # Weight vector that focuses entirely on objective i
            lam = np.full(n_obj, 1e-6)
            lam[i] = 1.0

            w, _, _ = solver.solve(
                lam,
                x0=None,
                n_random_starts=5,
                maxiter=self.maxiter,
            )
            payoff[i, :] = solver.raw_objectives(w)

        ideal = payoff.min(axis=0)
        nadir = payoff.max(axis=0)

        # Add a small margin to avoid degenerate normalisation
        margin = 0.05 * np.abs(nadir - ideal)
        margin = np.where(margin < 1e-10, 1e-6, margin)
        ideal -= margin
        nadir += margin

        logger.info(
            "Estimated ideal=%s, nadir=%s",
            np.array2string(ideal, precision=6),
            np.array2string(nadir, precision=6),
        )
        return ideal, nadir

    def _find_nearest_seed(
        self,
        lam: np.ndarray,
        ref_dirs: np.ndarray,
        seed_objectives: np.ndarray,
    ) -> np.ndarray | None:
        """Find the seed weight vector closest to the given reference direction.

        Uses cosine similarity in objective space between the reference
        direction and each seed's normalised objective vector.

        Parameters
        ----------
        lam : np.ndarray
            Target reference direction.
        ref_dirs : np.ndarray
            All reference directions (unused; kept for API symmetry).
        seed_objectives : np.ndarray
            Pre-computed objective vectors for all seeds.

        Returns
        -------
        np.ndarray | None
            Nearest seed weight vector, or ``None`` if no seeds.
        """
        if self.seed_weights is None or len(self.seed_weights) == 0:
            return None

        # Cosine similarity between lam and each seed's objective profile
        norms = np.linalg.norm(seed_objectives, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normalised = seed_objectives / norms

        lam_norm = lam / max(float(np.linalg.norm(lam)), 1e-12)
        similarities = normalised @ lam_norm
        best_idx = int(np.argmax(similarities))
        return np.asarray(self.seed_weights[best_idx], dtype=float).copy()

    @staticmethod
    def _to_platypus_solutions(
        weights_list: list[np.ndarray],
        mu: np.ndarray,
        cov: np.ndarray,
        l_assets: int,
        u_assets: int,
        hhi_as_constraint: bool,
        hhi_bound: float,
        constraint_slack: float,
        activity_tol: float,
        trading_costs: TradingCosts | None = None,
    ) -> list[Solution]:
        """Convert weight vectors to evaluated Platypus solutions.

        Parameters
        ----------
        weights_list : list[np.ndarray]
            Portfolio weight vectors.
        mu, cov, l_assets, u_assets, hhi_as_constraint, hhi_bound,
        constraint_slack, activity_tol
            BPRC model parameters.

        Returns
        -------
        list[Solution]
            Evaluated Platypus solutions compatible with existing
            selection and plotting infrastructure.
        """
        problem = BalancedPortfoliosRiskContributions(
            mu=mu,
            cov=cov,
            l_assets=l_assets,
            u_assets=u_assets,
            activity_tol=activity_tol,
            constraint_slack=constraint_slack,
            hhi_as_constraint=hhi_as_constraint,
            hhi_bound=hhi_bound,
            trading_costs=trading_costs,
        )
        solutions: list[Solution] = []
        for w in weights_list:
            sol = Solution(problem)
            sol.variables[:] = w.tolist()
            problem.evaluate(sol)
            solutions.append(sol)
        return solutions

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _plot_results(
        self,
        front: list[Solution],
        markowitz_front: list[Solution] | None,
        selected_solution: Solution | None = None,
    ) -> None:
        """Save diagnostic Pareto front plots.

        Parameters
        ----------
        front : list[Solution]
            Non-dominated decomposition front.
        markowitz_front : list[Solution] | None
            Optional Markowitz frontier for overlay.
        selected_solution : Solution | None, optional
            The solution chosen by the selection strategy.  When provided,
            it is highlighted with a distinct marker on every plot.
        """
        if self.output_mode == OutputMode.STANDARD:
            plot_final_front(
                front,
                markowitz_front,
                self.output_directory,
                self.date_label,
                selected_solution,
            )
            return

        plot_solution(
            front,
            "decomposition_pareto_front",
            self.output_directory,
            self.date_label,
            selected_solution=selected_solution,
        )
        if markowitz_front is not None:
            plot_solutions_with_markowitz_front(
                solutions=front,
                markowitz_front=markowitz_front,
                file_name="decomposition_front_vs_markowitz",
                output_directory=self.output_directory,
                date_label=self.date_label,
                selected_solution=selected_solution,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """Run the decomposition-based optimisation.

        Steps
        -----
        1. Build a ``ScalarizedSubproblemSolver`` with the BPRC parameters.
        2. Estimate ideal/nadir by solving per-objective anchor problems.
        3. Generate uniformly-spaced reference directions.
        4. Solve each scalarized subproblem via SLSQP (warm-started from
           the nearest seed when available).
        5. Convert results to Platypus solutions, extract non-dominated
           front, and save diagnostic plots.

        Returns
        -------
        OptimizationResult
            Pareto front, raw archives, and optional Markowitz frontier.
        """
        mu = np.asarray(self.expected_returns.values)
        cov = np.asarray(self.covariance_matrix.values)
        n_obj = 2 if self.hhi_as_constraint else 3

        logger.info(
            "Starting decomposition optimisation: %d objectives, "
            "%d directions, method=%s",
            n_obj,
            self.n_directions,
            self.scalarization_method.value,
        )

        # 1. Build solver (without ideal/nadir first)
        solver = ScalarizedSubproblemSolver(
            mu=mu,
            cov=cov,
            l_assets=self.l_assets,
            u_assets=self.u_assets,
            activity_tol=self.activity_tol,
            constraint_slack=self.constraint_slack,
            hhi_as_constraint=self.hhi_as_constraint,
            hhi_bound=self.hhi_bound,
            method=self.scalarization_method,
            rho=self.rho,
            trading_costs=self.trading_costs,
        )

        # 2. Estimate ideal/nadir
        ideal, nadir = self._estimate_ideal_nadir(solver)
        solver.ideal = ideal
        solver.nadir = nadir

        # 3. Generate reference directions
        ref_dirs = _generate_reference_directions(n_obj, self.n_directions)
        logger.info("Generated %d reference directions", ref_dirs.shape[0])

        # Pre-compute seed objectives for warm-start matching
        seed_objectives: np.ndarray | None = None
        if self.seed_weights is not None and len(self.seed_weights) > 0:
            n_assets = mu.size
            valid_seeds = [
                w
                for w in self.seed_weights
                if np.asarray(w, dtype=float).shape[0] == n_assets
            ]
            if valid_seeds:
                self.seed_weights = valid_seeds
                seed_objs = [solver.raw_objectives(w) for w in valid_seeds]
                seed_objectives = np.array(seed_objs)
            else:
                logger.warning(
                    "All seed weight vectors have wrong dimension for "
                    "%d assets; ignoring warm-start",
                    n_assets,
                )

        # 4. Solve each subproblem
        all_weights: list[np.ndarray] = []
        n_converged = 0

        for i, lam in enumerate(ref_dirs):
            # Warm-start: find nearest seed
            x0: np.ndarray | None = None
            if seed_objectives is not None:
                x0 = self._find_nearest_seed(lam, ref_dirs, seed_objectives)

            w, val, converged = solver.solve(
                lam,
                x0=x0,
                n_random_starts=self.n_random_starts,
                maxiter=self.maxiter,
            )
            all_weights.append(w)
            if converged:
                n_converged += 1

            if progress_callback is not None:
                progress_callback(
                    (i + 1) / ref_dirs.shape[0],
                    f"Subproblem {i + 1}/{ref_dirs.shape[0]}",
                )

            if (i + 1) % 25 == 0:
                logger.debug(
                    "Subproblem %d/%d solved (converged=%s, obj=%.6f)",
                    i + 1,
                    ref_dirs.shape[0],
                    converged,
                    val,
                )

        logger.info(
            "Decomposition complete: %d/%d subproblems converged",
            n_converged,
            ref_dirs.shape[0],
        )

        # 5. Convert to Platypus solutions and extract non-dominated front
        all_solutions = self._to_platypus_solutions(
            all_weights,
            mu=mu,
            cov=cov,
            trading_costs=self.trading_costs,
            l_assets=self.l_assets,
            u_assets=self.u_assets,
            hhi_as_constraint=self.hhi_as_constraint,
            hhi_bound=self.hhi_bound,
            constraint_slack=self.constraint_slack,
            activity_tol=self.activity_tol,
        )

        # Non-dominated sort + prune
        _ = nondominated_sort(all_solutions)
        front = list(nondominated_prune(all_solutions, len(all_solutions)))
        if not front:
            raise RuntimeError(
                "Decomposition optimisation produced no non-dominated solutions."
            )

        logger.info(
            "Decomposition front: %d non-dominated solutions from %d subproblems",
            len(front),
            ref_dirs.shape[0],
        )

        # 6. Compute Markowitz frontier for overlay plots (if not pre-supplied)
        markowitz_front = self.markowitz_front
        if markowitz_front is None:
            try:
                markowitz_front = compute_markowitz_front(
                    mu=np.asarray(self.expected_returns.values),
                    cov=np.asarray(self.covariance_matrix.values),
                    **self.cost_kwargs,
                )
            except Exception as exc:
                logger.exception("Failed to compute Markowitz frontier: %s", exc)

        # 7. Diagnostic plots (without selection highlight — selection happens after)
        self._plot_results(front, markowitz_front)

        return OptimizationResult(
            front=front,
            all_archives=[all_solutions],
            markowitz_front=markowitz_front,
        )

    def replot_with_selection(self, result: "OptimizationResult") -> None:
        """Re-save all diagnostic plots with the selected solution highlighted.

        Must be called **after** ``result.selected_solution`` has been set
        by the investment strategy layer.  Overwrites the plots produced
        by the initial ``optimize()`` call.

        Parameters
        ----------
        result : OptimizationResult
            The result object whose ``selected_solution`` has been populated.
        """
        if self.output_mode == OutputMode.STANDARD:
            plot_final_front(
                result.front,
                result.markowitz_front,
                self.output_directory,
                self.date_label,
                result.selected_solution,
                result.reference_objectives,
            )
            return
        if result.selected_solution is None:
            return
        self._plot_results(
            result.front, result.markowitz_front, result.selected_solution
        )
