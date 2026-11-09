"""Hybrid evolutionary + local-search portfolio optimisation strategy.

Combines the global exploration of evolutionary algorithms (via
:class:`PlatypusOptimizationStrategy`) with the exact local convergence
of SLSQP (via :class:`DecompositionOptimizationStrategy`) into a
two-phase approach:

**Phase 1 — Evolutionary exploration** (reduced budget):
    Run the existing island/ensemble multi-start EA to discover
    promising regions of the Pareto front.  Uses a fraction of the
    full evaluation budget (default 30%) to keep wall-time low.

**Phase 2 — SLSQP polishing**:
    Take each solution from the EA front, find the nearest reference
    direction, and run SLSQP to refine it.  Additionally, solve
    ``n_extra_directions`` uniformly-spaced subproblems (seeded from
    the nearest EA solution) to fill gaps in the front.

The two fronts are merged and the global non-dominated set is returned.
This hybrid approach addresses both weaknesses:

* EA alone: inconsistent convergence, solutions far from the frontier.
* Decomposition alone: may miss good regions if all starts converge
  to the same local optimum.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode
from investment_tool.output.front_plot import plot_final_front

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from platypus import (
    AbstractGeneticAlgorithm,
    NSGAIII,
    Solution,
)

from investment_tool.defaults import DECOMPOSITION, OPTIMIZER
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    BaseOptimizationStrategy,
    ProgressCallback,
)
from investment_tool.strategies.optimization_strategies.decomposition_optimization import (
    DecompositionOptimizationStrategy,
)
from investment_tool.strategies.optimization_strategies.model import PortfolioModel
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)
from investment_tool.strategies.optimization_strategies.platypus_optimization import (
    PlatypusOptimizationStrategy,
    WarmStartSource,
)
from investment_tool.strategies.optimization_strategies.scalarized_solver import (
    ScalarizationMethod,
)
from investment_tool.strategies.optimization_strategies.utils import (
    merge_nondominated,
    plot_solution,
    plot_solutions_with_markowitz_front,
)

logger = logging.getLogger(__name__)


class HybridOptimizationStrategy(BaseOptimizationStrategy):
    """Two-phase hybrid: evolutionary exploration + SLSQP polishing.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns per asset.
    covariance_matrix : pd.DataFrame
        Covariance matrix.
    risk_aversion : float
        Risk aversion parameter (passed to the EA).
    algorithm : type[AbstractGeneticAlgorithm]
        Default EA class for Phase 1.
    model : type[PortfolioModel]
        Portfolio model class (e.g., ``BalancedPortfoliosRiskContributions``).
    population_size : int, optional
        EA population size, by default 300.
    n_generations : int, optional
        *Total* evaluation budget; Phase 1 uses ``ea_budget_ratio``
        of this, by default 500 000.
    n_restarts : int, optional
        EA multi-start runs, by default 5.
    algorithm_kwargs : dict | None, optional
        Extra kwargs for the EA constructor.
    model_kwargs : dict | None, optional
        Extra kwargs for the model constructor.
    output_directory : Path | None, optional
        Output directory for plots.
    include_markowitz_front : bool, optional
        Whether to compute / overlay the Markowitz frontier, by default True.
    date_label : str | None, optional
        Date label for plot filenames.
    seed_weights : list[np.ndarray] | None, optional
        Prior-period front weights for warm-starting.
    warm_start_source : WarmStartSource, optional
        Warm-start source for the EA phase. ``NONE`` disables all
        seeding (cold start every period), ``PREVIOUS`` uses prior-period
        front only, ``MARKOWITZ`` uses the Markowitz frontier, ``BOTH``
        combines them. Default ``WarmStartSource.MARKOWITZ``.
    warm_start_generation_ratio : float, optional
        Warm-start evaluation ratio for the EA, by default 0.7.
    algorithm_portfolio : list[tuple[type, dict]] | None, optional
        Island/ensemble configuration for the EA.
    ea_budget_ratio : float, optional
        Fraction of ``n_generations`` allocated to the EA phase,
        by default 0.3.  The remaining budget is implicit in the
        number of SLSQP subproblems.
    n_decomposition_directions : int, optional
        Number of SLSQP subproblems in the polishing phase, by default 80.
    n_slsqp_random_starts : int, optional
        Random restarts per SLSQP subproblem, by default 2.
    slsqp_maxiter : int, optional
        Maximum SLSQP iterations per start, by default 500.
    scalarization_method : ScalarizationMethod, optional
        Scalarization for polishing, by default ``AUGMENTED_TCHEBYCHEFF``.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_aversion: float,
        algorithm: type[AbstractGeneticAlgorithm] = NSGAIII,
        model: type[PortfolioModel] | None = None,
        population_size: int = OPTIMIZER.pop_size,
        n_generations: int = OPTIMIZER.n_generations,
        n_restarts: int = OPTIMIZER.n_restarts,
        algorithm_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        output_directory: Path | None = None,
        include_markowitz_front: bool = True,
        date_label: str | None = None,
        seed_weights: list[np.ndarray] | None = None,
        warm_start_source: WarmStartSource = WarmStartSource.MARKOWITZ,
        warm_start_generation_ratio: float = OPTIMIZER.warm_start_generation_ratio,
        algorithm_portfolio: (
            list[tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]] | None
        ) = None,
        ea_budget_ratio: float = OPTIMIZER.ea_budget_ratio,
        n_decomposition_directions: int = DECOMPOSITION.n_decomposition_directions,
        n_slsqp_random_starts: int = 2,
        slsqp_maxiter: int = DECOMPOSITION.maxiter,
        scalarization_method: ScalarizationMethod = ScalarizationMethod.AUGMENTED_TCHEBYCHEFF,
    ) -> None:
        if model is None:
            model = BalancedPortfoliosRiskContributions

        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_aversion = risk_aversion
        self.algorithm = algorithm
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.model: type[PortfolioModel] = model
        self.model_kwargs = model_kwargs or {}
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_restarts = n_restarts
        self.output_directory = output_directory
        self.include_markowitz_front = include_markowitz_front
        self.date_label = date_label
        self.seed_weights = seed_weights
        self.warm_start_source = WarmStartSource(warm_start_source)
        self.warm_start_generation_ratio = warm_start_generation_ratio
        self.algorithm_portfolio = algorithm_portfolio
        self.ea_budget_ratio = float(ea_budget_ratio)
        self.n_decomposition_directions = n_decomposition_directions
        self.n_slsqp_random_starts = n_slsqp_random_starts
        self.slsqp_maxiter = slsqp_maxiter
        self.scalarization_method = ScalarizationMethod(scalarization_method)

    # ------------------------------------------------------------------
    # Phase 1: Evolutionary exploration
    # ------------------------------------------------------------------

    def _run_ea_phase(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """Run the evolutionary phase with a reduced evaluation budget.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Progress callback; fractions are in the EA's local [0, 1]
            range — the caller is responsible for rescaling.

        Returns
        -------
        OptimizationResult
            EA result with front, archives, and optional Markowitz front.
        """
        ea_evals = max(int(self.n_generations * self.ea_budget_ratio), 10_000)
        logger.info(
            "Hybrid Phase 1: EA exploration with %d evaluations "
            "(%.0f%% of total budget)",
            ea_evals,
            self.ea_budget_ratio * 100,
        )

        ea_strategy = PlatypusOptimizationStrategy(
            expected_returns=self.expected_returns,
            covariance_matrix=self.covariance_matrix,
            risk_aversion=self.risk_aversion,
            algorithm=self.algorithm,
            algorithm_kwargs=self.algorithm_kwargs,
            model=self.model,
            model_kwargs=self.model_kwargs,
            population_size=self.population_size,
            n_generations=ea_evals,
            n_restarts=self.n_restarts,
            output_directory=self.output_directory,
            include_markowitz_front=self.include_markowitz_front,
            date_label=self.date_label,
            seed_weights=self.seed_weights,
            warm_start_source=self.warm_start_source,
            warm_start_generation_ratio=self.warm_start_generation_ratio,
            algorithm_portfolio=self.algorithm_portfolio,
        )
        ea_strategy.output_mode = self.output_mode
        ea_strategy.trading_costs = self.trading_costs
        return ea_strategy.optimize(progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # Phase 2: SLSQP polishing
    # ------------------------------------------------------------------

    def _run_polishing_phase(
        self,
        ea_front_weights: list[np.ndarray],
        markowitz_front: list[Solution] | None,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """Run the SLSQP decomposition phase seeded from EA solutions.

        Parameters
        ----------
        ea_front_weights : list[np.ndarray]
            Weight vectors from Phase 1 EA front.
        markowitz_front : list[Solution] | None
            Markowitz frontier for reference.
        progress_callback : ProgressCallback | None, optional
            Progress callback; fractions are in the polishing phase's
            local [0, 1] range — the caller is responsible for rescaling.

        Returns
        -------
        OptimizationResult
            Decomposition result.
        """
        logger.info(
            "Hybrid Phase 2: SLSQP polishing with %d directions, "
            "seeded from %d EA solutions",
            self.n_decomposition_directions,
            len(ea_front_weights),
        )

        # Extract model kwargs for the decomposition strategy
        model_kwargs = dict(self.model_kwargs)

        decomp = DecompositionOptimizationStrategy(
            expected_returns=self.expected_returns,
            covariance_matrix=self.covariance_matrix,
            l_assets=model_kwargs.get("l_assets", 2),
            u_assets=model_kwargs.get("u_assets", 5),
            n_directions=self.n_decomposition_directions,
            n_random_starts=self.n_slsqp_random_starts,
            maxiter=self.slsqp_maxiter,
            activity_tol=model_kwargs.get("activity_tol", 1e-4),
            constraint_slack=model_kwargs.get("constraint_slack", 0.1),
            hhi_as_constraint=model_kwargs.get("hhi_as_constraint", False),
            hhi_bound=model_kwargs.get("hhi_bound", 0.5),
            scalarization_method=self.scalarization_method,
            seed_weights=ea_front_weights,
            output_directory=self.output_directory,
            date_label=self.date_label,
            markowitz_front=markowitz_front,
        )
        decomp.output_mode = self.output_mode
        decomp.trading_costs = self.trading_costs
        return decomp.optimize(progress_callback=progress_callback)

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_fronts(
        fronts: list[list[Solution]],
        target_size: int = 300,
    ) -> list[Solution]:
        """Merge multiple fronts and extract the global non-dominated set.

        Delegates to
        :meth:`PlatypusOptimizationStrategy._merge_nondominated` which
        implements the same deduplication + non-dominated pruning logic.

        Parameters
        ----------
        fronts : list[list[Solution]]
            Solution lists to merge.
        target_size : int, optional
            Maximum front size, by default 300.

        Returns
        -------
        list[Solution]
            Pruned non-dominated solutions.
        """
        result: list[Solution] = merge_nondominated(
            fronts,
            target_size=target_size,
        )
        return result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_hybrid_results(
        self,
        ea_front: list[Solution],
        polished_front: list[Solution],
        merged_front: list[Solution],
        markowitz_front: list[Solution] | None,
        selected_solution: Solution | None = None,
    ) -> None:
        """Save diagnostic plots for the hybrid approach.

        Parameters
        ----------
        ea_front : list[Solution]
            Phase 1 EA front.
        polished_front : list[Solution]
            Phase 2 decomposition front.
        merged_front : list[Solution]
            Final merged non-dominated front.
        markowitz_front : list[Solution] | None
            Optional Markowitz frontier overlay.
        selected_solution : Solution | None, optional
            The solution chosen by the selection strategy.  When provided,
            it is highlighted with a distinct marker on every plot.
        """
        if self.output_mode == OutputMode.STANDARD:
            plot_final_front(
                merged_front,
                markowitz_front,
                self.output_directory,
                self.date_label,
                selected_solution,
            )
            return

        plot_solution(
            ea_front,
            "hybrid_phase1_ea_front",
            self.output_directory,
            self.date_label,
            selected_solution=selected_solution,
        )
        plot_solution(
            polished_front,
            "hybrid_phase2_polished_front",
            self.output_directory,
            self.date_label,
            selected_solution=selected_solution,
        )
        plot_solution(
            merged_front,
            "hybrid_merged_front",
            self.output_directory,
            self.date_label,
            selected_solution=selected_solution,
        )
        if markowitz_front is not None:
            plot_solutions_with_markowitz_front(
                solutions=merged_front,
                markowitz_front=markowitz_front,
                file_name="hybrid_merged_front_vs_markowitz",
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
        """Run the two-phase hybrid optimisation.

        Phase 1: Evolutionary exploration with reduced budget.
        Phase 2: SLSQP polishing seeded from EA solutions.
        Merge: Global non-dominated set from both phases.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Called periodically with ``(fraction_complete, message)``.
            EA progress is mapped to [0, 0.5] and polishing to
            [0.5, 1.0].

        Returns
        -------
        OptimizationResult
            Final Pareto front, archives from both phases, and
            optional Markowitz frontier.
        """
        # Build scaled callbacks for each phase
        ea_cb: ProgressCallback | None = None
        polish_cb: ProgressCallback | None = None
        if progress_callback is not None:

            def ea_cb(frac: float, msg: str) -> None:
                progress_callback(frac * 0.5, f"[EA] {msg}")

            def polish_cb(frac: float, msg: str) -> None:
                progress_callback(0.5 + frac * 0.5, f"[Polish] {msg}")

        # Phase 1: EA exploration
        ea_result = self._run_ea_phase(progress_callback=ea_cb)
        ea_front_weights = [
            np.asarray(s.variables, dtype=float) for s in ea_result.front
        ]
        logger.info("Phase 1 complete: %d EA front solutions", len(ea_result.front))

        # Phase 2: SLSQP polishing
        polished_result = self._run_polishing_phase(
            ea_front_weights,
            ea_result.markowitz_front,
            progress_callback=polish_cb,
        )
        logger.info(
            "Phase 2 complete: %d polished solutions",
            len(polished_result.front),
        )

        # Merge both fronts
        merged_front = self._merge_fronts(
            [ea_result.front, polished_result.front],
            target_size=self.population_size,
        )

        if not merged_front:
            raise RuntimeError(
                "Hybrid optimisation produced no non-dominated solutions."
            )

        logger.info(
            "Hybrid optimisation complete: %d EA + %d polished → %d merged",
            len(ea_result.front),
            len(polished_result.front),
            len(merged_front),
        )

        # Diagnostic plots (without selection highlight — selection happens after)
        self._last_ea_front = ea_result.front
        self._last_polished_front = polished_result.front
        self._plot_hybrid_results(
            ea_result.front,
            polished_result.front,
            merged_front,
            ea_result.markowitz_front,
        )

        # Combine archives from both phases
        all_archives = ea_result.all_archives + polished_result.all_archives

        if progress_callback is not None:
            progress_callback(1.0, "Hybrid optimisation complete")

        return OptimizationResult(
            front=merged_front,
            all_archives=all_archives,
            markowitz_front=ea_result.markowitz_front,
        )

    def replot_with_selection(self, result: "OptimizationResult") -> None:
        """Re-save all diagnostic plots with the selected solution highlighted.

        Must be called **after** ``result.selected_solution`` has been set
        by the investment strategy layer.  Overwrites the plots produced
        by the initial ``optimize()`` call, adding an orange star marker
        at the selected portfolio's position.

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
            logger.debug("replot_with_selection: no selected solution — skipping")
            return

        ea_front = getattr(self, "_last_ea_front", result.front)
        polished_front = getattr(self, "_last_polished_front", result.front)
        self._plot_hybrid_results(
            ea_front=ea_front,
            polished_front=polished_front,
            merged_front=result.front,
            markowitz_front=result.markowitz_front,
            selected_solution=result.selected_solution,
        )
