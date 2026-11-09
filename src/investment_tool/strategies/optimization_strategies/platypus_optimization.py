from __future__ import annotations

from investment_tool.output.policy import OutputMode
from investment_tool.output.front_plot import plot_final_front

import logging
from enum import Enum, unique
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from platypus import (
    AbstractGeneticAlgorithm,
    Solution,
)

from investment_tool.defaults import MARKOWITZ, OPTIMIZER, SEED_NOISE_SCALES
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    BaseOptimizationStrategy,
    ProgressCallback,
)
from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    repair_portfolio,
)
from investment_tool.strategies.optimization_strategies.custom_algorithm.portfolio_sampling import (
    ArchivingEvaluator,
    PortfolioGenerator,
)
from investment_tool.strategies.optimization_strategies.model import (
    PortfolioModel,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)
from investment_tool.strategies.optimization_strategies.utils import (
    compute_markowitz_front,
    merge_nondominated,
    plot_solution,
    plot_solutions_with_markowitz_front,
)

logger = logging.getLogger(__name__)

# Markowitz is a 2-objective unconstrained problem — converges much faster
# than the 3-objective constrained BPRC.  Defaults imported from defaults.py.

# Perturbation noise scales for seeding across restarts — imported from
# defaults.py (earlier restarts exploit, later restarts explore).


@unique
class WarmStartSource(str, Enum):
    """Source of warm-start seeds for the portfolio optimizer.

    Controls which solutions are injected into the initial population
    of the evolutionary optimizer when starting the BPRC optimization.

    Members
    -------
    NONE
        Cold start every period — no seeds injected at all.  The
        Markowitz frontier is still computed for diagnostic plots if
        ``include_markowitz_front=True``, but its solutions are not
        used to initialise the population.
    PREVIOUS
        Use the Pareto front from the previous backtesting period only.
    MARKOWITZ
        Compute the Markowitz efficient frontier and use its solutions
        only (ignores any previous-period seeds).
    BOTH
        Combine Markowitz frontier solutions with previous-period seeds.
    """

    NONE = "none"
    PREVIOUS = "previous"
    MARKOWITZ = "markowitz"
    BOTH = "both"


class PlatypusOptimizationStrategy(BaseOptimizationStrategy):
    """Portfolio optimization using Platypus with optional warm-start.

    Responsibilities are split across focused methods:

    * ``optimize`` — public entry-point; runs the evolutionary optimizer
      and returns an ``OptimizationResult`` (Pareto front + archives +
      optional Markowitz frontier).  **Selection is not performed here.**
    * ``_run_multi_start_optimization`` — core evolutionary multi-start loop
      with island/ensemble support (different algorithms per restart).
    * ``_plot_optimization_results`` — saves diagnostic Pareto-front plots.
    * ``compute_markowitz_front`` — independent Markowitz efficient-frontier
      computation with its own (lighter) settings.

    Portfolio selection is delegated to the investment strategy layer via
    the selectors in :mod:`~investment_tool.strategies.selection_strategies`.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns per asset, indexed by symbol.
    covariance_matrix : pd.DataFrame
        Covariance matrix of asset returns.
    risk_aversion : float
        Risk aversion parameter.
    algorithm : type[AbstractGeneticAlgorithm]
        Default genetic algorithm class (e.g., NSGAIII).
    model : type[PortfolioModel]
        Portfolio model class (e.g., BalancedPortfoliosRiskContributions).
    population_size : int, optional
        GA population size (default 300).
    n_generations : int, optional
        Number of function evaluations per restart (default 500 000).
    n_restarts : int, optional
        Independent multi-start runs (default 5).
    algorithm_kwargs : dict, optional
        Extra kwargs for the default algorithm constructor.
    model_kwargs : dict, optional
        Extra kwargs for the model constructor.
    output_directory : Path | None
        Output directory for plots.
    include_markowitz_front : bool, optional
        Whether to overlay the Markowitz efficient frontier on plots.
    date_label : str | None, optional
        Date string appended to plot filenames for uniqueness.
    seed_weights : list[np.ndarray] | None, optional
        Previous-period Pareto front weight vectors for warm-starting.
        When provided, the optimizer seeds the initial population from
        these diverse solutions rather than a single point.
    warm_start_generation_ratio : float, optional
        Fraction of ``n_generations`` used on warm iterations (default 0.7).
    warm_start_source : WarmStartSource, optional
        Controls which solutions seed the initial population:
        ``NONE`` disables all seeding (cold start every period),
        ``PREVIOUS`` uses prior-period front only, ``MARKOWITZ``
        computes the Markowitz frontier and uses it only, ``BOTH``
        combines them.  Default ``WarmStartSource.MARKOWITZ``.
    algorithm_portfolio : list[tuple[type, dict]] | None, optional
        Island/ensemble configuration.  Each element is
        ``(algorithm_class, algorithm_kwargs)`` used for one restart.
        When provided, restarts cycle through this list instead of
        always using the default ``algorithm``.  This hedges against
        any single algorithm getting stuck.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_aversion: float,
        algorithm: type[AbstractGeneticAlgorithm],
        model: type[PortfolioModel],
        population_size: int = OPTIMIZER.pop_size,
        n_generations: int = OPTIMIZER.n_generations,
        n_restarts: int = OPTIMIZER.n_restarts,
        algorithm_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        output_directory: Path | None = None,
        include_markowitz_front: bool = True,
        date_label: str | None = None,
        seed_weights: list[np.ndarray] | None = None,
        warm_start_generation_ratio: float = OPTIMIZER.warm_start_generation_ratio,
        warm_start_source: WarmStartSource = WarmStartSource.MARKOWITZ,
        algorithm_portfolio: list[tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]]
        | None = None,
    ) -> None:
        self.expected_returns: pd.Series = expected_returns
        self.covariance_matrix: pd.DataFrame = covariance_matrix
        self.risk_aversion: float = risk_aversion
        self.algorithm: type[AbstractGeneticAlgorithm] = algorithm
        self.algorithm_kwargs: dict[str, Any] = (
            algorithm_kwargs if algorithm_kwargs is not None else {}
        )
        self.model: type[PortfolioModel] = model
        self.model_kwargs: dict[str, Any] = (
            model_kwargs if model_kwargs is not None else {}
        )
        self.population_size: int = population_size
        self.n_generations: int = n_generations
        self.n_restarts: int = n_restarts
        self.optimal_weights: pd.Series | None = None
        self.output_directory: Path | None = output_directory
        self.include_markowitz_front: bool = include_markowitz_front
        self.date_label: str | None = date_label
        self.seed_weights: list[np.ndarray] | None = seed_weights
        self.warm_start_generation_ratio: float = warm_start_generation_ratio
        self.warm_start_source: WarmStartSource = WarmStartSource(warm_start_source)
        self.algorithm_portfolio: (
            list[tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]] | None
        ) = algorithm_portfolio

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_nondominated(
        lists_of_solutions: list[list[Solution]],
        target_size: int,
        dedup_decimals: int = 8,
    ) -> list[Solution]:
        """Merge solution archives, deduplicate, and prune to the Pareto front.

        Delegates to :func:`~..utils.merge_nondominated`.

        Parameters
        ----------
        lists_of_solutions : list[list[Solution]]
            Archives from each restart.
        target_size : int
            Maximum number of solutions to keep after pruning.
        dedup_decimals : int, optional
            Rounding precision for deduplication (default 8).

        Returns
        -------
        list[Solution]
            Pruned non-dominated set.
        """
        result: list[Solution] = merge_nondominated(
            lists_of_solutions, target_size, dedup_decimals
        )
        return result

    def _build_seed_solutions(
        self,
        problem: Any,
        n_assets: int,
        noise_scale: float = 0.03,
    ) -> list[Solution]:
        """Build seed solutions from the previous Pareto front for warm-starting.

        Each front solution is injected as-is, plus a small number of
        perturbed copies are generated to maintain diversity around
        the previous frontier.  The total number of seeds is capped at
        ``population_size`` to avoid overwhelming the initial population.

        Parameters
        ----------
        problem : Any
            Platypus Problem instance.
        n_assets : int
            Number of portfolio assets.
        noise_scale : float, optional
            Standard deviation of Gaussian perturbation noise, by
            default 0.03.  Earlier restarts use smaller scales (exploit)
            while later restarts use larger scales (explore).

        Returns
        -------
        list[Solution]
            Seed solutions ready for population injection.
        """
        if self.seed_weights is None or len(self.seed_weights) == 0:
            return []

        # Filter out seeds with mismatched dimensions
        valid_seeds = [
            w
            for w in self.seed_weights
            if np.asarray(w, dtype=float).shape[0] == n_assets
        ]
        if not valid_seeds:
            logger.warning(
                "All seed weight vectors have wrong dimension for %d assets; "
                "ignoring warm-start for this period",
                n_assets,
            )
            return []

        seeds: list[Solution] = []

        # 1. Inject every front solution as-is
        for w in valid_seeds:
            s = Solution(problem)
            s.variables[:] = repair_portfolio(
                np.asarray(w, dtype=float).copy()
            ).tolist()
            seeds.append(s)

        # 2. Add perturbations to fill remaining population capacity
        remaining = max(0, self.population_size - len(seeds))
        if remaining > 0 and valid_seeds:
            n_perturbations_per_seed = max(1, remaining // len(valid_seeds))
            for w in valid_seeds:
                base = np.asarray(w, dtype=float)
                for _ in range(n_perturbations_per_seed):
                    if len(seeds) >= self.population_size:
                        break
                    noise = np.random.normal(0, noise_scale, size=n_assets)
                    perturbed = repair_portfolio(np.clip(base + noise, 0.0, None))
                    s = Solution(problem)
                    s.variables[:] = perturbed.tolist()
                    seeds.append(s)

        logger.info(
            "Warm-start: %d front solutions + %d perturbations = %d seeds "
            "(population_size=%d)",
            len(valid_seeds),
            len(seeds) - len(valid_seeds),
            len(seeds),
            self.population_size,
        )
        return seeds[: self.population_size]

    # ------------------------------------------------------------------
    # Core optimization
    # ------------------------------------------------------------------

    def _get_algorithm_for_restart(
        self, restart_idx: int
    ) -> tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]:
        """Select algorithm and kwargs for a given restart index.

        When ``algorithm_portfolio`` is set, restarts cycle through the
        portfolio entries.  Otherwise, the default algorithm is used for
        every restart.

        Parameters
        ----------
        restart_idx : int
            Zero-based restart index.

        Returns
        -------
        tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]
            ``(algorithm_class, algorithm_kwargs)`` for this restart.
        """
        if self.algorithm_portfolio:
            entry = self.algorithm_portfolio[
                restart_idx % len(self.algorithm_portfolio)
            ]
            return entry[0], dict(entry[1])
        return self.algorithm, dict(self.algorithm_kwargs)

    def _run_multi_start_optimization(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[Solution], list[list[Any]]]:
        """Run the island/ensemble multi-start evolutionary optimization.

        Each restart can optionally use a different algorithm from
        ``algorithm_portfolio`` (island model).  When warm-start seeds
        are available, *every* restart is seeded — but with increasing
        perturbation noise so that early restarts exploit the previous
        front while later restarts explore more broadly.

        Parameters
        ----------
        mu : np.ndarray
            Expected returns vector.
        cov : np.ndarray
            Covariance matrix.

        Returns
        -------
        tuple[list[Solution], list[list[Any]]]
            ``(global_front, all_archives)`` — the pruned non-dominated
            front and the raw archives from every restart.
        """
        n_assets = len(mu)
        is_warm = (
            self.seed_weights is not None
            and len(self.seed_weights) > 0
            and any(np.asarray(w).shape[0] == n_assets for w in self.seed_weights)
        )

        effective_generations = (
            int(self.n_generations * self.warm_start_generation_ratio)
            if is_warm
            else self.n_generations
        )
        if is_warm:
            logger.info(
                "Warm-starting optimizer: %d → %d evaluations (ratio=%.2f)",
                self.n_generations,
                effective_generations,
                self.warm_start_generation_ratio,
            )

        all_archives: list[list[Any]] = []

        for restart_idx in range(self.n_restarts):
            algo_cls, algo_kwargs = self._get_algorithm_for_restart(restart_idx)
            problem = self.model(
                mu=mu, cov=cov, **self.model_kwargs, **self.cost_kwargs
            )

            algo_kwargs.setdefault("generator", PortfolioGenerator())
            algorithm = algo_cls(
                problem,
                population_size=self.population_size,
                **algo_kwargs,
            )

            # Capture candidates before selection, including for ensemble algorithms.
            recorder = None
            if not hasattr(algorithm, "pareto_archive"):
                recorder = ArchivingEvaluator(algorithm.evaluator)
                algorithm.evaluator = recorder

            # Seed ALL restarts when warm (with increasing perturbation noise)
            if is_warm:
                noise_scale = (
                    SEED_NOISE_SCALES[restart_idx]
                    if restart_idx < len(SEED_NOISE_SCALES)
                    else SEED_NOISE_SCALES[-1]
                )
                seed_solutions = self._build_seed_solutions(
                    problem, n_assets, noise_scale=noise_scale
                )
                if seed_solutions and hasattr(algorithm, "_seed_solutions"):
                    algorithm._seed_solutions = seed_solutions
                    logger.info(
                        "Restart %d/%d [%s]: seeded with noise_scale=%.3f",
                        restart_idx + 1,
                        self.n_restarts,
                        algo_cls.__name__,
                        noise_scale,
                    )
                else:
                    logger.info(
                        "Restart %d/%d [%s]: no seed injection (unsupported or empty)",
                        restart_idx + 1,
                        self.n_restarts,
                        algo_cls.__name__,
                    )
            else:
                logger.info(
                    "Restart %d/%d [%s]: cold start",
                    restart_idx + 1,
                    self.n_restarts,
                    algo_cls.__name__,
                )

            algorithm.run(effective_generations)
            archive = (
                recorder.archive if recorder is not None else algorithm.pareto_archive
            )
            all_archives.append(list(archive))

            if progress_callback is not None:
                frac = (restart_idx + 1) / self.n_restarts
                progress_callback(
                    frac,
                    f"EA restart {restart_idx + 1}/{self.n_restarts} complete",
                )

        global_front = self._merge_nondominated(
            all_archives,
            target_size=self.population_size,
        )

        if not global_front:
            raise RuntimeError("No feasible solutions found by the optimization runs.")

        return global_front, all_archives

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot_optimization_results(
        self,
        global_front: list[Solution],
        all_archives: list[list[Any]],
        markowitz_front: list[Solution] | None = None,
        selected_solution: Solution | None = None,
    ) -> None:
        """Save all diagnostic Pareto-front plots.

        Parameters
        ----------
        global_front : list[Solution]
            Pruned non-dominated set from the main optimization.
        all_archives : list[list[Any]]
            Raw archives from every restart.
        markowitz_front : list[Solution] | None, optional
            Markowitz efficient-frontier solutions for overlay plots.
        selected_solution : Solution | None, optional
            The solution chosen by the selection strategy.  When provided,
            it is highlighted with a distinct marker on every plot.
        """
        if self.output_mode == OutputMode.STANDARD:
            plot_final_front(
                global_front,
                markowitz_front,
                self.output_directory,
                self.date_label,
                selected_solution,
            )
            return

        all_solutions = [s for archive in all_archives for s in archive]

        plot_solution(
            all_solutions,
            "pareto_front_all_runs",
            self.output_directory,
            self.date_label,
            selected_solution=selected_solution,
        )
        plot_solution(
            global_front,
            "pareto_front_non_dominated",
            self.output_directory,
            self.date_label,
            selected_solution=selected_solution,
        )

        if markowitz_front is not None:
            plot_solutions_with_markowitz_front(
                solutions=all_solutions,
                markowitz_front=markowitz_front,
                file_name="pareto_front_all_runs_vs_markowitz_front",
                output_directory=self.output_directory,
                date_label=self.date_label,
                selected_solution=selected_solution,
            )
            plot_solutions_with_markowitz_front(
                solutions=global_front,
                markowitz_front=markowitz_front,
                file_name="pareto_front_non_dominated_vs_markowitz_front",
                output_directory=self.output_directory,
                date_label=self.date_label,
                selected_solution=selected_solution,
            )

    # ------------------------------------------------------------------
    # Markowitz benchmark (independent, lightweight settings)
    # ------------------------------------------------------------------

    def compute_markowitz_front(
        self,
        n_evaluations: int = MARKOWITZ.n_evaluations,
        population_size: int = MARKOWITZ.pop_size,
        n_restarts: int = MARKOWITZ.n_restarts,
    ) -> list[Solution]:
        """Compute the Markowitz long-only efficient frontier.

        Delegates to :func:`~..utils.compute_markowitz_front`.  Kept as an
        instance method for backwards compatibility.

        Parameters
        ----------
        n_evaluations : int, optional
            Function evaluations per restart (default 20 000).
        population_size : int, optional
            GA population size (default 300).
        n_restarts : int, optional
            Independent multi-start runs (default 3).

        Returns
        -------
        list[Solution]
            Non-dominated Markowitz frontier solutions.
        """
        return cast(
            list[Solution],
            compute_markowitz_front(
                mu=np.asarray(self.expected_returns.values),
                cov=np.asarray(self.covariance_matrix.values),
                n_evaluations=n_evaluations,
                population_size=population_size,
                n_restarts=n_restarts,
                **self.cost_kwargs,
            ),
        )

    @staticmethod
    def _extract_weights_from_front(front: list[Solution]) -> list[np.ndarray]:
        """Extract weight vectors from a list of Platypus solutions.

        Parameters
        ----------
        front : list[Solution]
            Solutions whose ``variables`` represent portfolio weights.

        Returns
        -------
        list[np.ndarray]
            Weight vectors (one per solution).
        """
        return [np.asarray(s.variables, dtype=float) for s in front]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """Run the evolutionary optimization and return the full result.

        Orchestrates:

        1. Optionally compute the Markowitz efficient frontier.
        2. Assemble warm-start seeds according to ``warm_start_source``:
           - ``PREVIOUS``: use prior-period front only.
           - ``MARKOWITZ``: use Markowitz frontier solutions only.
           - ``BOTH``: combine Markowitz frontier + prior-period front.
        3. Multi-start evolutionary optimization (BPRC or other model).
        4. Save diagnostic plots.

        Portfolio **selection** is intentionally *not* performed here —
        that responsibility belongs to the investment strategy layer.

        Returns
        -------
        OptimizationResult
            Contains the Pareto front, raw archives, and optional
            Markowitz frontier.
        """
        mu = np.asarray(self.expected_returns.values)
        cov = np.asarray(self.covariance_matrix.values)
        use_markowitz = self.warm_start_source in (
            WarmStartSource.MARKOWITZ,
            WarmStartSource.BOTH,
        )

        # 1. Compute Markowitz frontier (for seeding and/or diagnostic plots)
        markowitz_front: list[Solution] | None = None
        if self.include_markowitz_front or use_markowitz:
            try:
                markowitz_front = self.compute_markowitz_front()
            except Exception as e:
                logger.exception("Failed to compute Markowitz frontier: %s", e)

        # 2. Assemble warm-start seeds based on warm_start_source
        markowitz_weights: list[np.ndarray] = (
            self._extract_weights_from_front(markowitz_front)
            if markowitz_front is not None
            else []
        )
        prior_weights: list[np.ndarray] = (
            list(self.seed_weights) if self.seed_weights is not None else []
        )

        if self.warm_start_source == WarmStartSource.NONE:
            effective_seeds = None
        elif self.warm_start_source == WarmStartSource.MARKOWITZ:
            effective_seeds = markowitz_weights or None
        elif self.warm_start_source == WarmStartSource.BOTH:
            combined = markowitz_weights + prior_weights
            effective_seeds = combined or None
        else:  # WarmStartSource.PREVIOUS
            effective_seeds = prior_weights or None

        if effective_seeds:
            logger.info(
                "Warm-start source=%s: %d Markowitz + %d prior = %d seeds",
                self.warm_start_source.value,
                len(markowitz_weights)
                if self.warm_start_source
                not in (WarmStartSource.PREVIOUS, WarmStartSource.NONE)
                else 0,
                len(prior_weights)
                if self.warm_start_source
                not in (WarmStartSource.MARKOWITZ, WarmStartSource.NONE)
                else 0,
                len(effective_seeds),
            )
        else:
            logger.info(
                "Warm-start source=%s: cold start", self.warm_start_source.value
            )

        # Temporarily swap seed_weights so _run_multi_start_optimization
        # picks up the assembled seeds without mutating the original
        original_seeds = self.seed_weights
        self.seed_weights = effective_seeds

        try:
            # 3. Core BPRC optimization
            global_front, all_archives = self._run_multi_start_optimization(
                mu,
                cov,
                progress_callback=progress_callback,
            )
        finally:
            # Restore original seeds (don't leak Markowitz weights into
            # the strategy's persistent state)
            self.seed_weights = original_seeds

        # 4. Diagnostic plots (without selection highlight — selection happens after)
        self._plot_optimization_results(global_front, all_archives, markowitz_front)

        return OptimizationResult(
            front=global_front,
            all_archives=all_archives,
            markowitz_front=markowitz_front,
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
        self._plot_optimization_results(
            result.front,
            result.all_archives,
            result.markowitz_front,
            selected_solution=result.selected_solution,
        )
