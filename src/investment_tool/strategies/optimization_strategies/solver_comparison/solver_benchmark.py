"""Solver benchmark runner.

Runs multiple evolutionary solvers on the same portfolio problem,
computes quality metrics, and generates comparison outputs.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from platypus import AbstractGeneticAlgorithm, Solution

from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.optimization_strategies.model import PortfolioModel
from investment_tool.strategies.optimization_strategies.solver_comparison.comparison_metrics import (
    compute_all_metrics,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.comparison_plots import (
    plot_metric_bars,
    plot_overlaid_fronts,
    plot_radar_chart,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.solver_config import (
    SolverConfig,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.solver_result import (
    SolverResult,
)
from investment_tool.strategies.optimization_strategies.utils import (
    merge_nondominated,
    sharpe_like,
)

logger = logging.getLogger(__name__)


class SolverBenchmark:
    """Run multiple solvers on the same portfolio problem and compare.

    All solvers receive the exact same ``(μ, Σ, model, model_kwargs)``
    so differences in results are purely from the solver.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected returns per asset, indexed by symbol.
    covariance_matrix : pd.DataFrame
        Covariance matrix of asset returns.
    model : type[PortfolioModel]
        Portfolio model class (e.g. ``BalancedPortfoliosRiskContributions``).
    model_kwargs : dict[str, Any] | None
        Extra kwargs for the model constructor.
    output_directory : Path | None
        Directory for comparison outputs.
    reference_front : list[Solution] | None
        Pre-computed reference front (e.g. Markowitz) for hypervolume.
    """

    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        model: type[PortfolioModel],
        model_kwargs: dict[str, Any] | None = None,
        output_directory: Path | None = None,
        reference_front: list[Solution] | None = None,
        seed_front: list[Solution] | None = None,
    ) -> None:
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.model = model
        self.model_kwargs = model_kwargs or {}
        self.output_directory = output_directory
        self.reference_front = reference_front
        self.seed_front = sorted(seed_front or [], key=lambda s: s.objectives[0])
        self._results: list[SolverResult] = []

    @property
    def results(self) -> list[SolverResult]:
        """Results from the last :meth:`run` call."""
        return self._results

    def run(
        self,
        solvers: list[SolverConfig],
        progress_callback: ProgressCallback | None = None,
        partial_sink: dict | None = None,
    ) -> list[SolverResult]:
        """Run all solvers and compute comparison metrics.

        Parameters
        ----------
        solvers : list[SolverConfig]
            Solver configurations to benchmark.
        progress_callback : ProgressCallback | None, optional
            Called after each solver completes with
            ``(fraction_complete, message)``.
        partial_sink : dict | None, optional
            If provided, after each solver completes the accumulated
            results list is written into ``partial_sink["results"]``.
            This allows callers to retrieve partial results if the run
            is interrupted mid-loop.

        Returns
        -------
        list[SolverResult]
            One result per solver, in the same order.

        Raises
        ------
        ValueError
            If *solvers* is empty.
        """
        if not solvers:
            raise ValueError("At least one SolverConfig is required.")

        mu = np.asarray(self.expected_returns.values)
        cov = np.asarray(self.covariance_matrix.values)

        # Compute a shared reference point for hypervolume across all solvers
        ref_point = self._compute_reference_point(solvers, mu, cov)

        results: list[SolverResult] = []
        n_solvers = len(solvers)
        for idx, config in enumerate(solvers):
            logger.info(
                "Benchmarking solver '%s' (pop=%d, evals=%d, restarts=%d)",
                config.label,
                config.population_size,
                config.n_generations,
                config.n_restarts,
            )

            def report(fraction: float, message: str, solver_idx: int = idx) -> None:
                if progress_callback is not None:
                    progress_callback((solver_idx + fraction) / n_solvers, message)

            report(0.0, f"Starting solver '{config.label}' ({idx + 1}/{n_solvers})")
            result = self._run_single_solver(config, mu, cov, ref_point, report)
            results.append(result)
            if partial_sink is not None:
                partial_sink["results"] = list(results)
            logger.info(
                "Solver '%s' completed in %.2fs — hypervolume=%.6f, "
                "best_sharpe=%.4f, cardinality=%d",
                config.label,
                result.wall_time_seconds,
                result.metrics.get("hypervolume", 0.0),
                result.metrics.get("best_sharpe", 0.0),
                int(result.metrics.get("cardinality", 0)),
            )
            if progress_callback is not None:
                progress_callback(
                    (idx + 1) / n_solvers,
                    f"Solver '{config.label}' complete ({idx + 1}/{n_solvers})",
                )

        self._results = results
        return results

    def summary_table(self) -> pd.DataFrame:
        """Build a DataFrame comparing all solvers side-by-side.

        Returns
        -------
        pd.DataFrame
            Rows = solvers, columns = metrics + wall_time.
        """
        rows: list[dict[str, Any]] = []
        for r in self._results:
            row: dict[str, Any] = {"solver": r.config.label}
            row.update(r.metrics)
            row["wall_time_s"] = round(r.wall_time_seconds, 2)
            rows.append(row)
        return pd.DataFrame(rows).set_index("solver")

    def save_comparison(
        self, output_mode: OutputMode = OutputMode.DIAGNOSTIC
    ) -> list[str]:
        """Save plots and CSV summary to ``output_directory``.

        Returns
        -------
        list[str]
            Paths to all saved files.
        """
        saved: list[str] = []

        # CSV summary
        if self._results:
            table = self.summary_table()
            out_dir = (
                Path(self.output_directory) / "solver_comparison"
                if self.output_directory
                else Path.cwd() / "solver_comparison"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "solver_comparison_summary.csv"
            table.to_csv(csv_path)
            logger.info("Saved comparison summary: %s", csv_path)
            saved.append(str(csv_path))

        if not output_mode.save_chart_exports:
            return saved

        # Plots
        path = plot_overlaid_fronts(
            self._results, self.reference_front, self.output_directory, self.seed_front
        )
        if path:
            saved.append(path)

        path = plot_metric_bars(self._results, self.output_directory)
        if path:
            saved.append(path)

        path = plot_radar_chart(self._results, self.output_directory)
        if path:
            saved.append(path)

        return saved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_reference_point(
        self,
        solvers: list[SolverConfig],
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Compute a shared hypervolume reference point.

        If a ``reference_front`` was provided at construction, use its
        extremes.  Otherwise, derive a conservative reference from the
        data itself.

        Parameters
        ----------
        solvers : list[SolverConfig]
            Solver configs (used for fallback budget estimation).
        mu, cov : np.ndarray
            Expected returns and covariance.

        Returns
        -------
        np.ndarray
            Reference point ``[max_var, max_neg_return]``.
        """
        costs = self.model_kwargs.get("trading_costs")
        if costs is not None and costs.rate > 0:
            # Long-only simplex portfolios have variance <= max diagonal and
            # turnover <= 1 + all held risky weight (including forced exits).
            # A convex MV curve need not bound the constrained search's worst
            # net return, so use shared conservative data bounds for this case.
            worst = np.array(
                [
                    float(np.max(np.diag(cov))),
                    -float(np.min(mu))
                    + costs.rate
                    * (1 + float(costs.current_weights.sum()) + costs.outside_weight),
                ]
            )
            bounded: np.ndarray = worst + np.maximum(np.abs(worst) * 0.2, 1e-6)
            return bounded
        if self.reference_front:
            points = np.array(
                [[s.objectives[0], s.objectives[1]] for s in self.reference_front]
            )
            margin = 0.2 * (points.max(axis=0) - points.min(axis=0))
            margin = np.maximum(margin, 1e-6)
            ref: np.ndarray = points.max(axis=0) + margin
            return ref

        # Fallback: use diagonal portfolio (equal weight) as a worst-case anchor
        n = len(mu)
        w_eq = np.full(n, 1.0 / n)
        var_eq = float(w_eq @ cov @ w_eq)
        ret_eq = float(w_eq @ mu)
        return np.array([var_eq * 3.0, abs(ret_eq) * 3.0])

    def _run_single_solver(
        self,
        config: SolverConfig,
        mu: np.ndarray,
        cov: np.ndarray,
        ref_point: np.ndarray,
        progress_callback: ProgressCallback | None = None,
    ) -> SolverResult:
        """Run one solver and compute its metrics.

        Parameters
        ----------
        config : SolverConfig
            Solver configuration.
        mu, cov : np.ndarray
            Expected returns and covariance.
        ref_point : np.ndarray
            Shared hypervolume reference point.

        Returns
        -------
        SolverResult
            Complete result with front, metrics, and best weights.
        """
        all_archives: list[list[Solution]] = []
        actual_evaluations = 0
        initial_population_size = 0
        t_start = time.perf_counter()

        for restart_idx in range(config.n_restarts):
            from contextlib import nullcontext
            from investment_tool.strategies.optimization_strategies.seeded_optimization import (
                _seeded_randomness,
            )

            context = (
                nullcontext()
                if config.random_seed is None
                else _seeded_randomness((config.random_seed + restart_idx) % 2**32)
            )
            with context:
                problem = self.model(mu=mu, cov=cov, **self.model_kwargs)
                from investment_tool.strategies.optimization_strategies.multi_greedy import (
                    SeededPortfolioGenerator,
                )

                kwargs = dict(config.algorithm_kwargs)
                # Fresh generator for every solver/restart; at least half remains random.
                seeds = sorted(self.seed_front, key=lambda s: s.objectives[0])
                algorithm = config.algorithm(
                    problem,
                    population_size=config.population_size,
                    **kwargs,
                )
                initial_population_size = algorithm.population_size
                if "generator" not in kwargs:
                    # NSGA-III may derive a smaller/larger population from directions.
                    algorithm.generator = SeededPortfolioGenerator(
                        seeds, initial_population_size // 2
                    )
                last_report = 0.0
                observed_population = False

                def report_generation(
                    algo: AbstractGeneticAlgorithm, restart: int = restart_idx
                ) -> None:
                    nonlocal last_report, initial_population_size, observed_population
                    if not observed_population:
                        # GWASFGA finalizes/rounds its size during initialize().
                        initial_population_size = algo.population_size
                        observed_population = True
                    now = time.monotonic()
                    if progress_callback is not None and now - last_report >= 1.0:
                        last_report = now
                        fraction = min(1.0, algo.nfe / max(1, config.n_generations))
                        progress_callback(
                            (restart + fraction) / config.n_restarts,
                            f"{config.label}: restart {restart + 1}/{config.n_restarts}, "
                            f"{algo.nfe}/{config.n_generations} evaluations",
                        )

                algorithm.run(config.n_generations, callback=report_generation)
                actual_evaluations += algorithm.nfe
                all_archives.append(list(algorithm.result))

        wall_time = time.perf_counter() - t_start

        # Merge and prune
        front = merge_nondominated(
            all_archives,
            target_size=config.population_size,
        )

        # Metrics
        metrics = compute_all_metrics(front, ref_point)
        metrics["wall_time"] = wall_time
        metrics["requested_evaluations"] = float(
            config.n_generations * config.n_restarts
        )
        metrics["actual_evaluations"] = float(actual_evaluations)
        metrics["initial_population_size"] = float(initial_population_size)

        # Best portfolio
        best_weights = pd.Series(dtype=float)
        if front:
            best_sol = max(front, key=sharpe_like)
            weights = np.asarray(best_sol.variables, dtype=float)
            best_weights = pd.Series(weights, index=self.expected_returns.index)
            total = best_weights.sum()
            if not np.isclose(total, 1.0, rtol=1e-5):
                best_weights = best_weights / total

        return SolverResult(
            config=config,
            front=front,
            all_archives=all_archives,
            wall_time_seconds=wall_time,
            metrics=metrics,
            best_weights=best_weights,
        )
