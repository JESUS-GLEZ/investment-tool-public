"""Solver comparison coordinator.

App-layer orchestrator that loads market data, builds the portfolio
problem, runs the solver benchmark, and persists outputs.  Called by
the CLI entrypoint.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode
from investment_tool.output.run_metadata import (
    write_run_metadata,
    record_input_snapshot,
)

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
from platypus import NSGAII, NSGAIII, SPEA2, EpsNSGAII

from investment_tool.config import AppConfig
from investment_tool.constants import DataSource, ExecutionMode, Frequency, StockFields
from investment_tool.data.assembly import ModelDataFactory
from investment_tool.data.validation import ValidationLevel
from investment_tool.defaults import (
    ALGORITHM,
    CONSTRUCTION,
    GENETIC,
    OPTIMIZER,
    PREDICTION,
)
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    PortfolioVariator,
)
from investment_tool.strategies.optimization_strategies.custom_algorithm.gwasfga import (
    GWASFGA,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.solver_comparison import (
    SolverBenchmark,
    SolverConfig,
)
from investment_tool.utils import (
    build_exp_weighted_covariance_matrix,
    recently_observed_symbols,
)

from .comparison_variant import SolverVariant, validate_names

logger = logging.getLogger(__name__)


_DEFAULT_SOLVERS = ["NSGA-II", "NSGA-III", "SPEA2", "EpsNSGAII", "GWASFGA"]


@dataclass
class SolverComparisonParameters:
    """Parameters for the solver comparison run.

    Parameters
    ----------
    data_source : DataSource
        Source of financial data.
    frequency : Frequency
        Data frequency.
    output_directory : Path
        Directory where comparison outputs will be saved.
    input_data_path : Path | None
        Path to local input data file.
    save_input_data : bool
        Whether to save fetched input data.
    validation_level : ValidationLevel
        Validation strictness.
    population_size : int
        GA population size for all solvers.
    n_generations : int
        Function evaluations per restart for all solvers.
    n_restarts : int
        Independent multi-start runs for all solvers.
    min_assets : int
        Minimum number of assets to hold.
    num_assets : int
        Maximum number of assets to hold.
    include_markowitz_reference : bool
        Whether to compute a Markowitz frontier as reference.
    solvers : list[str]
        Solver labels to include (e.g. ``["GWASFGA", "NSGA-II"]``).
    use_multi_greedy_seeds : bool
        Generate feasible risk-parity seeds with bounded subset search.
    seed_max_subsets : int
        Maximum unique subset solves, including unsuccessful solves.
    seed_beam_width : int
        Candidates retained for each of eleven normalized risk preferences.
    cfg : AppConfig | None, optional
        Typed application configuration.  When ``None``, the default
        config (from files/environment) is used.
    """

    data_source: DataSource
    frequency: Frequency
    output_directory: Path
    input_data_path: Path | None = None
    save_input_data: bool = False
    validation_level: ValidationLevel = ValidationLevel.WARN
    population_size: int = OPTIMIZER.pop_size
    n_generations: int = OPTIMIZER.n_generations
    n_restarts: int = OPTIMIZER.n_restarts
    min_assets: int = CONSTRUCTION.min_assets
    num_assets: int = CONSTRUCTION.max_assets
    include_markowitz_reference: bool = True
    solvers: list[str] = field(default_factory=lambda: list(_DEFAULT_SOLVERS))
    use_multi_greedy_seeds: bool = True
    seed_max_subsets: int = 2000
    seed_beam_width: int = 5
    cfg: AppConfig | None = None
    output_mode: OutputMode = OutputMode.STANDARD
    solver_variants: list[SolverVariant] | None = None


# Type for solver builder functions
_SolverBuilder = Callable[[SolverComparisonParameters], SolverConfig]

# Registry of available solver builders
_SOLVER_REGISTRY: dict[str, _SolverBuilder] = {}


def _register_solver(label: str) -> Callable[[_SolverBuilder], _SolverBuilder]:
    """Decorator to register a solver builder function."""

    def decorator(func: _SolverBuilder) -> _SolverBuilder:
        _SOLVER_REGISTRY[label.upper()] = func
        return func

    return decorator


@_register_solver("GWASFGA")
def _build_gwasfga(params: SolverComparisonParameters) -> SolverConfig:
    return SolverConfig(
        label="GWASFGA",
        algorithm=GWASFGA,
        algorithm_kwargs={
            "variator": PortfolioVariator(
                min_assets=params.min_assets,
                max_assets=params.num_assets,
                mutation_prob=GENETIC.mutation_prob,
            ),
        },
        population_size=params.population_size,
        n_generations=params.n_generations,
        n_restarts=params.n_restarts,
    )


@_register_solver("NSGA-II")
def _build_nsgaii(params: SolverComparisonParameters) -> SolverConfig:
    return SolverConfig(
        label="NSGA-II",
        algorithm=NSGAII,
        algorithm_kwargs={
            "variator": PortfolioVariator(
                min_assets=params.min_assets,
                max_assets=params.num_assets,
                mutation_prob=GENETIC.mutation_prob,
            ),
        },
        population_size=params.population_size,
        n_generations=params.n_generations,
        n_restarts=params.n_restarts,
    )


@_register_solver("NSGA-III")
def _build_nsgaiii(params: SolverComparisonParameters) -> SolverConfig:
    return SolverConfig(
        label="NSGA-III",
        algorithm=NSGAIII,
        algorithm_kwargs={
            # Two objectives: p divisions yield p+1 directions. Platypus rounds
            # the resulting population up to a multiple of four.
            "divisions_outer": max(1, params.population_size - 1),
            "variator": PortfolioVariator(
                min_assets=params.min_assets,
                max_assets=params.num_assets,
                mutation_prob=GENETIC.mutation_prob,
            ),
        },
        population_size=params.population_size,
        n_generations=params.n_generations,
        n_restarts=params.n_restarts,
    )


@_register_solver("SPEA2")
def _build_spea2(params: SolverComparisonParameters) -> SolverConfig:
    return SolverConfig(
        label="SPEA2",
        algorithm=SPEA2,
        algorithm_kwargs={
            "variator": PortfolioVariator(
                min_assets=params.min_assets,
                max_assets=params.num_assets,
                mutation_prob=GENETIC.mutation_prob,
            ),
        },
        population_size=params.population_size,
        n_generations=params.n_generations,
        n_restarts=params.n_restarts,
    )


@_register_solver("EPSNSGAII")
def _build_epsnsgaii(params: SolverComparisonParameters) -> SolverConfig:
    return SolverConfig(
        label="EpsNSGAII",
        algorithm=EpsNSGAII,
        algorithm_kwargs={
            "epsilons": list(ALGORITHM.epsilons),
            "variator": PortfolioVariator(
                min_assets=params.min_assets,
                max_assets=params.num_assets,
                mutation_prob=GENETIC.mutation_prob,
            ),
        },
        population_size=params.population_size,
        n_generations=params.n_generations,
        n_restarts=params.n_restarts,
    )


def get_available_solvers() -> list[str]:
    """Return labels of all registered solvers.

    Returns
    -------
    list[str]
        Available solver labels.
    """
    return sorted(_SOLVER_REGISTRY.keys())


def _build_solver_configs(
    params: SolverComparisonParameters,
) -> list[SolverConfig]:
    """Build SolverConfig list from parameter labels.

    Parameters
    ----------
    params : SolverComparisonParameters
        Run parameters with solver labels.

    Returns
    -------
    list[SolverConfig]
        Configured solvers ready for benchmarking.

    Raises
    ------
    ValueError
        If a requested solver label is not registered.
    """
    configs: list[SolverConfig] = []
    if params.solver_variants:
        validate_names([variant.name for variant in params.solver_variants])
        for variant in params.solver_variants:
            effective = replace(
                params,
                **{
                    key: getattr(variant, key)
                    for key in ("population_size", "n_generations", "n_restarts")
                    if getattr(variant, key) is not None
                },
            )
            config = _build_solver_configs(
                replace(effective, solver_variants=None, solvers=[variant.solver])
            )[0]
            kwargs = dict(config.algorithm_kwargs)
            if variant.mutation_prob is not None:
                kwargs["variator"] = PortfolioVariator(
                    min_assets=params.min_assets,
                    max_assets=params.num_assets,
                    mutation_prob=variant.mutation_prob,
                )
            if variant.divisions_outer is not None:
                kwargs["divisions_outer"] = variant.divisions_outer
            if variant.epsilons is not None:
                kwargs["epsilons"] = list(variant.epsilons)
            configs.append(
                replace(
                    config,
                    label=variant.name.strip(),
                    algorithm_kwargs=kwargs,
                    random_seed=variant.random_seed,
                )
            )
        return configs
    for label in params.solvers:
        builder = _SOLVER_REGISTRY.get(label.upper())
        if builder is None:
            available = ", ".join(get_available_solvers())
            raise ValueError(f"Unknown solver '{label}'. Available: {available}")
        configs.append(builder(params))
    return configs


def _build_problem_inputs(
    model_data: ModelData,
) -> tuple[pd.Series, pd.DataFrame]:
    """Extract expected returns and covariance from model data.

    Uses each eligible asset's EWM mean and the same regularized covariance
    estimator as the optimization investment strategy.

    Parameters
    ----------
    model_data : ModelData
        Loaded market data.

    Returns
    -------
    tuple[pd.Series, pd.DataFrame]
        ``(expected_returns, covariance_matrix)`` ready for the optimizer.
    """
    stocks = model_data.stocks

    # Build returns matrix
    returns_dict: dict[str, pd.Series] = {}
    for stock in stocks:
        returns_series = stock.data[StockFields.RETURNS].dropna()
        if not returns_series.empty:
            returns_dict[stock.symbol] = returns_series

    returns_df = pd.DataFrame(returns_dict).sort_index()
    eligible = recently_observed_symbols(returns_df)
    if not eligible:
        raise ValueError("No assets have enough recent return history")
    returns_df = returns_df[eligible]

    # EWM expected returns
    expected_returns = returns_df.ewm(span=PREDICTION.ewm_span).mean().iloc[-1]

    cov = build_exp_weighted_covariance_matrix(returns_df)

    return expected_returns, cov


def execute_solver_comparison(
    params: SolverComparisonParameters,
    progress_callback: ProgressCallback | None = None,
    partial_sink: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestrate the full solver comparison pipeline.

    1. Load market data.
    2. Compute expected returns and covariance.
    3. Optionally compute Markowitz reference frontier.
    4. Run all requested solvers.
    5. Save comparison outputs.

    Parameters
    ----------
    params : SolverComparisonParameters
        Execution parameters.
    progress_callback : ProgressCallback | None, optional
        Called periodically with ``(fraction_complete, message)``.
        Forwarded to the solver benchmark loop.

    Returns
    -------
    dict[str, Any]
        Results dictionary with ``summary_table``, ``saved_files``,
        and ``output_dir``.
    """
    logger.info("Starting solver comparison")
    solver_configs = _build_solver_configs(params)

    # 1. Load data
    write_run_metadata(params)
    model_data: ModelData = ModelDataFactory.create_model_data(
        data_source=params.data_source,
        execution_mode=ExecutionMode.INVESTING,
        frequency=params.frequency,
        input_data_path=params.input_data_path,
        save_input_data=params.save_input_data,
        validation_level=params.validation_level,
        cfg=params.cfg,
    )
    record_input_snapshot(params.output_directory, model_data.input_snapshot_path)

    # 2. Build problem inputs
    expected_returns, cov = _build_problem_inputs(model_data)
    logger.info(
        "Problem: %d assets, expected returns range [%.4f, %.4f]",
        len(expected_returns),
        expected_returns.min(),
        expected_returns.max(),
    )

    # 3. Optional Markowitz reference
    from investment_tool.domain.trading_costs import decision_trading_costs

    costs = decision_trading_costs(
        model_data.portfolio,
        model_data.stocks,
        model_data.instance.current_trading_period,
        expected_returns.index,
        model_data.corporate_action_events,
    )
    cost_kwargs = {"trading_costs": costs} if costs is not None else {}
    reference_front = None
    if params.include_markowitz_reference:
        logger.info("Computing Markowitz reference frontier")
        from investment_tool.strategies.optimization_strategies.convex_frontier import (
            convex_markowitz_front,
        )

        reference_front = convex_markowitz_front(
            expected_returns.values, cov.values, **cost_kwargs
        )

    # 4. Build solver configs and run benchmark
    benchmark = SolverBenchmark(
        expected_returns=expected_returns,
        covariance_matrix=cov,
        model=BalancedPortfoliosRiskContributions,
        model_kwargs={
            **cost_kwargs,
            "l_assets": params.min_assets,
            "u_assets": params.num_assets,
            "constraint_slack": OPTIMIZER.constraint_slack,
            # Solver comparison uses 2-objective mode (deliberate override)
            "hhi_as_constraint": True,
            "hhi_bound": CONSTRUCTION.hhi_bound,
        },
        output_directory=params.output_directory,
        reference_front=reference_front,
    )

    seed_seconds = 0.0
    seed_subsets = 0
    if params.use_multi_greedy_seeds:
        from investment_tool.strategies.optimization_strategies.multi_greedy import (
            multi_greedy_seeds,
        )

        if progress_callback:
            progress_callback(0.0, "Generating Multi-Greedy risk-parity seeds")
        started = time.perf_counter()
        seed_problem = BalancedPortfoliosRiskContributions(
            mu=expected_returns.values,
            cov=cov.values,
            **benchmark.model_kwargs,
        )
        benchmark.seed_front, seed_subsets = multi_greedy_seeds(
            seed_problem,
            max_subsets=params.seed_max_subsets,
            beam_width=params.seed_beam_width,
            progress=(lambda fraction, message: progress_callback(0.0, message))
            if progress_callback
            else None,
        )
        seed_seconds = time.perf_counter() - started
    results = benchmark.run(
        solver_configs, progress_callback=progress_callback, partial_sink=partial_sink
    )
    for result in results:
        result.metrics["seed_generation_seconds"] = seed_seconds
        result.metrics["total_time_with_seeds_s"] = (
            result.wall_time_seconds + seed_seconds
        )

    # 5. Save outputs
    saved_files = benchmark.save_comparison(output_mode=params.output_mode)

    # Log summary
    summary = benchmark.summary_table()
    logger.info("\n%s", summary.to_string())

    return {
        "summary_table": summary,
        "saved_files": saved_files,
        "output_dir": params.output_directory,
        "results": results,
        "reference_front": reference_front,
        "seed_front": benchmark.seed_front,
        "trading_costs": costs.record() if costs is not None else None,
        "covariance_diagnostics": cov.attrs.get("covariance_diagnostics", {}),
        "seed_diagnostics": {
            "enabled": params.use_multi_greedy_seeds,
            "subsets_evaluated": seed_subsets,
            "front_size": len(benchmark.seed_front),
            "wall_time_seconds": seed_seconds,
        },
    }
