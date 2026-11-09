"""Orchestrator for side-by-side strategy comparison.

Loads the market data **once**, deep-copies it for each strategy, runs
independent backtests, and feeds the collected results into
:class:`~investment_tool.output.strategy_comparator.StrategyComparator`.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode
from investment_tool.output.run_metadata import (
    write_run_metadata,
    record_input_snapshot,
)

import copy
import logging
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from investment_tool.app.backtesting_service import BacktestingService
from investment_tool.app.data_readiness import require_ready
from investment_tool.app.strategy_variant import (
    StrategyVariant,
    SelectionMethod,
    build_selector,
)
from investment_tool.strategies.optimization_strategies.front_cache import (
    CandidateFrontCache,
)
from investment_tool.app.execution_coordinator import _build_strategy
from investment_tool.config import AppConfig, get_config
from investment_tool.constants import (
    DataSource,
    ExecutionMode,
    Frequency,
    StrategyType,
    OptimizationReturnEstimator,
    PortfolioBalance,
)
from investment_tool.defaults import CONSTRUCTION, STRATEGY_OPTIMIZER
from investment_tool.data.assembly import ModelDataFactory
from investment_tool.data.validation import ValidationLevel
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.domain.backtesting.backtester import Backtester
from investment_tool.output.strategy_comparator import StrategyComparator
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.selection_strategies import (
    BaseSelectionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyComparisonParameters:
    """Configuration for a multi-strategy comparison run.

    Parameters
    ----------
    strategy_types : list[StrategyType]
        Strategies to backtest and compare.
    data_source : DataSource
        Source of financial data.
    frequency : Frequency
        Data frequency.
    output_directory : Path
        Root output directory.
    input_data_path : Path | None, optional
        Path to local input file.
    save_input_data : bool, optional
        Whether to save fetched input data.
    validation_level : ValidationLevel, optional
        Validation strictness.
    benchmark_asset : str, optional
        Asset symbol for the buy-and-hold benchmark, by default ``"SPY"``.
    include_benchmark : bool, optional
        Whether to include a BUY_AND_HOLD benchmark automatically,
        by default ``True``.
    min_assets : int, optional
        Minimum number of active assets in the portfolio (default 2).
    max_assets : int, optional
        Maximum number of active assets in the portfolio (default 5).
    cfg : AppConfig | None, optional
        Typed application configuration.  When ``None``, the default
        config (from files/environment) is used.
    """

    strategy_types: list[StrategyType]
    data_source: DataSource
    frequency: Frequency
    output_directory: Path
    input_data_path: Path | None = None
    save_input_data: bool = False
    validation_level: ValidationLevel = ValidationLevel.WARN
    benchmark_asset: str = "SPY"
    include_benchmark: bool = True
    min_assets: int = CONSTRUCTION.min_assets
    max_assets: int = CONSTRUCTION.max_assets
    population_size: int = STRATEGY_OPTIMIZER.pop_size
    n_generations: int = STRATEGY_OPTIMIZER.n_generations
    cfg: AppConfig | None = None
    optimization_selection: SelectionMethod = "sharpe"
    optimization_volatility_increase: float = 0.25
    optimization_return_target_fraction: float = 0.75
    strategy_variants: list[StrategyVariant] = field(default_factory=list)
    output_mode: OutputMode = OutputMode.STANDARD


def compare_strategies(
    params: StrategyComparisonParameters,
    progress_callback: ProgressCallback | None = None,
    partial_sink: dict[str, Any] | None = None,
    partial_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run backtests for multiple strategies and compare results.

    Parameters
    ----------
    params : StrategyComparisonParameters
        Comparison configuration.
    progress_callback : ProgressCallback | None, optional
        Called periodically with ``(fraction_complete, message)``.
        Each strategy's backtest receives a rescaled sub-callback.
    partial_sink : dict | None, optional
        If provided, completed period snapshots and portfolio history are
        written into ``partial_sink["individual_results"]``.
    partial_callback : Callable | None, optional
        Called after each period so the worker can persist a checkpoint.

    Returns
    -------
    dict[str, Any]
        ``metrics_df``, ``saved_files``, ``output_dir``, and per-strategy
        ``individual_results``.
    """
    # Validate labels before fetching data; every result/output folder is unique.
    names = [v.name.casefold() for v in params.strategy_variants]
    if len(names) != len(set(names)) or any(
        not v.name.strip()
        or v.name in {".", ".."}
        or any(c in v.name for c in "/\\<>")
        or any(ord(c) < 32 for c in v.name)
        or v.name.casefold() in {t.value for t in StrategyType}
        for v in params.strategy_variants
    ):
        raise ValueError(
            "Configuration names must be unique and distinct from built-in names"
        )
    if not params.strategy_types and not params.strategy_variants:
        raise ValueError("Select a strategy or configuration")
    for variant in params.strategy_variants:
        PortfolioBalance(variant.balance)
        if variant.news_top_assets < 1 or variant.news_min_articles < 1:
            raise ValueError("News shortlist settings must be positive")
        maximum_assets = (
            variant.max_assets if variant.max_assets is not None else params.max_assets
        )
        if variant.news_shortlist and variant.news_top_assets < maximum_assets:
            raise ValueError("News shortlist size must be at least maximum assets")
        if variant.strategy_type != StrategyType.OPTIMIZATION and (
            variant.balance != PortfolioBalance.RISK_CONTRIBUTIONS
            or variant.news_shortlist
        ):
            raise ValueError(
                "Balance and news selection overrides apply only to Optimization"
            )
        if variant.return_estimator is not None:
            OptimizationReturnEstimator(variant.return_estimator)
            if variant.strategy_type != StrategyType.OPTIMIZATION:
                raise ValueError(
                    "Return estimator overrides apply only to Optimization"
                )
    # 1. Load model data once
    risk_free_rate = (params.cfg or get_config()).model.risk_free_rate
    if partial_sink is not None:
        partial_sink.update(
            frequency=params.frequency.value, risk_free_rate=risk_free_rate
        )
    logger.info("Loading model data for strategy comparison …")
    write_run_metadata(params)
    model_data: ModelData = ModelDataFactory.create_model_data(
        data_source=params.data_source,
        execution_mode=ExecutionMode.BACKTESTING,
        frequency=params.frequency,
        input_data_path=params.input_data_path,
        save_input_data=params.save_input_data,
        validation_level=params.validation_level,
        cfg=params.cfg,
    )
    record_input_snapshot(params.output_directory, model_data.input_snapshot_path)
    if partial_sink is not None:
        partial_sink["validation_report"] = model_data.validation_report
        partial_sink["output_dir"] = params.output_directory
    require_ready(model_data.validation_report, params.output_directory)

    # 2. Determine the list of strategies to run
    configured_types = {v.strategy_type for v in params.strategy_variants}
    strategy_types = list(
        dict.fromkeys(t for t in params.strategy_types if t not in configured_types)
    )
    # A configured single-asset baseline can also be the reference; do not
    # execute the same asset twice merely because its preset was migrated.
    benchmark_variant = (
        next(
            (
                v
                for v in params.strategy_variants
                if v.strategy_type == StrategyType.BUY_AND_HOLD
                and (v.asset_symbol or "SPY") == params.benchmark_asset
            ),
            None,
        )
        if params.include_benchmark
        else None
    )
    if (
        params.include_benchmark
        and StrategyType.BUY_AND_HOLD not in strategy_types
        and benchmark_variant is None
    ):
        strategy_types.append(StrategyType.BUY_AND_HOLD)

    entries = [(t, t.value, None) for t in strategy_types] + [
        (v.strategy_type, v.name, v) for v in params.strategy_variants
    ]
    # One cache namespace per invocation: never reuse a previous run/retry's
    # front after a code update or when a new experiment is requested.
    front_cache_directory = (
        params.output_directory / "candidate_fronts" / uuid.uuid4().hex
    )
    # 3. Run each configuration through the same factory/backtester.
    all_results: dict[str, list[dict[str, Any]]] = {}
    individual_outputs: dict[str, dict[str, Any]] = {}

    benchmark_name: str | None = (
        benchmark_variant.name if benchmark_variant is not None else None
    )

    # Pre-check: verify benchmark asset is available for BUY_AND_HOLD
    available_symbols = {s.symbol for s in model_data.stocks}
    benchmark_asset = params.benchmark_asset
    if benchmark_asset not in available_symbols:
        logger.warning(
            "Benchmark asset '%s' not in available symbols %s — "
            "falling back to first available symbol '%s'",
            benchmark_asset,
            sorted(available_symbols),
            sorted(available_symbols)[0],
        )
        benchmark_asset = sorted(available_symbols)[0]

    for idx, (stype, name, variant) in enumerate(entries):
        logger.info("Running backtest for strategy: %s", name)

        md_copy = copy.deepcopy(model_data)
        output_dir = params.output_directory / name
        output_dir.mkdir(parents=True, exist_ok=True)

        extra_kwargs: dict[str, Any] = {}
        selector: BaseSelectionStrategy | None = None
        selection_diagnostics: list[dict[str, Any]] = []
        if stype == StrategyType.BUY_AND_HOLD:
            extra_kwargs["asset_symbol"] = benchmark_asset
        elif stype in (
            StrategyType.COMBINED,
            StrategyType.OPTIMIZATION,
            StrategyType.SENTIMENT_EWM,
        ):
            extra_kwargs["population_size"] = params.population_size
            extra_kwargs["n_generations"] = params.n_generations

        if variant is not None:
            if variant.uses_optimizer:
                selector = build_selector(
                    variant.selection,
                    variant.volatility_increase,
                    variant.return_target_fraction,
                )
        elif (
            stype == StrategyType.OPTIMIZATION
            and params.optimization_selection != "sharpe"
        ):
            selector = build_selector(
                params.optimization_selection,
                params.optimization_volatility_increase,
                params.optimization_return_target_fraction,
            )
        if selector is not None:
            extra_kwargs["selection_strategy"] = selector

        strategy = _build_strategy(
            strategy_type=stype,
            model_data=md_copy,
            output_directory=output_dir,
            output_mode=params.output_mode,
            min_assets=params.min_assets,
            max_assets=params.max_assets,
            configuration=variant,
            **extra_kwargs,
        )

        if variant is not None and variant.uses_optimizer:
            strategy.candidate_front_cache = CandidateFrontCache(
                front_cache_directory, stype.value, params.output_mode
            )

        configuration = (
            {
                **asdict(variant),
                "return_estimator": (
                    OptimizationReturnEstimator(
                        variant.return_estimator or OptimizationReturnEstimator.EWM_6
                    ).value
                    if stype == StrategyType.OPTIMIZATION
                    else None
                ),
                "population_size": (variant.population_size or params.population_size)
                if variant.uses_optimizer
                else None,
                "n_generations": (variant.n_generations or params.n_generations)
                if variant.uses_optimizer
                else None,
                "min_assets": (
                    variant.min_assets
                    if variant.min_assets is not None
                    else params.min_assets
                )
                if variant.uses_optimizer
                else None,
                "max_assets": (
                    variant.max_assets
                    if variant.max_assets is not None
                    else params.max_assets
                )
                if variant.uses_optimizer
                else None,
            }
            if variant is not None
            else {"strategy_type": stype.value}
        )
        if variant is not None:
            if stype == StrategyType.BUY_AND_HOLD:
                configuration["asset_symbol"] = variant.asset_symbol or "SPY"
            elif stype == StrategyType.MOMENTUM_RISK_PARITY:
                configuration.update(
                    {
                        key: getattr(strategy, key)
                        for key in (
                            "top_n",
                            "lookback",
                            "vol_window",
                            "trend_window",
                            "min_weight",
                        )
                    }
                )
        # Build a per-strategy sub-callback that scales into
        # this strategy's fraction of overall progress.
        strategy_cb: ProgressCallback | None = None
        if progress_callback is not None:
            n_strategies = len(entries)
            base = idx / n_strategies
            span = 1.0 / n_strategies

            def strategy_cb(
                frac: float, msg: str, _b: float = base, _s: float = span
            ) -> None:
                progress_callback(_b + frac * _s, f"[{name}] {msg}")

        def period_complete(backtester: Backtester) -> None:
            if selector is not None:
                selection_diagnostics.append(
                    {
                        "period_start": backtester.results[-1].get("period_start"),
                        **(
                            {"sharpe_weights": dict(selector.reference_weights)}
                            if hasattr(selector, "reference_weights")
                            else {}
                        ),
                        **getattr(selector, "last_diagnostics", {}),
                    }
                )
            individual_outputs[name] = {
                "results": list(backtester.results),
                "portfolio_history": list(md_copy.portfolio.history),
                "selection_diagnostics": list(selection_diagnostics),
                "configuration": configuration,
            }
            if partial_sink is not None:
                partial_sink["partial"] = True
                partial_sink["individual_results"] = dict(individual_outputs)
                partial_sink["output_dir"] = params.output_directory
                if partial_callback is not None:
                    partial_callback(partial_sink)

        service = BacktestingService(
            model_data=md_copy,
            investment_strategy=strategy,
        )
        backtester = service.run_backtest(
            progress_callback=strategy_cb,
            period_complete_callback=period_complete,
        )

        all_results[name] = backtester.results
        individual_outputs[name] = {
            "results": backtester.results,
            "portfolio_history": md_copy.portfolio.history,
            "selection_diagnostics": selection_diagnostics,
            "configuration": configuration,
        }

        if partial_sink is not None:
            partial_sink["individual_results"] = dict(individual_outputs)
            partial_sink["output_dir"] = params.output_directory

        if stype == StrategyType.BUY_AND_HOLD and variant is None:
            benchmark_name = name

    # 4. Compare
    comparator = StrategyComparator(
        results=all_results,
        frequency=params.frequency,
        output_dir=params.output_directory,
        output_mode=params.output_mode,
        benchmark_name=benchmark_name,
        risk_free_rate=risk_free_rate,
    )
    comparison = comparator.run()

    return {
        "metrics_df": comparison["metrics_df"],
        "saved_files": comparison["saved_files"],
        "output_dir": params.output_directory,
        "individual_results": individual_outputs,
        "validation_report": model_data.validation_report,
        "frequency": params.frequency.value,
        "risk_free_rate": risk_free_rate,
        "metric_definitions": comparison.get("metric_definitions", {}),
    }
