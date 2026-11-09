"""
Execution coordinator for the Investment Tool.

This module orchestrates the main execution flow, handling both live investing
and backtesting modes. It coordinates data loading, strategy execution, and
output persistence.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode
from investment_tool.output.run_metadata import (
    write_run_metadata,
    record_input_snapshot,
)

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from investment_tool.app.input_freshness import input_freshness
from investment_tool.app.strategy_variant import StrategyVariant
from investment_tool.app.backtesting_service import BacktestingService
from investment_tool.app.data_readiness import require_ready
from investment_tool.config import AppConfig, get_config
from investment_tool.constants import DataSource, ExecutionMode, Frequency, StrategyType
from investment_tool.defaults import CONSTRUCTION
from investment_tool.data.assembly import ModelDataFactory
from investment_tool.data.validation import ValidationLevel
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.orchestrators import (
    InvestmentStrategyBuyAndHold,
    InvestmentStrategyCombined,
    InvestmentStrategyMomentumRiskParity,
    InvestmentStrategyOptimization,
    InvestmentStrategySentimentEWM,
)
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.output import OutputManager

logger = logging.getLogger(__name__)


@dataclass
class Parameters:
    """Execution parameters container.

    Encapsulates all configuration needed to run the investment tool.

    Parameters
    ----------
    data_source : DataSource
        Source of financial data (Alpaca API or local file).
    execution_mode : ExecutionMode
        Mode of execution (INVESTING or BACKTESTING).
    frequency : Frequency
        Data frequency (DAILY, WEEKLY, MONTHLY).
    output_directory : Path
        Directory where output files will be saved.
    strategy_type : StrategyType, optional
        Investment strategy to use.  Default is ``COMBINED``.
    input_data_path : Path | None, optional
        Path to input data file for local data source.
    save_input_data : bool, optional
        Whether to save fetched input data locally.
    validation_level : ValidationLevel, optional
        How to handle validation errors (STRICT, WARN, SKIP).
    min_assets : int, optional
        Minimum number of active assets in the portfolio (default 2).
    max_assets : int, optional
        Maximum number of active assets in the portfolio (default 5).
    cfg : AppConfig | None, optional
        Typed application configuration.  When ``None``, the default
        config (from files/environment) is used.
    """

    data_source: DataSource
    execution_mode: ExecutionMode
    frequency: Frequency
    output_directory: Path
    strategy_type: StrategyType = StrategyType.COMBINED
    input_data_path: Path | None = None
    save_input_data: bool = False
    validation_level: ValidationLevel = ValidationLevel.WARN
    min_assets: int = CONSTRUCTION.min_assets
    max_assets: int = CONSTRUCTION.max_assets
    cfg: AppConfig | None = None
    output_mode: OutputMode = OutputMode.STANDARD
    strategy_configuration: StrategyVariant | None = None
    current_holdings: dict[str, float] = field(default_factory=dict)


def execute(
    parameters: Parameters,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Orchestrate investment strategy execution.

    Main entry point that coordinates data loading, strategy execution,
    and output generation based on the execution mode.

    Parameters
    ----------
    parameters : Parameters
        Execution configuration.
    progress_callback : ProgressCallback | None, optional
        Called periodically with ``(fraction_complete, message)``.
        Threaded to the strategy's ``run_strategy()`` call.

    Returns
    -------
    dict[str, Any]
        Results dictionary containing portfolio, weights, metrics, etc.

    Raises
    ------
    ValueError
        If an unsupported execution mode is provided.
    RuntimeError
        For critical errors during execution.
    """
    if (
        parameters.current_holdings
        and parameters.execution_mode != ExecutionMode.INVESTING
    ):
        raise ValueError("Current holdings apply only to investment recommendations")
    write_run_metadata(parameters)
    model_data: ModelData = ModelDataFactory.create_model_data(
        data_source=parameters.data_source,
        execution_mode=parameters.execution_mode,
        frequency=parameters.frequency,
        input_data_path=parameters.input_data_path,
        save_input_data=parameters.save_input_data,
        validation_level=parameters.validation_level,
        cfg=parameters.cfg,
    )
    record_input_snapshot(parameters.output_directory, model_data.input_snapshot_path)

    require_ready(model_data.validation_report, parameters.output_directory)
    if parameters.execution_mode == ExecutionMode.INVESTING:
        from investment_tool.app.recommendation import initialize_holdings
        from investment_tool.data.universes.loader import load_asset_universe

        universe = load_asset_universe(
            (parameters.cfg or get_config()).asset_config_file
        )
        if (
            set(parameters.current_holdings)
            - set(universe.stocks)
            - set(universe.crypto)
        ):
            raise ValueError("Current holdings must belong to the selected universe")
        initialize_holdings(model_data, parameters.current_holdings)
    investment_strategy = _build_strategy(
        strategy_type=parameters.strategy_type,
        model_data=model_data,
        output_directory=parameters.output_directory,
        min_assets=parameters.min_assets,
        max_assets=parameters.max_assets,
        output_mode=parameters.output_mode,
        configuration=parameters.strategy_configuration,
    )

    if parameters.execution_mode == ExecutionMode.INVESTING:
        return _execute_live_investment(
            model_data,
            investment_strategy,
            parameters.output_directory,
            progress_callback=progress_callback,
        )
    elif parameters.execution_mode == ExecutionMode.BACKTESTING:
        return _execute_backtesting(
            model_data,
            investment_strategy,
            parameters.output_directory,
            progress_callback=progress_callback,
            risk_free_rate=(parameters.cfg or get_config()).model.risk_free_rate,
        )
    else:
        raise ValueError(f"Unsupported execution mode: {parameters.execution_mode}")


def _build_strategy(
    strategy_type: StrategyType,
    model_data: ModelData,
    output_directory: Path,
    min_assets: int = CONSTRUCTION.min_assets,
    max_assets: int = CONSTRUCTION.max_assets,
    output_mode: OutputMode = OutputMode.STANDARD,
    configuration: StrategyVariant | None = None,
    **extra_kwargs: Any,
) -> BaseInvestmentStrategy:
    if configuration is not None:
        if configuration.strategy_type != strategy_type:
            raise ValueError("Configuration strategy must match strategy_type")
        extra_kwargs.update(
            configuration.constructor_kwargs(extra_kwargs.get("selection_strategy"))
        )
        min_assets = (
            configuration.min_assets
            if configuration.min_assets is not None
            else min_assets
        )
        max_assets = (
            configuration.max_assets
            if configuration.max_assets is not None
            else max_assets
        )
    strategy = _instantiate_strategy(
        strategy_type,
        model_data,
        output_directory,
        min_assets,
        max_assets,
        **extra_kwargs,
    )
    strategy.output_mode = output_mode
    if configuration is not None:
        strategy.optimization_settings = configuration.search_settings()
    return strategy


def _instantiate_strategy(
    strategy_type: StrategyType,
    model_data: ModelData,
    output_directory: Path,
    min_assets: int = CONSTRUCTION.min_assets,
    max_assets: int = CONSTRUCTION.max_assets,
    **extra_kwargs: Any,
) -> BaseInvestmentStrategy:
    """Instantiate the investment strategy selected by *strategy_type*.

    Parameters
    ----------
    strategy_type : StrategyType
        Which strategy to build.
    model_data : ModelData
        Loaded model data (instance, stocks, news, portfolio).
    output_directory : Path
        Run output directory.
    min_assets : int, optional
        Minimum number of active assets (default 2).
    max_assets : int, optional
        Maximum number of active assets (default 5).
    **extra_kwargs : Any
        Strategy-specific keyword arguments (e.g. ``asset_symbol`` for
        ``BUY_AND_HOLD``).

    Returns
    -------
    BaseInvestmentStrategy
        Ready-to-use strategy instance.

    Raises
    ------
    ValueError
        If *strategy_type* is not recognised.
    """
    common_kwargs: dict[str, Any] = dict(
        instance=model_data.instance,
        stocks=model_data.stocks,
        news=model_data.news,
        output_directory=output_directory,
    )
    asset_kwargs: dict[str, Any] = dict(
        min_assets=min_assets,
        max_assets=max_assets,
    )

    if strategy_type == StrategyType.COMBINED:
        return InvestmentStrategyCombined(
            **common_kwargs, **asset_kwargs, **extra_kwargs
        )
    if strategy_type == StrategyType.OPTIMIZATION:
        return InvestmentStrategyOptimization(
            **common_kwargs, **asset_kwargs, **extra_kwargs
        )
    if strategy_type == StrategyType.MOMENTUM_RISK_PARITY:
        return InvestmentStrategyMomentumRiskParity(**common_kwargs, **extra_kwargs)
    if strategy_type == StrategyType.BUY_AND_HOLD:
        return InvestmentStrategyBuyAndHold(**common_kwargs, **extra_kwargs)
    if strategy_type == StrategyType.SENTIMENT_EWM:
        return InvestmentStrategySentimentEWM(
            **common_kwargs, **asset_kwargs, **extra_kwargs
        )

    raise ValueError(f"Unsupported strategy type: {strategy_type}")


def _execute_live_investment(
    model_data: ModelData,
    investment_strategy: BaseInvestmentStrategy,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute live investment mode.

    Runs the investment strategy once, estimates trades on a copy, and
    generates output via the OutputManager.

    Parameters
    ----------
    model_data : ModelData
        Loaded model data with stocks, periods, and portfolio.
    investment_strategy : BaseInvestmentStrategy
        Strategy to generate optimal weights.
    output_dir : Path
        Directory for output files.
    progress_callback : ProgressCallback | None, optional
        Threaded to the strategy's ``run_strategy()`` call.

    Returns
    -------
    dict[str, Any]
        Dictionary with portfolio, optimal_weights, concentration, and output_dir.
    """
    investment_strategy.bind_portfolio(
        model_data.portfolio, model_data.corporate_action_events
    )
    optimal_weights = investment_strategy.run_strategy(
        progress_callback=progress_callback,
    )
    freshness = input_freshness(
        model_data.instance,
        investment_strategy.stocks,
        model_data.stocks,
        model_data.validation_report,
    )
    from investment_tool.app.recommendation import estimate_rebalance

    rebalance_plan = estimate_rebalance(model_data, optimal_weights)

    # Build portfolio snapshot
    snapshot: dict[str, Any] = {}
    try:
        latest = (
            model_data.portfolio.history[-1] if model_data.portfolio.history else {}
        )
        if "date" in latest:
            latest["date"] = str(latest["date"])
        snapshot = {
            "latest": latest,
            "cash": model_data.portfolio.cash,
            "positions": model_data.portfolio.positions,
            "data_freshness": freshness,
            "rebalance_plan": rebalance_plan,
        }
    except Exception as e:
        logger.warning(f"Failed building portfolio snapshot: {e}")

    # Generate output via OutputManager
    manager = OutputManager(
        output_dir=output_dir,
        frequency=model_data.instance.frequency,
        output_mode=investment_strategy.output_mode,
    )
    output = manager.generate_live_output(
        optimal_weights=optimal_weights,
        portfolio_snapshot=snapshot,
    )

    return {
        "frequency": model_data.instance.frequency.value,
        "data_asof": freshness["latest_available_at"],
        "data_freshness": freshness,
        "validation_report": model_data.validation_report,
        "portfolio": model_data.portfolio,
        "optimal_weights": optimal_weights,
        "rebalance_plan": rebalance_plan,
        "optimization_diagnostics": investment_strategy.last_optimization_diagnostics,
        "concentration": output.get("concentration"),
        "output_dir": output_dir,
    }


def _execute_backtesting(
    model_data: ModelData,
    investment_strategy: BaseInvestmentStrategy,
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
    risk_free_rate: float = 0.0,
) -> dict[str, Any]:
    """Execute backtesting mode.

    Runs the investment strategy across historical periods, computes
    comprehensive performance metrics, and generates all output artifacts.

    Parameters
    ----------
    model_data : ModelData
        Loaded model data with historical periods.
    investment_strategy : BaseInvestmentStrategy
        Strategy to apply at each rebalancing period.
    output_dir : Path
        Directory for output files.
    progress_callback : ProgressCallback | None, optional
        Threaded to the backtesting service.

    Returns
    -------
    dict[str, Any]
        Dictionary with metrics, results_df, saved_files, and output_dir.
    """
    backtesting_service = BacktestingService(
        model_data=model_data,
        investment_strategy=investment_strategy,
    )

    backtester = backtesting_service.run_backtest(
        progress_callback=progress_callback,
    )

    # Generate comprehensive output via OutputManager
    manager = OutputManager(
        output_dir=output_dir,
        frequency=model_data.instance.frequency,
        risk_free_rate=risk_free_rate,
        output_mode=investment_strategy.output_mode,
    )
    output = manager.generate_backtest_output(
        results=backtester.results,
        portfolio_history=model_data.portfolio.history,
    )

    return {
        "metrics": output["metrics"],
        "results_df": output["results_df"],
        "validation_report": model_data.validation_report,
        "saved_files": output.get("saved_files", []),
        "output_dir": output_dir,
    }
