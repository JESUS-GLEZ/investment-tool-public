"""Adapter layer: converts API request schemas to coordinator Parameters.

Handles infrastructure concerns (output directories, config loading)
that are internal to the API and not exposed to clients.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from investment_tool.app.comparison_variant import SolverVariant, PredictorVariant
from pathlib import Path

from .local_data import resolve_dataset_path

from investment_tool.app.execution_coordinator import (
    Parameters as ExecParams,
)
from investment_tool.app.prediction_comparison_coordinator import (
    PredictionComparisonParameters,
)
from investment_tool.app.solver_comparison_coordinator import (
    SolverComparisonParameters,
)
from investment_tool.app.strategy_comparison_coordinator import (
    StrategyComparisonParameters,
)
from investment_tool.app.strategy_variant import StrategyVariant
from investment_tool.config import load_config, AppConfig, ModelConfig
from investment_tool.constants import DataSource, ExecutionMode, Frequency, StrategyType
from investment_tool.data.validation import ValidationLevel
from investment_tool.output.policy import OutputMode
from investment_tool.exceptions import DataValidationError

from api.schemas import (
    ExecutionRequest,
    PredictionComparisonRequest,
    SolverComparisonRequest,
    StrategyComparisonRequest,
)

_OUTPUT_BASE = Path("/app/data/output")
_INPUT_BASE = Path("/app/data/input")


def _make_output_dir(run_id: str) -> Path:
    """Create a persistent output directory for a run under /app/data/output/<run_id>.

    Falls back to a temp directory when the persistent path is not writable
    (e.g. during local development without the volume mount).
    """
    d = _OUTPUT_BASE / run_id
    try:
        d.mkdir(parents=True, exist_ok=True)
        # Quick write-test
        (d / ".keep").touch()
        return d
    except OSError:
        fallback = Path(tempfile.mkdtemp(prefix=f"api_run_{run_id}_"))
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _resolve_local_data_path(filename: str | None = None) -> Path:
    """Return the path to a local .pkl dataset file.

    Local requests must include an explicit filename so the API never silently
    falls back to whichever dataset happened to be the newest on disk.
    """
    data_dir = _INPUT_BASE
    if not filename or not filename.strip():
        raise DataValidationError(
            "input_file is required when data_source is 'local'",
            field="input_file",
        )

    try:
        return resolve_dataset_path(data_dir, filename)
    except ValueError as exc:
        raise DataValidationError(
            str(exc),
            field="input_file",
        ) from exc


def _build_cfg(
    historical_start_date: str,
    historical_end_date: str,
    initial_cash: float,
    transaction_cost_rate: float,
    risk_free_rate: float,
    asset_config_file: str,
    backtest_start_date: str | None = None,
) -> AppConfig:
    """Build a custom AppConfig with user-supplied date and financial params."""
    base = load_config()
    # Replace only the model fields the user specified; keep everything else.
    custom_model = ModelConfig(
        historical_start_date=historical_start_date,
        historical_end_date=historical_end_date,
        initial_cash=initial_cash,
        min_backtest_periods=base.model.min_backtest_periods,
        start_index_percentage_backtesting=base.model.start_index_percentage_backtesting,
        transaction_cost_rate=transaction_cost_rate,
        risk_free_rate=risk_free_rate,
        backtest_start_date=backtest_start_date,
    )
    return AppConfig(
        alpaca=base.alpaca,
        alpha_vantage=base.alpha_vantage,
        model=custom_model,
        asset_config_file=asset_config_file,
        paths=base.paths,
        database=base.database,
        redis=base.redis,
        jwt=base.jwt,
    )


def build_exec_params(req: ExecutionRequest, run_id: str) -> ExecParams:
    """Convert an :class:`ExecutionRequest` to coordinator Parameters."""
    ds = DataSource(req.data_source)
    return ExecParams(
        data_source=ds,
        execution_mode=ExecutionMode(req.execution_mode),
        frequency=Frequency(req.frequency),
        output_directory=_make_output_dir(run_id),
        output_mode=OutputMode(req.output_mode),
        strategy_type=StrategyType(req.strategy_type),
        current_holdings={
            holding.symbol: holding.quantity for holding in req.current_holdings
        },
        strategy_configuration=StrategyVariant(
            **{
                **req.strategy_configuration.model_dump(),
                "strategy_type": StrategyType(req.strategy_configuration.strategy_type),
            }
        )
        if req.strategy_configuration is not None
        else None,
        validation_level=ValidationLevel(req.validation_level),
        min_assets=req.min_assets,
        max_assets=req.max_assets,
        save_input_data=ds != DataSource.LOCAL,
        input_data_path=_resolve_local_data_path(req.input_file)
        if ds == DataSource.LOCAL
        else None,
        cfg=_build_cfg(
            req.historical_start_date,
            req.historical_end_date,
            req.initial_cash,
            req.transaction_cost_rate,
            req.risk_free_rate,
            req.asset_config_file,
            req.backtest_start_date,
        ),
    )


def build_strategy_comparison_params(
    req: StrategyComparisonRequest,
    run_id: str,
) -> StrategyComparisonParameters:
    """Convert a :class:`StrategyComparisonRequest` to coordinator params."""
    ds = DataSource(req.data_source)
    return StrategyComparisonParameters(
        strategy_types=[StrategyType(s) for s in req.strategy_types],
        data_source=ds,
        frequency=Frequency(req.frequency),
        output_directory=_make_output_dir(run_id),
        output_mode=OutputMode(req.output_mode),
        validation_level=ValidationLevel(req.validation_level),
        benchmark_asset=req.benchmark_asset,
        include_benchmark=req.include_benchmark,
        min_assets=req.min_assets,
        max_assets=req.max_assets,
        population_size=req.population_size,
        n_generations=req.n_generations,
        optimization_selection=req.optimization_selection,
        optimization_volatility_increase=req.optimization_volatility_increase,
        optimization_return_target_fraction=req.optimization_return_target_fraction,
        strategy_variants=[
            StrategyVariant(
                **{**v.model_dump(), "strategy_type": StrategyType(v.strategy_type)}
            )
            for v in req.strategy_variants
        ],
        save_input_data=ds != DataSource.LOCAL,
        input_data_path=_resolve_local_data_path(req.input_file)
        if ds == DataSource.LOCAL
        else None,
        cfg=_build_cfg(
            req.historical_start_date,
            req.historical_end_date,
            req.initial_cash,
            req.transaction_cost_rate,
            req.risk_free_rate,
            req.asset_config_file,
            req.backtest_start_date,
        ),
    )


def build_solver_comparison_params(
    req: SolverComparisonRequest,
    run_id: str,
) -> SolverComparisonParameters:
    """Convert a :class:`SolverComparisonRequest` to coordinator params."""
    ds = DataSource(req.data_source)
    return SolverComparisonParameters(
        data_source=ds,
        frequency=Frequency(req.frequency),
        output_directory=_make_output_dir(run_id),
        output_mode=OutputMode(req.output_mode),
        validation_level=ValidationLevel(req.validation_level),
        population_size=req.population_size,
        n_generations=req.n_generations,
        n_restarts=req.n_restarts,
        min_assets=req.min_assets,
        num_assets=req.num_assets,
        include_markowitz_reference=req.include_markowitz_reference,
        use_multi_greedy_seeds=req.use_multi_greedy_seeds,
        seed_max_subsets=req.seed_max_subsets,
        seed_beam_width=req.seed_beam_width,
        solvers=req.solvers,
        solver_variants=[SolverVariant(**asdict(v)) for v in req.solver_variants]
        if req.solver_variants
        else None,
        save_input_data=ds != DataSource.LOCAL,
        input_data_path=_resolve_local_data_path(req.input_file)
        if ds == DataSource.LOCAL
        else None,
        cfg=_build_cfg(
            req.historical_start_date,
            req.historical_end_date,
            req.initial_cash,
            req.transaction_cost_rate,
            req.risk_free_rate,
            req.asset_config_file,
            req.backtest_start_date,
        ),
    )


def build_prediction_comparison_params(
    req: PredictionComparisonRequest,
    run_id: str,
) -> PredictionComparisonParameters:
    """Convert a :class:`PredictionComparisonRequest` to coordinator params."""
    ds = DataSource(req.data_source)
    return PredictionComparisonParameters(
        data_source=ds,
        frequency=Frequency(req.frequency),
        output_directory=_make_output_dir(run_id),
        output_mode=OutputMode(req.output_mode),
        validation_level=ValidationLevel(req.validation_level),
        n_eval_periods=req.n_eval_periods,
        min_history=req.min_history,
        top_k=req.top_k,
        predictors=req.predictors,
        predictor_variants=[
            PredictorVariant(**asdict(v)) for v in req.predictor_variants
        ]
        if req.predictor_variants
        else None,
        compute_sentiment=req.compute_sentiment,
        save_input_data=ds != DataSource.LOCAL,
        input_data_path=_resolve_local_data_path(req.input_file)
        if ds == DataSource.LOCAL
        else None,
        cfg=_build_cfg(
            req.historical_start_date,
            req.historical_end_date,
            req.initial_cash,
            req.transaction_cost_rate,
            req.risk_free_rate,
            req.asset_config_file,
            req.backtest_start_date,
        ),
    )
