"""Pydantic request / response schemas for the Investment Tool API.

These mirror the coordinator ``Parameters`` dataclasses from the engine,
with sensible defaults pulled from ``investment_tool.defaults``.
"""

from __future__ import annotations

from investment_tool.data.universes.loader import (
    DEFAULT_ASSET_CONFIG_FILE,
    load_asset_universe,
)
from investment_tool.defaults import STRATEGY_OPTIMIZER
from investment_tool.output.policy import OutputMode

import enum
from datetime import date, datetime, timezone
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from .local_data import validate_dataset_filename
from investment_tool.app.comparison_variant import (
    SolverVariant,
    PredictorVariant,
    validate_names,
)
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as validated_dataclass


from investment_tool.constants import (
    DataSource,
    ExecutionMode,
    Frequency,
    StrategyType,
    OptimizationReturnEstimator,
    PortfolioBalance,
)
from investment_tool.data.validation import ValidationLevel
from investment_tool.defaults import CONSTRUCTION, OPTIMIZER


# Reject misspelled/unsupported fields rather than silently ignoring a parameter.
@validated_dataclass(
    frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False)
)
class SolverVariantInput(SolverVariant):
    pass


@validated_dataclass(
    frozen=True, config=ConfigDict(extra="forbid", allow_inf_nan=False)
)
class PredictorVariantInput(PredictorVariant):
    pass


# ---------------------------------------------------------------------------
# Enum re-exports for OpenAPI docs
# ---------------------------------------------------------------------------


class RunStatus(str, enum.Enum):
    """Status of a run in the store."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunType(str, enum.Enum):
    """Kind of run."""

    EXECUTION = "execution"
    STRATEGY_COMPARISON = "strategy_comparison"
    SOLVER_COMPARISON = "solver_comparison"
    PREDICTION_COMPARISON = "prediction_comparison"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------


class _LocalInputFileRequest(BaseModel):
    model_config = {"allow_inf_nan": False}

    campaign_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{12}$")
    output_mode: OutputMode = OutputMode.STANDARD
    asset_config_file: str = DEFAULT_ASSET_CONFIG_FILE
    historical_start_date: str = "2020-01-01"
    backtest_start_date: str | None = None
    historical_end_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).date().isoformat()
    )

    @model_validator(mode="after")
    def _validate_historical_window(self) -> Self:
        start = date.fromisoformat(self.historical_start_date)
        end = date.fromisoformat(self.historical_end_date)
        if self.backtest_start_date is not None:
            trading_start = date.fromisoformat(self.backtest_start_date)
            if not start <= trading_start <= end:
                raise ValueError(
                    "Backtest start date must lie in the historical window"
                )
        if start > end:
            raise ValueError("Historical start date must be on or before end date")
        return self

    @model_validator(mode="after")
    def _require_input_file_for_local_data(self) -> Self:
        data_source = getattr(self, "data_source", None)
        input_file = getattr(self, "input_file", None)

        if data_source in (DataSource.LOCAL, DataSource.LOCAL.value):
            if not isinstance(input_file, str) or not input_file.strip():
                raise ValueError("input_file is required when data_source is 'local'")
            validate_dataset_filename(input_file)

        load_asset_universe(self.asset_config_file)
        variants = getattr(self, "solver_variants", None) or getattr(
            self, "predictor_variants", None
        )
        if variants:
            validate_names([variant.name for variant in variants])
            if hasattr(self, "solver_variants"):
                from investment_tool.app.solver_comparison_coordinator import (
                    get_available_solvers,
                )

                available = get_available_solvers()
                if any(variant.solver.upper() not in available for variant in variants):
                    raise ValueError("Unknown solver configuration")
            else:
                from investment_tool.app.prediction_comparison_coordinator import (
                    get_available_predictors,
                )

                available = get_available_predictors()
                if any(
                    variant.predictor.upper() not in available for variant in variants
                ):
                    raise ValueError("Unknown predictor configuration")
                if getattr(self, "frequency", None) != "monthly" and any(
                    variant.predictor.upper() in {"MOMENTUMRIDGE", "MOMENTUM12-2RIDGE"}
                    for variant in variants
                ):
                    raise ValueError("Momentum ridge predictors require monthly data")
        return self


class StrategyVariantRequest(BaseModel):
    """A named configuration of an existing investment strategy."""

    name: str = Field(min_length=1, max_length=80)
    strategy_type: Literal[
        "optimization",
        "combined",
        "sentiment_ewm",
        "buy_and_hold",
        "momentum_risk_parity",
    ] = "optimization"
    return_estimator: OptimizationReturnEstimator | None = None
    balance: PortfolioBalance = PortfolioBalance.RISK_CONTRIBUTIONS
    news_shortlist: bool = False
    news_top_assets: int = Field(default=50, ge=1, le=10000)
    news_min_articles: int = Field(default=3, ge=1, le=1000)
    selection: Literal["sharpe", "return_with_risk_budget", "return_target"] = "sharpe"
    return_target_fraction: float = Field(default=0.75, ge=0, le=1)
    volatility_increase: float = Field(default=0.25, ge=0, le=1)
    population_size: int | None = Field(default=None, ge=10)
    n_generations: int | None = Field(default=None, ge=1)
    min_assets: int | None = Field(default=None, ge=1)
    max_assets: int | None = Field(default=None, ge=1)
    n_restarts: int = Field(default=STRATEGY_OPTIMIZER.n_restarts, ge=1, le=20)
    random_seed: int = Field(
        default=STRATEGY_OPTIMIZER.random_seed, ge=0, le=4294967295
    )
    search_seconds: float | None = Field(
        default=STRATEGY_OPTIMIZER.search_seconds, gt=0
    )
    seed_max_subsets: int = Field(
        default=STRATEGY_OPTIMIZER.seed_max_subsets, ge=1, le=360000
    )
    seed_beam_width: int = Field(
        default=STRATEGY_OPTIMIZER.seed_beam_width, ge=1, le=100
    )
    refinement_supports: int = Field(
        default=STRATEGY_OPTIMIZER.refinement_supports, ge=0, le=500
    )
    asset_symbol: str | None = Field(
        default=None, min_length=1, max_length=30, pattern=r"^[A-Za-z0-9./_-]+$"
    )
    top_n: int | None = Field(default=None, ge=1)
    lookback: int | None = Field(default=None, ge=1)
    vol_window: int | None = Field(default=None, ge=2)
    trend_window: int | None = Field(default=None, ge=1)
    min_weight: float | None = Field(default=None, ge=0, lt=1)
    model_config = {"extra": "forbid", "allow_inf_nan": False}

    @model_validator(mode="after")
    def validate_name(self) -> Self:
        if self.return_estimator is not None and self.strategy_type != "optimization":
            raise ValueError("Return estimator overrides apply only to Optimization")
        if self.strategy_type != "optimization" and (
            self.balance != PortfolioBalance.RISK_CONTRIBUTIONS or self.news_shortlist
        ):
            raise ValueError(
                "Balance and news selection overrides apply only to Optimization"
            )
        if self.asset_symbol is not None and self.strategy_type != "buy_and_hold":
            raise ValueError("Asset symbol applies only to Buy & Hold")
        if (
            any(
                getattr(self, key) is not None
                for key in (
                    "top_n",
                    "lookback",
                    "vol_window",
                    "trend_window",
                    "min_weight",
                )
            )
            and self.strategy_type != "momentum_risk_parity"
        ):
            raise ValueError("Momentum settings apply only to Momentum Risk Parity")
        if (
            self.min_assets is not None
            and self.max_assets is not None
            and self.min_assets > self.max_assets
        ):
            raise ValueError("Minimum assets cannot exceed maximum assets")
        self.name = self.name.strip()
        if any(c in self.name for c in "/\\<>") or any(ord(c) < 32 for c in self.name):
            raise ValueError(
                "Configuration names cannot contain path separators, markup or control characters"
            )
        if (
            not self.name
            or self.name in {".", ".."}
            or self.name.casefold() in {s.value for s in StrategyType}
        ):
            raise ValueError(
                "Variant names must be nonblank and different from built-in strategy names"
            )
        return self


class HoldingInput(BaseModel):
    """Fractional shares/units already owned, valued in USD by the engine."""

    symbol: str = Field(min_length=1, max_length=30)
    quantity: float = Field(gt=0)
    model_config = {"extra": "forbid", "allow_inf_nan": False}


class ExecutionRequest(_LocalInputFileRequest):
    """Request body for ``POST /runs``."""

    data_source: DataSource = DataSource.ALPACA
    execution_mode: ExecutionMode = ExecutionMode.BACKTESTING
    frequency: Frequency = Frequency.MONTHLY
    strategy_type: StrategyType = StrategyType.COMBINED
    validation_level: ValidationLevel = ValidationLevel.WARN
    min_assets: int = Field(default=CONSTRUCTION.min_assets, ge=1)
    max_assets: int = Field(default=CONSTRUCTION.max_assets, ge=1)
    # Portfolio parameters
    initial_cash: float = Field(default=1000.0, ge=0)
    current_holdings: list[HoldingInput] = Field(default_factory=list, max_length=1000)
    transaction_cost_rate: float = Field(default=0.001, ge=0)
    risk_free_rate: float = Field(default=0.0, ge=0)
    # Local dataset selection (ignored when data_source != "local")
    input_file: str | None = None

    strategy_configuration: StrategyVariantRequest | None = None

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def validate_asset_bounds(self) -> Self:
        if self.current_holdings and self.execution_mode != ExecutionMode.INVESTING:
            raise ValueError(
                "Current holdings apply only to investment recommendations"
            )
        if self.initial_cash == 0 and not self.current_holdings:
            raise ValueError("Positive cash or current holdings are required")
        symbols = [holding.symbol for holding in self.current_holdings]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Current holding symbols must be unique")
        universe = load_asset_universe(self.asset_config_file)
        unknown = set(symbols) - set(universe.stocks) - set(universe.crypto)
        if unknown:
            raise ValueError(
                f"Holdings must belong to the selected universe: {sorted(unknown)}"
            )
        if self.min_assets > self.max_assets:
            raise ValueError("Minimum assets cannot exceed maximum assets")
        variant = self.strategy_configuration
        if variant is not None:
            if variant.strategy_type != self.strategy_type:
                raise ValueError("Configuration strategy must match strategy_type")
            minimum = (
                variant.min_assets
                if variant.min_assets is not None
                else self.min_assets
            )
            maximum = (
                variant.max_assets
                if variant.max_assets is not None
                else self.max_assets
            )
            if minimum > maximum:
                raise ValueError("Minimum assets cannot exceed maximum assets")
            if variant.news_shortlist:
                if variant.news_top_assets < maximum:
                    raise ValueError(
                        "News shortlist size must be at least maximum assets"
                    )
        return self


class StrategyComparisonRequest(_LocalInputFileRequest):
    """Request body for ``POST /comparisons/strategy``."""

    strategy_types: list[StrategyType] = Field(
        default_factory=lambda: [StrategyType.COMBINED],
    )
    data_source: DataSource = DataSource.ALPACA
    frequency: Frequency = Frequency.MONTHLY
    validation_level: ValidationLevel = ValidationLevel.WARN
    benchmark_asset: str = "SPY"
    include_benchmark: bool = True
    min_assets: int = Field(default=CONSTRUCTION.min_assets, ge=1)
    max_assets: int = Field(default=CONSTRUCTION.max_assets, ge=1)
    population_size: int = Field(default=STRATEGY_OPTIMIZER.pop_size, ge=10)
    n_generations: int = Field(default=STRATEGY_OPTIMIZER.n_generations, ge=1)
    strategy_variants: list[StrategyVariantRequest] = Field(
        default_factory=list, max_length=20
    )
    optimization_selection: Literal[
        "sharpe", "return_with_risk_budget", "return_target"
    ] = "sharpe"
    optimization_volatility_increase: float = Field(default=0.25, ge=0, le=1)
    optimization_return_target_fraction: float = Field(default=0.75, ge=0, le=1)
    # Portfolio parameters
    initial_cash: float = Field(default=1000.0, gt=0)
    transaction_cost_rate: float = Field(default=0.001, ge=0)
    risk_free_rate: float = Field(default=0.0, ge=0)
    # Local dataset selection (ignored when data_source != "local")
    input_file: str | None = None

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def validate_variants(self) -> Self:
        if not self.strategy_types and not self.strategy_variants:
            raise ValueError("Select at least one strategy or configuration")
        names = [v.name.casefold() for v in self.strategy_variants]
        if len(names) != len(set(names)):
            raise ValueError("Configuration names must be unique")
        if self.min_assets > self.max_assets or any(
            (v.min_assets if v.min_assets is not None else self.min_assets)
            > (v.max_assets if v.max_assets is not None else self.max_assets)
            for v in self.strategy_variants
        ):
            raise ValueError("Minimum assets cannot exceed maximum assets")
        if any(
            v.news_shortlist
            and v.news_top_assets
            < (v.max_assets if v.max_assets is not None else self.max_assets)
            for v in self.strategy_variants
        ):
            raise ValueError("News shortlist size must be at least maximum assets")
        return self


class SolverComparisonRequest(_LocalInputFileRequest):
    """Request body for ``POST /comparisons/solver``."""

    data_source: DataSource = DataSource.ALPACA
    frequency: Frequency = Frequency.MONTHLY
    validation_level: ValidationLevel = ValidationLevel.WARN
    solver_variants: list[SolverVariantInput] | None = Field(
        default=None, min_length=1, max_length=20
    )
    population_size: int = Field(default=OPTIMIZER.pop_size, ge=10)
    n_generations: int = Field(default=OPTIMIZER.n_generations, ge=1)
    n_restarts: int = Field(default=OPTIMIZER.n_restarts, ge=1)
    min_assets: int = Field(default=CONSTRUCTION.min_assets, ge=1)
    num_assets: int = Field(default=CONSTRUCTION.max_assets, ge=1)
    include_markowitz_reference: bool = True
    use_multi_greedy_seeds: bool = True
    seed_max_subsets: int = Field(default=2000, ge=1, le=100000)
    seed_beam_width: int = Field(default=5, ge=1, le=100)
    solvers: list[str] = Field(
        default_factory=lambda: [
            "NSGA-II",
            "NSGA-III",
            "SPEA2",
            "EpsNSGAII",
            "GWASFGA",
        ],
        min_length=1,
    )
    # Portfolio parameters
    initial_cash: float = Field(default=1000.0, gt=0)
    transaction_cost_rate: float = Field(default=0.001, ge=0)
    risk_free_rate: float = Field(default=0.0, ge=0)
    # Local dataset selection (ignored when data_source != "local")
    input_file: str | None = None

    model_config = {"use_enum_values": True}

    @model_validator(mode="after")
    def validate_asset_bounds(self) -> Self:
        if self.min_assets > self.num_assets:
            raise ValueError("Minimum assets cannot exceed number of assets")
        return self


class PredictionComparisonRequest(_LocalInputFileRequest):
    """Request body for ``POST /comparisons/prediction``."""

    data_source: DataSource = DataSource.ALPACA
    frequency: Frequency = Frequency.MONTHLY
    validation_level: ValidationLevel = ValidationLevel.WARN
    predictor_variants: list[PredictorVariantInput] | None = Field(
        default=None, min_length=1, max_length=20
    )
    n_eval_periods: int = Field(default=12, ge=1)
    min_history: int = Field(default=12, ge=1)
    top_k: int = Field(default=5, ge=1)
    predictors: list[str] = Field(
        default_factory=lambda: ["RidgeVARX", "EWM-3", "EWM-6", "EWM-12"],
        min_length=1,
    )
    compute_sentiment: bool = True
    # Portfolio parameters
    initial_cash: float = Field(default=1000.0, gt=0)
    transaction_cost_rate: float = Field(default=0.001, ge=0)
    risk_free_rate: float = Field(default=0.0, ge=0)
    # Local dataset selection (ignored when data_source != "local")
    input_file: str | None = None

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Lightweight run info returned in list endpoints."""

    id: str
    run_type: RunType
    status: RunStatus
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    campaign_id: str | None = None

    heartbeat_at: datetime | None = None
    worker_health: Literal["alive", "unresponsive", "unknown"] | None = None
    progress_fraction: float = 0.0
    progress_message: str = ""
    # Submitted settings are small; saved result/figure payloads stay detail-only.
    request_payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class RunDetail(RunSummary):
    """Full run info including request and result payload."""

    result: dict[str, Any] | None = None

    model_config = {"from_attributes": True}


class ErrorResponse(BaseModel):
    """Standard error envelope returned by exception handlers."""

    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    status_code: int


class CampaignFields(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    hypothesis: str = Field(default="", max_length=2000)
    notes: str = Field(default="", max_length=10000)


class CampaignSummary(CampaignFields):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    updated_at: datetime
    run_count: int = 0


class RunCampaignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    campaign_id: str | None = Field(pattern=r"^[a-f0-9]{12}$")
