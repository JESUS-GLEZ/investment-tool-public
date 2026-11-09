"""Prediction comparison coordinator.

App-layer orchestrator that loads market data, builds walk-forward
evaluation, runs the prediction benchmark, and persists outputs.
Called by the CLI entrypoint.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode
from investment_tool.output.run_metadata import (
    write_run_metadata,
    record_input_snapshot,
)

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any


from investment_tool.config import AppConfig, get_config
from investment_tool.data.universes.loader import load_asset_universe
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.data.assembly import ModelDataFactory
from investment_tool.data.validation import ValidationLevel
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
    AveragePredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.fuzzy_prediction_strategy import (
    FuzzyPredictionStrategy,
    FuzzySettings,
    PooledRidgePredictionStrategy,
    ReturnBaselinePredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.momentum_prediction_strategy import (
    MeanBlendPredictionStrategy,
    CalibratedMomentumPredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison import (
    PredictionBenchmark,
    PredictorConfig,
)
from investment_tool.strategies.prediction_strategies.ridge_varx_prediction_strategy import (
    RidgeVARXPredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.sentiment_ewm_prediction_strategy import (
    SentimentEWMPredictionStrategy,
)
from investment_tool.strategies.sentiment_analysis_strategies import (
    compute_sentiment_scores,
)

from .comparison_variant import PredictorVariant, validate_names
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.base_prediction_strategy import (
    BasePredictionStrategy,
)

logger = logging.getLogger(__name__)


_DEFAULT_PREDICTORS = ["RidgeVARX", "EWM-3", "EWM-6", "EWM-12"]


@dataclass
class PredictionComparisonParameters:
    """Parameters for the prediction comparison run.

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
    n_eval_periods : int
        Number of walk-forward evaluation steps.
    min_history : int
        Minimum historical periods before first evaluation.
    top_k : int
        Number of top assets for hit-rate metric.
    predictors : list[str]
        Predictor labels to include.
    compute_sentiment : bool
        Whether to run sentiment analysis (needed for RidgeVARX).
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
    n_eval_periods: int = 12
    min_history: int = 12
    top_k: int = 5
    predictors: list[str] = field(default_factory=lambda: list(_DEFAULT_PREDICTORS))
    compute_sentiment: bool = True
    cfg: AppConfig | None = None
    output_mode: OutputMode = OutputMode.STANDARD
    predictor_variants: list[PredictorVariant] | None = None


# ── Predictor factory type ───────────────────────────────────────────────────

_PredictorBuilder = Callable[[PredictionComparisonParameters], PredictorConfig]

# Registry of available predictor builders
_PREDICTOR_REGISTRY: dict[str, _PredictorBuilder] = {}


def _register_predictor(label: str) -> Callable[[_PredictorBuilder], _PredictorBuilder]:
    """Decorator to register a predictor builder function."""

    def decorator(func: _PredictorBuilder) -> _PredictorBuilder:
        _PREDICTOR_REGISTRY[label.upper()] = func
        return func

    return decorator


# Research variants share implementations; defaults above remain unchanged.
_RESEARCH_SETTINGS = {
    "FuzzyThesis-fixed": FuzzySettings(beta=0.0),
    "FuzzyThesis-adaptive": FuzzySettings(),
    "FuzzyThesis-no-trend": FuzzySettings(trend=False),
    "FuzzyRidge-fixed": FuzzySettings(beta=0.0),
    "FuzzyRidge-adaptive": FuzzySettings(),
}


def _research_builder(label: str) -> _PredictorBuilder:
    def build(params: PredictionComparisonParameters) -> PredictorConfig:
        if label in ("ZeroReturn", "HistoricalMean"):
            return PredictorConfig(
                label=label,
                factory=lambda stocks, sentiments: ReturnBaselinePredictionStrategy(
                    stocks, zero=label == "ZeroReturn"
                ),
            )
        settings = _RESEARCH_SETTINGS.get(label, FuzzySettings())
        if label.startswith("FuzzyThesis"):
            return PredictorConfig(
                label=label,
                factory=lambda stocks, sentiments: FuzzyPredictionStrategy(
                    stocks, settings
                ),
            )
        return PredictorConfig(
            label=label,
            factory=lambda stocks, sentiments: PooledRidgePredictionStrategy(
                stocks, settings, fuzzy=label != "PooledRidge"
            ),
        )

    return build


for _label in ("ZeroReturn", "HistoricalMean", "PooledRidge", *_RESEARCH_SETTINGS):
    _register_predictor(_label)(_research_builder(_label))


# Fixed hypotheses share predictor and training implementations.
_MEAN_BLEND_WEIGHTS = {
    "EWM12-Mean-25": 0.25,
    "EWM12-Mean-50": 0.5,
    "EWM12-Mean-75": 0.75,
}


def _momentum_builder(label: str) -> _PredictorBuilder:
    def build(params: PredictionComparisonParameters) -> PredictorConfig:
        if label in _MEAN_BLEND_WEIGHTS:
            return PredictorConfig(
                label=label,
                factory=lambda stocks, sentiments: MeanBlendPredictionStrategy(
                    stocks, momentum_weight=_MEAN_BLEND_WEIGHTS[label]
                ),
            )
        if params.frequency != Frequency.MONTHLY:
            raise ValueError(
                f"{label} requires monthly data for its 12-2 momentum signal"
            )
        return PredictorConfig(
            label=label,
            factory=lambda stocks, sentiments: CalibratedMomentumPredictionStrategy(
                stocks, signals="12-2" if label == "Momentum12-2Ridge" else "ewm"
            ),
        )

    return build


for _label in (*_MEAN_BLEND_WEIGHTS, "MomentumRidge", "Momentum12-2Ridge"):
    _register_predictor(_label)(_momentum_builder(_label))


@_register_predictor("RIDGEVARX")
def _build_ridge_varx(params: PredictionComparisonParameters) -> PredictorConfig:
    return PredictorConfig(
        label="RidgeVARX",
        factory=lambda stocks, sentiments: RidgeVARXPredictionStrategy(
            stocks=stocks,
            stock_sentiment_scores=sentiments,
            lag_order=3,
            alpha=10.0,
        ),
    )


@_register_predictor("RIDGEVARX-LAG5")
def _build_ridge_varx_lag5(params: PredictionComparisonParameters) -> PredictorConfig:
    return PredictorConfig(
        label="RidgeVARX-lag5",
        factory=lambda stocks, sentiments: RidgeVARXPredictionStrategy(
            stocks=stocks,
            stock_sentiment_scores=sentiments,
            lag_order=5,
            alpha=10.0,
        ),
    )


@_register_predictor("EWM-3")
def _build_ewm_3(params: PredictionComparisonParameters) -> PredictorConfig:
    return PredictorConfig(
        label="EWM-3",
        factory=lambda stocks, sentiments: AveragePredictionStrategy(
            stocks=stocks, span=3
        ),
    )


@_register_predictor("EWM-6")
def _build_ewm_6(params: PredictionComparisonParameters) -> PredictorConfig:
    return PredictorConfig(
        label="EWM-6",
        factory=lambda stocks, sentiments: AveragePredictionStrategy(
            stocks=stocks, span=6
        ),
    )


@_register_predictor("EWM-12")
def _build_ewm_12(params: PredictionComparisonParameters) -> PredictorConfig:
    return PredictorConfig(
        label="EWM-12",
        factory=lambda stocks, sentiments: AveragePredictionStrategy(
            stocks=stocks, span=12
        ),
    )


@_register_predictor("SENTIMENTEWM-3")
def _build_sentiment_ewm_3(params: PredictionComparisonParameters) -> PredictorConfig:
    return PredictorConfig(
        label="SentimentEWM-3",
        factory=lambda stocks, sentiments: SentimentEWMPredictionStrategy(
            stocks=stocks, sentiment_scores=sentiments, ewm_span=3
        ),
    )


@_register_predictor("SENTIMENTEWM-6")
def _build_sentiment_ewm_6(params: PredictionComparisonParameters) -> PredictorConfig:
    return PredictorConfig(
        label="SentimentEWM-6",
        factory=lambda stocks, sentiments: SentimentEWMPredictionStrategy(
            stocks=stocks, sentiment_scores=sentiments, ewm_span=6
        ),
    )


def get_available_predictors() -> list[str]:
    """Return labels of all registered predictors.

    Returns
    -------
    list[str]
        Available predictor labels.
    """
    return sorted(_PREDICTOR_REGISTRY.keys())


def _configured_predictor(
    variant: PredictorVariant, base: PredictorConfig
) -> PredictorConfig:
    def factory(
        stocks: Sequence[Stock], sentiments: dict[str, dict[date, float]]
    ) -> BasePredictionStrategy:
        original = base.factory(stocks, sentiments)
        if isinstance(original, AveragePredictionStrategy):
            return AveragePredictionStrategy(
                stocks, span=variant.span if variant.span is not None else original.span
            )
        if isinstance(original, SentimentEWMPredictionStrategy):
            return SentimentEWMPredictionStrategy(
                stocks,
                sentiments,
                ewm_span=variant.span
                if variant.span is not None
                else original.ewm_span,
            )
        if isinstance(original, MeanBlendPredictionStrategy):
            return MeanBlendPredictionStrategy(
                stocks,
                span=variant.span if variant.span is not None else original.span,
                momentum_weight=variant.momentum_weight
                if variant.momentum_weight is not None
                else original.momentum_weight,
            )
        if isinstance(original, RidgeVARXPredictionStrategy):
            return RidgeVARXPredictionStrategy(
                stocks,
                sentiments,
                lag_order=variant.lag_order
                if variant.lag_order is not None
                else original.lag_order,
                alpha=variant.alpha if variant.alpha is not None else original.alpha,
            )
        if isinstance(original, CalibratedMomentumPredictionStrategy):
            return CalibratedMomentumPredictionStrategy(
                stocks,
                signals=original.signals,
                alpha=variant.alpha if variant.alpha is not None else original.alpha,
                min_train_history=variant.min_train_history
                if variant.min_train_history is not None
                else original.min_train_history,
            )
        if isinstance(
            original, (FuzzyPredictionStrategy, PooledRidgePredictionStrategy)
        ):
            settings = replace(
                original.settings,
                **{
                    key: getattr(variant, key)
                    for key in (
                        "partitions",
                        "lags",
                        "sigma",
                        "beta",
                        "trend",
                        "grid_points",
                    )
                    if getattr(variant, key) is not None
                },
            )
            if isinstance(original, FuzzyPredictionStrategy):
                return FuzzyPredictionStrategy(stocks, settings)
            return PooledRidgePredictionStrategy(
                stocks,
                settings,
                fuzzy=original.fuzzy,
                alpha=variant.alpha if variant.alpha is not None else original.alpha,
                min_train_history=variant.min_train_history
                if variant.min_train_history is not None
                else original.min_train_history,
            )
        return original

    return PredictorConfig(label=variant.name.strip(), factory=factory)


def _build_predictor_configs(
    params: PredictionComparisonParameters,
) -> list[PredictorConfig]:
    """Build PredictorConfig list from parameter labels.

    Parameters
    ----------
    params : PredictionComparisonParameters
        Run parameters.

    Returns
    -------
    list[PredictorConfig]
        Configured predictors.

    Raises
    ------
    ValueError
        If a requested label is not registered.
    """
    configs: list[PredictorConfig] = []
    if params.predictor_variants:
        validate_names([variant.name for variant in params.predictor_variants])
        for variant in params.predictor_variants:
            base = _build_predictor_configs(
                replace(params, predictor_variants=None, predictors=[variant.predictor])
            )[0]
            configs.append(_configured_predictor(variant, base))
        return configs
    for label in params.predictors:
        builder = _PREDICTOR_REGISTRY.get(label.upper())
        if builder is None:
            available = ", ".join(get_available_predictors())
            raise ValueError(f"Unknown predictor '{label}'. Available: {available}")
        configs.append(builder(params))
    return configs


def execute_prediction_comparison(
    params: PredictionComparisonParameters,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Orchestrate the full prediction comparison pipeline.

    1. Load market data.
    2. Optionally compute sentiment scores.
    3. Build predictor configs.
    4. Run walk-forward benchmark.
    5. Save comparison outputs.

    Parameters
    ----------
    params : PredictionComparisonParameters
        Execution parameters.
    progress_callback : ProgressCallback | None, optional
        Called periodically with ``(fraction_complete, message)``.
        Forwarded to the prediction benchmark loop.

    Returns
    -------
    dict[str, Any]
        Results dictionary with ``summary_table``, ``saved_files``,
        and ``output_dir``.
    """
    logger.info("Starting prediction comparison")
    # Reject unsupported predictor/frequency combinations before loading data.
    predictor_configs = _build_predictor_configs(params)

    # This coordinator loads a static candidate set. Do not silently evaluate
    # a dated manifest's union as if every member existed at every origin.
    cfg = params.cfg if params.cfg is not None else get_config()
    universe = load_asset_universe(cfg.asset_config_file)
    if universe.membership is not None:
        raise ValueError(
            "Prediction comparison does not yet support dated universe membership. "
            "Use a reviewed static universe for this evaluation window; do not "
            "substitute the union of historical snapshots."
        )
    # 1. Load data
    write_run_metadata(params)
    model_data: ModelData = ModelDataFactory.create_model_data(
        data_source=params.data_source,
        execution_mode=ExecutionMode.INVESTING,
        frequency=params.frequency,
        input_data_path=params.input_data_path,
        save_input_data=params.save_input_data,
        validation_level=params.validation_level,
        cfg=cfg,
    )
    record_input_snapshot(params.output_directory, model_data.input_snapshot_path)

    candidates = set(universe.stocks) | set(universe.crypto)
    stocks = [stock for stock in model_data.stocks if stock.symbol in candidates]
    logger.info("Loaded %d stocks for prediction comparison", len(stocks))

    # 2. Sentiment scores (if any predictor needs them)
    sentiment_scores: dict[str, dict[date, float]] = {}
    needs_sentiment = any(
        label.upper().startswith("RIDGEVARX")
        or label.upper().startswith("SENTIMENTEWM")
        for label in (
            [v.predictor for v in params.predictor_variants]
            if params.predictor_variants
            else params.predictors
        )
    )

    if params.compute_sentiment and needs_sentiment and model_data.news:
        logger.info("Computing sentiment scores for prediction comparison")
        try:
            sentiment_scores = compute_sentiment_scores(model_data.news)
        except Exception as e:
            logger.warning("Sentiment computation failed: %s — using empty scores", e)
            sentiment_scores = {s.symbol: {} for s in stocks}
    else:
        sentiment_scores = {s.symbol: {} for s in stocks}

    # 4. Run benchmark
    benchmark = PredictionBenchmark(
        stocks=stocks,
        sentiment_scores=sentiment_scores,
        n_eval_periods=params.n_eval_periods,
        min_history=max(params.min_history, (universe.minimum_return_periods or 0) + 1),
        top_k=params.top_k,
        output_directory=params.output_directory,
        frequency=params.frequency,
    )

    results = benchmark.run(predictor_configs, progress_callback=progress_callback)

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
    }
