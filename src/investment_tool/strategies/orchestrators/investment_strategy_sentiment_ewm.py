"""Sentiment-tilted EWM optimization strategy with regression calibration.

Combines short-horizon exponentially-weighted return prediction with
transformer-based sentiment analysis and multi-objective portfolio
optimization.  The sentiment tilt strength is **calibrated via OLS
linear regression** on historical data rather than set as a fixed
constant:

    actual_return_t = α + β₁ · ewm_t + β₂ · (ewm_t × sentiment_t) + ε

The out-of-sample prediction for the next period is then:

    predicted = α̂ + β̂₁ · ewm + β̂₂ · (ewm × sentiment)

This learns from realised returns how much (if at all) sentiment
improves the EWM forecast — avoiding arbitrary constant choices.
When insufficient history is available, the strategy falls back to
a plain EWM prediction (no sentiment adjustment).

The optimization uses the BPRC (Balanced Portfolios Risk Contributions)
model with the same population/generation settings as the other
optimization-based strategies.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from platypus import NSGAII

from investment_tool.defaults import (
    CONSTRUCTION,
    DECOMPOSITION,
    GENETIC,
    OPTIMIZER,
    PREDICTION,
    STRATEGY_OPTIMIZER,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.exceptions import OptimizationError, PredictionError
from investment_tool.strategies.optimization_strategies import (
    DecompositionOptimizationStrategy,
    HybridOptimizationStrategy,
    PortfolioVariator,
    WarmStartSource,
)
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    BaseOptimizationStrategy,
    ProgressCallback,
)
from investment_tool.strategies.optimization_strategies.model import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
    find_solution_for_weights,
)
from investment_tool.strategies.optimization_strategies.seeded_optimization import (
    MultiGreedyOptimizationStrategy,
)
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)
from investment_tool.strategies.orchestrators.investment_strategy_optimization import (
    OptimizerType,
)
from investment_tool.strategies.prediction_strategies.sentiment_ewm_prediction_strategy import (
    SentimentEWMPredictionStrategy,
)
from investment_tool.strategies.selection_strategies import (
    BaseSelectionStrategy,
    SharpeSelection,
)
from investment_tool.strategies.sentiment_analysis_strategies import (
    compute_sentiment_scores,
)

logger = logging.getLogger(__name__)


class InvestmentStrategySentimentEWM(BaseInvestmentStrategy):
    """Sentiment-tilted EWM optimization strategy.

    Pipeline
    --------
    1. ``_compute_sentiment`` — transformer-based sentiment on news.
    2. ``_predict_returns`` — OLS-calibrated EWM + sentiment prediction.
    3. ``build_covariance_matrix`` — shared EWM covariance (base class).
    4. ``_build_optimizer`` — BPRC multi-objective optimization.
    5. ``run_strategy`` — orchestrate and select by Sharpe ratio.

    Regression-calibrated sentiment tilt
    -------------------------------------
    Instead of applying a fixed ``tilt_strength`` constant, the strategy
    fits per-asset OLS regressions on historical data:

        actual_return_t = α + β₁ · ewm_t + β₂ · (ewm_t × sentiment_t) + ε

    The learned coefficients determine how much (if any) sentiment
    information improves the EWM forecast.  When insufficient data is
    available (< 12 periods), the model falls back to a plain EWM
    prediction with an optional ``fallback_tilt``.

    Parameters
    ----------
    instance : Instance
        Execution instance configuration.
    stocks : Sequence[Stock]
        Stock entities for the strategy.
    news : dict[tuple[str, date], News] | None
        News data keyed by (symbol, date).
    num_assets_to_hold : int, optional
        Maximum assets in the portfolio (default 5).
        .. deprecated:: Use ``max_assets`` instead.
    min_assets : int, optional
        Minimum number of active assets (default 2).
    max_assets : int, optional
        Maximum number of active assets (default 5).
    risk_aversion : float, optional
        Risk aversion parameter (default 0.1).
    output_directory : Path | None, optional
        Directory for output files.
    selection_strategy : BaseSelectionStrategy | None, optional
        Portfolio selector.  Defaults to ``SharpeSelection()``.
    ewm_span : int, optional
        EWM span in periods for the return prediction (default 6).
    fallback_tilt : float, optional
        Fixed tilt strength used only when the regression cannot be
        fitted due to insufficient data (default 0.0 — pure EWM).
    population_size : int, optional
        GA population size (default 300).
    n_generations : int, optional
        Function evaluations per restart (default 300 000).
    n_restarts : int, optional
        Multi-start runs (default 3).
    optimizer_type : OptimizerType, optional
        Backend: ``EVOLUTIONARY``, ``DECOMPOSITION``, or ``HYBRID``.
        Default ``OptimizerType.HYBRID``.
    """

    def __init__(
        self,
        instance: Instance,
        stocks: Sequence[Stock],
        news: dict[tuple[str, date], News] | None,
        num_assets_to_hold: int | None = None,
        min_assets: int = CONSTRUCTION.min_assets,
        max_assets: int = CONSTRUCTION.max_assets,
        risk_aversion: float = CONSTRUCTION.risk_aversion,
        output_directory: Path | None = None,
        selection_strategy: BaseSelectionStrategy | None = None,
        ewm_span: int = PREDICTION.ewm_span,
        fallback_tilt: float = PREDICTION.fallback_tilt,
        population_size: int = STRATEGY_OPTIMIZER.pop_size,
        n_generations: int = STRATEGY_OPTIMIZER.n_generations,
        n_restarts: int = STRATEGY_OPTIMIZER.n_restarts,
        optimizer_type: OptimizerType = OptimizerType.EVOLUTIONARY,
    ) -> None:
        super().__init__(instance=instance, stocks=stocks, news=news)
        self._instance = instance
        self.min_assets = min_assets
        self.max_assets = (
            num_assets_to_hold if num_assets_to_hold is not None else max_assets
        )
        self.risk_aversion = risk_aversion
        self.output_directory = output_directory
        self._selector = selection_strategy or SharpeSelection()
        self.ewm_span = ewm_span
        self.fallback_tilt = fallback_tilt
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_restarts = n_restarts
        self.optimizer_type = OptimizerType(optimizer_type)

        # Store calibration diagnostics from the latest prediction
        self._calibration_diagnostics: dict[str, dict[str, float]] = {}

        logger.info(
            "Initialized InvestmentStrategySentimentEWM "
            "(ewm_span=%d, fallback_tilt=%.2f, pop=%d, gens=%d, "
            "optimizer=%s, selector=%s)",
            ewm_span,
            fallback_tilt,
            population_size,
            n_generations,
            self.optimizer_type.value,
            type(self._selector).__name__,
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _compute_sentiment(self) -> dict[str, dict[date, float]]:
        """Run transformer-based sentiment analysis on filtered news.

        Returns
        -------
        dict[str, dict[date, float]]
            Sentiment scores per (symbol, date).
        """
        scores: dict[str, dict[date, float]] = compute_sentiment_scores(self.news)
        logger.debug("Computed sentiment scores for %d stocks", len(scores))
        return scores

    def _predict_returns(
        self,
        sentiment_scores: dict[str, dict[date, float]],
    ) -> dict[str, float]:
        """Predict expected returns via OLS-calibrated EWM + sentiment.

        Delegates to ``SentimentEWMPredictionStrategy`` which fits a
        per-asset linear regression:

            actual_return = α + β₁·ewm + β₂·(ewm × sentiment) + ε

        and returns the out-of-sample prediction.  When insufficient
        history is available, falls back to pure EWM prediction.

        Parameters
        ----------
        sentiment_scores : dict[str, dict[date, float]]
            Per-date sentiment scores per symbol.

        Returns
        -------
        dict[str, float]
            Regression-calibrated predicted returns per symbol.

        Raises
        ------
        PredictionError
            If prediction fails.
        """
        try:
            strategy = SentimentEWMPredictionStrategy(
                stocks=self.stocks_with_recent_history(),
                sentiment_scores=sentiment_scores,
                ewm_span=self.ewm_span,
                fallback_tilt=self.fallback_tilt,
            )
            predictions: dict[str, float] = strategy.predict_returns()

            # Store calibration diagnostics for inspection / logging
            self._calibration_diagnostics = strategy.calibration_diagnostics

        except Exception as e:
            raise PredictionError(
                f"SentimentEWM prediction failed: {e!s}",
                model="SentimentEWMPredictionStrategy",
                original_error=e,
            ) from e

        if not predictions:
            raise PredictionError(
                "No expected returns could be predicted",
                model="SentimentEWMPredictionStrategy",
            )

        self._log_calibration_diagnostics()
        return predictions

    def _log_calibration_diagnostics(self) -> None:
        """Log the regression calibration results for the current period."""
        if not self._calibration_diagnostics:
            return

        calibrated = {
            s: d
            for s, d in self._calibration_diagnostics.items()
            if d.get("calibrated", 0.0) > 0.5
        }
        fallback = {
            s: d
            for s, d in self._calibration_diagnostics.items()
            if d.get("calibrated", 0.0) <= 0.5
        }

        if calibrated:
            avg_r2 = np.mean([d["r_squared"] for d in calibrated.values()])
            tilts = {s: d.get("effective_tilt", 0.0) for s, d in calibrated.items()}
            logger.info(
                "Regression calibrated for %d/%d assets (avg R²=%.4f). "
                "Effective tilts: %s",
                len(calibrated),
                len(self._calibration_diagnostics),
                avg_r2,
                {s: f"{t:+.4f}" for s, t in tilts.items()},
            )
        if fallback:
            logger.info(
                "Fallback (pure EWM) used for %d assets: %s",
                len(fallback),
                list(fallback.keys()),
            )

    def _build_optimizer(
        self,
        expected_returns: dict[str, float],
        covariance_matrix: pd.DataFrame,
    ) -> BaseOptimizationStrategy:
        """Construct the BPRC optimizer.

        Dispatches to the appropriate backend based on
        ``self.optimizer_type``.

        Parameters
        ----------
        expected_returns : dict[str, float]
            Sentiment-adjusted predicted returns.
        covariance_matrix : pd.DataFrame
            Covariance matrix.

        Returns
        -------
        BaseOptimizationStrategy
            Configured optimizer.
        """
        trading_period = self._instance.current_trading_period
        date_label = (
            str(trading_period.trading_date.date())
            if trading_period is not None
            else None
        )

        model_kwargs: dict[str, Any] = {
            "l_assets": self.min_assets,
            "u_assets": self.max_assets,
            "constraint_slack": OPTIMIZER.constraint_slack,
            "hhi_as_constraint": CONSTRUCTION.hhi_as_constraint,
            "hhi_bound": CONSTRUCTION.hhi_bound,
        }

        if self.optimizer_type == OptimizerType.DECOMPOSITION:
            return DecompositionOptimizationStrategy(
                expected_returns=pd.Series(expected_returns),
                covariance_matrix=covariance_matrix,
                l_assets=self.min_assets,
                u_assets=self.max_assets,
                constraint_slack=OPTIMIZER.constraint_slack,
                hhi_as_constraint=CONSTRUCTION.hhi_as_constraint,
                hhi_bound=CONSTRUCTION.hhi_bound,
                seed_weights=self._previous_front_weights,
                output_directory=self.output_directory,
                date_label=date_label,
            )

        if self.optimizer_type == OptimizerType.HYBRID:
            return HybridOptimizationStrategy(
                expected_returns=pd.Series(expected_returns),
                covariance_matrix=covariance_matrix,
                risk_aversion=self.risk_aversion,
                algorithm=NSGAII,
                algorithm_kwargs={
                    "variator": PortfolioVariator(
                        min_assets=self.min_assets,
                        max_assets=self.max_assets,
                        mutation_prob=GENETIC.mutation_prob,
                    ),
                },
                model=BalancedPortfoliosRiskContributions,
                model_kwargs=model_kwargs,
                population_size=self.population_size,
                n_generations=self.n_generations,
                n_restarts=self.n_restarts,
                output_directory=self.output_directory,
                date_label=date_label,
                seed_weights=None,  # Cold start: no previous-period seeding
                warm_start_source=WarmStartSource.NONE,
                ea_budget_ratio=OPTIMIZER.ea_budget_ratio,
                n_decomposition_directions=DECOMPOSITION.n_decomposition_directions,
            )

        # EVOLUTIONARY (default / fallback)
        return MultiGreedyOptimizationStrategy(
            expected_returns=pd.Series(expected_returns),
            covariance_matrix=covariance_matrix,
            risk_aversion=self.risk_aversion,
            algorithm=NSGAII,
            algorithm_kwargs={
                "variator": PortfolioVariator(
                    min_assets=self.min_assets,
                    max_assets=self.max_assets,
                    mutation_prob=GENETIC.mutation_prob,
                ),
            },
            model=BalancedPortfoliosRiskContributions,
            model_kwargs=model_kwargs,
            population_size=self.population_size,
            n_generations=self.n_generations,
            n_restarts=self.n_restarts,
            output_directory=self.output_directory,
            date_label=date_label,
            seed_weights=None,
            warm_start_source=WarmStartSource.NONE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.Series:
        """Execute the sentiment-tilted EWM strategy pipeline.

        Orchestrates:

        1. Sentiment analysis on news articles (cached incrementally).
        2. OLS-calibrated EWM + sentiment return prediction.
        3. Covariance matrix estimation.
        4. BPRC multi-objective optimization (warm-started after first
           iteration).
        5. Sharpe-ratio portfolio selection.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Called periodically with ``(fraction_complete, message)``.
            Threaded to the optimizer's ``optimize()`` call.

        Returns
        -------
        pd.Series
            Optimal portfolio weights indexed by symbol.

        Raises
        ------
        PredictionError
            If return prediction fails.
        OptimizationError
            If optimization fails.
        """
        logger.info(
            "Running sentiment-EWM strategy (iteration=%d, warm=%s)",
            self._iteration,
            self.is_warm_start,
        )

        # 1. Sentiment (cached across iterations)
        sentiment_scores = self._compute_sentiment()

        # 2. Regression-calibrated prediction (EWM + sentiment tilt)
        adjusted_returns = self._predict_returns(sentiment_scores)

        # 3. Covariance
        try:
            covariance_matrix = self.build_covariance_matrix()
        except Exception as e:
            raise OptimizationError(
                f"Failed to build covariance matrix: {e!s}",
                optimizer="covariance_builder",
                original_error=e,
            ) from e

        # 4. Optimization
        try:
            optimizer = self._build_optimizer(adjusted_returns, covariance_matrix)
            result: OptimizationResult = self.optimize_portfolios(
                optimizer,
                progress_callback=progress_callback,
            )
        except Exception as e:
            raise OptimizationError(
                f"Portfolio optimization failed: {e!s}",
                optimizer=self.optimizer_type.value,
                original_error=e,
            ) from e

        # 5. Portfolio selection
        self.last_optimization_diagnostics = result.search_diagnostics
        optimal_weights: pd.Series = self._selector.select(
            result.front, pd.Index(adjusted_returns.keys())
        )

        if optimal_weights.empty:
            raise OptimizationError(
                "Optimizer returned empty weights",
                optimizer=self.optimizer_type.value,
            )

        # Highlight the selected portfolio on diagnostic plots
        result.selected_solution = find_solution_for_weights(
            result.front, optimal_weights
        )
        optimizer.replot_with_selection(result)

        # Store selected weights + full front for diverse warm-starting
        front_weights = [np.asarray(s.variables, dtype=float) for s in result.front]
        self.store_optimal_weights(optimal_weights, front_weights=front_weights)
        self._log_weights(optimal_weights)
        return optimal_weights
