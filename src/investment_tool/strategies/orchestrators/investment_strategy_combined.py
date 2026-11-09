"""
Combined investment strategy with sentiment analysis.

This module implements a comprehensive strategy that combines transformer-based
sentiment analysis, VARX return prediction, and portfolio optimization.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from platypus import NSGAII

from investment_tool.defaults import (
    CONSTRUCTION,
    GENETIC,
    OPTIMIZER,
    STRATEGY_OPTIMIZER,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.optimization_strategies import (
    PlatypusOptimizationStrategy,
    PortfolioVariator,
    WarmStartSource,
)
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
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
from investment_tool.strategies.prediction_strategies import RidgeVARXPredictionStrategy
from investment_tool.strategies.selection_strategies import (
    BaseSelectionStrategy,
    SharpeSelection,
)
from investment_tool.strategies.sentiment_analysis_strategies import (
    compute_sentiment_scores,
)

logger = logging.getLogger(__name__)


class InvestmentStrategyCombined(BaseInvestmentStrategy):
    """Combined strategy with sentiment analysis, prediction, and optimization.

    Pipeline
    --------
    1. ``_compute_sentiment`` — Transformer-based sentiment on news articles.
    2. ``_predict_returns`` — Ridge VARX model with sentiment features.
    3. ``_build_optimizer`` — Construct the multi-objective optimizer.
    4. ``run_strategy`` — Orchestrate the above and return weights.

    Parameters
    ----------
    instance : Instance
        Execution instance configuration.
    stocks : Sequence[Stock]
        Stock entities for the strategy.
    news : dict[tuple[str, date], News] | None
        News data keyed by (symbol, date).
    num_assets_to_hold : int, optional
        Maximum number of assets in portfolio, by default 5.
        .. deprecated:: Use ``max_assets`` instead.
    min_assets : int, optional
        Minimum number of active assets (default 2).
    max_assets : int, optional
        Maximum number of active assets (default 5).
    risk_aversion : float, optional
        Risk aversion parameter (0-1), by default 0.1.
    output_directory : Path | None, optional
        Directory for output files.
    selection_strategy : BaseSelectionStrategy | None, optional
        Strategy for selecting a portfolio from the Pareto front.
        Defaults to ``SharpeSelection()``.
    population_size : int, optional
        GA population size (default 300).
    n_generations : int, optional
        Function evaluations per restart (default 300 000).
    n_restarts : int, optional
        Multi-start runs (default 3).
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
        population_size: int = STRATEGY_OPTIMIZER.pop_size,
        n_generations: int = STRATEGY_OPTIMIZER.n_generations,
        n_restarts: int = STRATEGY_OPTIMIZER.n_restarts,
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
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_restarts = n_restarts
        logger.info(
            "Initialized InvestmentStrategyCombined "
            "(min_assets=%d, max_assets=%d, risk_aversion=%.2f, selector=%s)",
            self.min_assets,
            self.max_assets,
            risk_aversion,
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

        Raises
        ------
        RuntimeError
            If sentiment analysis fails.
        """
        scores: dict[str, dict[date, float]] = compute_sentiment_scores(self.news)
        logger.debug("Computed sentiment scores for %d stocks", len(scores))
        return scores

    def _predict_returns(
        self,
        sentiment_scores: dict[str, dict[date, float]],
    ) -> dict[str, float]:
        """Predict expected returns using Ridge VARX with sentiment features.

        Parameters
        ----------
        sentiment_scores : dict[str, dict[date, float]]
            Output of ``_compute_sentiment``.

        Returns
        -------
        dict[str, float]
            Predicted next-period return per symbol.

        Raises
        ------
        ValueError
            If no returns could be predicted.
        """
        strategy = RidgeVARXPredictionStrategy(
            self.stocks_with_recent_history(), sentiment_scores
        )
        expected_returns: dict[str, float] = strategy.predict_returns()

        if not expected_returns:
            raise ValueError("No expected returns could be predicted")

        logger.debug("Predicted returns for %d stocks", len(expected_returns))
        return expected_returns

    def _build_optimizer(
        self,
        expected_returns: dict[str, float],
        covariance_matrix: pd.DataFrame,
    ) -> PlatypusOptimizationStrategy:
        """Construct the portfolio optimizer with current-period settings.

        Parameters
        ----------
        expected_returns : dict[str, float]
            Predicted returns per symbol.
        covariance_matrix : pd.DataFrame
            Covariance matrix of asset returns.

        Returns
        -------
        PlatypusOptimizationStrategy
            Configured optimizer ready to call ``optimize()``.
        """
        trading_period = self._instance.current_trading_period
        date_label = (
            str(trading_period.trading_date.date())
            if trading_period is not None
            else None
        )

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
            model_kwargs={
                "l_assets": self.min_assets,
                "u_assets": self.max_assets,
                "constraint_slack": OPTIMIZER.constraint_slack,
                "hhi_as_constraint": CONSTRUCTION.hhi_as_constraint,
                "hhi_bound": CONSTRUCTION.hhi_bound,
            },
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
        """Execute the combined strategy pipeline.

        Orchestrates:

        1. Sentiment analysis on news articles.
        2. Ridge VARX return prediction with sentiment features.
        3. Covariance matrix estimation.
        4. Multi-objective portfolio optimization (warm-started after
           the first iteration).

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
        RuntimeError
            If any pipeline step fails.
        """
        logger.info(
            "Running combined strategy (iteration=%d, warm=%s)",
            self._iteration,
            self.is_warm_start,
        )

        # 1. Sentiment → 2. Prediction → 3. Covariance → 4. Optimization
        sentiment_scores = self._compute_sentiment()
        expected_returns = self._predict_returns(sentiment_scores)
        covariance_matrix = self.build_covariance_matrix()

        optimizer = self._build_optimizer(expected_returns, covariance_matrix)
        result: OptimizationResult = self.optimize_portfolios(
            optimizer,
            progress_callback=progress_callback,
        )

        # 5. Portfolio selection (investment strategy responsibility)
        self.last_optimization_diagnostics = result.search_diagnostics
        optimal_weights: pd.Series = self._selector.select(
            result.front, pd.Index(expected_returns.keys())
        )

        if optimal_weights.empty:
            raise RuntimeError("Optimizer returned empty weights")

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
