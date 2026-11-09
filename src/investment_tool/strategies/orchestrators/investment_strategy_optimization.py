"""
Portfolio optimization investment strategy.

This module implements a multi-objective optimization strategy that combines
return predictions with portfolio optimization using evolutionary algorithms,
decomposition-based SLSQP, or a hybrid of both.

Optimizer types:

* **evolutionary** — island/ensemble multi-start EA (NSGA-III, NSGA-II,
  EpsNSGAII, SPEA2).
* **decomposition** — scalarized subproblems solved by SLSQP; exact local
  convergence per reference direction.
* **hybrid** — reduced-budget EA exploration followed by SLSQP polishing
  of the EA front; best balance of robustness and quality.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from enum import Enum, unique
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from platypus import (
    NSGAII,
    NSGAIII,
    SPEA2,
    AbstractGeneticAlgorithm,
    EpsNSGAII,
)

from investment_tool.defaults import (
    ALGORITHM,
    CONSTRUCTION,
    DECOMPOSITION,
    GENETIC,
    GENTLE_VARIATOR,
    OPTIMIZER,
    STRATEGY_OPTIMIZER,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.exceptions import (
    OptimizationError,
    PredictionError,
)
from investment_tool.strategies.optimization_strategies import (
    DecompositionOptimizationStrategy,
    HybridOptimizationStrategy,
    PortfolioVariator,
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
from investment_tool.strategies.optimization_strategies.platypus_optimization import (
    WarmStartSource,
)
from investment_tool.strategies.optimization_strategies.seeded_optimization import (
    MultiGreedyOptimizationStrategy,
)
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)
from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
    AveragePredictionStrategy,
)
from investment_tool.constants import (
    OptimizationReturnEstimator,
    PortfolioBalance,
)
from investment_tool.strategies.news_selection import news_candidates
from investment_tool.strategies.prediction_strategies.momentum_prediction_strategy import (
    MeanBlendPredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.base_prediction_strategy import (
    BasePredictionStrategy,
)
from investment_tool.strategies.selection_strategies import (
    BaseSelectionStrategy,
    SharpeSelection,
)


@unique
class OptimizerType(str, Enum):
    """Portfolio optimizer backend.

    Members
    -------
    EVOLUTIONARY
        Island/ensemble multi-start evolutionary algorithms.
    DECOMPOSITION
        Scalarized SLSQP subproblems — exact local convergence.
    HYBRID
        Reduced-budget EA exploration + SLSQP polishing.
    """

    EVOLUTIONARY = "evolutionary"
    DECOMPOSITION = "decomposition"
    HYBRID = "hybrid"


logger = logging.getLogger(__name__)


class InvestmentStrategyOptimization(BaseInvestmentStrategy):
    """Multi-objective portfolio optimization strategy.

    Pipeline
    --------
    1. ``_predict_returns`` — Exponentially-weighted average prediction.
    2. ``build_covariance_matrix`` — Shared EWM covariance (base class).
    3. ``_build_optimizer`` — Construct the multi-objective optimizer
       (evolutionary, decomposition, or hybrid).
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
        Function evaluations per restart (default 500 000).
    n_restarts : int, optional
        Multi-start runs (default 5).
    optimizer_type : OptimizerType, optional
        Backend: ``EVOLUTIONARY`` (EA only), ``DECOMPOSITION``
        (SLSQP only), or ``HYBRID`` (EA + SLSQP).
        Default ``OptimizerType.DECOMPOSITION``.
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
        optimizer_type: OptimizerType = OptimizerType.EVOLUTIONARY,
        return_estimator: OptimizationReturnEstimator = OptimizationReturnEstimator.EWM_6,
        balance: PortfolioBalance = PortfolioBalance.RISK_CONTRIBUTIONS,
        news_shortlist: bool = False,
        news_top_assets: int = 50,
        news_min_articles: int = 3,
    ) -> None:
        super().__init__(instance=instance, stocks=stocks, news=news)
        self._instance = instance
        self.balance = PortfolioBalance(balance)
        self.news_shortlist = news_shortlist
        self.news_top_assets = news_top_assets
        self.news_min_articles = news_min_articles
        self.last_news_selection: dict[str, Any] = {"enabled": False}
        self._news_selection_key: tuple[Any, ...] | None = None
        self._news_candidates: set[str] = set()
        if news_top_assets < 1 or news_min_articles < 1:
            raise ValueError("News shortlist settings must be positive")
        if (
            self.balance != PortfolioBalance.RISK_CONTRIBUTIONS
            and OptimizerType(optimizer_type) != OptimizerType.EVOLUTIONARY
        ):
            raise ValueError(
                "Capital-weight and hybrid balancing require the evolutionary backend"
            )
        self.return_estimator = OptimizationReturnEstimator(return_estimator)
        self.min_assets = min_assets
        self.max_assets = (
            num_assets_to_hold if num_assets_to_hold is not None else max_assets
        )
        if news_shortlist and news_top_assets < self.max_assets:
            raise ValueError("News shortlist size must be at least maximum assets")
        self.risk_aversion = risk_aversion
        self.output_directory = output_directory
        self._selector = selection_strategy or SharpeSelection()
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_restarts = n_restarts
        self.optimizer_type = OptimizerType(optimizer_type)
        logger.info(
            "Initialized InvestmentStrategyOptimization "
            "(min_assets=%d, max_assets=%d, risk_aversion=%.2f, "
            "optimizer=%s, selector=%s)",
            self.min_assets,
            self.max_assets,
            risk_aversion,
            self.optimizer_type.value,
            type(self._selector).__name__,
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def stocks_with_recent_history(self) -> list[Stock]:
        """Apply one candidate view to forecasts and covariance, preserving raw holdings."""
        stocks: list[Stock] = super().stocks_with_recent_history()
        if not getattr(self, "news_shortlist", False):
            return stocks
        period = self._instance.current_trading_period
        if period is None:
            raise ValueError("News shortlist requires a trading period")
        cutoff = period.decision_time or period.trading_date
        key = (period.trading_date, cutoff, tuple(s.symbol for s in stocks))
        if key != self._news_selection_key:
            self._news_candidates, self.last_news_selection = news_candidates(
                [s.symbol for s in stocks],
                self.news,
                period.trading_date,
                cutoff,
                frequency=self._instance.frequency,
                top_assets=self.news_top_assets,
                min_articles=self.news_min_articles,
            )
            self._news_selection_key = key
        selected = [s for s in stocks if s.symbol in self._news_candidates]
        if len(selected) < self.min_assets:
            raise ValueError(
                "News shortlist leaves fewer candidates than minimum assets"
            )
        return selected

    def _predict_returns(self) -> dict[str, float]:
        """Predict returns using the configured shared EWM or mean-blend estimator.

        Returns
        -------
        dict[str, float]
            Predicted return per symbol.

        Raises
        ------
        PredictionError
            If prediction fails or produces no results.
        """
        try:
            stocks = self.stocks_with_recent_history()
            strategy: BasePredictionStrategy
            if self.return_estimator == OptimizationReturnEstimator.EWM_12_MEAN_50:
                strategy = MeanBlendPredictionStrategy(
                    stocks, momentum_weight=0.5, span=12
                )
            else:
                span = (
                    12
                    if self.return_estimator == OptimizationReturnEstimator.EWM_12
                    else 6
                )
                strategy = AveragePredictionStrategy(stocks, span=span)
            expected_returns: dict[str, float] = strategy.predict_returns()
        except Exception as e:
            raise PredictionError(
                f"Failed to predict returns: {e!s}",
                model=self.return_estimator.value,
                original_error=e,
            ) from e

        if not expected_returns:
            raise PredictionError(
                "No expected returns could be predicted",
                model=self.return_estimator.value,
            )

        logger.debug("Predicted returns for %d stocks", len(expected_returns))
        return expected_returns

    def _build_algorithm_portfolio(
        self,
    ) -> list[tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]]:
        """Build the island/ensemble algorithm portfolio.

        Each entry is ``(algorithm_class, algorithm_kwargs)`` and restarts
        cycle through them.  This hedges against any single algorithm
        getting stuck on a particular problem landscape.

        Returns
        -------
        list[tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]]
            Algorithm portfolio for the multi-start optimizer.
        """
        variator = PortfolioVariator(
            min_assets=self.min_assets,
            max_assets=self.max_assets,
            mutation_prob=GENETIC.mutation_prob,
        )
        # Gentler variator variant for algorithms that benefit from less
        # aggressive mutation (more perturbation, less cardinality change)
        gentle_variator = PortfolioVariator(
            min_assets=self.min_assets,
            max_assets=self.max_assets,
            mutation_prob=GENTLE_VARIATOR.mutation_prob,
            perturb_prob=GENTLE_VARIATOR.perturb_prob,
            cardinality_prob=GENTLE_VARIATOR.cardinality_prob,
            noise_std=GENTLE_VARIATOR.noise_std,
        )

        portfolio: list[tuple[type[AbstractGeneticAlgorithm], dict[str, Any]]] = [
            # Island 0: NSGA-III — purpose-built for many-objective problems
            (
                NSGAIII,
                {
                    "divisions_outer": ALGORITHM.divisions_outer,
                    "variator": variator,
                },
            ),
            # Island 1: NSGA-II — robust workhorse with gentle variator
            (
                NSGAII,
                {
                    "variator": gentle_variator,
                },
            ),
            # Island 2: EpsNSGAII — epsilon-dominance archive for diversity
            (
                EpsNSGAII,
                {
                    "epsilons": list(ALGORITHM.epsilons),
                    "variator": variator,
                },
            ),
            # Island 3: SPEA2 — density-based archive, different selection pressure
            (
                SPEA2,
                {
                    "variator": variator,
                },
            ),
            # Island 4: NSGA-III with gentle variator — extra exploitation run
            (
                NSGAIII,
                {
                    "divisions_outer": ALGORITHM.divisions_outer,
                    "variator": gentle_variator,
                },
            ),
        ]
        return portfolio

    def _build_optimizer(
        self,
        expected_returns: dict[str, float],
        covariance_matrix: pd.DataFrame,
    ) -> BaseOptimizationStrategy:
        """Construct the portfolio optimizer for the current period.

        Dispatches to the appropriate backend based on
        ``self.optimizer_type``:

        * **EVOLUTIONARY** — island/ensemble multi-start EA.
        * **DECOMPOSITION** — scalarized SLSQP subproblems.
        * **HYBRID** — reduced-budget EA + SLSQP polishing.

        Parameters
        ----------
        expected_returns : dict[str, float]
            Predicted returns per symbol.
        covariance_matrix : pd.DataFrame
            Covariance matrix of asset returns.

        Returns
        -------
        BaseOptimizationStrategy
            Configured optimizer ready to call ``optimize()``.
        """
        trading_period = self._instance.current_trading_period
        date_label = (
            str(trading_period.trading_date.date())
            if trading_period is not None
            else None
        )

        model_kwargs: dict[str, Any] = {
            "balance": self.balance,
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
            algorithm_portfolio = self._build_algorithm_portfolio()
            return HybridOptimizationStrategy(
                expected_returns=pd.Series(expected_returns),
                covariance_matrix=covariance_matrix,
                risk_aversion=self.risk_aversion,
                algorithm=NSGAIII,
                algorithm_kwargs={
                    "divisions_outer": ALGORITHM.divisions_outer,
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
                algorithm_portfolio=algorithm_portfolio,
                ea_budget_ratio=OPTIMIZER.ea_budget_ratio,
                n_decomposition_directions=DECOMPOSITION.n_decomposition_directions,
            )

        # EVOLUTIONARY (default / fallback)
        return MultiGreedyOptimizationStrategy(
            expected_returns=pd.Series(expected_returns),
            covariance_matrix=covariance_matrix,
            risk_aversion=self.risk_aversion,
            # Shared bounded Multi-Greedy + NSGA-II strategy profile
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
        """Execute the optimization strategy pipeline.

        Orchestrates:

        1. EWM return prediction.
        2. Covariance matrix estimation.
        3. Multi-objective portfolio optimization (warm-started after
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
        PredictionError
            If return prediction fails.
        OptimizationError
            If covariance estimation or portfolio optimization fails.
        """
        logger.info(
            "Running optimization strategy (iteration=%d, warm=%s)",
            self._iteration,
            self.is_warm_start,
        )

        # 1. Prediction → 2. Covariance → 3. Optimization
        expected_returns = self._predict_returns()

        try:
            covariance_matrix = self.build_covariance_matrix()
        except Exception as e:
            raise OptimizationError(
                f"Failed to build covariance matrix: {e!s}",
                optimizer="covariance_builder",
                original_error=e,
            ) from e

        try:
            optimizer = self._build_optimizer(expected_returns, covariance_matrix)
            result: OptimizationResult = self.optimize_portfolios(
                optimizer,
                progress_callback=progress_callback,
            )
        except Exception as e:
            raise OptimizationError(
                f"Failed to optimize portfolio: {e!s}",
                optimizer=self.optimizer_type.value,
                original_error=e,
            ) from e

        # Portfolio selection (investment strategy responsibility)
        self.last_optimization_diagnostics = {
            **result.search_diagnostics,
            "balance": self.balance.value,
            "news_selection": self.last_news_selection,
        }
        optimal_weights: pd.Series = self._selector.select(
            result.front, pd.Index(expected_returns.keys())
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
