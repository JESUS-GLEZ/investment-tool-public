"""
Base investment strategy interface.

This module defines the abstract base class for all investment strategies,
providing common data filtering, preprocessing, iteration tracking, and
shared covariance matrix computation.
"""

from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from investment_tool.output.front_record import front_record, write_record
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields
from investment_tool.output.policy import OutputMode
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.security_action import SecurityAction
from investment_tool.domain.trading_costs import TradingCosts, decision_trading_costs
from investment_tool.domain.return_history import (
    adjacent_observations,
    has_return_history,
)
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
    BaseOptimizationStrategy,
)
from investment_tool.strategies.optimization_strategies.front_cache import (
    CandidateFrontCache,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)
from investment_tool.strategies.optimization_strategies.seeded_optimization import (
    MultiGreedyOptimizationStrategy,
)
from investment_tool.utils import (
    build_exp_weighted_covariance_matrix,
    recently_observed_symbols,
)

logger = logging.getLogger(__name__)


class BaseInvestmentStrategy(ABC):
    """Abstract base class for investment strategies.

    Handles data filtering to prevent look-ahead bias by ensuring only
    historical data (before the current trading period) is used.

    At construction time the full (unfiltered) stock and news references
    are stored so that ``refresh_data`` can re-filter them each backtesting
    period — avoiding the "stale data" problem where a single filter at
    init would freeze the dataset for all future periods.

    Iteration tracking
    ------------------
    ``_iteration`` counts how many times ``refresh_data()`` has been called
    (starts at 0 after construction).  ``_previous_optimal_weights`` stores
    the result of the last ``run_strategy()`` call so subclasses can pass
    it to the optimizer as a warm-start seed.

    Parameters
    ----------
    instance : Instance
        Instance configuration with current trading period.
    stocks : Sequence[Stock]
        Stock entities to include in strategy.
    news : dict[tuple[str, date], News] | None, optional
        News data keyed by (symbol, period_start_date).

    Attributes
    ----------
    stocks : list[Stock]
        Filtered stock data up to current trading period.
    news : dict[tuple[str, date], News]
        Filtered news data up to current trading period.
    _iteration : int
        Number of completed rebalancing iterations (0 = first period).
    _previous_optimal_weights : np.ndarray | None
        Optimal weights from the previous period (``None`` on first call).
    _previous_front_weights : list[np.ndarray] | None
        Weight vectors from the previous Pareto front for diverse
        warm-starting of the optimizer.

    Raises
    ------
    ValueError
        If instance is None or has no current trading period.
        If stocks sequence is empty.
    """

    def __init__(
        self,
        instance: Instance,
        stocks: Sequence[Stock],
        news: dict[tuple[str, date], News] | None = None,
    ) -> None:
        if instance is None or instance.current_trading_period is None:
            raise ValueError("Instance must have a current_trading_period")

        if not stocks:
            raise ValueError("At least one stock must be provided")

        # Keep unfiltered references for refresh_data()
        self._instance_ref = instance
        self._raw_stocks: Sequence[Stock] = stocks
        self._raw_news: dict[tuple[str, date], News] | None = news

        # Iteration tracking for warm-start support
        self._iteration: int = 0
        self._previous_optimal_weights: np.ndarray | None = None
        self._previous_front_weights: list[np.ndarray] | None = None
        self.last_optimization_diagnostics: dict[str, Any] = {}
        self.last_covariance_diagnostics: dict[str, Any] = {}
        self.output_mode = OutputMode.STANDARD
        self.optimization_settings: dict[str, Any] = {}
        self.candidate_front_cache: CandidateFrontCache | None = None
        self._portfolio: Portfolio | None = None
        self._security_actions: Sequence[SecurityAction] = ()
        self._trading_costs: TradingCosts | None = None
        self._cost_returns: pd.Series | None = None
        self._optimization_covariance: np.ndarray | None = None

        self._apply_data_filter(instance.current_trading_period)

    def bind_portfolio(
        self, portfolio: Portfolio, events: Sequence[SecurityAction] = ()
    ) -> None:
        """Read the actual evolving holdings when building each decision problem."""
        self._portfolio, self._security_actions = portfolio, events

    def optimize_portfolios(
        self,
        optimizer: BaseOptimizationStrategy,
        progress_callback: ProgressCallback | None = None,
    ) -> OptimizationResult:
        """Apply search configuration and optionally share a current-period front."""
        optimizer.output_mode = self.output_mode
        if self._portfolio is not None:
            optimizer.trading_costs = decision_trading_costs(
                self._portfolio,
                self._raw_stocks,
                self._instance_ref.current_trading_period,
                getattr(optimizer, "expected_returns").index,
                self._security_actions,
            )
        self._trading_costs = optimizer.trading_costs
        self._cost_returns = getattr(optimizer, "expected_returns", None)
        covariance = getattr(optimizer, "covariance_matrix", None)
        self._optimization_covariance = (
            np.asarray(covariance, dtype=float)
            if isinstance(covariance, (pd.DataFrame, np.ndarray))
            else None
        )
        if self.optimization_settings or self.candidate_front_cache is not None:
            if not isinstance(optimizer, MultiGreedyOptimizationStrategy):
                raise ValueError(
                    "Configured front sharing requires the seeded NSGA-II optimizer"
                )
            for setting, value in self.optimization_settings.items():
                if setting not in {
                    "random_seed",
                    "search_seconds",
                    "seed_max_subsets",
                    "seed_beam_width",
                    "refinement_supports",
                }:
                    raise ValueError(f"Unsupported optimizer setting: {setting}")
                setattr(optimizer, setting, value)
            if self.candidate_front_cache is not None:
                result = self.candidate_front_cache.optimize(
                    optimizer, progress_callback
                )
            else:
                result = optimizer.optimize(progress_callback=progress_callback)
        else:
            result = optimizer.optimize(progress_callback=progress_callback)
        if self.last_covariance_diagnostics:
            result.search_diagnostics["covariance"] = dict(
                self.last_covariance_diagnostics
            )
        if optimizer.trading_costs is not None:
            result.search_diagnostics["trading_costs"] = (
                optimizer.trading_costs.record()
            )
            result.search_diagnostics["cost_assets"] = list(
                map(str, getattr(optimizer, "expected_returns").index)
            )
            result.search_diagnostics["cost_price_basis"] = (
                "latest_completed_available_close"
            )
            current = self._instance_ref.current_trading_period
            result.search_diagnostics["cost_decision_time"] = (
                current.decision_time.isoformat()
                if current is not None and current.decision_time is not None
                else None
            )
        directory = getattr(self, "output_directory", None)
        if directory is not None and self.candidate_front_cache is None:
            returns = getattr(optimizer, "expected_returns")
            covariance = getattr(optimizer, "covariance_matrix")
            date_label = getattr(optimizer, "date_label", None)
            assets = list(map(str, returns.index))
            digest = hashlib.sha256()
            digest.update("\0".join(assets).encode())
            digest.update(np.asarray(returns, dtype="<f8").tobytes())
            digest.update(np.asarray(covariance, dtype="<f8").tobytes())
            if optimizer.trading_costs is not None:
                digest.update(
                    json.dumps(
                        optimizer.trading_costs.record(), sort_keys=True
                    ).encode()
                )
            path = (
                Path(directory)
                / "candidate_fronts"
                / f"{date_label}_{digest.hexdigest()[:16]}.json.gz"
            )
            write_record(
                path,
                {
                    **front_record(result, self.output_mode),
                    "assets": assets,
                    "date_label": date_label,
                    "problem_fingerprint": digest.hexdigest(),
                },
            )
            result.search_diagnostics["candidate_front_file"] = str(path)
        return result

    # ------------------------------------------------------------------
    # Data filtering
    # ------------------------------------------------------------------

    def refresh_data(self) -> None:
        """Re-filter stocks and news for the current trading period.

        Must be called before ``run_strategy()`` whenever the instance's
        ``current_trading_period`` has advanced (e.g. in backtesting).
        This prevents using stale data from a previous period.

        Also increments ``_iteration`` so subclasses can distinguish
        between the first call (cold start) and subsequent calls
        (warm start with ``_previous_optimal_weights``).
        """
        current = self._instance_ref.current_trading_period
        if current is None:
            raise ValueError("Instance must have a current_trading_period")
        self._apply_data_filter(current)
        self._iteration += 1

    def store_optimal_weights(
        self,
        weights: pd.Series,
        front_weights: list[np.ndarray] | None = None,
    ) -> None:
        """Cache optimal weights and Pareto front for the next iteration.

        Parameters
        ----------
        weights : pd.Series
            Selected optimal portfolio weights from the current period.
        front_weights : list[np.ndarray] | None, optional
            Weight vectors from the full Pareto front.  When provided,
            the optimizer can warm-start from the entire frontier
            instead of a single point, preserving diversity.
        """
        self._previous_optimal_weights = np.asarray(weights.values)
        self._previous_front_weights = front_weights
        if self._trading_costs is not None and self._cost_returns is not None:
            aligned = weights.reindex(
                self._cost_returns.index, fill_value=0.0
            ).to_numpy(dtype=float)
            gross = float(aligned @ self._cost_returns.to_numpy(dtype=float))
            fee = self._trading_costs.fraction(aligned)
            self.last_optimization_diagnostics.update(
                {
                    "selected_gross_expected_return": gross,
                    "selected_estimated_cost_fraction": fee,
                    "selected_net_expected_return": gross - fee,
                }
            )
        if self._cost_returns is not None and self._optimization_covariance is not None:
            aligned = weights.reindex(
                self._cost_returns.index, fill_value=0.0
            ).to_numpy(dtype=float)
            variance = float(aligned @ self._optimization_covariance @ aligned)
            if np.isfinite(variance) and variance >= 0:
                self.last_optimization_diagnostics.update(
                    selected_variance=variance,
                    selected_volatility=float(np.sqrt(variance)),
                )

    @property
    def is_warm_start(self) -> bool:
        """Whether warm-start data is available from a previous iteration.

        Returns
        -------
        bool
            ``True`` if ``_previous_optimal_weights`` is not ``None``.
        """
        return self._previous_optimal_weights is not None

    def build_covariance_matrix(self) -> pd.DataFrame:
        """Build the exponentially-weighted covariance matrix from current stocks.

        Uses only already-filtered observed returns. Individual history is
        retained when another asset becomes eligible; missing returns are not
        copied forward as if they were fresh observations.

        Returns
        -------
        pd.DataFrame
            Exponentially weighted covariance matrix indexed by symbol.
        """
        eligible_stocks = self.stocks_with_recent_history()
        returns: pd.DataFrame = pd.concat(
            [stock.data[StockFields.RETURNS] for stock in eligible_stocks], axis=1
        )
        returns.columns = [stock.symbol for stock in eligible_stocks]

        cov: pd.DataFrame = build_exp_weighted_covariance_matrix(returns)
        self.last_covariance_diagnostics = dict(
            cov.attrs.get("covariance_diagnostics", {})
        )
        return cov

    def stocks_with_recent_history(self) -> list[Stock]:
        """Exclude assets without enough fresh returns for optimization."""
        returns = pd.concat(
            [stock.data[StockFields.RETURNS] for stock in self.stocks], axis=1
        )
        returns.columns = [stock.symbol for stock in self.stocks]
        eligible = set(recently_observed_symbols(returns))
        stocks = [stock for stock in self.stocks if stock.symbol in eligible]
        if not stocks:
            raise ValueError("No assets have enough recent return history")
        if len(stocks) < len(self.stocks):
            logger.info(
                "Excluded %d assets without a complete recent return window",
                len(self.stocks) - len(stocks),
            )
        return stocks

    def _apply_data_filter(self, current_trading_period: Period) -> None:
        """Filter stocks and news up to *current_trading_period*.

        Parameters
        ----------
        current_trading_period : Period
            Only data strictly before this period's trading date is kept.
        """
        logger.info(
            f"Filtering strategy data for trading date: "
            f"{current_trading_period.trading_date}"
        )

        execution_candidates = current_trading_period.execution_eligible_symbols
        candidates = [
            Stock(
                stock.symbol,
                self._adjust_stock_data(stock, current_trading_period),
            )
            for stock in self._raw_stocks
            if (
                current_trading_period.eligible_symbols is None
                or stock.symbol in current_trading_period.eligible_symbols
            )
            and (execution_candidates is None or stock.symbol in execution_candidates)
        ]
        self.stocks = candidates
        minimum = current_trading_period.minimum_return_periods
        if minimum is not None:
            frequency = self._instance_ref.frequency
            calendars = self._instance_ref.history_calendars
            for stock in candidates:
                if stock.data.empty:
                    continue
                # Do not bridge older holes after recent eligibility recovers.
                adjacent = adjacent_observations(
                    stock.data.index, frequency, calendars.get(stock.symbol)
                )
                stock.data.loc[
                    ~adjacent, [StockFields.RETURNS, StockFields.LOG_RETURNS]
                ] = np.nan
            self.stocks = [
                stock
                for stock in candidates
                if has_return_history(
                    stock.data,
                    current_trading_period.trading_date,
                    minimum,
                    frequency,
                    calendars.get(stock.symbol),
                    self._instance_ref.history_calendar_end,
                )
            ]
            if not self.stocks:
                raise ValueError(
                    "No dated candidates have the required completed monthly history "
                    f"span ({minimum} months of {frequency.value} bars)"
                )

        if self._raw_news:
            self.news: dict[tuple[str, date], News] = self._filter_news(
                self._raw_news, current_trading_period
            )
            logger.debug(
                f"Processed {len(self.stocks)} stocks and {len(self.news)} news items"
            )
        else:
            self.news = {}
            logger.debug(f"Processed {len(self.stocks)} stocks (no news)")

    @staticmethod
    def _adjust_stock_data(
        stock: Stock, current_trading_period: Period
    ) -> pd.DataFrame:
        """
        Adjust stock data to include only data before the current trading period.

        Parameters
        ----------
        stock : Stock
            The stock whose data is to be adjusted.
        current_trading_period : Period
            The current trading period object with attribute `trading_date`.

        Returns
        -------
        pd.DataFrame
            Adjusted stock data sorted by ascending date.

        Notes
        -----
        This method filters out any data points on or after the current trading date.
        """
        if current_trading_period.decision_time is not None:
            from investment_tool.domain.market_time import observable_history

            return pd.DataFrame(
                observable_history(
                    stock.data,
                    current_trading_period.trading_date,
                    current_trading_period.decision_time,
                )
            )
        # Compatibility for directly constructed, untimed domain entities.
        # All repository/factory entry points require the explicit time contract.
        trading_date = pd.to_datetime(current_trading_period.trading_date)
        stock_data_adjusted: pd.DataFrame = stock.data[stock.data.index < trading_date]

        if stock_data_adjusted.empty:
            logger.warning(
                f"No historical data available for {stock.symbol} before {trading_date}"
            )

        return pd.DataFrame(stock_data_adjusted.sort_index())

    @staticmethod
    def _filter_news(
        news: dict[tuple[str, date], News], current_trading_period: Period
    ) -> dict[tuple[str, date], News]:
        """
        Filter stock news articles to include only those before the current trading period.

        Parameters
        ----------
        news: dict[tuple[str, date], News]
            Dictionary mapping (stock symbol, date) tuples to News objects.
        current_trading_period: Period
            The current trading period object with attribute `trading_date`.

        Returns
        -------
        dict[tuple[str, date], News]
            Filtered news dictionary containing only entries before the trading date.
        """
        cutoff = (
            current_trading_period.decision_time or current_trading_period.trading_date
        )
        trading_date = current_trading_period.trading_date.date()
        filtered_news = {
            key: value.available_before(cutoff)
            for key, value in news.items()
            if key[1] < trading_date
        }

        if len(filtered_news) < len(news):
            logger.info(
                f"Filtered out {len(news) - len(filtered_news)} news items on or after {trading_date}"
            )

        return filtered_news

    @abstractmethod
    def run_strategy(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.Series:
        """
        Run the investment strategy to generate optimal portfolio weights.

        This method must be implemented by subclasses to define the specific
        strategy combination logic.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Called periodically with ``(fraction_complete, message)``.
            Subclasses should thread this to their optimizer's
            ``optimize()`` call.

        Returns
        -------
        pd.Series
            A pandas Series containing the optimal weights for each asset in the portfolio.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement run_strategy method")

    @staticmethod
    def _log_weights(weights: pd.Series) -> None:
        """Log selected portfolio weights (non-dust positions only).

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by symbol.
        """
        active = {s: w for s, w in weights.items() if w >= 0.01}
        logger.info(
            "Optimization complete. Selected %d assets: %s",
            len(active),
            ", ".join(f"{s}:{w:.4f}" for s, w in active.items()),
        )
