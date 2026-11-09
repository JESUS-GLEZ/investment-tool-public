"""
Backtester for historical portfolio simulation.

This module provides the core backtesting functionality, simulating
portfolio performance over historical periods using a given investment
strategy and computing performance metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from investment_tool.constants import Frequency, StockFields
from investment_tool.defaults import PORTFOLIO
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.domain.market_quotes import (
    execution_prices as extract_execution_prices,
    period_prices,
    valuation_time as period_valuation_time,
)
from investment_tool.exceptions import (
    BacktestingError,
    BacktestReadinessError,
    ConfigurationError,
)
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)

logger = logging.getLogger(__name__)


class Backtester:
    """Historical portfolio backtester.

    Simulates portfolio performance by iterating through historical periods,
    applying an investment strategy, and tracking portfolio value and returns.

    Parameters
    ----------
    model_data : ModelData
        Historical data containing periods, stocks, and portfolio.
    investment_strategy : BaseInvestmentStrategy
        Strategy to generate portfolio weights at each rebalancing.
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe ratio calculation, by default 0.0.

    Attributes
    ----------
    model_data : ModelData
        The model data being backtested.
    investment_strategy : BaseInvestmentStrategy
        The strategy being applied.
    results : list[dict[str, Any]]
        List of period snapshots with date, total_value, and return.
    frequency : Frequency
        Data frequency (daily, weekly, monthly).
    risk_free_rate : float
        Annual risk-free rate.
    metrics : dict[str, float]
        Computed performance metrics after calling compute_additional_metrics().
    """

    def __init__(
        self,
        model_data: ModelData,
        investment_strategy: BaseInvestmentStrategy,
        risk_free_rate: float = PORTFOLIO.risk_free_rate,
    ) -> None:
        self.model_data = model_data
        self.investment_strategy = investment_strategy
        self.investment_strategy.bind_portfolio(
            model_data.portfolio, model_data.corporate_action_events
        )
        self.results: list[dict[str, Any]] = []
        self.frequency = self.model_data.instance.frequency
        self.risk_free_rate = risk_free_rate
        self.metrics: dict[str, float] = {}
        self._starting_value: float | None = None

    def run_backtest(
        self,
        progress_callback: ProgressCallback | None = None,
        period_complete_callback: Callable[[Backtester], None] | None = None,
    ) -> None:
        """Run the backtest over all historical periods.

        Iterates from the current trading period through all remaining
        periods, applying the strategy and recording snapshots.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Called periodically with ``(fraction_complete, message)``.
            Threaded to the strategy's ``run_strategy()`` call.
        """
        if self.model_data.validation_report.get("readiness", {}).get("ready") is False:
            raise BacktestReadinessError(self.model_data.validation_report)
        current = self.model_data.instance.current_trading_period
        if current is None:
            raise BacktestingError("No current trading period")
        current_trading_period_index = self.model_data.periods.index(current)
        total_periods = len(self.model_data.periods) - current_trading_period_index
        periods_done = 0
        while current_trading_period_index < len(self.model_data.periods):
            current_period = self.model_data.periods[current_trading_period_index]

            # Build a per-period sub-callback that scales optimizer
            # progress within the current period's fraction of the total.
            period_cb: ProgressCallback | None = None
            if progress_callback is not None:
                base_frac = periods_done / total_periods
                period_span = 1.0 / total_periods

                def period_cb(
                    frac: float,
                    msg: str,
                    _b: float = base_frac,
                    _s: float = period_span,
                ) -> None:
                    progress_callback(
                        _b + frac * _s,
                        f"Period {periods_done + 1}/{total_periods}: {msg}",
                    )

            self._process_trading_period(current_period, progress_callback=period_cb)
            current_trading_period_index += 1
            periods_done += 1

            if period_complete_callback is not None:
                period_complete_callback(self)

            if progress_callback is not None:
                progress_callback(
                    periods_done / total_periods,
                    f"Completed period {periods_done}/{total_periods}",
                )
        logger.info("Backtest completed.")

    def _process_trading_period(
        self,
        current_period: Period,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Process a single trading period.

        Re-filters strategy data for the current period, generates optimal
        weights, records a portfolio snapshot, and updates portfolio state.

        Parameters
        ----------
        current_period : Period
            The trading period to process.
        progress_callback : ProgressCallback | None, optional
            Threaded to the strategy's ``run_strategy()`` call.
        """
        events = self.model_data.corporate_action_events
        portfolio = self.model_data.portfolio
        held_at_open = set(portfolio.positions)
        if events and current_period.execution_time is not None:
            try:
                held_at_open = portfolio.validate_security_actions(
                    events, current_period.execution_time
                )
            except ValueError as exc:
                raise BacktestingError(str(exc)) from exc
        execution_prices = extract_execution_prices(
            current_period,
            self.model_data.crypto_symbols,
            self.model_data.corporate_action_events,
        )
        # At the scheduled open, constrain the search to executable securities.
        # Only the opening quote's validity is inspected, never this month's
        # close/return. Historical features retain the earlier decision cutoff.
        current_period.execution_eligible_symbols = frozenset(
            symbol
            for symbol, price in execution_prices.items()
            if np.isfinite(price) and price > 0
        )
        action_start = len(portfolio.corporate_actions)
        missing_held = sorted(
            symbol
            for symbol in held_at_open
            if symbol not in current_period.execution_eligible_symbols
        )
        if missing_held:
            raise BacktestingError(
                f"Missing or invalid execution prices for held assets: {missing_held}. "
                "Provide reviewed corporate-action settlement or replacement quotes; "
                "holdings cannot be dropped or sold at a stale price."
            )
        self.investment_strategy.refresh_data()
        optimal_weights = self.investment_strategy.run_strategy(
            progress_callback=progress_callback,
        )
        valuation_time = period_valuation_time(current_period)
        selected = set(optimal_weights[optimal_weights > 0].index)
        if events and valuation_time is not None:
            # Events processed at the open (even if unheld then) must not be
            # credited to a new purchase. Check only the remaining holding path.
            pending = tuple(
                event
                for event in events
                if current_period.execution_time is None
                or max(event.effective_at, event.published_at)
                > current_period.execution_time
            )
            try:
                portfolio.validate_security_actions(
                    pending, valuation_time, held_symbols=selected
                )
            except ValueError as exc:
                raise BacktestingError(str(exc)) from exc
        redeemable = {
            event.symbol
            for event in events
            if not event.retain_original
            and valuation_time is not None
            and max(event.effective_at, event.published_at) <= valuation_time
        }
        asset_prices = self._extract_period_prices(current_period)
        invalid_closes = sorted(
            symbol
            for symbol in selected
            if symbol not in redeemable
            and (
                symbol not in asset_prices
                or not np.isfinite(asset_prices[symbol])
                or asset_prices[symbol] <= 0
            )
        )
        if invalid_closes:
            raise BacktestingError(
                f"Missing current-period close prices: {invalid_closes}"
            )
        try:
            # Defer actual opening credits until prospective holding accounting
            # is validated too; neither failure may commit part of this period.
            if events and current_period.execution_time is not None:
                portfolio.apply_security_actions(events, current_period.execution_time)
            self.model_data.update(
                optimal_weights,
                execution_prices=pd.Series(execution_prices, dtype=float),
            )
        except ValueError as exc:
            raise BacktestingError(str(exc)) from exc
        history = self.model_data.portfolio.history[-1]
        history["execution_price_basis"] = "current_period_open"
        history["execution_time"] = (
            current_period.execution_time.isoformat()
            if current_period.execution_time
            else None
        )
        # A crypto fill may follow the stock open by up to two minutes.
        # Keep the scheduled open and each asset's actual fill time distinct.
        history["execution_times"] = {
            symbol: actual.isoformat()
            for symbol, quantity in history["trades"].items()
            if quantity != 0
            and (
                actual := current_period.crypto_execution_times.get(
                    symbol, current_period.execution_time
                )
            )
            is not None
        }
        if self._starting_value is None:
            self._starting_value = float(
                self.model_data.portfolio.history[-1]["pre_rebalance_value"]
            )
        if events and valuation_time is not None:
            self.model_data.portfolio.apply_security_actions(events, valuation_time)
        missing_prices = set(self.model_data.portfolio.positions) - set(asset_prices)
        if missing_prices:
            raise BacktestingError(
                "Missing current-period prices for held assets: "
                f"{sorted(missing_prices)}"
            )
        snapshot = self._record_snapshot(
            pd.Timestamp(current_period.period_end_date).to_pydatetime()
            if current_period.period_end_date is not None
            else current_period.trading_date,
            pd.Series(asset_prices, dtype=float),
            optimal_weights,
            asset_prices,
        )
        snapshot["positions"] = dict(self.model_data.portfolio.positions)
        snapshot["cash"] = float(self.model_data.portfolio.cash)
        snapshot["corporate_actions"] = self.model_data.portfolio.corporate_actions[
            action_start:
        ]
        snapshot["execution_price_basis"] = history["execution_price_basis"]
        snapshot["execution_time"] = history["execution_time"]
        snapshot["execution_times"] = history["execution_times"]
        snapshot["period_start"] = current_period.trading_date.date().isoformat()
        snapshot["period_end"] = (
            current_period.period_end_date.isoformat()
            if current_period.period_end_date is not None
            else None
        )
        snapshot["date_basis"] = (
            "period_end_label"
            if current_period.period_end_date is not None
            else "legacy_period_label"
        )
        snapshot["valuation_basis"] = "period_close"
        snapshot["optimization_diagnostics"] = (
            self.investment_strategy.last_optimization_diagnostics
        )
        snapshot["decision_time"] = (
            current_period.decision_time.isoformat()
            if current_period.decision_time
            else None
        )
        history["period_start"] = snapshot["period_start"]
        history["period_end"] = snapshot["period_end"]
        history["decision_time"] = snapshot["decision_time"]
        history["date_basis"] = "rebalance_period_label"
        self.results.append(snapshot)

    def _record_snapshot(
        self,
        current_date: datetime,
        prices: pd.Series,
        optimal_weights: pd.Series,
        asset_prices: dict[str, float],
    ) -> dict[str, Any]:
        """Record portfolio snapshot for a trading date.

        Parameters
        ----------
        current_date : datetime
            The trading date.
        prices : pd.Series
            Current-period asset prices used to value the new holdings.
        optimal_weights : pd.Series
            Optimal weights selected by the strategy.
        asset_prices : dict[str, float]
            Per-asset close prices at the current trading period.

        Returns
        -------
        dict[str, Any]
            Snapshot with date, total_value, return, selected_assets,
            and asset_prices.
        """
        total_value = self.model_data.portfolio.calculate_portfolio_value(prices)
        period_return = self._calculate_daily_return(total_value)

        # Build selected-assets dict: symbol → weight (only active positions)
        selected_assets = {
            sym: float(w) for sym, w in optimal_weights.items() if float(w) >= 0.01
        }

        snapshot: dict[str, Any] = {
            "date": current_date,
            "total_value": total_value,
            "starting_value": self._starting_value,
            "return": period_return,
            "selected_assets": selected_assets,
            "asset_prices": asset_prices,
        }
        return snapshot

    def _calculate_daily_return(self, current_total_value: float) -> float:
        """Calculate period return from previous portfolio value.

        Parameters
        ----------
        current_total_value : float
            Current portfolio value.

        Returns
        -------
        float
            Period return (0.0 if no previous value).
        """
        if self.results:
            previous_value = float(self.results[-1]["total_value"])
        elif self._starting_value is not None:
            previous_value = self._starting_value
        else:
            return 0.0
        return (current_total_value - previous_value) / previous_value

    @staticmethod
    def _extract_period_prices(
        period: Period, field: StockFields = StockFields.CLOSE
    ) -> dict[str, float]:
        """Extract exact-period prices for the requested OHLC field.

        Parameters
        ----------
        period : Period
            Trading period with market data.
        field : StockFields
            Price field, defaulting to the period-end close.

        Returns
        -------
        dict[str, float]
            Symbol → price mapping. Empty dict on failure.
        """
        prices: dict[str, float] = period_prices(period, field)
        return prices

    def compute_additional_metrics(self) -> None:
        """Compute and store performance metrics.

        Calculates annualized return, volatility, and Sharpe ratio,
        storing results in the ``metrics`` attribute.
        """
        from investment_tool.output.metrics import MetricsCalculator

        computed = MetricsCalculator(
            self.results, [], self.frequency, self.risk_free_rate
        ).compute_performance()
        self.metrics = {
            key: float(computed[key])
            for key in (
                "average_return",
                "std_return",
                "annualized_return",
                "annualized_volatility",
                "sharpe_ratio",
            )
        }
        logger.info("Performance metrics: %s", self.metrics)

    def _get_periods_per_year(self) -> int:
        """Get the number of trading periods per year.

        Returns
        -------
        int
            252 for daily, 52 for weekly, 12 for monthly.

        Raises
        ------
        ValueError
            If frequency is not supported.
        """
        if self.frequency == Frequency.DAILY:
            return 252
        elif self.frequency == Frequency.WEEKLY:
            return 52
        elif self.frequency == Frequency.MONTHLY:
            return 12
        raise ConfigurationError(
            f"Unsupported frequency: {self.frequency}",
            details={"frequency": str(self.frequency)},
        )

    def get_results(self) -> pd.DataFrame:
        """Get backtest results as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame with total_value, return, and cumulative_return.

        Raises
        ------
        BacktestingError
            If backtest has not been run yet.
        """
        if not self.results:
            raise BacktestingError(
                "No results to return. Please run the backtest first."
            )

        df_results = pd.DataFrame(self.results)
        df_results.set_index("date", inplace=True)
        df_results["cumulative_return"] = (1 + df_results["return"]).cumprod() - 1
        return df_results
