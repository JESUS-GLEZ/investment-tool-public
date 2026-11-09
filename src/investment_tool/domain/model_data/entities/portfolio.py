"""
Portfolio entity for investment position management.

This module defines the Portfolio class which manages cash and asset
positions, handles rebalancing with transaction costs, and maintains
trade history.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields
from investment_tool.domain.model_data.entities.security_action import SecurityAction
from investment_tool.domain.market_time import utc_instant
from investment_tool.domain.model_data.entities.instance import Instance

logger = logging.getLogger(__name__)


class Portfolio:
    """Investment portfolio with cash and asset positions.

    Manages portfolio state, executes rebalancing trades with transaction
    costs, and maintains a history of all transactions.

    Parameters
    ----------
    instance : Instance
        Execution instance with market data references.
    initial_cash : float
        Starting cash balance.
    transaction_cost_rate : float, optional
        Nonnegative transaction cost as fraction of trade value; 0.001 = 0.1%.
    min_position : float, optional
        Minimum position size threshold for cleanup, by default 1e-6.

    Attributes
    ----------
    instance : Instance
    cash : float
        Current cash balance.
    transaction_cost_rate : float
    min_position : float
    positions : dict[str, float]
        Current asset positions (symbol -> quantity).
    history : list[dict[str, Any]]
        List of historical snapshots after each rebalancing.
    """

    def __init__(
        self,
        instance: Instance,
        initial_cash: float,
        transaction_cost_rate: float = 0.0,
        min_position: float = 1e-6,
    ) -> None:
        if not np.isfinite(transaction_cost_rate) or transaction_cost_rate < 0:
            raise ValueError("Transaction cost rate must be finite and nonnegative")
        self.instance = instance
        self.cash = initial_cash
        self.transaction_cost_rate = transaction_cost_rate
        self.min_position = min_position
        self.positions: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        self.corporate_actions: list[dict[str, Any]] = []
        self._applied_actions: set[str] = set()

    def settle_cash_actions(
        self, events: Sequence[SecurityAction], cutoff: datetime
    ) -> None:
        """Compatibility entry point for existing cash-redemption callers."""
        self.apply_security_actions(events, cutoff)

    def validate_security_actions(
        self,
        events: Sequence[SecurityAction],
        cutoff: datetime,
        *,
        held_symbols: set[str] | None = None,
    ) -> set[str]:
        """Check entitlement completeness along a holding path without crediting it.

        Project received securities in accounting order, including chains and
        retained parents. A caller can supply prospective rebalance holdings.
        Already applied events are ignored; unheld events are skipped in this
        projection. The accounting caller records them when actually processed.
        This checks declared completeness, not quote or amount validity.
        """
        instant = utc_instant(cutoff)
        held = (
            {symbol for symbol, quantity in self.positions.items() if quantity}
            if held_symbols is None
            else set(held_symbols)
        )
        for event in sorted(
            events, key=lambda e: (max(e.effective_at, e.published_at), e.key)
        ):
            if (
                max(event.effective_at, event.published_at) > instant
                or event.key in self._applied_actions
                or event.symbol not in held
            ):
                continue
            if event.unresolved_entitlements:
                raise ValueError(
                    f"Incomplete security action for {event.symbol}: "
                    + "; ".join(event.unresolved_entitlements)
                )
            if not event.retain_original:
                held.discard(event.symbol)
            held.update(symbol for symbol, _ in event.successor_shares)
        return held

    def apply_security_actions(
        self, events: Sequence[SecurityAction], cutoff: datetime
    ) -> None:
        """Credit sourced entitlements once, after effectiveness and availability.

        Received shares are holdings even when outside candidate membership.
        A distribution never repeats for shares bought after its effective date.
        """
        # Check the entire ordered batch before crediting any earlier component.
        self.validate_security_actions(events, cutoff)
        instant = utc_instant(cutoff)
        for event in sorted(
            events, key=lambda e: (max(e.effective_at, e.published_at), e.key)
        ):
            if (
                max(event.effective_at, event.published_at) > instant
                or event.key in self._applied_actions
            ):
                continue
            quantity = self.positions.get(event.symbol, 0.0)
            proceeds = quantity * event.cash_per_share
            received = {
                symbol: quantity * ratio for symbol, ratio in event.successor_shares
            }
            if (
                not np.isfinite(proceeds)
                or proceeds < 0
                or any(not np.isfinite(q) or q < 0 for q in received.values())
            ):
                raise ValueError("Invalid corporate-action entitlement")
            self._applied_actions.add(event.key)
            if not quantity:
                continue
            if not event.retain_original:
                self.positions.pop(event.symbol)
            self.cash += proceeds
            for symbol, shares in received.items():
                self.positions[symbol] = self.positions.get(symbol, 0.0) + shares
            self.corporate_actions.append(
                {
                    "symbol": event.symbol,
                    "quantity": quantity,
                    "cash_per_share": event.cash_per_share,
                    "proceeds": proceeds,
                    "successor_shares": received,
                    "retain_original": event.retain_original,
                    "effective_at": event.effective_at.isoformat(),
                    "published_at": event.published_at.isoformat(),
                    "accounted_at": instant.isoformat(),
                    "source": event.source,
                }
            )

    def update(
        self, optimal_weights: pd.Series, *, execution_prices: pd.Series | None = None
    ) -> None:
        """Rebalance towards target weights, reducing buys to fund nonzero fees.

        Parameters
        ----------
        optimal_weights : pd.Series
            Target weights for each asset (should sum to ~1.0).
        execution_prices : pd.Series | None
            Explicit simulated fill prices. Mark existing holdings at these
            prices before sizing new positions and validate held/selected assets.
            If omitted, retain the previous-close behavior for other callers.

        Raises
        ------
        TypeError
            If optimal_weights is not a pandas Series.
        ValueError
            If prices cannot be retrieved.
        """
        if not isinstance(optimal_weights, pd.Series):
            raise TypeError("optimal_weights must be a pandas Series")
        if not np.isclose(optimal_weights.sum(), 1.0, atol=0.001):
            logger.warning(f"Optimal weights sum to {optimal_weights.sum():.4f}")

        prices = self.get_prices() if execution_prices is None else execution_prices
        if prices is None:
            raise ValueError("Could not retrieve valid prices for rebalancing")
        if execution_prices is not None:
            required = set(self.positions) | set(
                optimal_weights[optimal_weights > 0].index
            )
            invalid = sorted(
                symbol
                for symbol in required
                if symbol not in prices
                or not np.isfinite(prices[symbol])
                or prices[symbol] <= 0
            )
            if invalid:
                raise ValueError(f"Missing or invalid execution prices: {invalid}")
            # Zero-weight assets need no quote. Previously held assets do,
            # including those removed from the new eligible universe.
            prices = prices.loc[sorted(required)]
            optimal_weights = optimal_weights[optimal_weights > 0]

        current_value = self.calculate_portfolio_value(prices)
        target_quantities = self._calculate_target_quantities(
            optimal_weights, current_value, prices
        )
        current_quantities = pd.Series(self.positions)
        trades = self._calculate_trades(current_quantities, target_quantities)
        trades = self._fund_trades(trades, prices)
        self._execute_trades(trades, prices)
        self._record_history(prices, current_value, trades)

    def calculate_portfolio_value(self, prices: pd.Series) -> float:
        """Calculate total portfolio value (cash + positions).

        Parameters
        ----------
        prices : pd.Series
            Current asset prices.

        Returns
        -------
        float
            Total portfolio value.
        """
        positions_value: float = sum(
            qty * float(prices.get(symbol, 0.0))
            for symbol, qty in self.positions.items()
        )
        return float(positions_value + self.cash)

    def get_prices(self) -> pd.Series | None:
        """Get asset prices from previous trading period.

        Returns
        -------
        pd.Series | None
            Close prices indexed by symbol, or None if retrieval fails.
        """
        previous_day = self.instance.previous_trading_period
        if previous_day is None:
            logger.error("No previous trading period available")
            return None

        try:
            prices: pd.Series = previous_day.market_data.xs(
                StockFields.CLOSE, level=0, axis=1
            ).loc[previous_day.trading_date]
            prices = prices.replace(0.0, np.nan).dropna()
            return prices
        except KeyError as e:
            logger.error(f"Price retrieval failed: {e}")
            return None

    def _calculate_target_quantities(
        self, optimal_weights: pd.Series, portfolio_value: float, prices: pd.Series
    ) -> pd.Series:
        """Convert target weights to share quantities.

        Parameters
        ----------
        optimal_weights : pd.Series
            Target portfolio weights.
        portfolio_value : float
            Current total portfolio value.
        prices : pd.Series
            Current asset prices.

        Returns
        -------
        pd.Series
            Target quantities for each asset.
        """
        valid_weights = optimal_weights[optimal_weights.index.isin(prices.index)]
        target_values = valid_weights * portfolio_value
        prices_non_zero = prices.replace(0.0, np.inf)
        return (target_values / prices_non_zero).fillna(0.0).clip(lower=0.0)

    def _calculate_trades(self, current: pd.Series, target: pd.Series) -> pd.Series:
        """Calculate required trades to reach target positions.

        Parameters
        ----------
        current : pd.Series
            Current position quantities.
        target : pd.Series
            Target position quantities.

        Returns
        -------
        pd.Series
            Trade quantities (positive = buy, negative = sell).
        """
        symbols = current.index.union(target.index)
        return target.reindex(symbols, fill_value=0.0) - current.reindex(
            symbols, fill_value=0.0
        )

    def _fund_trades(self, trades: pd.Series, prices: pd.Series) -> pd.Series:
        """Pay selling fees, then proportionally reduce unaffordable purchases.

        Target sales remain unchanged. This adjusts execution quantities only;
        it does not introduce a turnover objective or change portfolio selection.
        """
        rate = self.transaction_cost_rate
        if rate == 0:
            return trades
        sale_value = float((-trades.clip(upper=0) * prices).sum())
        available = self.cash + sale_value * (1 - rate)
        if available < 0:
            raise ValueError("Insufficient cash to pay transaction costs on sales")
        purchase_cost = float((trades.clip(lower=0) * prices).sum()) * (1 + rate)
        if purchase_cost <= available or purchase_cost == 0:
            return trades
        funded = trades.copy()
        funded.loc[funded > 0] *= available / purchase_cost
        return funded

    def _execute_trades(self, trades: pd.Series, prices: pd.Series) -> None:
        """Execute trades and update portfolio state.

        Parameters
        ----------
        trades : pd.Series
            Trade quantities to execute.
        prices : pd.Series
            Current asset prices.
        """
        trade_values = trades * prices
        transaction_costs = trade_values.abs() * self.transaction_cost_rate
        cash = float(self.cash - (trade_values.sum() + transaction_costs.sum()))
        if self.transaction_cost_rate > 0 and cash < 0:
            # Only absorb floating-point cancellation after funding trades.
            tolerance = (
                8
                * np.finfo(float).eps
                * max(1.0, abs(self.cash), float(trade_values.abs().sum()))
            )
            if cash < -tolerance:
                raise ValueError("Insufficient cash for trades and transaction costs")
            cash = 0.0
        self.cash = cash
        new_positions = trades.add(pd.Series(self.positions), fill_value=0.0).to_dict()
        self.positions = {
            k: v for k, v in new_positions.items() if abs(v) >= self.min_position
        }

    def _record_history(
        self, prices: pd.Series, pre_rebalance_value: float, trades: pd.Series
    ) -> None:
        """Record portfolio state snapshot after rebalancing.

        Parameters
        ----------
        prices : pd.Series
            Asset prices at time of rebalancing.
        pre_rebalance_value : float
            Portfolio value before rebalancing.
        trades : pd.Series
            Trades executed in this rebalancing.
        """
        current_period = self.instance.current_trading_period
        self.history.append(
            {
                "date": current_period.trading_date if current_period else None,
                "cash": self.cash,
                "positions": self.positions.copy(),
                "trades": trades.to_dict(),
                "transaction_costs": float(
                    ((trades * prices).abs() * self.transaction_cost_rate).sum()
                ),
                "prices": prices.to_dict(),
                "pre_rebalance_value": pre_rebalance_value,
                "post_rebalance_value": self.calculate_portfolio_value(prices),
            }
        )
