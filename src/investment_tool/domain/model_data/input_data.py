"""
Input data container for investment tool entities.

This module defines the InputData class which serves as a container
for all the core entities used in the investment tool: instance config,
trading periods, stocks, news, and portfolio.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any
from pathlib import Path

from investment_tool.domain.model_data.entities.cash_settlement import CashSettlement
from investment_tool.domain.model_data.entities.security_action import SecurityAction
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.stock import Stock


class InputData:
    """Container for investment tool input data.

    Holds references to all core entities needed for strategy execution
    and backtesting.

    Attributes
    ----------
    instance : Instance
        Configuration and trading period references.
    periods : list[Period]
        Chronologically ordered trading periods.
    stocks : list[Stock]
        Stock entities with historical data.
    news : dict[tuple[str, date], News]
        News data keyed by (symbol, period_start_date).
    portfolio : Portfolio
        Current portfolio state.
    """

    _instance: Instance
    _periods: list[Period]
    _stocks: list[Stock]
    _news: dict[tuple[str, date], News]
    _portfolio: Portfolio

    def __init__(self) -> None:
        """Initialize empty InputData container."""
        self.input_snapshot_path: Path | None = None
        self.crypto_symbols: set[str] = set()
        self.cash_settlements: tuple[CashSettlement, ...] = ()
        self.security_actions: tuple[SecurityAction, ...] = ()
        self.validation_report: dict[str, Any] = {}

    @property
    def corporate_action_events(self) -> tuple[SecurityAction, ...]:
        return tuple(
            sorted(
                (*self.cash_settlements, *self.security_actions),
                key=lambda event: (
                    max(event.effective_at, event.published_at),
                    event.key,
                ),
            )
        )

    @property
    def instance(self) -> Instance:
        """Get the instance configuration."""
        return self._instance

    @instance.setter
    def instance(self, instance: Instance) -> None:
        """Set the instance configuration.

        Parameters
        ----------
        instance : Instance
            Instance configuration to assign.
        """
        self._instance = instance

    @property
    def periods(self) -> list[Period]:
        """Get the list of trading periods."""
        return self._periods

    @periods.setter
    def periods(self, periods: Sequence[Period]) -> None:
        """Set the trading periods.

        Parameters
        ----------
        periods : Sequence[Period]
            Sequence of Period objects.
        """
        self._periods = list(periods)

    @property
    def stocks(self) -> list[Stock]:
        """Get the list of stock entities."""
        return self._stocks

    @stocks.setter
    def stocks(self, stocks: Sequence[Stock]) -> None:
        """Set the stock entities.

        Parameters
        ----------
        stocks : Sequence[Stock]
            Sequence of Stock objects.
        """
        self._stocks = list(stocks)

    @property
    def news(self) -> dict[tuple[str, date], News]:
        """Get the news data dictionary."""
        return self._news

    @news.setter
    def news(self, news: dict[tuple[str, date], News]) -> None:
        """Set the news data.

        Parameters
        ----------
        news : dict[tuple[str, date], News]
            News data keyed by (symbol, period_start_date).
        """
        self._news = dict(news)

    @property
    def portfolio(self) -> Portfolio:
        """Get the portfolio."""
        return self._portfolio

    @portfolio.setter
    def portfolio(self, portfolio: Portfolio) -> None:
        """Set the portfolio.

        Parameters
        ----------
        portfolio : Portfolio
            Portfolio object to assign.
        """
        self._portfolio = portfolio
