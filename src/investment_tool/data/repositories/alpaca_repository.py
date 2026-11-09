"""
Alpaca API data repository implementation.

This module provides the AlpacaRepository class for fetching financial
data from the Alpaca trading API.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.data.repositories._alpaca_client import AlpacaDataFetcher
from investment_tool.data.repositories.base_repository import (
    BaseDataRepository,
    ExecutionQuote,
)

logger = logging.getLogger(__name__)


class AlpacaRepository(BaseDataRepository):
    """Repository for fetching data from Alpaca API.

    Wraps the existing AlpacaDataFetcher to provide a clean repository
    interface, enabling consistent data access patterns and easier testing.

    Parameters
    ----------
    frequency : Frequency
        Data frequency (DAILY, WEEKLY, MONTHLY).
    rate_limit_sleep : float, optional
        Time to sleep between API calls, by default 0.01.

    Attributes
    ----------
    _fetcher : AlpacaDataFetcher
        The underlying data fetcher instance.
    """

    def __init__(
        self,
        frequency: Frequency,
        rate_limit_sleep: float = 0.01,
        symbol_identity_date: date | None = None,
    ) -> None:
        super().__init__(frequency, symbol_identity_date)
        self._fetcher = AlpacaDataFetcher(rate_limit_sleep=rate_limit_sleep)
        self._logger.info("AlpacaRepository initialized")

    @property
    def source_name(self) -> str:
        """Return the data source name."""
        return "Alpaca"

    def fetch_stock_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical stock data from Alpaca."""
        self._logger.info(f"Fetching stock data for {len(symbols)} symbols")
        self.validate_date_range(start_date, end_date)
        identity_options = (
            {"symbol_identity_date": self.symbol_identity_date}
            if self.symbol_identity_date is not None
            else {}
        )
        stock_dict: dict[str, pd.DataFrame] = self._fetcher.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=self.frequency,
            **identity_options,
        )
        self._logger.info(f"Retrieved stock data for {len(stock_dict)} symbols")
        return stock_dict

    def fetch_execution_data(
        self, crypto_symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, ExecutionQuote]:
        self.validate_date_range(start_date, end_date)
        quotes: dict[str, ExecutionQuote] = self._fetcher.get_execution_data(
            crypto_symbols, start_date, end_date, self.frequency
        )
        return quotes

    def fetch_crypto_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical crypto data from Alpaca."""
        self._logger.info(f"Fetching crypto data for {len(symbols)} symbols")
        self.validate_date_range(start_date, end_date)
        crypto_dict: dict[str, pd.DataFrame] = self._fetcher.get_historical_crypto_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=self.frequency,
        )
        self._logger.info(f"Retrieved crypto data for {len(crypto_dict)} symbols")
        return crypto_dict

    def fetch_news_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[tuple[str, date], Any]:
        """Fetch news data from Alpaca."""
        self._logger.info(f"Fetching news data for {len(symbols)} symbols")
        self.validate_date_range(start_date, end_date)
        news_dict: dict[tuple[str, date], Any] = self._fetcher.get_news_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=self.frequency,
        )
        self._logger.info("Retrieved news data")
        return news_dict
