"""
Stock entity for financial data representation.

This module defines the Stock class which holds historical price data
and computes common technical indicators used in investment analysis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields


class Stock:
    """Represents a stock's historical data with technical indicators.

    Loads historical OHLCV data and automatically computes common technical
    indicators including returns, moving averages, volatility, and RSI.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., "AAPL").
    data : pd.DataFrame
        Historical OHLCV data with DatetimeIndex or 'date' column.

    Attributes
    ----------
    symbol : str
        The stock ticker symbol.
    data : pd.DataFrame
        Historical data with computed technical indicators.

    Raises
    ------
    ValueError
        If data lacks a DatetimeIndex or 'date' column.
    """

    def __init__(self, symbol: str, data: pd.DataFrame) -> None:
        self.symbol: str = symbol
        self.data: pd.DataFrame = data.copy()
        self._prepare_data()

    def __str__(self) -> str:
        """Return string representation (ticker symbol)."""
        return self.symbol

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return f"Stock(symbol='{self.symbol}', periods={len(self.data)})"

    def _prepare_data(self) -> None:
        """Prepare data by ensuring datetime index and computing indicators."""
        # Ensure the index is DatetimeIndex
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if StockFields.DATE in self.data.columns:
                self.data[StockFields.DATE] = pd.to_datetime(
                    self.data[StockFields.DATE]
                )
                self.data.set_index(StockFields.DATE, inplace=True)
            else:
                raise ValueError(
                    f"{self.symbol} data must have a DatetimeIndex or "
                    f"'{StockFields.DATE}' column."
                )

        # Sort chronologically
        self.data.sort_index(inplace=True)

        # A carried price with no trading is not a fresh market observation.
        # Keep its calendar row for auditing/gap detection, but never use its
        # OHLC values for estimation, valuation or simulated execution.
        if StockFields.VOLUME in self.data:
            volume = self.data[StockFields.VOLUME]
            invalid = ~np.isfinite(volume) | (volume <= 0)
            prices = [
                column
                for column in (
                    StockFields.OPEN,
                    StockFields.HIGH,
                    StockFields.LOW,
                    StockFields.CLOSE,
                )
                if column in self.data
            ]
            self.data.loc[invalid, prices] = np.nan

        # Compute technical indicators
        self.calculate_returns()
        self.calculate_log_returns()
        self.calculate_moving_averages()
        self.calculate_volatility()
        self.calculate_rsi()

    def calculate_returns(self) -> None:
        """Calculate simple period returns from close prices."""
        self.data[StockFields.RETURNS] = self.data[StockFields.CLOSE].pct_change(
            fill_method=None
        )

    def calculate_log_returns(self) -> None:
        """Calculate logarithmic returns from close prices."""
        self.data[StockFields.LOG_RETURNS] = np.log(
            self.data[StockFields.CLOSE] / self.data[StockFields.CLOSE].shift(1)
        )

    def calculate_moving_averages(
        self, windows: Sequence[int] = (5, 10, 20, 50, 100, 200)
    ) -> None:
        """Calculate simple moving averages for specified windows.

        Parameters
        ----------
        windows : Sequence[int], optional
            Window sizes for moving averages, by default (5, 10, 20, 50, 100, 200).
        """
        for window in windows:
            ma_field = f"{StockFields.MOVING_AVERAGE}_{window}"
            self.data[ma_field] = (
                self.data[StockFields.CLOSE].rolling(window=window).mean()
            )

    def calculate_volatility(self, window: int = 20) -> None:
        """Calculate rolling volatility (annualized standard deviation).

        Parameters
        ----------
        window : int, optional
            Rolling window size, by default 20.
        """
        self.data[StockFields.VOLATILITY] = self.data[StockFields.RETURNS].rolling(
            window=window
        ).std() * np.sqrt(window)

    def calculate_rsi(self, window: int = 14) -> None:
        """Calculate Relative Strength Index (RSI).

        Parameters
        ----------
        window : int, optional
            RSI period, by default 14.
        """
        delta = self.data[StockFields.CLOSE].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        average_gain = gain.rolling(window=window).mean()
        average_loss = loss.rolling(window=window).mean()
        rs = average_gain / average_loss
        self.data[StockFields.RSI] = 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, window: int = 20, num_std_dev: int = 2) -> None:
        """Calculate Bollinger Bands.

        Parameters
        ----------
        window : int, optional
            Moving average window, by default 20.
        num_std_dev : int, optional
            Number of standard deviations for bands, by default 2.
        """
        self.data[StockFields.MIDDLE_BAND] = (
            self.data[StockFields.CLOSE].rolling(window=window).mean()
        )
        std_dev = self.data[StockFields.CLOSE].rolling(window=window).std()
        self.data[StockFields.UPPER_BAND] = (
            self.data[StockFields.MIDDLE_BAND] + std_dev * num_std_dev
        )
        self.data[StockFields.LOWER_BAND] = (
            self.data[StockFields.MIDDLE_BAND] - std_dev * num_std_dev
        )

    def calculate_macd(self) -> None:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_12 = self.data[StockFields.CLOSE].ewm(span=12, adjust=False).mean()
        ema_26 = self.data[StockFields.CLOSE].ewm(span=26, adjust=False).mean()
        self.data[StockFields.MACD] = ema_12 - ema_26
        self.data[StockFields.SIGNAL_LINE] = (
            self.data[StockFields.MACD].ewm(span=9, adjust=False).mean()
        )
