"""
Private Alpaca API client for the Alpaca repository.

This module provides the AlpacaDataFetcher class — the low-level HTTP
client that talks to the Alpaca REST API.  It is an **implementation
detail** of :class:`AlpacaRepository` and should not be imported
directly by other packages.
"""

import time
import logging
from typing import Sequence, Any, Callable
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment
from alpaca.data.historical import (
    CryptoHistoricalDataClient,
    NewsClient,
    StockHistoricalDataClient,
)
from alpaca.data.requests import CryptoBarsRequest, NewsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from investment_tool.config import AlpacaConfig, get_config
from investment_tool.defaults import INFRA
from investment_tool.domain.market_time import (
    MAX_CRYPTO_EXECUTION_DELAY,
    normalize_bars,
    period_code,
)
from investment_tool.constants import Frequency
from investment_tool.data.repositories.base_repository import ExecutionQuote

logger = logging.getLogger(__name__)


class AlpacaDataFetcher:
    """
    Class for fetching data from the Alpaca API.

    Attributes
    ----------
    frequency_map : dict
        Maps Frequency enum values to Alpaca TimeFrame values.
    stock_client, crypto_client, news_client, calendar_client
        Supported Alpaca SDK clients for each read-only data endpoint.
    """

    # Mapping between our frequency enum and Alpaca's TimeFrame
    frequency_map = {
        Frequency.DAILY: TimeFrame.Day,
        Frequency.WEEKLY: TimeFrame.Week,
        Frequency.MONTHLY: TimeFrame.Month,
    }

    def __init__(
        self,
        rate_limit_sleep: float = INFRA.rate_limit_sleep,
        alpaca_config: AlpacaConfig | None = None,
    ) -> None:
        """
        Initializes the AlpacaDataFetcher with API credentials.

        Parameters
        ----------
        rate_limit_sleep : float, optional
            Time to sleep between API calls to avoid rate limiting, by default 0.01 seconds.
        alpaca_config : AlpacaConfig | None, optional
            Typed Alpaca credentials.  When ``None`` (the default), falls back
            to the default :func:`~investment_tool.config.get_config` result.

        Raises
        ------
        ValueError
            If API credentials are not found.
        """
        cfg = alpaca_config if alpaca_config is not None else get_config().alpaca
        api_key = cfg.api_key
        api_secret = cfg.api_secret

        if not api_key or not api_secret:
            raise ValueError(
                "API keys not found. Please set ALPACA_API_KEY and ALPACA_API_SECRET in your environment variables."
            )

        self.stock_client = StockHistoricalDataClient(api_key, api_secret)
        self.crypto_client = CryptoHistoricalDataClient(api_key, api_secret)
        self.news_client = NewsClient(api_key, api_secret)
        # The SDK appends /v2; the configured trading host is used only for calendar.
        self.calendar_client = TradingClient(
            api_key,
            api_secret,
            url_override=cfg.base_url.rstrip("/").removesuffix("/v2"),
        )
        self.rate_limit_sleep = rate_limit_sleep

    def get_execution_data(
        self,
        crypto_symbols: list[str],
        start_date: str,
        end_date: str,
        frequency: Frequency = Frequency.MONTHLY,
    ) -> dict[str, ExecutionQuote]:
        """Get the first valid crypto minute open within two minutes of stock open.

        Calendar dates/open times are New York local time. No midnight crypto
        open or last-known quote is used as a fallback. Actual fill times are
        returned separately from the scheduled stock opening time.
        Missing quotes remain missing for the backtest preflight to reject.
        """
        code = period_code(frequency)
        start = pd.Timestamp(start_date).to_period(code).start_time
        end = pd.Timestamp(end_date).to_period(code).end_time
        sessions = self._handle_api_request(
            self.calendar_client.get_calendar,
            filters=GetCalendarRequest(start=start.date(), end=end.date()),
        )
        result: dict[str, ExecutionQuote] = {}
        for session in sorted(sessions, key=lambda item: item.date):
            period = pd.Timestamp(session.date).to_period(code).start_time
            key = period.date().isoformat()
            if key in result:
                continue
            opening = pd.Timestamp(
                datetime.combine(session.date, session.open.time()),
                tz=ZoneInfo("America/New_York"),
            ).tz_convert("UTC")
            prices: dict[str, float] = {}
            timestamps: dict[str, str] = {}
            if crypto_symbols:
                bars = self._handle_api_request(
                    self.crypto_client.get_crypto_bars,
                    request_params=CryptoBarsRequest(
                        symbol_or_symbols=crypto_symbols,
                        timeframe=TimeFrame.Minute,
                        start=opening.to_pydatetime(),
                        end=(opening + MAX_CRYPTO_EXECUTION_DELAY).to_pydatetime(),
                    ),
                ).df
                if not bars.empty:
                    # The SDK returns a (symbol, timestamp) MultiIndex.
                    # Also accept timestamp-indexed fixtures with a symbol column.
                    if "symbol" in bars.columns:
                        bars = bars.set_index("symbol", append=True)
                    for symbol in crypto_symbols:
                        if symbol not in bars.index.get_level_values("symbol"):
                            continue
                        quotes = bars.xs(symbol, level="symbol")
                        quotes.index = pd.to_datetime(quotes.index, utc=True)
                        quotes = quotes.loc[
                            (quotes.index >= opening)
                            & (quotes.index <= opening + MAX_CRYPTO_EXECUTION_DELAY)
                        ].sort_index()
                        for timestamp, row in quotes.iterrows():
                            price = float(row["open"])
                            if not np.isfinite(price) or price <= 0:
                                continue
                            prices[symbol] = price
                            timestamps[symbol] = pd.Timestamp(timestamp).isoformat()
                            break
            result[key] = {
                "timestamp": opening.isoformat(),
                "crypto_prices": prices,
                "crypto_timestamps": timestamps,
            }
            missing = sorted(set(crypto_symbols) - prices.keys())
            if missing:
                logger.warning(
                    "Missing crypto execution quotes within two minutes for %s: %s",
                    key,
                    missing,
                )
            else:
                logger.info("Loaded %s execution quotes for %s", frequency, key)
        if frequency == Frequency.DAILY:
            for day in pd.date_range(start, end.normalize(), freq="D"):
                result.setdefault(
                    day.date().isoformat(),
                    {"market_closed": True, "crypto_prices": {}},
                )
        return result

    def _validate_frequency(self, frequency: Frequency) -> None:
        """
        Validates if the given frequency is supported.

        Parameters
        ----------
        frequency : Frequency
            The frequency to validate.

        Raises
        ------
        ValueError
            If the frequency is not supported.
        """
        if frequency not in self.frequency_map:
            raise ValueError(
                f"Unsupported frequency '{frequency}'. "
                f"Supported frequencies are: {list(self.frequency_map.keys())}"
            )

    def _handle_api_request(self, api_call: Callable[..., Any], **kwargs: Any) -> Any:
        """
        Generic method to handle API calls with error handling and rate limiting.

        Parameters
        ----------
        api_call : Callable
            The API method to call.
        **kwargs : Any
            Arguments to pass to the API call.

        Returns
        -------
        Any
            The result of the API call.

        Raises
        ------
        Exception
            Re-raises any exception from the API call.
        """
        try:
            result = api_call(**kwargs)
            # Throttle API calls
            time.sleep(self.rate_limit_sleep)
            return result
        except Exception as e:
            symbol = kwargs.get("symbol", "Unknown")
            logger.error(f"API error for {symbol}: {str(e)}")
            raise

    def _get_historical_data_generic(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        frequency: Frequency,
        data_type: str = "stock",
        symbol_identity_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generic method to retrieve historical data from Alpaca API.

        Parameters
        ----------
        symbols : Sequence[str]
            Asset symbols.
        start_date : str
            Start date in ISO format.
        end_date : str
            End date in ISO format.
        frequency : Frequency
            Data frequency.
        data_type : str
            Type of data to retrieve ('stock' or 'crypto').

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping each symbol to its DataFrame of historical data.
        """
        self._validate_frequency(frequency)

        timeframe = self.frequency_map[frequency]
        all_data: dict[str, pd.DataFrame] = {}
        as_of = pd.Timestamp.now(tz="UTC")
        market_timezone = "UTC" if data_type == "crypto" else "America/New_York"
        coverage_start = pd.Timestamp(start_date).tz_localize(market_timezone)
        coverage_end = (pd.Timestamp(end_date) + pd.DateOffset(days=1)).tz_localize(
            market_timezone
        )

        for symbol in symbols:
            try:
                request_options: dict[str, Any] = {
                    "symbol_or_symbols": symbol,
                    "timeframe": timeframe,
                    "start": coverage_start.to_pydatetime(),
                    "end": coverage_end.to_pydatetime(),
                }
                if data_type == "crypto":
                    bars = self._handle_api_request(
                        self.crypto_client.get_crypto_bars,
                        request_params=CryptoBarsRequest(**request_options),
                    ).df
                else:
                    bars = self._handle_api_request(
                        self.stock_client.get_stock_bars,
                        request_params=StockBarsRequest(
                            **request_options,
                            adjustment=Adjustment.SPLIT,
                            asof=symbol_identity_date.isoformat()
                            if symbol_identity_date is not None
                            else None,
                        ),
                    ).df
                # Keep the repository's per-symbol timestamp-indexed cache format.
                if not bars.empty:
                    bars = bars.xs(symbol, level="symbol")

                all_data[symbol] = normalize_bars(
                    bars,
                    frequency,
                    market_timezone=market_timezone,
                    coverage_start=coverage_start,
                    coverage_end=coverage_end,
                    as_of=as_of,
                )

                logger.debug(f"Successfully fetched {data_type} data for {symbol}")

            except ValueError as e:
                raise ValueError(
                    f"Invalid {data_type} bar timing for {symbol}: {e}"
                ) from e
            except Exception as e:
                logger.error(f"Error fetching {data_type} data for {symbol}: {e}")
                all_data[symbol] = pd.DataFrame()  # Empty DataFrame for failed requests

        return all_data

    def get_historical_data(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        frequency: Frequency,
        symbol_identity_date: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Retrieve historical stock data from the Alpaca API.

        Parameters
        ----------
        symbols : Sequence[str]
            Stock symbols.
        start_date : str
            Start date in ISO format.
        end_date : str
            End date in ISO format.
        frequency : Frequency
            Data frequency (e.g., HOURLY, DAILY, WEEKLY, MONTHLY).

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping each symbol to its DataFrame of historical data.
        """
        return self._get_historical_data_generic(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            data_type="stock",
            symbol_identity_date=symbol_identity_date,
        )

    def get_historical_crypto_data(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        frequency: Frequency,
    ) -> dict[str, pd.DataFrame]:
        """
        Retrieve historical cryptocurrency data from the Alpaca API.

        Parameters
        ----------
        symbols : Sequence[str]
            Cryptocurrency symbols.
        start_date : str
            Start date in ISO format.
        end_date : str
            End date in ISO format.
        frequency : Frequency
            Data frequency.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping each symbol to its DataFrame of historical cryptocurrency data.
        """
        return self._get_historical_data_generic(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            data_type="crypto",
        )

    # TODO: Review dates handling in get_news_data
    def get_news_data(
        self,
        symbols: Sequence[str],
        start_date: str,
        end_date: str,
        frequency: Frequency,
        sort: str = "desc",
        limit: int = INFRA.news_limit,
    ) -> dict[str, dict[tuple[str, str], list[dict[str, Any]]]]:
        """
        Fetch news articles for specified stocks using the Alpaca News API, aggregated by period.

        Parameters
        ----------
        symbols : Sequence[str]
            List of stock symbols.
        start_date : str
            Start date in "YYYY-MM-DD" format.
        end_date : str
            End date in "YYYY-MM-DD" format.
        frequency : Frequency
            Frequency of data retrieval (e.g., DAILY, WEEKLY, MONTHLY).
        sort : str, optional
            Sort order ("asc" or "desc"), by default "desc".
        limit : int, optional
            Maximum number of news articles to fetch per symbol per period, by default 50.

        Returns
        -------
        dict[str, dict[tuple[str, str], list[dict[str, Any]]]]
            Dictionary mapping each symbol to a dictionary that maps period start/end (YYYY-MM-DD, YYYY-MM-DD)
            (formatted as "YYYY-MM-DD") to lists of news article dictionaries.

        Raises
        ------
        ValueError
            If the date format is invalid or frequency is not supported.
        """
        try:
            current_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            overall_end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                "start_date and end_date must be in 'YYYY-MM-DD' format"
            ) from exc

        # Check if the frequency is supported
        if frequency not in [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY]:
            raise ValueError(
                f"Unsupported frequency '{frequency}' for news data. "
                f"Supported frequencies are: DAILY, WEEKLY, MONTHLY"
            )

        news_by_symbol_date: dict[str, dict[tuple[str, str], list[dict[str, Any]]]] = {}

        while current_date <= overall_end:
            period_end = self._get_period_end(current_date, frequency, overall_end)
            period_key: tuple[str, str] = (
                current_date.strftime("%Y-%m-%d"),
                period_end.strftime("%Y-%m-%d"),
            )

            for symbol in symbols:
                try:
                    period_news = self._handle_api_request(
                        self.news_client.get_news,
                        request_params=NewsRequest(
                            symbols=symbol,
                            start=datetime.combine(current_date, datetime.min.time()),
                            end=datetime.combine(period_end, datetime.min.time()),
                            sort=sort,
                            limit=limit,
                            include_content=False,
                            exclude_contentless=False,
                        ),
                    ).data["news"]

                    logger.info(
                        f"Fetched {len(period_news)} news items for {symbol} "
                        f"from {current_date} to {period_end}"
                    )

                    # Process and format news items
                    news_for_symbol_period: list[dict[str, Any]] = []
                    for item in period_news:
                        updated_at = getattr(item, "updated_at", None)
                        news_item = {
                            "id": getattr(item, "id", None),
                            "symbol": symbol,
                            "headline": item.headline,
                            "summary": item.summary,
                            "author": item.author,
                            "published_at": item.created_at.isoformat(),
                            "updated_at": updated_at.isoformat()
                            if updated_at
                            else None,
                            "source": item.source,
                            "content": item.content,
                            "url": item.url,
                            "images": [
                                image.model_dump(mode="json") for image in item.images
                            ]
                            if item.images is not None
                            else None,
                        }
                        news_for_symbol_period.append(news_item)

                    if symbol not in news_by_symbol_date:
                        news_by_symbol_date[symbol] = {}
                    news_by_symbol_date[symbol][period_key] = news_for_symbol_period

                except Exception as e:
                    logger.error(
                        f"Error fetching news for {symbol} during {period_key}: {e}"
                    )
                    if symbol not in news_by_symbol_date:
                        news_by_symbol_date[symbol] = {}
                    news_by_symbol_date[symbol][period_key] = []

            current_date = period_end

        return dict(news_by_symbol_date)

    @staticmethod
    def _get_period_end(
        start_dt: date, frequency: Frequency, overall_end: date
    ) -> date:
        """
        Calculate the end date for a given period based on the frequency.

        Parameters
        ----------
        start_dt : date
            Period start date.
        frequency : Frequency
            Data frequency.
        overall_end : date
            Overall end date limit.

        Returns
        -------
        date
            Period end date, not exceeding the overall end date.

        Raises
        ------
        ValueError
            If the frequency is not supported.
        """
        period_end = pd.Timestamp(start_dt).to_period(
            period_code(frequency)
        ).end_time.date() + timedelta(days=1)
        return min(period_end, overall_end + timedelta(days=1))
