"""
Model data factory for constructing execution data.

This module provides the ModelDataFactory class which assembles all the
data components needed for strategy execution, including loading data,
validation, and building entity objects.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.data.universes.loader import load_asset_universe
from investment_tool.data.universes.models import AssetUniverse
from investment_tool.config import AppConfig, get_config
from investment_tool.constants import (
    DataSource,
    ExecutionMode,
    Frequency,
)
from investment_tool.data.repositories.local_data_saver import LocalDataSaver
from investment_tool.data.repositories.base_repository import BaseDataRepository
from investment_tool.data.repositories.repository_factory import RepositoryFactory
from investment_tool.data.validation import (
    NewsDataValidator,
    ValidationLevel,
    ValidationPipeline,
    ValidationReport,
)
from investment_tool.data.validation.security_identity import (
    identity_history,
    restrict_identity_execution,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.domain.market_time import (
    MAX_CRYPTO_EXECUTION_DELAY,
    TIME_CONTRACT_VERSION,
    historical_window,
    validate_bar_times,
    utc_instant,
)
from investment_tool.exceptions import (
    ConfigurationError,
    DataRetrievalError,
    MissingDataError,
)

logger = logging.getLogger(__name__)


class ModelDataFactory:
    """Factory for constructing complete ModelData objects.

    Handles data loading, validation, and entity construction for both
    live investing and backtesting modes.

    Parameters
    ----------
    data_source : DataSource
        Source of financial data (ALPACA or LOCAL).
    execution_mode : ExecutionMode
        Execution mode (INVESTING or BACKTESTING).
    frequency : Frequency
        Data frequency (DAILY, WEEKLY, MONTHLY).
    transaction_cost_rate : float
        Transaction cost as fraction of trade value.
    input_data_path : Path | None, optional
        Path to local data file (required for LOCAL source).
    save_input_data : bool, optional
        Whether to save fetched data locally.
    validation_level : ValidationLevel, optional
        Validation strictness (STRICT, WARN, SKIP).
    cfg : AppConfig | None, optional
        Typed application configuration.  When ``None`` (the default),
        the lazy default via :func:`~investment_tool.config.get_config`
        is used.

    Attributes
    ----------
    data_source : DataSource
    execution_mode : ExecutionMode
    frequency : Frequency
    input_data_path : Path | None
    save_input_data : bool
    validation_level : ValidationLevel
    transaction_cost_rate : float
    model_data : ModelData
    """

    def __init__(
        self,
        data_source: DataSource,
        execution_mode: ExecutionMode,
        frequency: Frequency,
        transaction_cost_rate: float,
        input_data_path: Path | None = None,
        save_input_data: bool = False,
        validation_level: ValidationLevel = ValidationLevel.WARN,
        cfg: AppConfig | None = None,
        enable_cache: bool = False,
    ) -> None:
        if data_source == DataSource.LOCAL and input_data_path is None:
            raise ConfigurationError(
                "input_data_path must be provided when data_source is 'local'",
                details={"data_source": str(data_source)},
            )
        if data_source != DataSource.LOCAL and input_data_path is not None:
            logger.warning(
                "input_data_path will be ignored when data_source is not 'local'."
            )

        self._cfg = cfg if cfg is not None else get_config()
        self.data_source = data_source
        self.execution_mode = execution_mode
        self.frequency = frequency
        self.input_data_path = input_data_path
        self.save_input_data = save_input_data
        self.validation_level = validation_level
        self.transaction_cost_rate = transaction_cost_rate
        # One observation boundary for the entire assembly, including slow fetches.
        self._as_of = pd.Timestamp.now(tz="UTC")
        self.enable_cache = enable_cache
        self.model_data = ModelData()
        self._input_data: dict[str, Any] = {}
        self._validation_report: ValidationReport | None = None
        self._input_repository: BaseDataRepository | None = None
        self._asset_universe: AssetUniverse | None = None

    @classmethod
    def create_model_data(
        cls,
        data_source: DataSource,
        execution_mode: ExecutionMode,
        frequency: Frequency,
        input_data_path: Path | None = None,
        save_input_data: bool = False,
        validation_level: ValidationLevel = ValidationLevel.WARN,
        cfg: AppConfig | None = None,
        enable_cache: bool = False,
    ) -> ModelData:
        """Create model data with configurable parameters.

        Parameters
        ----------
        data_source : DataSource
            Source of financial data.
        execution_mode : ExecutionMode
            Execution mode (INVESTING or BACKTESTING).
        frequency : Frequency
            Data frequency.
        input_data_path : Path | None, optional
            Path for local data source.
        save_input_data : bool, optional
            Whether to save fetched data locally.
        validation_level : ValidationLevel, optional
            Validation strictness.
        cfg : AppConfig | None, optional
            Typed application configuration.  Defaults to ``get_config()``.
        enable_cache : bool, optional
            When ``True`` and a database is configured, wrap the data
            repository with :class:`CachedRepository` so that OHLCV bars
            are transparently cached in the ``ohlcv_cache`` table.

        Returns
        -------
        ModelData
            Fully constructed model data.
        """
        effective_cfg = cfg if cfg is not None else get_config()
        transaction_cost_rate = effective_cfg.model.transaction_cost_rate
        factory = cls(
            data_source,
            execution_mode,
            frequency,
            transaction_cost_rate,
            input_data_path,
            save_input_data,
            validation_level,
            cfg=effective_cfg,
            enable_cache=enable_cache,
        )
        return factory._create_model_data()

    def _create_model_data(self) -> ModelData:
        """Orchestrate building the model data with all components.

        Returns
        -------
        ModelData
            The assembled model data.

        Raises
        ------
        DataRetrievalError
            If model data creation fails.
        """
        try:
            logger.info("Loading and validating input data...")
            self._load_input_data()

            self._build_instance()
            self._build_stocks()
            self._build_periods()
            self._build_portfolio()
            self.model_data.news = {}
            report = self._validation_report or ValidationReport()
            if self.execution_mode == ExecutionMode.BACKTESTING:
                from investment_tool.data.validation.lifecycle import (
                    readiness_summary,
                    validate_lifecycle,
                )

                validate_lifecycle(
                    self.model_data, report, universe=self._get_asset_universe()
                )
                self.model_data.validation_report = report.to_dict()
                self.model_data.validation_report["readiness"] = readiness_summary(
                    self.model_data.validation_report
                )
                if not self.model_data.validation_report["readiness"]["ready"]:
                    # Return the audited dataset for reporting. App/domain run
                    # entry points enforce readiness before strategy execution.
                    # Do not spend time fetching/scoring news for a blocked run.
                    return self.model_data
            self._load_news_and_save_input()
            self._build_news()
            self.model_data.validation_report.update(report.to_dict())

            logger.info("Model data construction completed successfully")
            return self.model_data
        except Exception as e:
            logger.error(f"Model data creation failed: {e}")
            raise DataRetrievalError(
                source=str(self.data_source),
                message=f"Failed to create model data: {e}",
                original_error=e,
            ) from e

    def _get_configured_dates(self) -> tuple[str, str]:
        """Read historical date range from config.

        Returns
        -------
        tuple[str, str]
            (start_date, end_date) in YYYY-MM-DD format.
        """
        return (
            self._cfg.model.historical_start_date,
            self._cfg.model.historical_end_date,
        )

    def _get_configured_symbols(self) -> tuple[list[str], list[str]]:
        """Read the selected stock and crypto universe from its JSON manifest."""
        universe = self._get_asset_universe()
        return universe.quote_stocks, universe.crypto

    def _get_asset_universe(self) -> AssetUniverse:
        if self._asset_universe is None:
            self._asset_universe = load_asset_universe(self._cfg.asset_config_file)
        return self._asset_universe

    def _load_input_data(self) -> None:
        """Load and validate raw input data.

        Uses the RepositoryFactory directly to fetch data from the
        configured source, then validates and optionally saves locally.
        When ``enable_cache`` is ``True``, wraps the repository with
        :class:`CachedRepository` for DB-backed OHLCV caching.

        Raises
        ------
        MissingDataError
            If data structure is invalid or empty after validation.
        """
        logger.info(f"Loading data from source: {self.data_source}")

        universe = self._get_asset_universe()
        identity_options = (
            {"symbol_identity_date": universe.symbol_identity_date}
            if universe.symbol_identity_date is not None
            else {}
        )
        repository = RepositoryFactory.create(
            data_source=self.data_source,
            frequency=self.frequency,
            file_path=self.input_data_path,
            **identity_options,
        )

        # Optionally wrap with DB-backed cache
        # The current shared OHLCV cache is keyed by ticker, not security era.
        if self.enable_cache and universe.symbol_identity_date is None:
            try:
                from investment_tool.data.repositories.cached_repository import (
                    CachedRepository,
                )
                from api.database import get_session_factory

                session_factory = get_session_factory()
                repository = CachedRepository(repository, session_factory)
                logger.info("OHLCV caching enabled via CachedRepository")
            except Exception:
                logger.warning(
                    "Could not enable OHLCV caching — falling back to "
                    "direct repository access",
                    exc_info=True,
                )
        stock_symbols, crypto_symbols = self._get_configured_symbols()
        start_date, end_date = self._get_configured_dates()
        self._input_repository = repository

        result = repository.fetch_all(
            stock_symbols=stock_symbols,
            crypto_symbols=crypto_symbols,
            start_date=start_date,
            end_date=end_date,
            include_news=False,
        )

        logger.info(
            f"Data retrieved via {repository.source_name} repository: "
            f"{result.symbol_count} symbols"
        )
        self._input_data = result.to_dict()
        if result.metadata.get("time_contract_version") != TIME_CONTRACT_VERSION:
            raise ValueError(
                "Repository must return the supported bar time contract version"
            )
        # Timing must be checked before validators can forward-fill any fields.
        for key in ("stock_dict", "crypto_dict"):
            for symbol, frame in self._input_data.get(key, {}).items():
                self._input_data[key][symbol] = validate_bar_times(
                    frame, self.frequency
                )
        if self.execution_mode == ExecutionMode.BACKTESTING or (
            self.frequency == Frequency.DAILY
            and universe.minimum_return_periods is not None
        ):
            # Persist separately: these minute quotes are fills, never features.
            self._input_data["execution_dict"] = repository.fetch_execution_data(
                list(result.crypto_dict)
                if self.execution_mode == ExecutionMode.BACKTESTING
                else [],
                start_date,
                end_date,
            )

        # Run validation pipeline
        pipeline = ValidationPipeline(
            level=self.validation_level,
            point_in_time_history=True,
        )
        self._input_data, self._validation_report = pipeline.validate(self._input_data)
        if self.validation_level == ValidationLevel.SKIP:
            # Skipping diagnostics never licenses impossible prices as fills.
            from investment_tool.data.validation.bar_quality import BarQualityValidator

            self._input_data = BarQualityValidator().validate(
                self._input_data, self._validation_report
            )

        if not isinstance(self._input_data, dict) or not self._input_data:
            raise MissingDataError(
                data_type="input_data",
                message="Invalid or empty input data structure",
            )

        # Check if we have any valid data after validation
        stock_count = len(self._input_data.get("stock_dict", {}))
        crypto_count = len(self._input_data.get("crypto_dict", {}))
        if stock_count + crypto_count == 0:
            raise MissingDataError(
                data_type="symbols",
                message="No valid symbols remaining after validation. "
                "Check validation report for details.",
                details={"stock_count": stock_count, "crypto_count": crypto_count},
            )

    def _build_instance(self) -> None:
        """
        Builds an Instance with the relevant date range and interval settings.
        """
        start_date_str = self._cfg.model.historical_start_date
        end_date_str = self._cfg.model.historical_end_date
        self.model_data.instance = Instance(
            historical_start_date=self._parse_date(start_date_str),
            historical_end_date=self._parse_date(end_date_str),
            frequency=self.frequency,
            initial_trading_period=None,
            previous_trading_period=None,
            current_trading_period=None,
        )

    def _build_stocks(self) -> None:
        """Build indicators only after applying the shared complete-bar window."""
        start = pd.Timestamp(self.model_data.instance.historical_start_date)
        end = pd.Timestamp(self.model_data.instance.historical_end_date)

        self.model_data.stocks = []
        self.model_data.crypto_symbols = set(self._input_data.get("crypto_dict", {}))
        universe = self._get_asset_universe()
        if self._validation_report is None:
            self._validation_report = ValidationReport()
        for symbol, stock_data in {
            **self._input_data.get("stock_dict", {}),
            **self._input_data.get("crypto_dict", {}),
        }.items():
            if not self._validate_stock_data(stock_data):
                logger.warning(f"Skipping invalid data for stock {symbol}")
                continue

            canonical = validate_bar_times(stock_data, self.frequency)
            processed_data = historical_window(
                canonical, self.frequency, start.date(), end.date(), as_of=self._as_of
            )
            # Apply the declared lifetime to the whole bar, before any features.
            processed_data = identity_history(
                processed_data,
                symbol=symbol,
                start=universe.price_history_starts.get(symbol),
                end=universe.price_history_ends.get(symbol),
                source=universe.price_history_sources.get(symbol),
                report=self._validation_report,
            )
            processed_data = self._process_stock_data(processed_data)
            if processed_data.empty:
                logger.warning(
                    f"No data for {symbol} in [{start.date()}, {end.date()}]. Skipping."
                )
                continue

            stock = Stock(symbol=symbol, data=processed_data)
            self.model_data.stocks.append(stock)

    def _build_periods(self) -> None:
        """
        Constructs a series of Period objects from merged stock data.
        """
        merged_data = self._merge_stock_data()
        if merged_data.empty:
            raise ValueError("No valid historical data available")

        self.model_data.periods = []
        for idx_value, group in merged_data.groupby(level=0):
            # idx_value may be a tuple or a scalar depending on groupby result
            if isinstance(idx_value, tuple):
                trading_dt = pd.Timestamp(idx_value[0]).to_pydatetime()
            else:
                trading_dt = pd.Timestamp(idx_value).to_pydatetime()
            schedule = self._input_data.get("execution_dict", {}).get(
                trading_dt.date().isoformat(), {}
            )
            if schedule.get("market_closed") is True:
                if schedule.get("timestamp") or schedule.get("crypto_prices"):
                    raise ValueError("A closed execution session cannot contain fills")
                stock_symbols = list(
                    set(self._input_data.get("stock_dict", {}))
                    & set(group["open"].columns)
                )
                if (
                    stock_symbols
                    and (group["open"].iloc[0][stock_symbols].dropna() > 0).any()
                ):
                    raise ValueError(
                        f"Closed session contradicts observed stock opening data for {trading_dt.date()}"
                    )
                # Keep original bars for observable history. Only the trading
                # calendar follows the repository's explicit closed sessions.
                continue
            self.model_data.periods.append(
                Period(
                    trading_date=trading_dt,
                    market_data=group,
                    period_end_date=(
                        pd.Timestamp(group["period_end"].iloc[0].dropna().iloc[0])
                        - pd.Timedelta(days=1)
                    ).date(),
                    decision_time=max(
                        pd.Timestamp(trading_dt).tz_localize(zone).tz_convert("UTC")
                        for zone in group["market_timezone"].iloc[0].dropna().unique()
                    ).to_pydatetime(),
                )
            )
        if self.execution_mode == ExecutionMode.INVESTING:
            self._add_current_period(merged_data)
        elif self.model_data.periods:
            actual = {
                p.trading_date.date().isoformat() for p in self.model_data.periods
            }
            first, last = min(actual), max(actual)
            missing = sorted(
                key
                for key, quote in self._input_data.get("execution_dict", {}).items()
                if first <= key <= last
                and quote.get("market_closed") is not True
                and key not in actual
            )
            if missing:
                raise ValueError(
                    f"Missing bar data for scheduled trading periods: {missing}"
                )
        self._set_trading_periods()
        self._prepare_universe_membership()
        if self.execution_mode == ExecutionMode.BACKTESTING:
            self._prepare_execution_quotes()

    def _prepare_universe_membership(self) -> None:
        """Apply explicit history eligibility without implying dated provenance."""
        universe = self._get_asset_universe()
        self.model_data.cash_settlements = universe.cash_settlements
        self.model_data.security_actions = universe.security_actions
        membership = universe.membership
        minimum = universe.minimum_return_periods
        if (
            minimum is None
            and not universe.price_history_ends
            and not universe.cash_settlements
            and not universe.security_actions
            and not universe.valuation_symbols
        ):
            return
        if minimum is not None and self.frequency == Frequency.DAILY:
            self._prepare_history_calendars()
        current = self.model_data.instance.current_trading_period
        if current is None:
            raise ValueError("No current period for dated universe")
        first = self.model_data.periods.index(current)
        loaded = {s.symbol for s in self.model_data.stocks}
        for period in self.model_data.periods[first:]:
            if period.decision_time is None:
                raise ValueError("Dated universe requires an explicit decision cutoff")
            stocks = (
                membership.symbols_at(period.decision_time)
                if membership is not None
                else frozenset(universe.stocks)
            )
            symbols = stocks | frozenset(universe.crypto)
            # An ended identity remains in earlier decisions; it cannot re-enter
            # later under an unrelated issuer sharing its ticker. Held positions
            # still require real execution/settlement data rather than vanish.
            symbols = frozenset(
                symbol
                for symbol in symbols
                if symbol not in universe.price_history_ends
                or period.trading_date.date() < universe.price_history_ends[symbol]
            )
            missing = symbols - loaded
            if missing:
                from investment_tool.data.validation import (
                    IssueSeverity,
                    ValidationIssue,
                )

                if self._validation_report is not None:
                    self._validation_report.add_issue(
                        ValidationIssue(
                            IssueSeverity.WARNING,
                            "UniverseAvailability",
                            "Configured candidates without a usable series cannot be purchased",
                            action="exclude_unavailable_candidates",
                            details={
                                "period": period.trading_date.isoformat(),
                                "symbols": sorted(missing),
                            },
                        )
                    )
            retired = {
                event.symbol
                for event in (*universe.cash_settlements, *universe.security_actions)
                if not event.retain_original
                and max(event.effective_at, event.published_at) <= period.decision_time
            }
            period.eligible_symbols = symbols - retired - missing
            period.minimum_return_periods = minimum

    def _prepare_history_calendars(self) -> None:
        """Use repository sessions for equities and calendar days for crypto."""
        instance = self.model_data.instance
        days = pd.date_range(
            instance.historical_start_date, instance.historical_end_date, freq="D"
        )
        schedule = self._input_data.get("execution_dict", {})
        missing = [
            day.date().isoformat()
            for day in days
            if day.date().isoformat() not in schedule
        ]
        if missing:
            raise ValueError(
                "Daily dated-universe history requires an explicit session calendar; "
                "fetch or enrich the dataset with execution schedules "
                f"(first missing date: {missing[0]})"
            )
        for day in days:
            quote = schedule[day.date().isoformat()]
            if quote.get("market_closed") is not True:
                if not quote.get("timestamp"):
                    raise ValueError("An open history session requires a timestamp")
                utc_instant(quote["timestamp"])
        sessions = pd.DatetimeIndex(
            [
                day
                for day in days
                if schedule[day.date().isoformat()].get("market_closed") is not True
            ]
        )
        instance.history_calendars = {
            stock.symbol: days
            if stock.symbol in self.model_data.crypto_symbols
            else sessions
            for stock in self.model_data.stocks
        }
        # Calendar coverage includes explicitly closed days beyond the last
        # trading session, so weekends/holidays cannot hide a stale schedule.
        instance.history_calendar_end = instance.historical_end_date

    def _prepare_execution_quotes(self) -> None:
        """Validate timed fills and security lifetimes before news or solvers."""
        current = self.model_data.instance.current_trading_period
        if current is None:
            raise ValueError("No current period for execution quotes")
        first = self.model_data.periods.index(current)
        execution_dict = self._input_data.get("execution_dict", {})
        universe = self._get_asset_universe()
        for period in self.model_data.periods[first:]:
            key = period.trading_date.date().isoformat()
            quote = execution_dict.get(key)
            if self.frequency != Frequency.MONTHLY and not quote:
                raise ValueError(
                    f"Missing synchronized execution schedule for {key}. "
                    "Fetch Alpaca data or enrich the local dataset."
                )
            if quote:
                if "timestamp" not in quote:
                    raise ValueError(f"Missing execution timestamp for {key}")
                timestamp = utc_instant(quote["timestamp"])
                if timestamp >= pd.Timestamp(
                    period.market_data["bar_end"].iloc[0].dropna().min()
                ):
                    raise ValueError(f"Invalid execution timestamp for {key}")
                period.execution_time = timestamp.to_pydatetime()
                if period.decision_time is not None and timestamp <= pd.Timestamp(
                    period.decision_time
                ):
                    raise ValueError(
                        f"Execution must follow the decision cutoff for {key}"
                    )
                period.crypto_execution_prices = dict(quote["crypto_prices"])
                actual_times = quote.get("crypto_timestamps")
                for symbol in period.crypto_execution_prices:
                    if actual_times is not None and symbol not in actual_times:
                        raise ValueError(
                            f"Missing actual crypto execution timestamp for {symbol} in {key}"
                        )
                    # Existing datasets used the exact scheduled open.
                    actual = (
                        timestamp
                        if actual_times is None
                        else utc_instant(actual_times[symbol])
                    )
                    if not (
                        timestamp <= actual <= timestamp + MAX_CRYPTO_EXECUTION_DELAY
                    ):
                        raise ValueError(
                            f"Crypto execution for {symbol} in {key} must occur "
                            "at the scheduled open or within two minutes after it"
                        )
                    if actual >= pd.Timestamp(
                        period.market_data["bar_end"].iloc[0].dropna().min()
                    ):
                        raise ValueError(
                            f"Invalid crypto execution timestamp for {key}"
                        )
                    period.crypto_execution_times[symbol] = actual.to_pydatetime()
            # Only require quotes for crypto assets observed in this period.
            # Missing pre-listing bars do not make an asset eligible for trading.
            required = {
                stock.symbol
                for stock in self.model_data.stocks
                if stock.symbol in self.model_data.crypto_symbols
                and period.trading_date in stock.data.index
            }
            missing = {
                symbol
                for symbol in required
                if symbol not in period.crypto_execution_prices
                or not np.isfinite(period.crypto_execution_prices[symbol])
                or period.crypto_execution_prices[symbol] <= 0
            }
            if missing:
                raise ValueError(
                    f"Missing synchronized crypto execution quotes for {key}: {sorted(missing)}. "
                    "A valid crypto fill is required at the stock-session open or within two minutes after it. "
                    "Calendar-bar midnight opens and stale prices cannot substitute for execution quotes. "
                    "Fetch a new Alpaca dataset or enrich the local file with "
                    "python -m investment_tool.data.enrich_execution_data INPUT.pkl OUTPUT.pkl. "
                    "Synchronized execution supports all configured frequencies."
                )
            if self._validation_report is not None:
                restrict_identity_execution(period, universe, self._validation_report)

    def _load_news_and_save_input(self) -> None:
        """Fetch news only after price/execution preflight; persist reusable input."""
        if self._input_repository is None:
            raise ValueError("No input repository available for news")
        stock_symbols = self._get_asset_universe().stocks
        start_date, end_date = self._get_configured_dates()
        self._input_data["news_dict"] = self._input_repository.fetch_news_data(
            stock_symbols, start_date, end_date
        )
        self._input_data, news_report = ValidationPipeline(
            validators=[NewsDataValidator(check_future_dates=True)],
            level=self.validation_level,
        ).validate(self._input_data)
        if self._validation_report is not None:
            self._validation_report.issues.extend(news_report.issues)
        if self.save_input_data:
            try:
                self.model_data.input_snapshot_path = LocalDataSaver.save_data(
                    self._input_data,
                    self.frequency.value,
                    save_format="pickle",
                    cfg=self._cfg,
                )
            except ValueError as e:
                logger.error(f"Failed to save input data locally: {e}")

    def _build_news(self) -> None:
        """Build news data filtered to the configured date range.

        Only news whose ``period_start_date`` falls within
        ``[historical_start_date, historical_end_date]`` is kept.
        """
        hist_start = self.model_data.instance.historical_start_date
        hist_end = self.model_data.instance.historical_end_date

        news_dict = {}
        skipped = 0
        for stock in self.model_data.stocks:
            if stock.symbol in self._input_data["news_dict"]:
                news_items = self._input_data["news_dict"].get(stock.symbol, {})
                if news_items:
                    for period_dates_str, news_data in news_items.items():
                        period_start_date = self._parse_date(period_dates_str[0])
                        period_end_date = self._parse_date(period_dates_str[1])

                        # Skip news outside the configured date window
                        if (
                            period_start_date < hist_start
                            or period_start_date > hist_end
                        ):
                            skipped += 1
                            continue

                        news_item = News(
                            symbol=stock.symbol,
                            period_start_date=period_start_date,
                            period_end_date=period_end_date,
                            data=news_data,
                        )
                        key = (stock.symbol, period_start_date)
                        news_dict[key] = news_item
                else:
                    logger.warning(
                        f"No news data available for stock {stock.symbol}. Skipping."
                    )

        if skipped:
            logger.info(
                f"Filtered out {skipped} news items outside "
                f"[{hist_start}, {hist_end}]"
            )
        self.model_data.news = news_dict

    def _build_portfolio(self) -> None:
        """
        Initializes the Portfolio object with the constructed model data.
        """
        if not self.model_data.periods:
            raise ValueError("Cannot build portfolio - no historical periods available")
        initial_cash = self._cfg.model.initial_cash
        self.model_data.portfolio = Portfolio(
            instance=self.model_data.instance,
            initial_cash=initial_cash,
            transaction_cost_rate=self.transaction_cost_rate,
        )

    def _parse_date(self, date_str: str) -> date:
        """
        Converts a date string to a date object.

        Parameters
        ----------
        date_str : str
            The date string in YYYY-MM-DD format.

        Returns
        -------
        date
            The parsed date object.

        Raises
        ------
        ValueError
            If the date string is in an invalid format.
        """
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError as exc:
            logger.error(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")
            raise exc

    def _validate_stock_data(self, data: pd.DataFrame) -> bool:
        """
        Checks basic validity of stock data.

        Note: Detailed validation (missing data, time series consistency, etc.)
        is performed by the ValidationPipeline in _load_input_data().
        This method provides a final sanity check.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing stock price data.

        Returns
        -------
        bool
            True if the data is valid, False otherwise.
        """
        # Basic sanity check - detailed validation done by pipeline
        return isinstance(data, pd.DataFrame) and not data.empty

    def _merge_stock_data(self) -> pd.DataFrame:
        """
        Merges data from all stocks into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            The merged DataFrame (indexed by date), retaining missing quotes.
        """
        data_frames: list[pd.DataFrame] = []
        for stock in self.model_data.stocks:
            stock_data = stock.data.copy()
            stock_data.columns = pd.MultiIndex.from_product(
                [stock_data.columns, [stock.symbol]]
            )
            data_frames.append(stock_data)

        if not data_frames:
            return pd.DataFrame()

        merged = pd.concat(data_frames, axis=1)
        merged.sort_index(inplace=True)
        # A missing period's OHLC values must not become stale execution prices.
        result = merged.dropna(how="all")
        return pd.DataFrame(result)

    def _add_current_period(self, historical_data: pd.DataFrame) -> None:
        """
        Appends a Period representing today's data based on the last known historical date.

        Parameters
        ----------
        historical_data : pd.DataFrame
            The merged historical DataFrame used as a reference.
        """
        decision = self._as_of
        current_date = decision.tz_localize(None).normalize().to_pydatetime()
        last_historical = historical_data.iloc[-1:].copy()
        last_historical.index = pd.DatetimeIndex([current_date])
        self.model_data.periods.append(
            Period(
                trading_date=current_date,
                market_data=last_historical,
                decision_time=decision.to_pydatetime(),
            )
        )

    def _set_trading_periods(self) -> None:
        """
        Configures the trading periods according to the current execution mode.

        Raises
        ------
        ValueError
            If there are no trading periods for the chosen mode.
        """
        periods = self.model_data.periods
        if self.execution_mode == ExecutionMode.INVESTING:
            if len(periods) >= 2:
                self.model_data.instance.initial_trading_period = periods[-1]
                self.model_data.instance.previous_trading_period = periods[-2]
                self.model_data.instance.current_trading_period = periods[-1]
            elif periods:
                self.model_data.instance.initial_trading_period = periods[0]
                self.model_data.instance.previous_trading_period = periods[0]
                self.model_data.instance.current_trading_period = periods[0]
            else:
                raise ValueError("No trading periods available for investment mode.")
        elif self.execution_mode == ExecutionMode.BACKTESTING:
            total_periods = len(periods)
            explicit_start = self._cfg.model.backtest_start_date
            if explicit_start is None:
                start_idx = self._calculate_backtest_start(total_periods)
            else:
                matches = [
                    i
                    for i, period in enumerate(periods)
                    if period.trading_date.date() == self._parse_date(explicit_start)
                ]
                if not matches:
                    raise ValueError(
                        "Backtest start must match an available period boundary"
                    )
                start_idx = matches[0]
                if start_idx < self._cfg.model.min_backtest_periods:
                    raise ValueError(
                        "Insufficient warm-up before explicit backtest start"
                    )

            if start_idx <= 0:
                self.model_data.instance.initial_trading_period = periods[0]
                self.model_data.instance.previous_trading_period = periods[0]
                self.model_data.instance.current_trading_period = periods[0]
            else:
                self.model_data.instance.initial_trading_period = periods[start_idx]
                self.model_data.instance.previous_trading_period = periods[
                    start_idx - 1
                ]
                self.model_data.instance.current_trading_period = periods[start_idx]
        else:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

    def _calculate_backtest_start(self, total_periods: int) -> int:
        """
        Determines a valid backtest start index within the available range.

        Parameters
        ----------
        total_periods : int
            The total number of available trading periods.

        Returns
        -------
        int
            The computed start index for backtesting.

        Raises
        ------
        ValueError
            If there is insufficient historical data.
        """
        min_periods = self._cfg.model.min_backtest_periods
        if total_periods < min_periods:
            raise ValueError(
                f"Insufficient historical data ({total_periods} < {min_periods})."
            )
        percentage = self._cfg.model.start_index_percentage_backtesting
        start_index = int(total_periods * percentage)
        start_index = max(start_index, min_periods)
        return int(min(start_index, total_periods - 1))

    @staticmethod
    def _process_stock_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans, normalizes, and sorts stock price data.

        Parameters
        ----------
        data : pd.DataFrame
            The stock data.

        Returns
        -------
        pd.DataFrame
            The cleaned DataFrame indexed by its canonical calendar label.
        """
        # Preserve quarantined calendar rows: deleting a missing month would
        # turn the next price change into a fake one-month return/indicator.
        result = data.copy().sort_index()
        return pd.DataFrame(result)
