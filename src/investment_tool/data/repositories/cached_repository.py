"""Database-backed caching repository (decorator pattern).

``CachedRepository`` wraps any :class:`BaseDataRepository` and
transparently caches OHLCV bar data in the ``ohlcv_cache`` database
table.  On each fetch it:

1. Queries the DB for cached rows matching (symbol, frequency, source,
   date range).
2. If validated timing metadata proves coverage of the requested window,
   returns from cache (fast path).
3. Otherwise refetches the whole window for affected symbols and caches only
   completed, available bars with explicit coverage metadata.

News data is **not** cached — it is always fetched fresh from the inner
repository.

Usage
-----
::

    inner = AlpacaRepository(Frequency.DAILY)
    cached = CachedRepository(inner, session_factory)
    result = cached.fetch_all(stocks, cryptos, start, end)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import IntegrityError

from investment_tool.domain.market_time import (
    TIME_COLUMNS,
    TIME_CONTRACT_VERSION,
    validate_bar_times,
)

from investment_tool.data.repositories.base_repository import (
    BaseDataRepository,
    ExecutionQuote,
)

logger = logging.getLogger(__name__)

# Columns stored in the cache (order matters — must match DataFrame columns).
_OHLCV_COLS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]


class CachedRepository(BaseDataRepository):
    """Caching decorator for any :class:`BaseDataRepository`.

    Parameters
    ----------
    inner : BaseDataRepository
        The upstream repository to delegate cache-misses to.
    session_factory : sessionmaker
        SQLAlchemy session factory for DB access.
    """

    def __init__(
        self,
        inner: BaseDataRepository,
        session_factory: sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        super().__init__(inner.frequency, inner.symbol_identity_date)
        self._inner = inner
        self._session_factory = session_factory

    @property
    def source_name(self) -> str:
        return f"Cached({self._inner.source_name})"

    # ------------------------------------------------------------------
    # Public API (overrides BaseDataRepository)
    # ------------------------------------------------------------------

    def fetch_stock_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        if self.symbol_identity_date is not None:
            # Ticker-only cache entries cannot distinguish reused symbols.
            prices: dict[str, pd.DataFrame] = self._inner.fetch_stock_data(
                symbols, start_date, end_date
            )
            return prices
        return self._fetch_with_cache(symbols, start_date, end_date, asset_type="stock")

    def fetch_crypto_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        return self._fetch_with_cache(
            symbols, start_date, end_date, asset_type="crypto"
        )

    def fetch_news_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[tuple[str, date], Any]:
        # News is not cached — always delegate.
        return self._inner.fetch_news_data(symbols, start_date, end_date)  # type: ignore[no-any-return]

    def fetch_execution_data(
        self, crypto_symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, ExecutionQuote]:
        # Minute execution quotes must not pollute the monthly OHLCV cache.
        quotes: dict[str, ExecutionQuote] = self._inner.fetch_execution_data(
            crypto_symbols, start_date, end_date
        )
        return quotes

    # ------------------------------------------------------------------
    # Cache logic
    # ------------------------------------------------------------------

    def _fetch_with_cache(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        *,
        asset_type: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data, using the DB cache where possible."""
        if not symbols:
            return {}

        db: Session = self._session_factory()
        try:
            cached = self._read_cache(db, symbols, start_date, end_date)

            # Determine which symbols have no cached data at all
            missing_symbols = [s for s in symbols if s not in cached]

            if not missing_symbols:
                logger.info("Cache hit for all %d %s symbols", len(symbols), asset_type)
                return cached

            # Fetch missing symbols from upstream
            logger.info(
                "Cache miss for %d/%d %s symbols — fetching from %s",
                len(missing_symbols),
                len(symbols),
                asset_type,
                self._inner.source_name,
            )
            if asset_type == "stock":
                fresh = self._inner.fetch_stock_data(
                    missing_symbols, start_date, end_date
                )
            else:
                fresh = self._inner.fetch_crypto_data(
                    missing_symbols, start_date, end_date
                )

            # Write fresh data to cache
            self._write_cache(db, fresh, start_date, end_date)

            # Merge cached + fresh
            merged = {**cached, **fresh}
            return merged
        finally:
            db.close()

    def _read_cache(
        self,
        db: Session,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Read only versioned bars with a proven, complete fetched range."""
        from api.models import OHLCVCacheRow

        rows = (
            db.query(OHLCVCacheRow)
            .filter(
                OHLCVCacheRow.symbol.in_(symbols),
                OHLCVCacheRow.frequency == str(self.frequency),
                OHLCVCacheRow.source == self._inner.source_name,
                OHLCVCacheRow.bar_date >= date.fromisoformat(start_date),
                OHLCVCacheRow.bar_date <= date.fromisoformat(end_date),
                OHLCVCacheRow.timing.is_not(None),
            )
            .order_by(OHLCVCacheRow.symbol, OHLCVCacheRow.bar_date)
            .all()
        )
        grouped: dict[str, list[Any]] = {}
        for row in rows:
            grouped.setdefault(row.symbol, []).append(row)
        result: dict[str, pd.DataFrame] = {}
        for symbol, symbol_rows in grouped.items():
            try:
                if not any(
                    isinstance(r.timing, dict)
                    and r.timing.get("version") == TIME_CONTRACT_VERSION
                    and r.timing.get("coverage_start", "9999") <= start_date
                    and r.timing.get("coverage_end", "0000") >= end_date
                    for r in symbol_rows
                ):
                    continue
                records = [
                    {
                        "date": r.bar_date,
                        **{c: getattr(r, c) for c in _OHLCV_COLS},
                        **{c: r.timing[c] for c in TIME_COLUMNS},
                    }
                    for r in symbol_rows
                    if isinstance(r.timing, dict)
                    and r.timing.get("version") == TIME_CONTRACT_VERSION
                ]
                frame = pd.DataFrame(records).set_index("date")
                result[symbol] = validate_bar_times(frame, self.frequency)
            except (ValueError, KeyError, TypeError):
                logger.warning(
                    "Invalid time metadata in cache for %s; refetching", symbol
                )
        return result

    def _write_cache(
        self,
        db: Session,
        data: dict[str, pd.DataFrame],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        """Upsert complete canonical bars; legacy/incomplete entries cannot hit.

        Range coverage is attached to one upstream fetch, never inferred by
        joining two disjoint cached ranges. Current incomplete periods cap it.
        """
        from api.models import OHLCVCacheRow

        for symbol, raw in data.items():
            frame = validate_bar_times(raw, self.frequency)
            if frame.empty:
                continue
            lower = start_date or frame.index.min().date().isoformat()
            upper = (
                end_date
                or (frame["period_end"].max() - pd.Timedelta(days=1)).date().isoformat()
            )
            now = pd.Timestamp.now(tz="UTC")
            completed = frame["is_complete"] & (frame["available_at"] <= now)
            incomplete = frame.loc[~completed]
            if not incomplete.empty:
                upper = min(
                    upper,
                    (incomplete.index.min() - pd.Timedelta(days=1)).date().isoformat(),
                )
            # An upstream request extending into the future does not prove
            # coverage of data which has not happened yet.
            upper = min(
                upper,
                min(
                    (
                        now.tz_convert(zone).normalize().tz_localize(None)
                        - pd.Timedelta(days=1)
                    )
                    .date()
                    .isoformat()
                    for zone in frame["market_timezone"].unique()
                ),
            )
            existing = {
                row.bar_date: row
                for row in db.query(OHLCVCacheRow)
                .filter(
                    OHLCVCacheRow.symbol == symbol,
                    OHLCVCacheRow.frequency == str(self.frequency),
                    OHLCVCacheRow.source == self._inner.source_name,
                    OHLCVCacheRow.bar_date >= frame.index.min().date(),
                    OHLCVCacheRow.bar_date <= frame.index.max().date(),
                )
                .all()
            }
            for idx, values in frame.loc[completed].iterrows():
                timing = {
                    "version": TIME_CONTRACT_VERSION,
                    "coverage_start": lower,
                    "coverage_end": upper,
                    **{
                        c: (
                            values[c].isoformat()
                            if isinstance(values[c], (datetime, pd.Timestamp))
                            else values[c]
                        )
                        for c in TIME_COLUMNS
                    },
                }
                row = existing.get(idx.date())
                if row is None:
                    row = OHLCVCacheRow(
                        symbol=symbol,
                        bar_date=idx.date(),
                        frequency=str(self.frequency),
                        source=self._inner.source_name,
                    )
                    db.add(row)
                for column in _OHLCV_COLS:
                    value = values.get(column)
                    setattr(row, column, float(value) if pd.notna(value) else None)
                row.timing = timing
                row.fetched_at = datetime.now(timezone.utc)
        try:
            db.commit()
        except IntegrityError:
            # Concurrent fetchers may race to insert the same bar. Roll back
            # this batch; future reads/refetches remain correct.
            db.rollback()
            logger.info("Concurrent bar-cache insert; using the upstream result")
