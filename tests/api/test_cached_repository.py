"""Tests for :mod:`investment_tool.data.repositories.cached_repository`.

Uses SQLite in-memory DB + a mock inner repository to test caching
behaviour without network calls or a real Postgres instance.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.models import Base, OHLCVCacheRow
from investment_tool.domain.market_time import normalize_bars
from investment_tool.constants import Frequency
from investment_tool.data.repositories.base_repository import (
    BaseDataRepository,
)
from investment_tool.data.repositories.cached_repository import CachedRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(
    dates: list[str],
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame."""
    n = len(dates)
    data = {
        "open": [base_price + i for i in range(n)],
        "high": [base_price + i + 1 for i in range(n)],
        "low": [base_price + i - 1 for i in range(n)],
        "close": [base_price + i + 0.5 for i in range(n)],
        "volume": [1000.0 * (i + 1) for i in range(n)],
        "trade_count": [50.0 * (i + 1) for i in range(n)],
        "vwap": [base_price + i + 0.25 for i in range(n)],
    }
    idx = pd.to_datetime(dates)
    return normalize_bars(
        pd.DataFrame(data, index=idx.tz_localize("UTC")),
        Frequency.DAILY,
        market_timezone="UTC",
    )


class FakeInnerRepository(BaseDataRepository):
    """A controllable fake repository for testing the cache wrapper."""

    def __init__(
        self,
        frequency: Frequency = Frequency.DAILY,
        stock_data: dict[str, pd.DataFrame] | None = None,
        crypto_data: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        super().__init__(frequency)
        self._stock_data = stock_data or {}
        self._crypto_data = crypto_data or {}
        self.stock_call_count = 0
        self.crypto_call_count = 0

    @property
    def source_name(self) -> str:
        return "FakeSource"

    def fetch_stock_data(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        self.stock_call_count += 1
        return {s: self._stock_data[s] for s in symbols if s in self._stock_data}

    def fetch_crypto_data(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        self.crypto_call_count += 1
        return {s: self._crypto_data[s] for s in symbols if s in self._crypto_data}

    def fetch_news_data(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> dict[tuple[str, date], Any]:
        return {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session_factory() -> sessionmaker:  # type: ignore[type-arg]
    """Create an in-memory SQLite session factory with schema."""
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    return factory


@pytest.fixture()
def db_session(db_session_factory: sessionmaker) -> Session:  # type: ignore[type-arg, misc]
    session = db_session_factory()
    try:
        yield session  # type: ignore[misc]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dated_symbol_identity_never_reads_or_populates_ticker_only_cache(
    db_session_factory,
):
    prices = _make_ohlcv_df(["2024-01-02", "2024-01-03"])
    inner = FakeInnerRepository(stock_data={"ABC": prices})
    inner.symbol_identity_date = date(2021, 1, 1)
    cached = CachedRepository(inner, db_session_factory)
    for _ in range(2):
        actual = cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-03")
        pd.testing.assert_frame_equal(actual["ABC"], prices)
    assert inner.stock_call_count == 2
    with db_session_factory() as session:
        assert session.query(OHLCVCacheRow).count() == 0


class TestCachedRepositorySource:
    """Verify metadata / source name delegation."""

    def test_source_name(self, db_session_factory: sessionmaker) -> None:  # type: ignore[type-arg]
        inner = FakeInnerRepository()
        cached = CachedRepository(inner, db_session_factory)
        assert cached.source_name == "Cached(FakeSource)"

    def test_frequency_inherited(self, db_session_factory: sessionmaker) -> None:  # type: ignore[type-arg]
        inner = FakeInnerRepository(frequency=Frequency.MONTHLY)
        cached = CachedRepository(inner, db_session_factory)
        assert cached.frequency == Frequency.MONTHLY


class TestCacheMiss:
    """First fetch (cold cache) delegates to inner repo and writes cache."""

    def test_stock_cache_miss_delegates(
        self,
        db_session_factory: sessionmaker,
        db_session: Session,  # type: ignore[type-arg]
    ) -> None:
        aapl_df = _make_ohlcv_df(["2024-01-02", "2024-01-03"])
        inner = FakeInnerRepository(stock_data={"AAPL": aapl_df})
        cached = CachedRepository(inner, db_session_factory)

        result = cached.fetch_stock_data(["AAPL"], "2024-01-02", "2024-01-03")

        assert "AAPL" in result
        assert len(result["AAPL"]) == 2
        assert inner.stock_call_count == 1

        # Verify rows were written to DB
        rows = db_session.query(OHLCVCacheRow).all()
        assert len(rows) == 2
        assert all(r.symbol == "AAPL" for r in rows)
        assert all(r.source == "FakeSource" for r in rows)

    def test_crypto_cache_miss_delegates(
        self,
        db_session_factory: sessionmaker,
        db_session: Session,  # type: ignore[type-arg]
    ) -> None:
        btc_df = _make_ohlcv_df(["2024-01-02"], base_price=40000.0)
        inner = FakeInnerRepository(crypto_data={"BTC/USD": btc_df})
        cached = CachedRepository(inner, db_session_factory)

        result = cached.fetch_crypto_data(["BTC/USD"], "2024-01-02", "2024-01-03")

        assert "BTC/USD" in result
        assert inner.crypto_call_count == 1

    def test_empty_symbols_returns_empty(
        self,
        db_session_factory: sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        inner = FakeInnerRepository()
        cached = CachedRepository(inner, db_session_factory)

        result = cached.fetch_stock_data([], "2024-01-02", "2024-01-03")
        assert result == {}
        assert inner.stock_call_count == 0


class TestCacheHit:
    """Second fetch (warm cache) should NOT call the inner repo."""

    def test_stock_cache_hit_avoids_inner_call(
        self,
        db_session_factory: sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        aapl_df = _make_ohlcv_df(["2024-01-02", "2024-01-03"])
        inner = FakeInnerRepository(stock_data={"AAPL": aapl_df})
        cached = CachedRepository(inner, db_session_factory)

        # First call — cache miss
        cached.fetch_stock_data(["AAPL"], "2024-01-02", "2024-01-03")
        assert inner.stock_call_count == 1

        # Second call — cache hit
        result = cached.fetch_stock_data(["AAPL"], "2024-01-02", "2024-01-03")
        assert inner.stock_call_count == 1  # NOT incremented
        assert "AAPL" in result
        assert len(result["AAPL"]) == 2

    def test_partial_hit_fetches_only_missing(
        self,
        db_session_factory: sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        """If AAPL is cached but MSFT is not, only MSFT should be fetched."""
        aapl_df = _make_ohlcv_df(["2024-01-02"])
        msft_df = _make_ohlcv_df(["2024-01-02"], base_price=300.0)
        inner = FakeInnerRepository(stock_data={"AAPL": aapl_df, "MSFT": msft_df})
        cached = CachedRepository(inner, db_session_factory)

        # Prime AAPL
        cached.fetch_stock_data(["AAPL"], "2024-01-02", "2024-01-02")
        assert inner.stock_call_count == 1

        # Now fetch both — inner should be called once more for MSFT only
        result = cached.fetch_stock_data(["AAPL", "MSFT"], "2024-01-02", "2024-01-02")
        assert inner.stock_call_count == 2
        assert "AAPL" in result
        assert "MSFT" in result


class TestCacheDataIntegrity:
    """Cached data should match original data values."""

    def test_round_trip_values(
        self,
        db_session_factory: sessionmaker,
        db_session: Session,  # type: ignore[type-arg]
    ) -> None:
        original = _make_ohlcv_df(["2024-06-01"])
        inner = FakeInnerRepository(stock_data={"SPY": original})
        cached = CachedRepository(inner, db_session_factory)

        # Write to cache
        cached.fetch_stock_data(["SPY"], "2024-06-01", "2024-06-01")

        # Read from cache (fresh CachedRepository to ensure no in-memory state)
        inner2 = FakeInnerRepository()  # empty — forces cache-only
        cached2 = CachedRepository(inner2, db_session_factory)
        result = cached2.fetch_stock_data(["SPY"], "2024-06-01", "2024-06-01")

        assert "SPY" in result
        cached_df = result["SPY"]

        assert float(cached_df.iloc[0]["open"]) == pytest.approx(
            original.iloc[0]["open"]
        )
        assert float(cached_df.iloc[0]["close"]) == pytest.approx(
            original.iloc[0]["close"]
        )
        assert float(cached_df.iloc[0]["volume"]) == pytest.approx(
            original.iloc[0]["volume"]
        )

    def test_nullable_fields_round_trip(
        self,
        db_session_factory: sessionmaker,  # type: ignore[type-arg]
    ) -> None:
        """trade_count and vwap can be NaN/None."""
        df = _make_ohlcv_df(["2024-06-01"])
        df.loc[df.index[0], "trade_count"] = np.nan
        df.loc[df.index[0], "vwap"] = np.nan

        inner = FakeInnerRepository(stock_data={"XYZ": df})
        cached = CachedRepository(inner, db_session_factory)

        cached.fetch_stock_data(["XYZ"], "2024-06-01", "2024-06-01")

        # Read back
        inner2 = FakeInnerRepository()
        cached2 = CachedRepository(inner2, db_session_factory)
        result = cached2.fetch_stock_data(["XYZ"], "2024-06-01", "2024-06-01")

        row = result["XYZ"].iloc[0]
        assert row["trade_count"] is None or pd.isna(row["trade_count"])
        assert row["vwap"] is None or pd.isna(row["vwap"])


class TestNewsDelegation:
    """News data should always go to the inner repo (never cached)."""

    def test_news_delegates(self, db_session_factory: sessionmaker) -> None:  # type: ignore[type-arg]
        inner = FakeInnerRepository()
        cached = CachedRepository(inner, db_session_factory)

        result = cached.fetch_news_data(["AAPL"], "2024-01-01", "2024-01-31")
        assert result == {}  # FakeInnerRepository returns empty dict


def test_expanded_request_refetches_a_symbol_with_only_partial_cached_coverage(
    db_session_factory,
):
    frame = _make_ohlcv_df(["2024-01-02", "2024-01-03"])
    inner = FakeInnerRepository(stock_data={"ABC": frame})
    cached = CachedRepository(inner, db_session_factory)
    cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-02")
    result = cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-03")
    assert inner.stock_call_count == 2
    assert len(result["ABC"]) == 2
    cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-03")
    assert inner.stock_call_count == 2


def test_cache_preserves_timezone_and_publication_delay(db_session_factory):
    frame = _make_ohlcv_df(["2024-01-02", "2024-01-03"])
    frame["available_at"] += pd.Timedelta(hours=1)
    inner = FakeInnerRepository(stock_data={"ABC": frame})
    cached = CachedRepository(inner, db_session_factory)
    cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-03")
    loaded = cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-03")["ABC"]
    pd.testing.assert_frame_equal(frame, loaded, check_freq=False)


def test_incomplete_bar_is_never_cached_as_final(db_session_factory, db_session):
    frame = _make_ohlcv_df(["2024-01-02"])
    frame["is_complete"] = False
    inner = FakeInnerRepository(stock_data={"ABC": frame})
    cached = CachedRepository(inner, db_session_factory)
    cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-02")
    assert db_session.query(OHLCVCacheRow).count() == 0
    frame["is_complete"] = True
    frame["close"] = 999.0
    result = cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-02")
    assert inner.stock_call_count == 2
    assert result["ABC"].iloc[0].close == 999.0


def test_legacy_cache_rows_require_refetch_not_guessed_timing(
    db_session_factory, db_session
):
    db_session.add(
        OHLCVCacheRow(
            symbol="ABC",
            bar_date=date(2024, 1, 2),
            frequency="daily",
            source="FakeSource",
            open=1.0,
            high=2.0,
            low=1.0,
            close=2.0,
            volume=10.0,
            timing=None,
        )
    )
    db_session.commit()
    inner = FakeInnerRepository(stock_data={"ABC": _make_ohlcv_df(["2024-01-02"])})
    cached = CachedRepository(inner, db_session_factory)
    result = cached.fetch_stock_data(["ABC"], "2024-01-02", "2024-01-02")
    assert inner.stock_call_count == 1
    assert result["ABC"].iloc[0].close != 2.0
    db_session.expire_all()
    assert db_session.query(OHLCVCacheRow).count() == 1
    assert db_session.query(OHLCVCacheRow).one().timing["version"] == 1
