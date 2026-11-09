"""
News entity for market news data.

This module defines the News class which holds news articles and
sentiment data associated with a stock for a specific time period.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


class News:
    """News data associated with a stock for a time period.

    Holds raw news articles and provides methods for sentiment analysis.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    period_start_date : date
        Start of the news collection period.
    period_end_date : date
        Exclusive end of the repository's news collection period.
    data : list[dict[str, Any]] | dict[str, Any]
        Raw news data (articles, headlines, etc.).

    Attributes
    ----------
    symbol : str
    period_start_date : date
    period_end_date : date
    data : list[dict[str, Any]]
    """

    def __init__(
        self,
        symbol: str,
        period_start_date: date,
        period_end_date: date,
        data: list[dict[str, Any]] | dict[str, Any],
    ) -> None:
        self.symbol = symbol
        self.period_start_date = period_start_date
        self.period_end_date = period_end_date
        # Older callers supplied a single article; repositories supply a list.
        self.data = [data] if isinstance(data, dict) else list(data)

    def available_before(self, cutoff: datetime | None = None) -> News:
        """Return a copy with articles available in this bucket before cutoff.

        Repository bucket bounds are UTC with an exclusive end date. Period labels alone
        cannot establish availability. Reject missing/naive/bad timestamps.
        Where a modification time is provided, reject versions updated later
        than the cutoff as well: the historical text cannot be reconstructed.
        """
        start = datetime.combine(self.period_start_date, time.min, timezone.utc)
        # Alpaca stores (April 1, May 1), not (April 1, April 30).
        # Legacy single-day entities use equal start/end dates.
        end_date = self.period_end_date
        if end_date == self.period_start_date:
            end_date += timedelta(days=1)
        end = datetime.combine(end_date, time.min, timezone.utc)
        boundary = end if cutoff is None else min(end, self._utc_cutoff(cutoff))
        articles = []
        invalid = unavailable = 0
        for article in self.data:
            if not isinstance(article, dict):
                invalid += 1
                continue
            published = self._article_time(
                article.get("published_at") or article.get("created_at")
            )
            updated_raw = article.get("updated_at")
            updated = self._article_time(updated_raw) if updated_raw else published
            if published is None or updated is None:
                invalid += 1
                continue
            if not start <= published < boundary or updated >= boundary:
                unavailable += 1
                continue
            articles.append(article)
        if invalid:
            logger.warning(
                "Excluded %d news articles with missing/invalid availability timestamps for %s (%s)",
                invalid,
                self.symbol,
                self.period_start_date,
            )
        if unavailable:
            logger.debug(
                "Excluded %d unavailable/out-of-bucket articles for %s (%s)",
                unavailable,
                self.symbol,
                self.period_start_date,
            )
        return News(self.symbol, self.period_start_date, self.period_end_date, articles)

    @staticmethod
    def _utc_cutoff(cutoff: datetime) -> datetime:
        # Factory period labels are naive UTC; aware callers preserve their instant.
        return (
            cutoff.replace(tzinfo=timezone.utc)
            if cutoff.tzinfo is None
            else cutoff.astimezone(timezone.utc)
        )

    @staticmethod
    def _article_time(value: Any) -> datetime | None:
        if not isinstance(value, (str, datetime)):
            return None
        try:
            parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
            if parsed.tzinfo is None:
                return None
            return parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            return None

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return f"News for {self.symbol} from {self.period_start_date} to {self.period_end_date}"

    def __repr__(self) -> str:
        """Return detailed string representation."""
        return (
            f"News(symbol={self.symbol!r}, "
            f"period_start_date={self.period_start_date}, "
            f"period_end_date={self.period_end_date})"
        )

    def calculate_sentiment_score(self) -> float | None:
        """Calculate sentiment score from news data.

        Returns
        -------
        float | None
            Sentiment score between -1 (negative) and 1 (positive),
            or None if not implemented.

        Notes
        -----
        This method is a placeholder for sentiment analysis implementation.
        """
        # TODO: Implement sentiment analysis
        return None
