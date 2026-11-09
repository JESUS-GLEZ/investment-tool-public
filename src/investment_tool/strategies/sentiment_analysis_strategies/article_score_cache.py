"""Persistent, content-addressed sentiment scores using the standard library."""

from __future__ import annotations

import logging
import math
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class ArticleScoreCache:
    """Short SQLite transactions allow independent workers to share scores.

    Cache failure only disables reuse; it must not fail a long backtest or
    manufacture neutral predictions. Inference occurs outside transactions.
    """

    def __init__(self, path: Path) -> None:
        self._connection: sqlite3.Connection | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(path, timeout=10)
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS article_scores "
                "(cache_key TEXT PRIMARY KEY, score REAL NOT NULL)"
            )
            self._connection.commit()
        except (OSError, sqlite3.Error) as exc:
            self._disable(exc)

    def get_many(self, keys: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        connection = self._connection
        if connection is None:
            return scores
        try:
            # Stay below SQLite's parameter limit even for large universes.
            for offset in range(0, len(keys), 500):
                chunk = keys[offset : offset + 500]
                parameters = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT cache_key, score FROM article_scores WHERE cache_key IN ({parameters})",
                    chunk,
                )
                for key, value in rows:
                    try:
                        score = float(value)
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(score) and -1 <= score <= 1:
                        scores[str(key)] = score
            return scores
        except sqlite3.Error as exc:
            self._disable(exc)
            return scores

    def put_many(self, scores: dict[str, float]) -> None:
        connection = self._connection
        if connection is None or not scores:
            return
        try:
            with connection:
                connection.executemany(
                    "INSERT OR REPLACE INTO article_scores (cache_key, score) VALUES (?, ?)",
                    list(scores.items()),
                )
        except sqlite3.Error as exc:
            self._disable(exc)

    def _disable(self, error: Exception) -> None:
        logger.warning(
            "Sentiment cache unavailable; using uncached inference: %s", error
        )
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
