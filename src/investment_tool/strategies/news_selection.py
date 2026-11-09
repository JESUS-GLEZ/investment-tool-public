"""Causal prior-period candidate selection; missing coverage remains unscored."""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.market_time import period_code
from investment_tool.constants import Frequency
import pandas as pd


def news_candidates(
    symbols: list[str],
    news: dict[tuple[str, date], News],
    trading_date: datetime,
    decision_time: datetime,
    *,
    frequency: Frequency = Frequency.MONTHLY,
    top_assets: int = 50,
    min_articles: int = 3,
) -> tuple[set[str], dict[str, Any]]:
    """Rank by the previous complete calendar period, retaining unscored assets.

    Reapply availability before consulting the article-score cache. Text is
    deduplicated per symbol, matching the existing classifier's preprocessing.
    Older history and the target period's articles never affect this ranking.
    """
    from investment_tool.strategies.sentiment_analysis_strategies.transformers_sentiment_analysis import (
        compute_sentiment_scores,
        MODEL_NAME,
        MODEL_REVISION,
        PREPROCESSING_VERSION,
    )

    if top_assets < 1 or min_articles < 1:
        raise ValueError(
            "News shortlist size and minimum article count must be positive"
        )
    current = pd.Timestamp(trading_date).to_period(period_code(frequency))
    # Investing observations may be labelled today, inside an open period.
    # Use the containing calendar interval, not the synthetic observation label.
    end = current.start_time.date()
    start = (current - 1).start_time.date()
    buckets: dict[tuple[str, date], News] = {}
    counts: dict[str, int] = {}
    for symbol in symbols:
        unique: dict[str, dict[str, Any]] = {}
        for (article_symbol, _), bucket in news.items():
            bucket_end = bucket.period_end_date
            if bucket_end == bucket.period_start_date:
                bucket_end += timedelta(days=1)
            if (
                article_symbol != symbol
                or bucket.period_start_date >= end
                or bucket_end <= start
            ):
                continue
            # Re-bucket overlapping saved intervals explicitly. This also handles
            # older snapshots whose news buckets were not calendar-aligned.
            bounded = News(
                symbol,
                max(start, bucket.period_start_date),
                min(end, bucket_end),
                bucket.data,
            )
            for article in bounded.available_before(decision_time).data:
                headline, summary = (
                    article.get("headline") or "",
                    article.get("summary") or "",
                )
                if not isinstance(headline, str) or not isinstance(summary, str):
                    continue
                if not headline.strip() and not summary.strip():
                    continue
                text = ("HEADLINE: " + headline + " SUMMARY: " + summary).strip()
                unique.setdefault(text, article)
        counts[symbol] = len(unique)
        if len(unique) >= min_articles:
            buckets[(symbol, start)] = News(symbol, start, end, list(unique.values()))
    scores_by_symbol = compute_sentiment_scores(buckets, omit_failed_periods=True)
    scores = {
        symbol: float(periods[start])
        for symbol, periods in scores_by_symbol.items()
        if start in periods and math.isfinite(periods[start])
    }
    ranked = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
    selected = ranked[:top_assets]
    unscored = sorted(set(symbols) - scores.keys())
    eligible = set(selected) | set(unscored)
    diagnostics = {
        "enabled": True,
        "classifier": MODEL_NAME,
        "classifier_revision": MODEL_REVISION,
        "preprocessing": PREPROCESSING_VERSION,
        "frequency": str(frequency),
        "period_start": start.isoformat(),
        "period_end": end.isoformat(),
        **({"month": start.isoformat()} if frequency == Frequency.MONTHLY else {}),
        "decision_time": decision_time.isoformat(),
        "top_assets": top_assets,
        "min_articles": min_articles,
        "candidate_count_before": len(symbols),
        "candidate_count_after": len(eligible),
        "ranked_selected": selected,
        "unscored_passthrough": unscored,
        "inference_unavailable": sorted(set(s for s, _ in buckets) - scores.keys()),
        "article_counts": counts,
        "scores": scores,
    }
    return eligible, diagnostics
