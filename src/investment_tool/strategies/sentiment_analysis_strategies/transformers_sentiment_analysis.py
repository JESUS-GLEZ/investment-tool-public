"""Batch sentiment inference with persistent article scores and causal aggregation.

The classifier, text construction, 512-character limit and signed confidence
are unchanged. Cache only successful article inferences, never monthly means.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import pipeline as hf_pipeline

from investment_tool.domain.model_data.entities.news import News
from investment_tool.strategies.sentiment_analysis_strategies.article_score_cache import (
    ArticleScoreCache,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
# Revision in the existing local HuggingFace snapshot; pin for reproducible reuse.
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
PREPROCESSING_VERSION = "headline-summary-512chars-signed-confidence-v1"
_BATCH_SIZE = 64
_SENTIMENT_PIPELINE: Any | None = None


def _get_sentiment_pipeline() -> Any:
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        device: int = 0 if torch.cuda.is_available() else -1
        task: Literal["text-classification"] = "text-classification"
        logger.info(
            "Loading sentiment pipeline '%s' revision=%s (device=%s)",
            MODEL_NAME,
            MODEL_REVISION,
            device,
        )
        _SENTIMENT_PIPELINE = hf_pipeline(
            task=task, model=MODEL_NAME, revision=MODEL_REVISION, device=device
        )
    return _SENTIMENT_PIPELINE


def reset_sentiment_cache() -> None:
    """Reset the process-local model; valid persisted article scores are retained."""
    global _SENTIMENT_PIPELINE
    _SENTIMENT_PIPELINE = None


def _cache_key(text: str) -> str:
    identity = json.dumps(
        [MODEL_NAME, MODEL_REVISION, PREPROCESSING_VERSION, text],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _signed_score(result: dict[str, Any]) -> float:
    label = result["label"]
    score = float(result["score"])
    if (
        label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Invalid sentiment inference result")
    return score if label == "POSITIVE" else -score if label == "NEGATIVE" else 0.0


def _score_batch(texts: list[str]) -> list[float | None]:
    pipeline = _get_sentiment_pipeline()
    try:
        results = pipeline(texts, batch_size=_BATCH_SIZE)
        if len(results) != len(texts):
            raise ValueError("Sentiment batch returned an incomplete result")
        return [_signed_score(result) for result in results]
    except Exception as exc:
        logger.warning(
            "Batch sentiment inference failed; retrying articles individually: %s", exc
        )
    scores: list[float | None] = []
    for text in texts:
        try:
            scores.append(_signed_score(pipeline(text)[0]))
        except Exception as exc:
            # Preserve neutral fallback for this call, but retry next time.
            logger.warning(
                "Sentiment inference failed; temporary neutral fallback (not cached): %s",
                exc,
            )
            scores.append(None)
    return scores


def compute_sentiment_scores(
    news_dict: dict[tuple[str, date], News],
    *,
    cache_path: Path | None = None,
    omit_failed_periods: bool = False,
) -> dict[str, dict[date, float]]:
    """Score unseen article text, then aggregate eligible scores for this call.

    Strategies already apply their decision cutoff. Enforce each news bucket's
    own end boundary here too, including callers that precompute full history.
    Cache keys include text, classifier revision and preprocessing version, so
    dataset/window changes cannot reuse a stale period aggregate. No cache hit
    ever bypasses article availability filtering.
    """
    period_keys: dict[tuple[str, date], list[str]] = {}
    article_texts: dict[str, str] = {}
    for period_key, news in news_dict.items():
        keys = []
        for article in news.available_before().data:
            text = (
                "HEADLINE: "
                + (article.get("headline") or "")
                + " SUMMARY: "
                + (article.get("summary") or "")
            ).strip()
            key = _cache_key(text)
            keys.append(key)
            article_texts[key] = text[:512]
        period_keys[period_key] = keys

    output: dict[str, dict[date, float]] = {}
    if not article_texts:
        for symbol, period in news_dict:
            output.setdefault(symbol, {})[period] = 0.0
        return output

    if cache_path is None:
        default = (
            Path(__file__).resolve().parents[4] / "data/output/sentiment_scores.sqlite3"
        )
        cache_path = Path(os.environ.get("SENTIMENT_CACHE_PATH", str(default)))
    cache = ArticleScoreCache(cache_path)
    try:
        scores = cache.get_many(list(article_texts))
        missing = [key for key in article_texts if key not in scores]
        logger.info(
            "Sentiment articles: %d unique, %d cached, %d to score",
            len(article_texts),
            len(scores),
            len(missing),
        )
        failed: set[str] = set()
        for offset in range(0, len(missing), _BATCH_SIZE):
            batch = missing[offset : offset + _BATCH_SIZE]
            try:
                predictions = _score_batch([article_texts[key] for key in batch])
            except Exception:
                if not omit_failed_periods:
                    raise
                logger.exception(
                    "Sentiment unavailable; affected shortlist candidates remain unscored"
                )
                predictions = [None] * len(batch)
            successful = {}
            for key, prediction in zip(batch, predictions):
                if prediction is None:
                    failed.add(key)
                scores[key] = 0.0 if prediction is None else prediction
                if prediction is not None:
                    successful[key] = prediction
            # Persist each completed batch so cancellation doesn't discard work.
            cache.put_many(successful)
        for (symbol, period), keys in period_keys.items():
            if omit_failed_periods and failed.intersection(keys):
                continue
            value = sum(scores[key] for key in keys) / len(keys) if keys else 0.0
            output.setdefault(symbol, {})[period] = value
        return output
    finally:
        cache.close()
