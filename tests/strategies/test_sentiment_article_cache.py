"""Causal news selection and safe inference reuse across datasets/workers."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)
from investment_tool.strategies.sentiment_analysis_strategies import (
    transformers_sentiment_analysis as sentiment,
)
from investment_tool.strategies.sentiment_analysis_strategies.article_score_cache import (
    ArticleScoreCache,
)

KEY = ("AAPL", date(2024, 4, 1))


def article(text="good", published="2024-04-30T23:59:59Z", **fields):
    return {"headline": text, "summary": "summary", "published_at": published, **fields}


def news(articles, symbol="AAPL"):
    return News(symbol, date(2024, 4, 1), date(2024, 5, 1), articles)


@pytest.fixture
def model(monkeypatch):
    def predict(texts, **kwargs):
        def result(text):
            return {"label": "NEGATIVE" if "bad" in text else "POSITIVE", "score": 0.8}

        return (
            [result(texts)]
            if isinstance(texts, str)
            else [result(text) for text in texts]
        )

    pipeline = MagicMock(side_effect=predict)
    monkeypatch.setattr(sentiment, "_get_sentiment_pipeline", lambda: pipeline)
    return pipeline


@pytest.mark.parametrize(
    "published",
    [
        None,
        "garbage",
        "2024-04-30 23:59:59",
        "2024-05-01T00:00:00Z",
        "2024-05-01T12:00:00Z",
        "2024-03-31T23:59:59Z",
    ],
)
def test_unknown_future_and_out_of_bucket_articles_excluded(published):
    raw = news([article("valid"), article("excluded", published)])
    filtered = raw.available_before(datetime(2024, 5, 1))
    assert [item["headline"] for item in filtered.data] == ["valid"]
    assert len(raw.data) == 2  # Raw dataset is retained for other decisions.


def test_timezone_conversion_uses_actual_publication_instant():
    raw = news(
        [
            article("late", "2024-04-30T23:30:00-04:00"),
            article("early", "2024-05-01T00:30:00+02:00"),
        ]
    )
    assert [
        item["headline"] for item in raw.available_before(datetime(2024, 5, 1)).data
    ] == ["early"]


def test_known_later_article_revision_cannot_enter_historical_features():
    raw = news(
        [
            article("original", updated_at="2024-04-30T23:59:59Z"),
            article("future revision", updated_at="2024-05-02T00:00:00Z"),
        ]
    )
    assert [
        item["headline"] for item in raw.available_before(datetime(2024, 6, 1)).data
    ] == ["original"]


def test_legacy_created_at_and_single_article_supported():
    raw = news({"headline": "legacy", "created_at": "2024-04-15T10:00:00Z"})
    assert len(raw.available_before(datetime(2024, 5, 1)).data) == 1


def test_strategy_filter_applies_exact_cutoff_not_just_bucket_label():
    raw = {
        KEY: news(
            [
                article("early", "2024-04-15T00:00:00Z"),
                article("late", "2024-04-30T15:00:00Z"),
            ]
        )
    }
    period = Period(datetime(2024, 4, 30, 12), pd.DataFrame())
    selected = BaseInvestmentStrategy._filter_news(raw, period)
    assert [item["headline"] for item in selected[KEY].data] == ["early"]
    assert len(raw[KEY].data) == 2


def test_empty_or_ineligible_news_does_not_load_model_or_create_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        sentiment,
        "_get_sentiment_pipeline",
        lambda: pytest.fail("Model should not load"),
    )
    path = tmp_path / "cache.sqlite3"
    assert sentiment.compute_sentiment_scores(
        {KEY: news([article(published=None)])}, cache_path=path
    ) == {"AAPL": {KEY[1]: 0}}
    assert not path.exists()


def test_article_scored_once_across_symbols_and_worker_restarts(
    tmp_path, model, monkeypatch
):
    path = tmp_path / "cache.sqlite3"
    data = {KEY: news([article()]), ("MSFT", KEY[1]): news([article()], "MSFT")}
    first = sentiment.compute_sentiment_scores(data, cache_path=path)
    assert first == {"AAPL": {KEY[1]: 0.8}, "MSFT": {KEY[1]: 0.8}}
    assert len(model.call_args.args[0]) == 1
    sentiment.reset_sentiment_cache()
    monkeypatch.setattr(
        sentiment,
        "_get_sentiment_pipeline",
        lambda: pytest.fail("Persisted cache should avoid model load"),
    )
    assert sentiment.compute_sentiment_scores(data, cache_path=path) == first


def test_same_period_different_dataset_reaggregates_only_current_articles(
    tmp_path, model
):
    path = tmp_path / "cache.sqlite3"
    sentiment.compute_sentiment_scores({KEY: news([article("good")])}, cache_path=path)
    result = sentiment.compute_sentiment_scores(
        {KEY: news([article("bad")])}, cache_path=path
    )
    assert result == {"AAPL": {KEY[1]: -0.8}}
    assert model.call_count == 2
    result = sentiment.compute_sentiment_scores(
        {KEY: news([article("good"), article("bad")])}, cache_path=path
    )
    assert result == {"AAPL": {KEY[1]: 0}}
    assert model.call_count == 2


def test_cutoff_changes_aggregate_without_rescoring_existing_articles(tmp_path, model):
    path = tmp_path / "cache.sqlite3"
    raw = news(
        [
            article("good", "2024-04-15T00:00:00Z"),
            article("bad", "2024-04-25T00:00:00Z"),
        ]
    )
    before = sentiment.compute_sentiment_scores(
        {KEY: raw.available_before(datetime(2024, 4, 20))}, cache_path=path
    )
    later = sentiment.compute_sentiment_scores(
        {KEY: raw.available_before(datetime(2024, 5, 1))}, cache_path=path
    )
    assert before["AAPL"][KEY[1]] == 0.8
    assert later["AAPL"][KEY[1]] == 0
    assert model.call_count == 2
    assert len(model.call_args.args[0]) == 1


def test_future_article_already_cached_still_cannot_change_earlier_decision(
    tmp_path, model
):
    path = tmp_path / "cache.sqlite3"
    future = article("bad", "2024-05-02T10:00:00Z")
    may_key = ("AAPL", date(2024, 5, 1))
    may_news = News("AAPL", may_key[1], date(2024, 6, 1), [future])
    sentiment.compute_sentiment_scores({may_key: may_news}, cache_path=path)
    result = sentiment.compute_sentiment_scores(
        {KEY: news([article("good"), future])}, cache_path=path
    )
    assert result["AAPL"][KEY[1]] == 0.8
    assert len(model.call_args.args[0]) == 1


@pytest.mark.parametrize(
    "setting", ["MODEL_REVISION", "PREPROCESSING_VERSION", "MODEL_NAME"]
)
def test_cache_invalidates_for_model_and_preprocessing_changes(
    tmp_path, model, monkeypatch, setting
):
    path = tmp_path / "cache.sqlite3"
    data = {KEY: news([article()])}
    sentiment.compute_sentiment_scores(data, cache_path=path)
    monkeypatch.setattr(sentiment, setting, "different-version")
    sentiment.compute_sentiment_scores(data, cache_path=path)
    assert model.call_count == 2


def test_corrupt_cache_does_not_kill_run_or_manufacture_neutral_score(
    tmp_path, model, caplog
):
    path = tmp_path / "cache.sqlite3"
    path.write_text("corrupt database")
    result = sentiment.compute_sentiment_scores(
        {KEY: news([article()])}, cache_path=path
    )
    assert result["AAPL"][KEY[1]] == 0.8
    assert "uncached inference" in caplog.text


def test_batch_failure_retries_individual_articles(tmp_path, model):
    model.side_effect = [
        RuntimeError("batch failed"),
        [{"label": "POSITIVE", "score": 0.8}],
    ]
    result = sentiment.compute_sentiment_scores(
        {KEY: news([article()])}, cache_path=tmp_path / "cache.sqlite3"
    )
    assert result["AAPL"][KEY[1]] == 0.8
    assert model.call_count == 2


def test_unwritable_cache_uses_real_inference(tmp_path, model, caplog):
    parent = tmp_path / "file-not-directory"
    parent.write_text("existing file")
    result = sentiment.compute_sentiment_scores(
        {KEY: news([article()])}, cache_path=parent / "cache.sqlite3"
    )
    assert result["AAPL"][KEY[1]] == 0.8
    assert "uncached inference" in caplog.text


@pytest.mark.parametrize(
    "bad_result",
    [
        [],
        [{"label": "POSITIVE", "score": float("nan")}],
        [{"label": "UNKNOWN", "score": 0.8}],
    ],
)
def test_invalid_outputs_are_not_persisted_as_valid_scores(tmp_path, model, bad_result):
    path = tmp_path / "cache.sqlite3"
    model.side_effect = None
    model.return_value = bad_result
    data = {KEY: news([article()])}
    assert (
        sentiment.compute_sentiment_scores(data, cache_path=path)["AAPL"][KEY[1]] == 0
    )
    model.return_value = [{"label": "POSITIVE", "score": 0.8}]
    assert (
        sentiment.compute_sentiment_scores(data, cache_path=path)["AAPL"][KEY[1]] == 0.8
    )


def test_failed_inference_fallback_is_retried_and_never_persisted(tmp_path, model):
    path = tmp_path / "cache.sqlite3"
    data = {KEY: news([article()])}
    model.side_effect = RuntimeError("temporary inference error")
    assert (
        sentiment.compute_sentiment_scores(data, cache_path=path)["AAPL"][KEY[1]] == 0
    )
    model.side_effect = None
    model.return_value = [{"label": "NEGATIVE", "score": 0.7}]
    assert (
        sentiment.compute_sentiment_scores(data, cache_path=path)["AAPL"][KEY[1]]
        == -0.7
    )


def test_completed_batches_survive_interruption(tmp_path, model):
    path = tmp_path / "cache.sqlite3"
    data = {KEY: news([article(str(i)) for i in range(65)])}
    model.side_effect = [
        [{"label": "POSITIVE", "score": 0.8}] * 64,
        KeyboardInterrupt(),
    ]
    with pytest.raises(KeyboardInterrupt):
        sentiment.compute_sentiment_scores(data, cache_path=path)
    model.side_effect = None
    model.return_value = [{"label": "POSITIVE", "score": 0.8}]
    result = sentiment.compute_sentiment_scores(data, cache_path=path)
    assert result["AAPL"][KEY[1]] == pytest.approx(0.8)
    assert len(model.call_args.args[0]) == 1


def test_simultaneous_workers_keep_valid_cache_entries(tmp_path):
    path = tmp_path / "cache.sqlite3"
    # Initialize before the workers race, as in an existing deployment.
    ArticleScoreCache(path).close()

    def save(index):
        cache = ArticleScoreCache(path)
        try:
            cache.put_many({f"key-{index}": 0.8})
        finally:
            cache.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(save, range(20)))
    cache = ArticleScoreCache(path)
    try:
        assert len(cache.get_many([f"key-{i}" for i in range(20)])) == 20
    finally:
        cache.close()
