from datetime import date, datetime
from unittest.mock import Mock

import pytest

from investment_tool.domain.model_data.entities.news import News
from investment_tool.constants import Frequency
from investment_tool.strategies.news_selection import news_candidates
from investment_tool.strategies.sentiment_analysis_strategies import (
    transformers_sentiment_analysis as sentiment,
)

START = date(2024, 4, 1)
END = date(2024, 5, 1)


def article(text, **kwargs):
    return {"headline": text, "published_at": "2024-04-20T12:00:00Z", **kwargs}


def bucket(symbol, articles):
    return News(symbol, START, END, articles)


@pytest.fixture
def scorer(monkeypatch, tmp_path):
    monkeypatch.setenv("SENTIMENT_CACHE_PATH", str(tmp_path / "scores.sqlite3"))

    def infer(texts):
        return [0.8 if "positive" in text else -0.8 for text in texts]

    model = Mock(side_effect=infer)
    monkeypatch.setattr(sentiment, "_score_batch", model)
    return model


def test_previous_month_dedup_cutoff_ties_missing_coverage_and_cache(scorer):
    symbols = ["B", "A", "C", "THIN", "BTC/USD", "GLD"]
    data = {}
    for symbol in ["A", "B", "C"]:
        data[(symbol, START)] = bucket(
            symbol,
            [
                article(f"positive {i}" if symbol != "C" else f"bad {i}")
                for i in range(3)
            ],
        )
    data[("THIN", START)] = bucket(
        "THIN",
        [article("positive duplicate")] * 4
        + [
            article("future", updated_at="2024-05-02T00:00:00Z"),
            article("naive", published_at="2024-04-21T00:00:00"),
        ],
    )
    data[("C", date(2024, 5, 1))] = News(
        "C",
        END,
        date(2024, 6, 1),
        [article("future winner", published_at="2024-05-05T00:00:00Z")] * 50,
    )
    selected, audit = news_candidates(
        symbols, data, datetime(2024, 5, 1), datetime(2024, 5, 1), top_assets=1
    )
    assert selected == {"A", "THIN", "BTC/USD", "GLD"}
    assert audit["ranked_selected"] == ["A"]  # alphabetic tie break
    assert audit["article_counts"]["THIN"] == 1
    assert audit["scores"] == pytest.approx({"A": 0.8, "B": 0.8, "C": -0.8})
    assert len(data[("THIN", START)].data) == 6  # raw input untouched
    calls = scorer.call_count
    assert (
        news_candidates(
            symbols, data, datetime(2024, 5, 1), datetime(2024, 5, 1), top_assets=1
        )[0]
        == selected
    )
    assert scorer.call_count == calls  # persisted successful scores only


@pytest.mark.parametrize("load_failure", [False, True])
def test_failed_scores_remain_unscored_not_neutral(scorer, load_failure):
    scorer.side_effect = (
        RuntimeError("model unavailable")
        if load_failure
        else lambda texts: [None] * len(texts)
    )
    data = {("A", START): bucket("A", [article(str(i)) for i in range(3)])}
    selected, audit = news_candidates(
        ["A"], data, datetime(2024, 5, 1), datetime(2024, 5, 1)
    )
    assert selected == {"A"} and audit["scores"] == {}
    assert audit["inference_unavailable"] == ["A"]


def test_cutoff_precedes_bucket_end_and_old_history_is_ignored(scorer):
    data = {
        ("A", START): bucket(
            "A",
            [article(str(i), published_at="2024-04-30T23:00:00Z") for i in range(3)],
        )
    }
    selected, audit = news_candidates(
        ["A"], data, datetime(2024, 5, 1), datetime(2024, 4, 30, 22)
    )
    assert selected == {"A"} and audit["article_counts"]["A"] == 0
    assert not scorer.called


@pytest.mark.parametrize(
    "frequency,trading,start,end",
    [
        (Frequency.DAILY, "2024-05-02", "2024-05-01", "2024-05-02"),
        (Frequency.WEEKLY, "2024-05-06", "2024-04-29", "2024-05-06"),
        (Frequency.MONTHLY, "2024-05-01", "2024-04-01", "2024-05-01"),
        (Frequency.WEEKLY, "2024-05-08", "2024-04-29", "2024-05-06"),
        (Frequency.MONTHLY, "2024-05-23", "2024-04-01", "2024-05-01"),
    ],
)
def test_each_frequency_scores_only_the_prior_calendar_period(
    scorer, frequency, trading, start, end
):
    from datetime import timedelta
    import pandas as pd

    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    # One broad stored interval crosses the selection boundary. It must be
    # bounded before inference/cache lookup, rather than assumed aligned.
    articles = [
        article("positive included", published_at=f"{start}T12:00:00Z"),
        article("current future", published_at=f"{end}T00:00:00Z"),
        article("old", published_at=f"{first - timedelta(days=1)}T23:59:00Z"),
        article(
            "revised", published_at=f"{start}T13:00:00Z", updated_at=f"{end}T00:00:00Z"
        ),
    ]
    data = {
        ("A", first): News(
            "A", first - timedelta(days=1), last + timedelta(days=1), articles
        )
    }
    selected, audit = news_candidates(
        ["A", "BTC/USD", "GLD"],
        data,
        datetime.fromisoformat(trading),
        pd.Timestamp(f"{end}T05:00:00Z").to_pydatetime(),
        frequency=frequency,
        min_articles=1,
    )
    assert audit["article_counts"]["A"] == 1
    assert audit["period_start"] == start and audit["period_end"] == end
    assert audit["scores"] == pytest.approx({"A": 0.8})
    assert {"BTC/USD", "GLD"} <= selected
    assert len(data[("A", first)].data) == 4


def test_daily_legacy_equal_start_end_bucket_is_still_bounded(scorer):
    first = date(2024, 5, 1)
    data = {
        ("A", first): News(
            "A",
            first,
            first,
            [article("positive", published_at="2024-05-01T12:00:00Z")],
        )
    }
    _, audit = news_candidates(
        ["A"],
        data,
        datetime(2024, 5, 2),
        datetime(2024, 5, 2),
        frequency=Frequency.DAILY,
        min_articles=1,
    )
    assert audit["article_counts"]["A"] == 1
