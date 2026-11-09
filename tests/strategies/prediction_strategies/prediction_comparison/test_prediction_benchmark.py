"""Unit tests for PredictionBenchmark.

Uses synthetic stock data with known patterns so tests are fast and
deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency
from investment_tool.domain.market_time import normalize_bars
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
    AveragePredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.prediction_benchmark import (
    PredictionBenchmark,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_config import (
    PredictorConfig,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_stock(symbol: str, n_periods: int = 40) -> Stock:
    """Build a synthetic stock with trending close prices."""
    np.random.seed(hash(symbol) % 2**31)
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS", tz="UTC")
    close = 100.0 + np.cumsum(np.random.randn(n_periods) * 2)
    close = np.maximum(close, 1.0)
    df = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.random.randint(1000, 10000, n_periods),
        },
        index=dates,
    )
    return Stock(
        symbol=symbol,
        data=normalize_bars(
            df, Frequency.MONTHLY, market_timezone="UTC", as_of="2030-01-01T00:00:00Z"
        ),
    )


def _make_stocks(n: int = 5, n_periods: int = 40) -> list[Stock]:
    symbols = [f"TST{i}" for i in range(n)]
    return [_make_stock(s, n_periods) for s in symbols]


def _ewm_config(span: int = 6) -> PredictorConfig:
    return PredictorConfig(
        label=f"EWM-{span}",
        factory=lambda stocks, sentiments: AveragePredictionStrategy(
            stocks=stocks, span=span
        ),
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestPredictionBenchmarkInit:
    def test_basic_init(self) -> None:
        stocks = _make_stocks(3)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        assert bench.results == []

    def test_default_sentiments(self) -> None:
        stocks = _make_stocks(2)
        bench = PredictionBenchmark(stocks=stocks)
        assert set(bench.sentiment_scores.keys()) == {"TST0", "TST1"}


class TestPredictionBenchmarkRun:
    def test_raises_on_empty_predictors(self) -> None:
        stocks = _make_stocks()
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        with pytest.raises(ValueError, match="At least one"):
            bench.run([])

    def test_single_predictor(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        results = bench.run([_ewm_config(6)])
        assert len(results) == 1
        assert results[0].config.label == "EWM-6"

    def test_multiple_predictors(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        results = bench.run([_ewm_config(3), _ewm_config(6), _ewm_config(12)])
        assert len(results) == 3

    def test_result_has_metrics(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        results = bench.run([_ewm_config(6)])
        r = results[0]
        assert "mae" in r.metrics
        assert "rmse" in r.metrics
        assert "directional_accuracy" in r.metrics
        assert "information_coefficient" in r.metrics

    def test_predictions_df_shape(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        results = bench.run([_ewm_config(6)])
        r = results[0]
        assert not r.predictions_df.empty
        assert not r.actuals_df.empty
        assert r.predictions_df.shape == r.actuals_df.shape

    def test_wall_time_recorded(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        results = bench.run([_ewm_config(6)])
        assert results[0].wall_time_seconds > 0

    def test_per_asset_metrics(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        results = bench.run([_ewm_config(6)])
        per = results[0].per_asset_metrics
        assert not per.empty
        assert len(per) == 3  # 3 stocks

    def test_insufficient_data_raises(self) -> None:
        stocks = _make_stocks(2, n_periods=5)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=10, min_history=10)
        with pytest.raises(ValueError, match="Insufficient data"):
            bench.run([_ewm_config(6)])


class TestPredictionBenchmarkSummary:
    def test_summary_table(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        bench.run([_ewm_config(3), _ewm_config(6)])
        table = bench.summary_table()

        assert isinstance(table, pd.DataFrame)
        assert list(table.index) == ["EWM-3", "EWM-6"]
        assert "mae" in table.columns
        assert "wall_time_s" in table.columns

    def test_save_comparison(self, tmp_path: Path) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(
            stocks=stocks,
            n_eval_periods=5,
            min_history=10,
            output_directory=tmp_path,
        )
        bench.run([_ewm_config(6)])
        saved = bench.save_comparison()

        assert len(saved) >= 1
        csv_files = [f for f in saved if f.endswith(".csv")]
        assert len(csv_files) >= 1
        assert Path(csv_files[0]).exists()


class TestPredictionBenchmarkProgressCallback:
    """Test that progress_callback is called correctly during run()."""

    def test_callback_called_per_predictor(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        calls: list[tuple[float, str]] = []

        def cb(frac: float, msg: str) -> None:
            calls.append((frac, msg))

        bench.run(
            [_ewm_config(3), _ewm_config(6), _ewm_config(12)], progress_callback=cb
        )

        assert len(calls) == 3
        assert calls[0][0] == pytest.approx(1 / 3)
        assert calls[1][0] == pytest.approx(2 / 3)
        assert calls[2][0] == pytest.approx(1.0)

    def test_callback_single_predictor(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        calls: list[tuple[float, str]] = []

        bench.run(
            [_ewm_config(6)],
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        assert len(calls) == 1
        assert calls[0][0] == pytest.approx(1.0)
        assert "EWM-6" in calls[0][1]

    def test_callback_message_contains_label(self) -> None:
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        calls: list[tuple[float, str]] = []

        bench.run(
            [_ewm_config(6)],
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        assert "EWM-6" in calls[0][1]

    def test_no_callback_still_works(self) -> None:
        """run() without progress_callback should behave as before."""
        stocks = _make_stocks(3, n_periods=40)
        bench = PredictionBenchmark(stocks=stocks, n_eval_periods=5, min_history=10)
        results = bench.run([_ewm_config(6)])
        assert len(results) == 1


class TestAuditedWalkForward:
    @pytest.mark.parametrize("periods", [1, 5, 12])
    def test_requested_count_and_actual_target_labels(self, periods: int) -> None:
        stock = _make_stock("A", 40)
        result = PredictionBenchmark([stock], n_eval_periods=periods).run(
            [_ewm_config()]
        )[0]
        assert len(result.predictions_df) == periods
        assert list(result.actuals_df.index) == list(stock.data.index[-periods:])
        expected = stock.data["close"].pct_change(fill_method=None).iloc[-periods:]
        np.testing.assert_allclose(result.actuals_df["A"], expected)
        assert result.metrics["scored_pairs"] == periods

    def test_exact_warmup_boundary_is_evaluable(self) -> None:
        result = PredictionBenchmark([_make_stock("A", 13)], min_history=12).run(
            [_ewm_config()]
        )[0]
        assert len(result.actuals_df) == 1
        assert result.metrics["scored_pairs"] == 1

    def test_missing_target_is_not_replaced_by_a_later_month(self) -> None:
        a, b = _make_stock("A", 15), _make_stock("B", 15)
        missing = a.data.index[-2]
        b = Stock(b.symbol, b.data.drop(index=missing))
        result = PredictionBenchmark([a, b], n_eval_periods=3, min_history=2).run(
            [_ewm_config()]
        )[0]
        assert np.isnan(result.actuals_df.loc[missing, "B"])
        assert np.isnan(result.actuals_df.iloc[-1]["B"])
        assert result.metrics["eligible_pairs"] == 5
        assert result.metrics["scored_pairs"] == 4

    def test_stock_order_does_not_change_targets_or_metrics(self) -> None:
        a, b = _make_stock("A", 20), _make_stock("B", 15)
        first = PredictionBenchmark([a, b], n_eval_periods=5).run([_ewm_config()])[0]
        second = PredictionBenchmark([b, a], n_eval_periods=5).run([_ewm_config()])[0]
        pd.testing.assert_frame_equal(first.predictions_df, second.predictions_df)
        pd.testing.assert_frame_equal(first.actuals_df, second.actuals_df)
        assert first.metrics["mae"] == second.metrics["mae"]

    def test_partial_final_bar_is_not_scored(self) -> None:
        stock = _make_stock("A", 15)
        stock.data.loc[stock.data.index[-1], "is_complete"] = False
        result = PredictionBenchmark([stock], n_eval_periods=1).run([_ewm_config()])[0]
        assert result.actuals_df.index[0] == stock.data.index[-2]

    def test_late_bar_cannot_enter_features_or_a_stored_return(self) -> None:
        stock = _make_stock("A", 15)
        target = stock.data.index[-1]
        # Hide an older bar, beyond the recent warm-up requirement, until later.
        late = stock.data.index[-4]
        stock.data.loc[late, "available_at"] = target.tz_localize("UTC") + pd.Timedelta(
            days=1
        )
        seen: list[Stock] = []

        def factory(stocks: list[Stock], sentiments: dict) -> AveragePredictionStrategy:
            seen.extend(stocks)
            return AveragePredictionStrategy(stocks)

        config = PredictorConfig("capture", factory)
        result = PredictionBenchmark([stock], n_eval_periods=1, min_history=2).run(
            [config]
        )[0]
        assert np.isnan(seen[0].data.loc[late, "close"])
        assert np.isnan(seen[0].data.loc[stock.data.index[-3], "returns"])
        assert target not in seen[0].data.index
        assert result.decision_times == [target.tz_localize("UTC").isoformat()]

    def test_target_price_cannot_change_its_forecast(self) -> None:
        stock = _make_stock("A", 15)
        original = PredictionBenchmark([stock], n_eval_periods=1).run([_ewm_config()])[
            0
        ]
        stock.data.loc[stock.data.index[-1], "close"] *= 10
        changed = PredictionBenchmark([stock], n_eval_periods=1).run([_ewm_config()])[0]
        pd.testing.assert_frame_equal(original.predictions_df, changed.predictions_df)
        assert original.actuals_df.iloc[0, 0] != changed.actuals_df.iloc[0, 0]

    def test_failure_uses_common_sample_and_retains_audit(self, tmp_path: Path) -> None:
        stock = _make_stock("A", 15)
        failed_origin = stock.data.index[-3]

        def factory(stocks: list[Stock], sentiments: dict) -> AveragePredictionStrategy:
            if stocks[0].data.index[-1] == failed_origin:
                raise ValueError("intentional forecast failure")
            return AveragePredictionStrategy(stocks)

        benchmark = PredictionBenchmark(
            [stock], n_eval_periods=3, min_history=2, output_directory=tmp_path
        )
        results = benchmark.run([_ewm_config(), PredictorConfig("fails once", factory)])
        assert (
            results[0].metrics["scored_pairs"]
            == results[1].metrics["scored_pairs"]
            == 2
        )
        assert results[0].metrics["prediction_pairs"] == 3
        assert results[1].metrics["prediction_pairs"] == 2
        assert results[1].metrics["failed_periods"] == 1
        assert results[1].metrics["coverage"] == pytest.approx(2 / 3)
        np.testing.assert_array_equal(
            results[0].actuals_df.isna(), results[1].actuals_df.isna()
        )
        from investment_tool.output.policy import OutputMode

        saved = benchmark.save_comparison(OutputMode.STANDARD)
        audit = pd.read_csv(
            next(path for path in saved if path.endswith("forecast_pairs.csv"))
        )
        assert len(audit) == 6
        assert len(audit[audit.scored]) == 4
        assert (
            audit.failure.fillna("").str.contains("intentional forecast failure").sum()
            == 1
        )
        # The complete predictor's excluded forecast survives in the audit CSV.
        excluded = audit[(audit.predictor == "EWM-6") & ~audit.scored]
        assert excluded.prediction.notna().all()
        assert not any(path.endswith(".png") for path in saved)

    def test_missing_and_infinite_predictions_share_the_same_mask(self) -> None:
        a, b = _make_stock("A", 15), _make_stock("B", 15)

        class Partial:
            def predict_returns(self) -> dict[str, float]:
                return {"A": 0.1, "B": float("inf"), "unknown": 0.5}

        results = PredictionBenchmark([a, b], n_eval_periods=2).run(
            [
                _ewm_config(),
                PredictorConfig("partial", lambda stocks, sentiments: Partial()),
            ]
        )
        for result in results:
            assert result.metrics["scored_pairs"] == 2
            assert result.predictions_df["B"].isna().all()
            assert "unknown" not in result.predictions_df
            assert np.isfinite(result.metrics["mae"])
            assert np.isnan(result.metrics["hit_rate_top_k"])

    def test_no_common_sample_preserves_failure_details(self) -> None:
        def factory(stocks: list[Stock], sentiments: dict) -> AveragePredictionStrategy:
            raise RuntimeError("cannot predict")

        results = PredictionBenchmark([_make_stock("A", 15)], n_eval_periods=1).run(
            [_ewm_config(), PredictorConfig("fails", factory)]
        )
        assert results[0].metrics["scored_pairs"] == 0
        assert np.isnan(results[0].metrics["mae"])
        assert len(results[1].failed_periods) == 1

    def test_predictor_mutations_do_not_contaminate_another_predictor(self) -> None:
        stock = _make_stock("A", 15)
        baseline = PredictionBenchmark([stock], n_eval_periods=2).run([_ewm_config()])[
            0
        ]

        def factory(stocks: list[Stock], sentiments: dict) -> AveragePredictionStrategy:
            stocks[0].data.loc[:, "returns"] = 0.5
            return AveragePredictionStrategy(stocks)

        result = PredictionBenchmark([stock], n_eval_periods=2).run(
            [PredictorConfig("mutates", factory), _ewm_config()]
        )[1]
        pd.testing.assert_frame_equal(baseline.predictions_df, result.predictions_df)

    def test_daily_next_session_does_not_insert_weekend_zero_returns(self) -> None:
        index = pd.date_range("2024-01-01", periods=15, freq="B", tz="UTC")
        frame = pd.DataFrame({"close": np.arange(100, 115)}, index=index)
        stock = Stock(
            "A",
            normalize_bars(
                frame,
                Frequency.DAILY,
                market_timezone="UTC",
                as_of="2030-01-01T00:00:00Z",
            ),
        )
        result = PredictionBenchmark(
            [stock], frequency=Frequency.DAILY, n_eval_periods=5, min_history=3
        ).run([_ewm_config()])[0]
        expected = stock.data["close"].pct_change(fill_method=None).iloc[-5:]
        np.testing.assert_allclose(result.actuals_df["A"], expected)
        assert result.metrics["scored_pairs"] == 5

    def test_unversioned_time_labels_are_rejected(self) -> None:
        stock = _make_stock("A", 15)
        stock.data = stock.data[["open", "high", "low", "close", "volume", "returns"]]
        with pytest.raises(ValueError, match="Missing bar time contract"):
            PredictionBenchmark([stock])

    @pytest.mark.parametrize(
        "overrides", [{"n_eval_periods": 0}, {"min_history": 0}, {"top_k": 0}]
    )
    def test_invalid_search_inputs_are_rejected(
        self, overrides: dict[str, int]
    ) -> None:
        with pytest.raises(ValueError, match="positive"):
            PredictionBenchmark([_make_stock("A", 15)], **overrides)


@pytest.mark.parametrize(
    "label",
    [
        "ZeroReturn",
        "HistoricalMean",
        "EWM12-Mean-25",
        "EWM12-Mean-50",
        "EWM12-Mean-75",
        "MomentumRidge",
        "Momentum12-2Ridge",
        "FuzzyThesis-fixed",
        "FuzzyThesis-adaptive",
        "FuzzyThesis-no-trend",
        "PooledRidge",
        "FuzzyRidge-fixed",
        "FuzzyRidge-adaptive",
    ],
)
def test_research_forecasts_are_unchanged_when_unobserved_target_prices_change(
    label, tmp_path
):
    from investment_tool.app.prediction_comparison_coordinator import (
        PredictionComparisonParameters,
        _build_predictor_configs,
    )
    from investment_tool.constants import DataSource

    params = PredictionComparisonParameters(
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=tmp_path,
        predictors=[label],
    )
    assets = _make_stocks(2, n_periods=40)
    before = PredictionBenchmark(assets, n_eval_periods=1, min_history=13).run(
        _build_predictor_configs(params)
    )[0]
    for asset in assets:
        asset.data.loc[asset.data.index[-1], ["open", "high", "low", "close"]] *= 20
    after = PredictionBenchmark(assets, n_eval_periods=1, min_history=13).run(
        _build_predictor_configs(params)
    )[0]
    pd.testing.assert_frame_equal(before.predictions_df, after.predictions_df)
    assert not before.actuals_df.equals(after.actuals_df)
    assert not before.failed_periods
