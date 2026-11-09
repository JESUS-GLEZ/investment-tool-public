"""Unit tests for prediction_comparison_coordinator (app layer)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from investment_tool.app.prediction_comparison_coordinator import (
    PredictionComparisonParameters,
    _build_predictor_configs,
    get_available_predictors,
)
from investment_tool.constants import DataSource, Frequency


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_params(**overrides: object) -> PredictionComparisonParameters:
    from pathlib import Path

    defaults = dict(
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=Path("/tmp/test"),
        n_eval_periods=6,
        min_history=6,
        top_k=3,
        predictors=["EWM-3", "EWM-6"],
    )
    defaults.update(overrides)
    return PredictionComparisonParameters(**defaults)  # type: ignore[arg-type]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestGetAvailablePredictors:
    def test_returns_list(self) -> None:
        result = get_available_predictors()
        assert isinstance(result, list)
        assert len(result) >= 4

    def test_contains_expected(self) -> None:
        result = get_available_predictors()
        for expected in ["EWM-3", "EWM-6", "EWM-12", "RIDGEVARX"]:
            assert expected in result


class TestBuildPredictorConfigs:
    def test_builds_requested(self) -> None:
        params = _make_params(predictors=["EWM-3", "EWM-6"])
        configs = _build_predictor_configs(params)
        assert len(configs) == 2
        labels = {c.label for c in configs}
        assert "EWM-3" in labels
        assert "EWM-6" in labels

    def test_unknown_raises(self) -> None:
        params = _make_params(predictors=["NONEXISTENT"])
        with pytest.raises(ValueError, match="Unknown predictor"):
            _build_predictor_configs(params)

    def test_case_insensitive(self) -> None:
        params = _make_params(predictors=["ewm-3", "ewm-6"])
        configs = _build_predictor_configs(params)
        assert len(configs) == 2


class TestPredictionComparisonParameters:
    def test_defaults(self) -> None:
        from pathlib import Path

        p = PredictionComparisonParameters(
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=Path("/tmp/test"),
        )
        assert p.n_eval_periods == 12
        assert p.min_history == 12
        assert p.top_k == 5
        assert p.compute_sentiment is True
        assert len(p.predictors) == 4

    def test_custom(self) -> None:
        from pathlib import Path

        p = PredictionComparisonParameters(
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=Path("/tmp/test"),
            predictors=["EWM-12"],
            n_eval_periods=6,
        )
        assert p.predictors == ["EWM-12"]
        assert p.n_eval_periods == 6


def test_execute_filters_quote_only_assets_and_uses_declared_frequency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd
    import investment_tool.app.prediction_comparison_coordinator as coordinator
    from investment_tool.config import AppConfig
    from investment_tool.data.universes.models import AssetUniverse
    from investment_tool.domain.market_time import normalize_bars
    from investment_tool.domain.model_data.model_data import ModelData
    from investment_tool.domain.model_data.entities.stock import Stock

    universe = AssetUniverse(
        "test.json",
        "test",
        ["A"],
        [],
        minimum_return_periods=3,
        valuation_symbols=("B",),
    )
    monkeypatch.setattr(coordinator, "load_asset_universe", lambda file: universe)
    model = ModelData()
    model.news = {}
    frame = pd.DataFrame(
        {"close": range(100, 115)},
        index=pd.date_range("2022-01-01", periods=15, freq="MS", tz="UTC"),
    )
    bars = normalize_bars(
        frame, Frequency.MONTHLY, market_timezone="UTC", as_of="2030-01-01T00:00:00Z"
    )
    model.stocks = [Stock("A", bars), Stock("B", bars)]
    calls: list[dict[str, Any]] = []

    def load(**kwargs: Any) -> ModelData:
        calls.append(kwargs)
        return model

    monkeypatch.setattr(coordinator.ModelDataFactory, "create_model_data", load)
    params = _make_params(
        cfg=AppConfig(), output_directory=tmp_path, n_eval_periods=2, min_history=2
    )
    output = coordinator.execute_prediction_comparison(params)
    assert calls[0]["frequency"] == Frequency.MONTHLY
    assert calls[0]["cfg"] is params.cfg
    for result in output["results"]:
        assert list(result.predictions_df.columns) == ["A"]
        assert result.metrics["history_bars"] == 4
        assert result.metrics["scored_pairs"] == 2


def test_dated_universe_is_not_silently_evaluated_as_static(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import investment_tool.app.prediction_comparison_coordinator as coordinator
    from investment_tool.config import AppConfig
    from investment_tool.data.universes.models import AssetUniverse, DatedMembership

    universe = AssetUniverse(
        "dated.json",
        "dated",
        ["A"],
        [],
        membership=DatedMembership(12, 100, frozenset(), ()),
    )
    monkeypatch.setattr(coordinator, "load_asset_universe", lambda file: universe)

    def unexpected_load(**kwargs: Any) -> None:
        pytest.fail("Dated membership must be rejected before loading data")

    monkeypatch.setattr(
        coordinator.ModelDataFactory, "create_model_data", unexpected_load
    )
    params = _make_params(cfg=AppConfig(), output_directory=tmp_path)
    with pytest.raises(ValueError, match="does not yet support dated universe"):
        coordinator.execute_prediction_comparison(params)


def test_research_predictors_are_opt_in_and_share_registered_factories(tmp_path):
    labels = [
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
    ]
    params = _make_params(output_directory=tmp_path)
    params.predictors = labels
    configs = _build_predictor_configs(params)
    assert [config.label for config in configs] == labels
    assert all(config.factory([], {}).predict_returns() == {} for config in configs)
    defaults = PredictionComparisonParameters(
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=tmp_path,
    )
    assert defaults.predictors == ["RidgeVARX", "EWM-3", "EWM-6", "EWM-12"]


@pytest.mark.parametrize("label", ["MomentumRidge", "Momentum12-2Ridge"])
@pytest.mark.parametrize("frequency", [Frequency.DAILY, Frequency.WEEKLY])
def test_month_based_signals_reject_other_frequencies_before_loading(
    label, frequency, monkeypatch
):
    import investment_tool.app.prediction_comparison_coordinator as coordinator

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid frequency must fail before loading data")

    monkeypatch.setattr(coordinator, "load_asset_universe", unexpected_load)
    with pytest.raises(ValueError, match="requires monthly data"):
        coordinator.execute_prediction_comparison(
            _make_params(predictors=[label], frequency=frequency)
        )
