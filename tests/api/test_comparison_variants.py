"""Named configurations round-trip without creating algorithm implementations."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from api.adapters import (
    build_prediction_comparison_params,
    build_solver_comparison_params,
)
from api.schemas import PredictionComparisonRequest, SolverComparisonRequest
from api.routers.presets import PresetRequest
from investment_tool.app.prediction_comparison_coordinator import (
    _build_predictor_configs,
)
from investment_tool.app.solver_comparison_coordinator import _build_solver_configs
from investment_tool.output.run_metadata import write_run_metadata


def test_solver_instances_resolve_independent_budgets_and_shared_constraints(tmp_path):
    import json

    request = SolverComparisonRequest(
        population_size=40,
        n_generations=100,
        n_restarts=2,
        min_assets=2,
        num_assets=4,
        solver_variants=[
            {"name": "NSGA baseline", "solver": "nsga-ii", "random_seed": 7},
            {
                "name": "NSGA larger",
                "solver": "NSGA-II",
                "population_size": 80,
                "n_generations": 300,
                "n_restarts": 3,
                "mutation_prob": 0.45,
                "random_seed": 11,
            },
        ],
    )
    params = build_solver_comparison_params(request, "named-solvers-test")
    params.output_directory = tmp_path
    configs = _build_solver_configs(params)
    assert [c.label for c in configs] == ["NSGA baseline", "NSGA larger"]
    assert [c.population_size for c in configs] == [40, 80]
    assert [c.n_generations for c in configs] == [100, 300]
    assert [c.n_restarts for c in configs] == [2, 3]
    assert [c.random_seed for c in configs] == [7, 11]
    assert configs[1].algorithm_kwargs["variator"].mutation_prob == 0.45
    assert configs[0].algorithm is configs[1].algorithm
    assert params.min_assets == 2 and params.num_assets == 4
    write_run_metadata(params)
    recorded = json.loads((tmp_path / "run_manifest.json").read_text())
    assert recorded["parameters"]["solver_variants"][1]["population_size"] == 80


def test_named_solver_specific_options_reach_existing_builders():
    params = build_solver_comparison_params(
        SolverComparisonRequest(
            solver_variants=[
                {"name": "Directions", "solver": "NSGA-III", "divisions_outer": 19},
                {"name": "Epsilon", "solver": "EPSNSGAII", "epsilons": [0.0001, 0.002]},
            ]
        ),
        "specific-solvers-test",
    )
    configs = _build_solver_configs(params)
    assert configs[0].algorithm_kwargs["divisions_outer"] == 19
    assert configs[1].algorithm_kwargs["epsilons"] == [0.0001, 0.002]


@pytest.mark.parametrize(
    "schema,field,key",
    [
        (SolverComparisonRequest, "solver_variants", "solver"),
        (PredictionComparisonRequest, "predictor_variants", "predictor"),
    ],
)
def test_names_must_be_unique_safe_artifact_labels(schema, field, key):
    method = "NSGA-II" if key == "solver" else "EWM-6"
    for name in ["", "../escape", "x/y", "x\\y", "\x00hidden", ".."]:
        with pytest.raises(ValidationError):
            schema(**{field: [{"name": name, key: method}]})
    with pytest.raises(ValidationError, match="unique"):
        schema(**{field: [{"name": " A ", key: method}, {"name": "a", key: method}]})


@pytest.mark.parametrize(
    "variant",
    [
        {"predictor": "EWM-6", "alpha": 10},
        {"predictor": "EWM-6", "span": 0},
        {"predictor": "RidgeVARX", "alpha": 0},
        {"predictor": "RidgeVARX", "alpha": float("inf")},
        {"predictor": "MomentumRidge", "min_train_history": 11},
        {"predictor": "PooledRidge", "lags": 15, "min_train_history": 12},
        {"predictor": "FuzzyRidge-adaptive", "trend": False},
        {"predictor": "FuzzyRidge-adaptive", "grid_points": 513},
        {"predictor": "HistoricalMean", "span": 3},
        {"predictor": "EWM-6", "typo": 1},
        {"predictor": "unknown"},
    ],
)
def test_rejects_ignored_or_invalid_predictor_parameters(variant):
    with pytest.raises(ValidationError):
        PredictionComparisonRequest(predictor_variants=[{"name": "Invalid", **variant}])


@pytest.mark.parametrize(
    "variant",
    [
        {"solver": "NSGA-II", "population_size": 0},
        {"solver": "NSGA-II", "n_generations": 1.5},
        {"solver": "NSGA-II", "mutation_prob": 1.1},
        {"solver": "NSGA-II", "random_seed": 4294967296},
        {"solver": "NSGA-II", "divisions_outer": 19},
        {"solver": "NSGA-II", "epsilons": [0.01, 0.01]},
        {"solver": "EPSNSGAII", "epsilons": [0.01]},
        {"solver": "unknown"},
        {"solver": "NSGA-II", "mutation_prob": float("nan")},
    ],
)
def test_rejects_ignored_or_invalid_solver_parameters(variant):
    with pytest.raises(ValidationError):
        SolverComparisonRequest(solver_variants=[{"name": "Invalid", **variant}])


@pytest.mark.parametrize(
    "run_type,schema,field,variants",
    [
        (
            "solver_comparison",
            SolverComparisonRequest,
            "solver_variants",
            [
                {"name": "First", "solver": "NSGA-II", "random_seed": 1},
                {"name": "Second", "solver": "NSGA-II", "n_generations": 100},
            ],
        ),
        (
            "prediction_comparison",
            PredictionComparisonRequest,
            "predictor_variants",
            [
                {"name": "Fast", "predictor": "EWM-6", "span": 3},
                {"name": "Slow", "predictor": "EWM-6", "span": 12},
            ],
        ),
    ],
)
def test_named_preset_json_roundtrip(run_type, schema, field, variants):
    payload = {field: variants}
    preset = PresetRequest(name="Named", run_type=run_type, parameters=payload)
    request = schema(**payload)
    assert schema(**preset.parameters).model_dump(mode="json") == request.model_dump(
        mode="json"
    )


def test_predictor_parameters_reach_existing_implementations():
    request = PredictionComparisonRequest(
        predictor_variants=[
            {"name": "Fast", "predictor": "EWM-6", "span": 2},
            {"name": "Slow", "predictor": "EWM-6", "span": 12},
            {"name": "Ridge", "predictor": "RidgeVARX", "alpha": 2.5, "lag_order": 5},
            {
                "name": "Blend",
                "predictor": "EWM12-Mean-50",
                "momentum_weight": 0.8,
                "span": 9,
            },
            {
                "name": "Momentum",
                "predictor": "MomentumRidge",
                "alpha": 3,
                "min_train_history": 18,
            },
            {
                "name": "Fuzzy",
                "predictor": "FuzzyThesis-adaptive",
                "partitions": 5,
                "sigma": 4,
                "beta": 1,
                "lags": 2,
                "grid_points": 101,
                "trend": False,
            },
            {
                "name": "Pooled",
                "predictor": "PooledRidge",
                "alpha": 4,
                "lags": 5,
                "min_train_history": 18,
            },
        ]
    )
    params = build_prediction_comparison_params(request, "predictor-settings-test")
    configs = _build_predictor_configs(params)
    models = [c.factory([], {}) for c in configs]
    assert [m.span for m in models[:2]] == [2, 12]
    assert type(models[0]) is type(models[1])
    assert models[2].alpha == 2.5 and models[2].lag_order == 5
    assert models[3].momentum_weight == 0.8 and models[3].span == 9
    assert models[4].alpha == 3 and models[4].min_train_history == 18
    assert models[5].settings.partitions == 5 and not models[5].settings.trend
    assert models[6].settings.lags == 5 and models[6].alpha == 4


def test_custom_momentum_is_still_monthly_only():
    with pytest.raises(ValidationError, match="monthly"):
        PredictionComparisonRequest(
            frequency="weekly",
            predictor_variants=[
                {"name": "Momentum", "predictor": "MomentumRidge", "alpha": 3}
            ],
        )


def test_custom_forecasts_share_targets_and_keep_distinct_results(tmp_path):
    from investment_tool.constants import Frequency
    from investment_tool.domain.market_time import normalize_bars
    from investment_tool.domain.model_data.entities.stock import Stock
    from investment_tool.strategies.prediction_strategies.prediction_comparison.prediction_benchmark import (
        PredictionBenchmark,
    )

    dates = pd.date_range("2021-01-01", periods=40, freq="MS", tz="UTC")
    values = np.exp(np.cumsum(np.random.default_rng(7).normal(0.01, 0.08, 40))) * 100
    frame = pd.DataFrame(
        {
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "volume": 1000,
        },
        index=dates,
    )
    stock = Stock(
        "SYNTH",
        normalize_bars(
            frame,
            Frequency.MONTHLY,
            market_timezone="UTC",
            as_of="2030-01-01T00:00:00Z",
        ),
    )
    params = build_prediction_comparison_params(
        PredictionComparisonRequest(
            predictor_variants=[
                {"name": "Fast", "predictor": "EWM-6", "span": 2},
                {"name": "Slow", "predictor": "EWM-6", "span": 12},
            ]
        ),
        "paired-target-test",
    )
    benchmark = PredictionBenchmark(
        [stock], n_eval_periods=4, min_history=12, output_directory=tmp_path
    )
    results = benchmark.run(_build_predictor_configs(params))
    pd.testing.assert_frame_equal(results[0].actuals_df, results[1].actuals_df)
    assert results[0].predictions_df.index.equals(results[1].predictions_df.index)
    assert not np.allclose(results[0].predictions_df, results[1].predictions_df)
    assert list(benchmark.summary_table().index) == ["Fast", "Slow"]
