import pytest
from pydantic import ValidationError

from api.adapters import build_strategy_comparison_params
from api.schemas import StrategyComparisonRequest
from api.routers.presets import PresetRequest


def test_pooled_discovery_budget_roundtrips_and_keeps_explicit_overrides():
    from investment_tool.defaults import STRATEGY_OPTIMIZER

    request = StrategyComparisonRequest(
        strategy_types=[],
        strategy_variants=[
            {"name": "Pooled default"},
            {"name": "Smaller saved budget", "seed_max_subsets": 30000},
        ],
    )
    assert STRATEGY_OPTIMIZER.seed_max_subsets == 360000
    assert request.strategy_variants[0].seed_max_subsets == 360000
    params = build_strategy_comparison_params(request, "pooled-test")
    assert [v.seed_max_subsets for v in params.strategy_variants] == [360000, 30000]
    preset = PresetRequest(
        name="Pooled",
        run_type="strategy_comparison",
        parameters=request.model_dump(mode="json"),
    )
    restored = StrategyComparisonRequest(**preset.parameters)
    assert restored.strategy_variants == request.strategy_variants
    with pytest.raises(ValidationError):
        StrategyComparisonRequest(
            strategy_types=[],
            strategy_variants=[{"name": "Over budget", "seed_max_subsets": 360001}],
        )


@pytest.mark.parametrize("balance", ["capital_weights", "hybrid"])
def test_named_variants_roundtrip_through_api_adapter_and_presets(balance):
    payload = {
        "strategy_types": [],
        "strategy_variants": [
            {
                "name": "Sharpe reference",
                "selection": "sharpe",
                "search_seconds": None,
                "random_seed": 123,
                "seed_beam_width": 7,
                "refinement_supports": 40,
            },
            {
                "name": "Return target",
                "return_estimator": "ewm_12_mean_50",
                "balance": balance,
                "news_shortlist": True,
                "news_top_assets": 50,
                "news_min_articles": 3,
                "selection": "return_target",
                "return_target_fraction": 0.75,
                "search_seconds": None,
                "random_seed": 123,
            },
        ],
    }
    request = StrategyComparisonRequest(**payload)
    params = build_strategy_comparison_params(request, "variants-test")
    assert params.strategy_types == []
    assert len(params.strategy_variants) == 2
    assert params.strategy_variants[0].return_estimator is None
    assert params.strategy_variants[1].return_estimator == "ewm_12_mean_50"
    assert params.strategy_variants[1].return_target_fraction == 0.75
    assert params.strategy_variants[1].balance == balance
    assert params.strategy_variants[1].news_shortlist
    assert params.strategy_variants[1].news_top_assets == 50
    assert params.strategy_variants[1].news_min_articles == 3
    assert params.strategy_variants[0].search_seconds is None
    assert params.strategy_variants[0].seed_beam_width == 7
    assert params.strategy_variants[0].refinement_supports == 40
    preset = PresetRequest(
        name="Paired", run_type="strategy_comparison", parameters=payload
    )
    assert StrategyComparisonRequest(**preset.parameters).model_dump(
        mode="json"
    ) == request.model_dump(mode="json")


@pytest.mark.parametrize(
    "variants",
    [
        [{"name": ""}],
        [{"name": ".."}],
        [{"name": "../outside"}],
        [{"name": "<script>"}],
        [{"name": "optimization"}],
        [{"name": "One"}, {"name": "one"}],
        [{"name": "X", "return_target_fraction": 1.1}],
        [{"name": "X", "search_seconds": 0}],
        [{"name": "X", "min_assets": 6}],
        [{"name": "X", "seed_max_subsets": 0}],
        [{"name": "X", "seed_beam_width": 0}],
        [{"name": "X", "refinement_supports": -1}],
        [{"name": "X", "refinement_supports": 501}],
        [{"name": "X", "return_estimator": "unknown"}],
        [{"name": "X", "balance": "unknown"}],
        [{"name": "X", "news_top_assets": 0}],
        [{"name": "X", "news_min_articles": 0}],
        [{"name": "X", "strategy_type": "combined", "news_shortlist": True}],
        [{"name": "X", "strategy_type": "sentiment_ewm", "balance": "capital_weights"}],
        [{"name": "X", "strategy_type": "combined", "return_estimator": "ewm_6"}],
        [
            {
                "name": "X",
                "strategy_type": "sentiment_ewm",
                "return_estimator": "ewm_12_mean_50",
                "balance": "capital_weights",
                "news_shortlist": True,
                "news_top_assets": 50,
                "news_min_articles": 3,
            }
        ],
        [{"name": "X", "population_sze": 100}],
        [{"name": "X", "strategy_type": "buy_and_hold", "asset_symbol": "<invalid>"}],
    ],
)
def test_invalid_variant_configuration_is_rejected(variants):
    with pytest.raises(ValidationError):
        StrategyComparisonRequest(strategy_types=[], strategy_variants=variants)


def test_no_strategy_or_variant_is_rejected():
    with pytest.raises(ValidationError):
        StrategyComparisonRequest(strategy_types=[])


@pytest.mark.parametrize("estimator", ["ewm_6", "ewm_12", "ewm_12_mean_50"])
def test_each_estimator_roundtrips_and_retains_separate_history_and_trading_dates(
    estimator,
):
    request = StrategyComparisonRequest(
        strategy_types=[],
        strategy_variants=[{"name": "Candidate", "return_estimator": estimator}],
        historical_start_date="2016-01-01",
        historical_end_date="2026-09-30",
        backtest_start_date="2023-10-01",
    )
    params = build_strategy_comparison_params(request, "estimator-test")
    assert params.strategy_variants[0].return_estimator == estimator
    assert params.cfg.model.historical_start_date == "2016-01-01"
    assert params.cfg.model.backtest_start_date == "2023-10-01"


def test_cli_rejects_estimator_override_for_other_strategy_before_data_loading(
    tmp_path, monkeypatch
):
    from investment_tool.app.strategy_comparison_coordinator import (
        StrategyComparisonParameters,
        compare_strategies,
    )
    from investment_tool.app.strategy_variant import StrategyVariant
    from investment_tool.constants import StrategyType, DataSource, Frequency

    def unexpected_load(*args, **kwargs):
        pytest.fail("Must validate configuration before loading data")

    monkeypatch.setattr(
        "investment_tool.app.strategy_comparison_coordinator.ModelDataFactory.create_model_data",
        unexpected_load,
    )
    params = StrategyComparisonParameters(
        strategy_types=[],
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=tmp_path,
        strategy_variants=[
            StrategyVariant(
                name="Invalid",
                strategy_type=StrategyType.COMBINED,
                return_estimator="ewm_12",
            )
        ],
    )
    with pytest.raises(ValueError, match="only to Optimization"):
        compare_strategies(params)


@pytest.mark.parametrize("frequency", ["daily", "weekly", "monthly"])
def test_news_shortlist_accepts_every_configurable_frequency(frequency):
    request = StrategyComparisonRequest(
        frequency=frequency,
        strategy_types=[],
        strategy_variants=[{"name": "News", "news_shortlist": True}],
    )
    assert request.frequency == frequency


@pytest.mark.parametrize("maximum", [None, 4])
def test_news_shortlist_must_cover_inherited_or_overridden_maximum_assets(maximum):
    with pytest.raises(ValidationError, match="at least maximum assets"):
        StrategyComparisonRequest(
            strategy_types=[],
            max_assets=5,
            strategy_variants=[
                {
                    "name": "News",
                    "news_shortlist": True,
                    "news_top_assets": 3,
                    "max_assets": maximum,
                }
            ],
        )
    # The variant's own maximum takes precedence over the run default.
    request = StrategyComparisonRequest(
        strategy_types=[],
        max_assets=5,
        strategy_variants=[
            {
                "name": "News",
                "news_shortlist": True,
                "news_top_assets": 3,
                "max_assets": 3,
            }
        ],
    )
    assert request.strategy_variants[0].news_top_assets == 3


@pytest.mark.parametrize("kind", ["optimization", "combined", "sentiment_ewm"])
def test_single_and_comparison_build_the_same_configured_strategy(
    kind, tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from api.adapters import build_exec_params
    from api.schemas import ExecutionRequest
    from investment_tool.app.execution_coordinator import _build_strategy
    from tests.strategies.orchestrators.test_investment_strategy_optimization import (
        _make_instance,
        _make_stock,
    )

    monkeypatch.setattr("api.adapters._OUTPUT_BASE", tmp_path)
    variant = {
        "name": "Candidate",
        "strategy_type": kind,
        "min_assets": 3,
        "max_assets": 4,
        "population_size": 80,
        "n_generations": 4321,
        "n_restarts": 7,
        "random_seed": 29,
        "search_seconds": None,
        "seed_max_subsets": 2500,
        "seed_beam_width": 8,
        "refinement_supports": 40,
        "selection": "return_target",
        "return_target_fraction": 0.75,
    }
    single = ExecutionRequest(strategy_type=kind, strategy_configuration=variant)
    execution = build_exec_params(single, "single")
    comparison = build_strategy_comparison_params(
        StrategyComparisonRequest(strategy_types=[], strategy_variants=[variant]),
        "paired",
    )
    preset = PresetRequest(
        name="Single candidate",
        run_type="execution",
        parameters=single.model_dump(mode="json"),
    )
    assert (
        preset.parameters["strategy_configuration"]
        == single.model_dump(mode="json")["strategy_configuration"]
    )
    md = SimpleNamespace(
        instance=_make_instance(),
        stocks=[_make_stock(s) for s in ["A", "B", "C", "D"]],
        news=None,
    )
    first = _build_strategy(
        execution.strategy_type,
        md,
        tmp_path,
        configuration=execution.strategy_configuration,
    )
    second = _build_strategy(
        comparison.strategy_variants[0].strategy_type,
        md,
        tmp_path,
        population_size=comparison.population_size,
        n_generations=comparison.n_generations,
        configuration=comparison.strategy_variants[0],
    )
    for strategy in (first, second):
        assert strategy.min_assets == 3
        assert strategy.max_assets == 4
        assert strategy.population_size == 80
        assert strategy.n_generations == 4321
        assert strategy.n_restarts == 7
        assert strategy._selector.fraction == 0.75
        assert strategy.optimization_settings == {
            "random_seed": 29,
            "search_seconds": None,
            "seed_max_subsets": 2500,
            "seed_beam_width": 8,
            "refinement_supports": 40,
        }
    assert first._selector is not second._selector


@pytest.mark.parametrize(
    "variant",
    [
        {"name": "Mismatch", "strategy_type": "combined"},
        {"name": "Bounds", "min_assets": 6},
        {"name": "Shortlist", "news_shortlist": True, "news_top_assets": 3},
    ],
)
def test_single_run_rejects_incompatible_configuration(variant):
    from api.schemas import ExecutionRequest

    with pytest.raises(ValidationError):
        ExecutionRequest(strategy_type="optimization", strategy_configuration=variant)


@pytest.mark.parametrize("frequency", ["daily", "weekly", "monthly"])
def test_single_run_accepts_news_shortlist_at_every_frequency(frequency):
    from api.schemas import ExecutionRequest

    request = ExecutionRequest(
        strategy_type="optimization",
        frequency=frequency,
        strategy_configuration={"name": "News", "news_shortlist": True},
    )
    assert request.frequency == frequency


@pytest.mark.parametrize(
    "variant",
    [
        {"name": "Wrong symbol", "asset_symbol": "SPY"},
        {"name": "Wrong momentum", "top_n": 5},
        {
            "name": "Bad window",
            "strategy_type": "momentum_risk_parity",
            "vol_window": 1,
        },
        {
            "name": "Bad weight",
            "strategy_type": "momentum_risk_parity",
            "min_weight": 1,
        },
    ],
)
def test_rejects_inapplicable_baseline_settings(variant):
    with pytest.raises(ValidationError):
        StrategyComparisonRequest(strategy_types=[], strategy_variants=[variant])


def test_baseline_variants_use_existing_factory_without_optimizer_settings(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from api.adapters import build_exec_params
    from api.schemas import ExecutionRequest
    from investment_tool.app.execution_coordinator import _build_strategy
    from tests.strategies.orchestrators.test_investment_strategy_optimization import (
        _make_instance,
        _make_stock,
    )

    monkeypatch.setattr("api.adapters._OUTPUT_BASE", tmp_path)
    md = SimpleNamespace(
        instance=_make_instance(),
        stocks=[_make_stock("SPY"), _make_stock("QQQ")],
        news=None,
    )
    buy = {
        "name": "QQQ baseline",
        "strategy_type": "buy_and_hold",
        "asset_symbol": "QQQ",
    }
    momentum = {
        "name": "Momentum five",
        "strategy_type": "momentum_risk_parity",
        "top_n": 5,
        "lookback": 12,
        "vol_window": 8,
        "trend_window": 10,
        "min_weight": 0.03,
    }
    for payload in (buy, momentum):
        params = build_exec_params(
            ExecutionRequest(
                strategy_type=payload["strategy_type"], strategy_configuration=payload
            ),
            payload["name"],
        )
        strategy = _build_strategy(
            params.strategy_type,
            md,
            tmp_path,
            configuration=params.strategy_configuration,
        )
        assert strategy.optimization_settings == {}
        assert strategy.candidate_front_cache is None
        if payload is buy:
            assert strategy._asset_symbol == "QQQ"
            assert strategy.run_strategy()["QQQ"] == 1
        else:
            assert strategy.top_n == 5
            assert strategy.lookback == 12
            assert strategy.vol_window == 8
            assert strategy.trend_window == 10
            assert strategy.min_weight == 0.03


def test_comparison_retains_multiple_baselines_and_the_separate_benchmark(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from investment_tool.app.strategy_comparison_coordinator import compare_strategies
    from tests.strategies.orchestrators.test_investment_strategy_optimization import (
        _make_instance,
        _make_stock,
    )

    req = StrategyComparisonRequest(
        strategy_types=[],
        include_benchmark=True,
        strategy_variants=[
            {
                "name": "QQQ reference",
                "strategy_type": "buy_and_hold",
                "asset_symbol": "QQQ",
            },
            {
                "name": "SPY reference",
                "strategy_type": "buy_and_hold",
                "asset_symbol": "SPY",
            },
            {
                "name": "Momentum five",
                "strategy_type": "momentum_risk_parity",
                "top_n": 5,
            },
        ],
    )
    params = build_strategy_comparison_params(req, "baseline-comparison")
    params.output_directory = tmp_path
    md = SimpleNamespace(
        instance=_make_instance(),
        stocks=[_make_stock("SPY"), _make_stock("QQQ")],
        news=None,
        input_snapshot_path=None,
        validation_report={},
        portfolio=SimpleNamespace(history=[]),
    )
    module = "investment_tool.app.strategy_comparison_coordinator"
    with patch(f"{module}.ModelDataFactory.create_model_data", return_value=md), patch(
        f"{module}.BacktestingService"
    ) as service, patch(f"{module}.StrategyComparator") as comparator:
        service.return_value.run_backtest.return_value.results = []
        comparator.return_value.run.return_value = {
            "metrics_df": None,
            "saved_files": [],
        }
        result = compare_strategies(params)
    assert list(result["individual_results"]) == [
        "QQQ reference",
        "SPY reference",
        "Momentum five",
    ]
    assert (
        result["individual_results"]["QQQ reference"]["configuration"]["asset_symbol"]
        == "QQQ"
    )
    assert result["individual_results"]["Momentum five"]["configuration"]["top_n"] == 5
    momentum_config = result["individual_results"]["Momentum five"]["configuration"]
    assert momentum_config["lookback"] == 6
    assert momentum_config["min_assets"] is None
    assert momentum_config["max_assets"] is None
    assert comparator.call_args.kwargs["benchmark_name"] == "SPY reference"


@pytest.mark.parametrize("include_benchmark", [False, True])
def test_migrated_legacy_buy_and_hold_keeps_asset_and_comparison_reference(
    include_benchmark, tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import patch
    from investment_tool.app.strategy_comparison_coordinator import compare_strategies
    from tests.strategies.orchestrators.test_investment_strategy_optimization import (
        _make_instance,
        _make_stock,
    )

    monkeypatch.setattr("api.adapters._OUTPUT_BASE", tmp_path)
    legacy = StrategyComparisonRequest(
        strategy_types=["combined", "buy_and_hold"],
        benchmark_asset="QQQ",
        include_benchmark=include_benchmark,
    )
    migrated = StrategyComparisonRequest(
        strategy_types=[],
        benchmark_asset="QQQ",
        include_benchmark=True,
        strategy_variants=[
            {"name": "Combined 1", "strategy_type": "combined"},
            {
                "name": "Buy & Hold 1",
                "strategy_type": "buy_and_hold",
                "asset_symbol": "QQQ",
            },
        ],
    )
    md = SimpleNamespace(
        instance=_make_instance(),
        stocks=[_make_stock("SPY"), _make_stock("QQQ")],
        news=None,
        input_snapshot_path=None,
        validation_report={},
        portfolio=SimpleNamespace(history=[]),
    )
    module = "investment_tool.app.strategy_comparison_coordinator"
    for request, expected_reference in (
        (legacy, "buy_and_hold"),
        (migrated, "Buy & Hold 1"),
    ):
        params = build_strategy_comparison_params(request, expected_reference)
        with patch(
            f"{module}.ModelDataFactory.create_model_data", return_value=md
        ), patch(f"{module}.BacktestingService") as service, patch(
            f"{module}.StrategyComparator"
        ) as comparator:
            service.return_value.run_backtest.return_value.results = []
            comparator.return_value.run.return_value = {
                "metrics_df": None,
                "saved_files": [],
            }
            result = compare_strategies(params)
        # The first strategy must not silently become the reference after migration.
        assert len(result["individual_results"]) == 2
        assert len(service.call_args_list) == 2
        assert (
            service.call_args_list[1].kwargs["investment_strategy"]._asset_symbol
            == "QQQ"
        )
        assert comparator.call_args.kwargs["benchmark_name"] == expected_reference
