from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from investment_tool.app.strategy_comparison_coordinator import (
    StrategyComparisonParameters,
    compare_strategies,
)
from investment_tool.app.strategy_variant import StrategyVariant
from investment_tool.constants import DataSource, Frequency, StockFields, StrategyType
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.strategies.optimization_strategies.seeded_optimization import (
    MultiGreedyOptimizationStrategy,
)
from investment_tool.strategies.orchestrators.investment_strategy_optimization import (
    InvestmentStrategyOptimization,
)
from tests.strategies.orchestrators.test_investment_strategy_optimization import (
    _make_instance,
    _make_stock,
)


@pytest.mark.parametrize("maximum", [None, 4])
def test_small_news_shortlist_is_rejected_before_loading_data(tmp_path, maximum):
    params = StrategyComparisonParameters(
        strategy_types=[],
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=tmp_path,
        max_assets=5,
        strategy_variants=[
            StrategyVariant(
                name="News",
                news_shortlist=True,
                news_top_assets=3,
                max_assets=maximum,
            )
        ],
    )
    with patch(
        "investment_tool.app.strategy_comparison_coordinator.ModelDataFactory.create_model_data"
    ) as load:
        with pytest.raises(ValueError, match="at least maximum assets"):
            compare_strategies(params)
        load.assert_not_called()


@pytest.mark.parametrize("different_estimators", [False, True])
def test_variants_share_search_only_when_forecasts_match_and_keep_separate_portfolios(
    tmp_path, monkeypatch, different_estimators
):
    dates = [datetime(2024, m, 1) for m in (1, 2, 3)]
    columns = pd.MultiIndex.from_product(
        [[StockFields.OPEN, StockFields.CLOSE], list("ABCD")]
    )
    periods = [
        Period(
            d,
            pd.DataFrame(
                [[100, 100, 100, 100, 100 + i, 110 + i, 120 + i, 130 + i]],
                index=[d],
                columns=columns,
            ),
        )
        for i, d in enumerate(dates)
    ]
    md = ModelData()
    md.periods = periods
    md.instance = _make_instance(dates[1])
    md.instance.initial_trading_period = md.instance.current_trading_period = periods[1]
    md.instance.previous_trading_period = periods[0]
    md.portfolio = Portfolio(md.instance, 1000.0)
    md.stocks = [_make_stock(s) for s in "ABCD"]
    md.news = {}
    monkeypatch.setattr(
        InvestmentStrategyOptimization,
        "_predict_returns",
        lambda strategy: dict(
            zip(
                "ABCD",
                np.array([0.01, 0.02, 0.03, 0.04])
                * (2 if strategy.return_estimator == "ewm_12" else 1),
            )
        ),
    )
    monkeypatch.setattr(
        InvestmentStrategyOptimization,
        "build_covariance_matrix",
        lambda _: pd.DataFrame(
            np.diag([0.01, 0.04, 0.09, 0.16]), index=list("ABCD"), columns=list("ABCD")
        ),
    )
    monkeypatch.setattr(
        MultiGreedyOptimizationStrategy,
        "_plot_optimization_results",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "investment_tool.strategies.optimization_strategies.seeded_optimization.convex_markowitz_front",
        lambda *a: [],
    )
    settings = {
        "n_restarts": 1,
        "n_generations": 60,
        "population_size": 20,
        "seed_max_subsets": 20,
        "search_seconds": None,
        "refinement_supports": 3,
        "seed_beam_width": 2,
    }
    params = StrategyComparisonParameters(
        strategy_types=[StrategyType.OPTIMIZATION],
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=tmp_path,
        include_benchmark=False,
        benchmark_asset="A",
        min_assets=2,
        max_assets=3,
        strategy_variants=[
            StrategyVariant(name="Reference", **settings),
            StrategyVariant(
                name="Target",
                selection="return_target",
                return_estimator="ewm_12" if different_estimators else None,
                **settings,
            ),
        ],
    )
    calls = []
    original = MultiGreedyOptimizationStrategy.optimize

    def counted(self, *args, **kwargs):
        calls.append(self.date_label)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(MultiGreedyOptimizationStrategy, "optimize", counted)
    partial = {}
    with patch(
        "investment_tool.app.strategy_comparison_coordinator.ModelDataFactory.create_model_data",
        return_value=md,
    ), patch(
        "investment_tool.app.strategy_comparison_coordinator.StrategyComparator"
    ) as comparator:
        comparator.return_value.run.return_value = {
            "metrics_df": None,
            "saved_files": [],
        }
        result = compare_strategies(params, partial_sink=partial)
    assert calls == [
        "2024-02-01",
        "2024-03-01",
    ] * (2 if different_estimators else 1)
    individual = result["individual_results"]
    assert set(individual) == {"Reference", "Target"}
    assert len(individual["Reference"]["portfolio_history"]) == 2
    assert len(individual["Target"]["portfolio_history"]) == 2
    for reference, target, diagnostic in zip(
        individual["Reference"]["results"],
        individual["Target"]["results"],
        individual["Target"]["selection_diagnostics"],
    ):
        assert (
            reference["optimization_diagnostics"]["shared_front_key"]
            == target["optimization_diagnostics"]["shared_front_key"]
        ) == (not different_estimators)
        assert not reference["optimization_diagnostics"]["reused_front"]
        assert target["optimization_diagnostics"]["reused_front"] == (
            not different_estimators
        )
        assert (
            diagnostic["selected_expected_return"]
            >= diagnostic["target_expected_return"]
        )
        assert 2 <= len(target["selected_assets"]) <= 3
    assert (
        partial["individual_results"]["Target"]["configuration"]["selection"]
        == "return_target"
    )
    configuration = individual["Target"]["configuration"]
    assert configuration["return_estimator"] == (
        "ewm_12" if different_estimators else "ewm_6"
    )
    assert configuration["population_size"] == 20
    assert configuration["n_generations"] == 60
    assert configuration["min_assets"] == 2
    assert configuration["max_assets"] == 3
    assert configuration["refinement_supports"] == 3
    assert configuration["seed_beam_width"] == 2
    assert (
        individual["Reference"]["results"][0]["optimization_diagnostics"]["refinement"][
            "subproblems"
        ]
        > 0
    )
    assert md.portfolio.positions == {}  # original and sibling portfolios were isolated
