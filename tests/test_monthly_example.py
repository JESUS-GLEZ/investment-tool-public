"""The documented small run uses the real request-to-engine configuration path."""

import json
from pathlib import Path

from api import adapters
from api.schemas import StrategyComparisonRequest
from investment_tool.constants import DataSource, Frequency, StrategyType
from investment_tool.data.universes.loader import load_asset_universe


def test_monthly_example_reaches_engine_without_inheriting_research_budgets(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples/monthly_backtest.json").read_text())
    # Top-level API requests accept extra fields for compatibility. Catch typos
    # here rather than silently replacing an example setting with a default.
    assert not payload.keys() - StrategyComparisonRequest.model_fields.keys()
    request = StrategyComparisonRequest.model_validate(payload)
    monkeypatch.setattr(adapters, "_OUTPUT_BASE", tmp_path)
    parameters = adapters.build_strategy_comparison_params(request, "example")

    assert parameters.data_source == DataSource.ALPACA
    assert parameters.frequency == Frequency.MONTHLY
    assert parameters.cfg is not None
    universe = load_asset_universe(parameters.cfg.asset_config_file)
    assert 3 <= len(universe.stocks) + len(universe.crypto) <= 10
    assert "GLD" in universe.stocks and "BTC/USD" in universe.crypto
    assert parameters.benchmark_asset in universe.stocks
    assert parameters.include_benchmark
    assert parameters.cfg.model.historical_start_date == "2022-01-01"
    assert parameters.cfg.model.historical_end_date == "2024-12-31"
    assert parameters.cfg.model.backtest_start_date == "2024-10-01"
    assert parameters.cfg.model.transaction_cost_rate == 0.001
    assert parameters.save_input_data
    assert parameters.strategy_types == []
    assert len(parameters.strategy_variants) == 1
    variant = parameters.strategy_variants[0]
    assert variant.strategy_type == StrategyType.OPTIMIZATION
    assert (variant.min_assets, variant.max_assets) == (2, 3)
    kwargs = variant.constructor_kwargs()
    assert kwargs["population_size"] == parameters.population_size == 40
    assert kwargs["n_generations"] == parameters.n_generations == 800
    assert kwargs["n_restarts"] == 1
    assert not kwargs["news_shortlist"]
    assert variant.search_settings() == {
        "random_seed": 0,
        "search_seconds": None,
        "seed_max_subsets": 64,
        "seed_beam_width": 3,
        "refinement_supports": 0,
    }
