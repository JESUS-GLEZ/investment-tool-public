"""Artifact policy cannot change search, selection, metrics or saved progress."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from api import adapters, schemas
from api.sanitize import sanitize
from api.result_presenter import _add_execution_figures
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.app.execution_coordinator import Parameters
from investment_tool.output.front_record import read_record, write_record
from investment_tool.output.output_manager import OutputManager
from investment_tool.output.policy import OutputMode
from investment_tool.output.run_metadata import (
    record_input_snapshot,
    write_run_metadata,
)
from investment_tool.output.strategy_comparator import StrategyComparator
from investment_tool.strategies.optimization_strategies.front_cache import (
    CandidateFrontCache,
)
from investment_tool.strategies.optimization_strategies.platypus_optimization import (
    PlatypusOptimizationStrategy,
)
from investment_tool.strategies.selection_strategies.sharpe_selection import (
    SharpeSelection,
)
from tests.output.test_output_manager import _make_results, _make_portfolio_history
from tests.output.test_strategy_comparator import _make_two_strategies
from tests.strategies.optimization_strategies.test_seeded_optimization import (
    make_optimizer,
)


@pytest.mark.parametrize(
    "schema, adapter",
    [
        (schemas.ExecutionRequest, adapters.build_exec_params),
        (schemas.StrategyComparisonRequest, adapters.build_strategy_comparison_params),
        (schemas.SolverComparisonRequest, adapters.build_solver_comparison_params),
        (
            schemas.PredictionComparisonRequest,
            adapters.build_prediction_comparison_params,
        ),
    ],
)
def test_mode_defaults_and_explicit_policy_reach_every_coordinator(
    tmp_path, monkeypatch, schema, adapter
):
    monkeypatch.setattr(adapters, "_OUTPUT_BASE", tmp_path)
    assert adapter(schema(), "standard").output_mode == OutputMode.STANDARD
    assert (
        adapter(schema(output_mode="diagnostic"), "diagnostic").output_mode
        == OutputMode.DIAGNOSTIC
    )
    with pytest.raises(ValidationError):
        schema(output_mode="none")


def test_both_modes_keep_identical_metrics_and_exports_and_ui_figures(tmp_path):
    outputs = []
    for mode in OutputMode:
        with patch(
            "investment_tool.output.plots.PlotBuilder.build_all", return_value=[]
        ) as export:
            output = OutputManager(
                tmp_path / mode.value, Frequency.MONTHLY, output_mode=mode
            ).generate_backtest_output(_make_results(), _make_portfolio_history())
            assert export.called == (mode == OutputMode.DIAGNOSTIC)
        outputs.append(output)
        assert (tmp_path / mode.value / "backtest_results.csv").exists()
        assert any("rebalancing" in p for p in output["saved_files"])
        enriched = _add_execution_figures(output)
        assert enriched["figures"]
    pd.testing.assert_frame_equal(outputs[0]["results_df"], outputs[1]["results_df"])
    assert sanitize(outputs[0]["metrics"]) == sanitize(outputs[1]["metrics"])
    assert not list((tmp_path / "standard").rglob("*.html"))
    assert not list((tmp_path / "standard").rglob("*.png"))


def test_standard_comparison_has_same_metrics_without_chart_exports(tmp_path):
    frames = []
    for mode in OutputMode:
        comparator = StrategyComparator(
            _make_two_strategies(),
            Frequency.MONTHLY,
            tmp_path / mode.value,
            output_mode=mode,
        )
        with patch.object(comparator, "_plot_equity_curves", return_value=[]) as plot:
            result = comparator.run()
            assert plot.called == (mode == OutputMode.DIAGNOSTIC)
        frames.append(result["metrics_df"])
        assert (tmp_path / mode.value / "comparison_metrics.csv").exists()
    pd.testing.assert_frame_equal(*frames)
    assert not list((tmp_path / "standard").rglob("*.html"))


def test_standard_cache_keeps_exact_selected_weights_and_mv_curve(tmp_path):
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    optimizer.include_markowitz_front = True
    optimizer.output_mode = OutputMode.STANDARD
    cache = CandidateFrontCache(tmp_path, "optimization", OutputMode.STANDARD)
    original = cache.optimize(optimizer, None)
    with patch.object(optimizer, "optimize", side_effect=AssertionError("must reuse")):
        reused = cache.optimize(optimizer, None)
    np.testing.assert_array_equal(
        [list(s.variables) for s in original.front],
        [list(s.variables) for s in reused.front],
    )
    selector = SharpeSelection()
    pd.testing.assert_series_equal(
        selector.select(original.front, optimizer.expected_returns.index),
        selector.select(reused.front, optimizer.expected_returns.index),
    )
    np.testing.assert_array_equal(
        [list(s.objectives) for s in original.markowitz_front],
        reused.reference_objectives,
    )
    assert reused.markowitz_front is None  # No invented executable MV portfolios.
    assert reused.all_archives == []
    record = read_record(next(tmp_path.glob("*.json.gz")))
    assert "archives" not in record and "reference" not in record
    assert record["front"] and record["assets"] and record["diagnostics"]
    assert all(s.feasible for s in reused.front)
    reused.selected_solution = reused.front[0]
    with patch(
        "investment_tool.output.front_plot.plot_solutions_with_markowitz_front"
    ) as plot:
        optimizer.replot_with_selection(reused)
        assert (
            plot.call_args.kwargs["reference_objectives"] == reused.reference_objectives
        )


def test_standard_and_diagnostic_search_and_selection_are_identical(tmp_path):
    results = []
    for mode in OutputMode:
        optimizer = make_optimizer()
        optimizer.search_seconds = None
        optimizer.output_mode = mode
        results.append(
            CandidateFrontCache(tmp_path / mode.value, "optimization", mode).optimize(
                optimizer, None
            )
        )
    np.testing.assert_array_equal(
        [list(s.variables) for s in results[0].front],
        [list(s.variables) for s in results[1].front],
    )
    np.testing.assert_array_equal(
        [list(s.objectives) for s in results[0].front],
        [list(s.objectives) for s in results[1].front],
    )
    assert read_record(next((tmp_path / "diagnostic").glob("*.json.gz")))["archives"]


def test_standard_renders_only_once_after_selection(tmp_path):
    optimizer = make_optimizer()
    optimizer.output_mode = OutputMode.STANDARD
    optimizer.search_seconds = None
    optimizer.output_directory = tmp_path
    # Restore the real method replaced by the helper fixture.
    optimizer._plot_optimization_results = (
        PlatypusOptimizationStrategy._plot_optimization_results.__get__(optimizer)
    )
    with patch(
        "investment_tool.output.front_plot.plot_solutions_with_markowitz_front"
    ) as plot:
        result = optimizer.optimize()
        plot.assert_not_called()
        result.selected_solution = result.front[0]
        optimizer.replot_with_selection(result)
        plot.assert_called_once()
        assert plot.call_args.args[2] == "pareto_front_selected"


def test_manifest_identifies_sources_inputs_and_universe_without_credentials(tmp_path):
    dataset = tmp_path / "prices.pkl"
    dataset.write_bytes(b"a saved immutable market dataset")
    params = Parameters(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        tmp_path,
        input_data_path=dataset,
    )
    write_run_metadata(params)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["parameters"]["output_mode"] == "standard"
    assert manifest["input_dataset"]["bytes"] == dataset.stat().st_size
    assert len(manifest["input_dataset"]["sha256"]) == 64
    assert len(manifest["source_sha256"]) == 64
    assert manifest["asset_universe"]["sha256"]
    assert "cfg" not in manifest["parameters"]
    assert set(manifest) == {
        "format_version",
        "parameters",
        "model_configuration",
        "asset_universe",
        "input_dataset",
        "source_sha256",
        "package_versions",
    }


def test_compression_and_legacy_read_preserve_full_precision(tmp_path):
    record = {
        "front": [
            {
                "weights": {"0": 0.12345678901234567},
                "objectives": [1e-12, -0.012345678901234567],
            }
        ]
        * 1000
    }
    compressed = tmp_path / "front.json.gz"
    legacy = tmp_path / "front.json"
    legacy.write_text(json.dumps(record))
    write_record(compressed, record)
    assert read_record(compressed) == read_record(legacy) == record
    assert compressed.stat().st_size < legacy.stat().st_size / 10


def test_fetched_snapshot_identity_is_added_without_copying_data(tmp_path):
    directory = tmp_path / "run"
    snapshot = tmp_path / "shared-input.pkl"
    snapshot.write_bytes(b"shared market and news snapshot")
    parameters = Parameters(
        DataSource.ALPACA, ExecutionMode.BACKTESTING, Frequency.MONTHLY, directory
    )
    write_run_metadata(parameters)
    record_input_snapshot(directory, snapshot)
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["input_dataset"]["path"] == str(snapshot)
    assert manifest["input_dataset"]["bytes"] == snapshot.stat().st_size
    assert list(directory.iterdir()) == [directory / "run_manifest.json"]


def test_standard_hybrid_suppresses_both_internal_optimizer_plot_suites():
    from investment_tool.strategies.optimization_strategies.hybrid_optimization import (
        HybridOptimizationStrategy,
    )
    from investment_tool.strategies.optimization_strategies.model.bprc import (
        BalancedPortfoliosRiskContributions,
    )

    source = make_optimizer()
    optimizer = HybridOptimizationStrategy(
        source.expected_returns,
        source.covariance_matrix,
        risk_aversion=1.0,
        model=BalancedPortfoliosRiskContributions,
    )
    optimizer.output_mode = OutputMode.STANDARD
    module = "investment_tool.strategies.optimization_strategies.hybrid_optimization"
    with patch(module + ".PlatypusOptimizationStrategy") as ea, patch(
        module + ".DecompositionOptimizationStrategy"
    ) as polish:
        optimizer._run_ea_phase()
        optimizer._run_polishing_phase([], None)
        assert ea.return_value.output_mode == OutputMode.STANDARD
        assert polish.return_value.output_mode == OutputMode.STANDARD


def test_nonshared_strategy_also_retains_compact_final_front(tmp_path):
    from investment_tool.strategies.orchestrators.investment_strategy_optimization import (
        InvestmentStrategyOptimization,
    )
    from tests.strategies.orchestrators.test_investment_strategy_optimization import (
        _make_instance,
        _make_stock,
    )

    strategy = InvestmentStrategyOptimization(
        _make_instance(),
        [_make_stock(s) for s in "ABCD"],
        news=None,
        output_directory=tmp_path,
    )
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    optimizer.expected_returns.index = list("ABCD")
    optimizer.covariance_matrix.index = list("ABCD")
    optimizer.covariance_matrix.columns = list("ABCD")
    result = strategy.optimize_portfolios(optimizer)
    record = read_record(next((tmp_path / "candidate_fronts").glob("*.json.gz")))
    assert record["assets"] == list("ABCD")
    assert len(record["front"]) == len(result.front)
    assert "archives" not in record
    assert result.search_diagnostics["candidate_front_file"]
