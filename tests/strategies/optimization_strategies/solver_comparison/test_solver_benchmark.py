"""Unit tests for SolverBenchmark.

Uses lightweight settings (tiny population, few generations) so tests
complete in seconds.
"""

from __future__ import annotations

import pandas as pd
import pytest
from platypus import NSGAII

from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    PortfolioVariator,
)
from investment_tool.strategies.optimization_strategies.custom_algorithm.gwasfga import (
    GWASFGA,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.solver_benchmark import (
    SolverBenchmark,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.solver_config import (
    SolverConfig,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

SYMBOLS = ["A", "B", "C"]
_MU = pd.Series([0.01, 0.02, 0.015], index=SYMBOLS)
_COV = pd.DataFrame(
    [
        [0.04, 0.005, 0.002],
        [0.005, 0.09, 0.003],
        [0.002, 0.003, 0.01],
    ],
    index=SYMBOLS,
    columns=SYMBOLS,
)


def _quick_config(label: str, algo: type = NSGAII) -> SolverConfig:
    """Build a minimal SolverConfig for fast tests."""
    return SolverConfig(
        label=label,
        algorithm=algo,
        population_size=20,
        n_generations=200,
        n_restarts=1,
    )


def _quick_gwasfga_config() -> SolverConfig:
    """GWASFGA with PortfolioVariator for BPRC model."""
    return SolverConfig(
        label="GWASFGA",
        algorithm=GWASFGA,
        algorithm_kwargs={
            "variator": PortfolioVariator(min_assets=1, max_assets=3),
        },
        population_size=20,
        n_generations=200,
        n_restarts=1,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSolverBenchmarkInit:
    """Test benchmark construction."""

    def test_basic_init(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=BalancedPortfoliosRiskContributions,
            model_kwargs={"l_assets": 1, "u_assets": 3},
        )
        assert bench.results == []

    def test_init_with_reference(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=BalancedPortfoliosRiskContributions,
            model_kwargs={"l_assets": 1, "u_assets": 3},
            reference_front=[],
        )
        assert bench.reference_front == []


class TestSolverBenchmarkRun:
    """Test the run method."""

    def test_raises_on_empty_solvers(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=BalancedPortfoliosRiskContributions,
            model_kwargs={"l_assets": 1, "u_assets": 3},
        )
        with pytest.raises(ValueError, match="At least one"):
            bench.run([])

    def test_single_solver(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=BalancedPortfoliosRiskContributions,
            model_kwargs={"l_assets": 1, "u_assets": 3},
        )
        results = bench.run([_quick_gwasfga_config()])
        assert len(results) == 1
        assert results[0].config.label == "GWASFGA"

    def test_multiple_solvers(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        configs = [
            _quick_config("NSGA-II-A", NSGAII),
            _quick_config("NSGA-II-B", NSGAII),
        ]
        results = bench.run(configs)
        assert len(results) == 2
        assert results[0].config.label == "NSGA-II-A"
        assert results[1].config.label == "NSGA-II-B"

    def test_result_has_front_and_metrics(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        results = bench.run([_quick_config("NSGA-II")])
        r = results[0]

        assert len(r.front) > 0
        assert r.wall_time_seconds > 0
        assert "hypervolume" in r.metrics
        assert "best_sharpe" in r.metrics
        assert "cardinality" in r.metrics

    def test_best_weights_valid(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        results = bench.run([_quick_config("NSGA-II")])
        w = results[0].best_weights

        assert isinstance(w, pd.Series)
        assert abs(w.sum() - 1.0) < 1e-4
        assert (w >= -1e-8).all()
        assert list(w.index) == SYMBOLS

    def test_wall_time_recorded(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        results = bench.run([_quick_config("NSGA-II")])
        assert results[0].wall_time_seconds > 0
        assert results[0].metrics["wall_time"] > 0


class TestSolverBenchmarkSummary:
    """Test summary_table and save_comparison."""

    def test_summary_table(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        bench.run(
            [
                _quick_config("A"),
                _quick_config("B"),
            ]
        )
        table = bench.summary_table()

        assert isinstance(table, pd.DataFrame)
        assert list(table.index) == ["A", "B"]
        assert "hypervolume" in table.columns
        assert "wall_time_s" in table.columns

    def test_save_comparison(self, tmp_path: object) -> None:
        from pathlib import Path

        out_dir = Path(str(tmp_path)) / "test_out"
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
            output_directory=out_dir,
        )
        bench.run([_quick_config("X")])
        saved = bench.save_comparison()

        assert len(saved) >= 1  # At least the CSV
        # Check CSV exists
        csv_files = [f for f in saved if f.endswith(".csv")]
        assert len(csv_files) == 1
        assert Path(csv_files[0]).exists()


class TestSolverBenchmarkProgressCallback:
    """Test that progress_callback is called correctly during run()."""

    def test_callback_called_per_solver(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        calls: list[tuple[float, str]] = []

        def cb(frac: float, msg: str) -> None:
            calls.append((frac, msg))

        configs = [_quick_config("A"), _quick_config("B")]
        bench.run(configs, progress_callback=cb)

        completed = [item for item in calls if "complete" in item[1]]
        assert [item[0] for item in completed] == pytest.approx([0.5, 1.0])
        assert calls[0][0] == 0

    def test_callback_single_solver(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        calls: list[tuple[float, str]] = []

        bench.run(
            [_quick_config("only")],
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        assert calls[-1][0] == pytest.approx(1.0)
        assert all("only" in message for _, message in calls)

    def test_callback_message_contains_label(self) -> None:
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        calls: list[tuple[float, str]] = []

        bench.run(
            [_quick_config("MySolver")],
            progress_callback=lambda f, m: calls.append((f, m)),
        )
        assert "MySolver" in calls[0][1]

    def test_no_callback_still_works(self) -> None:
        """run() without progress_callback should behave as before."""
        bench = SolverBenchmark(
            expected_returns=_MU,
            covariance_matrix=_COV,
            model=MarkowitzLongOnly,
        )
        results = bench.run([_quick_config("X")])
        assert len(results) == 1


def test_reports_evaluations_during_solver_run():
    benchmark = SolverBenchmark(_MU, _COV, MarkowitzLongOnly)
    updates = []
    benchmark.run(
        [_quick_config("NSGA-II")],
        progress_callback=lambda f, m: updates.append((f, m)),
    )
    assert updates[0][0] == 0
    assert any(
        "evaluations" in message and 0 < fraction < 1 for fraction, message in updates
    )
    assert updates[-1][0] == 1
    assert all(a[0] <= b[0] for a, b in zip(updates, updates[1:]))


def test_named_seeds_are_reproducible_isolated_and_independent_of_order():
    import random
    from dataclasses import replace
    import numpy as np

    first = replace(_quick_config("Same algorithm A"), random_seed=91, n_generations=60)
    second = replace(first, label="Same algorithm B", random_seed=17)
    random.seed(5)
    np.random.seed(5)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    bench = SolverBenchmark(
        expected_returns=_MU, covariance_matrix=_COV, model=MarkowitzLongOnly
    )
    original = bench.run([first, second])
    assert random.getstate() == python_before
    after = np.random.get_state()
    assert after[0] == numpy_before[0]
    np.testing.assert_array_equal(after[1], numpy_before[1])
    assert after[2:] == numpy_before[2:]
    reordered = bench.run([second, first])
    for left, right in zip(original, reversed(reordered), strict=True):
        assert sorted(tuple(s.variables) for s in left.front) == sorted(
            tuple(s.variables) for s in right.front
        )
        assert left.config.label == right.config.label
    assert list(bench.summary_table().index) == [second.label, first.label]


def test_nsga3_reports_actual_work_and_seeds_half_the_derived_population(monkeypatch):
    from platypus import NSGAIII
    from investment_tool.strategies.optimization_strategies import multi_greedy

    limits = []
    original = multi_greedy.SeededPortfolioGenerator

    class ObservedGenerator(original):
        def __init__(self, seeds, limit):
            limits.append(limit)
            super().__init__(seeds, limit)

    monkeypatch.setattr(multi_greedy, "SeededPortfolioGenerator", ObservedGenerator)
    config = SolverConfig(
        label="Few directions",
        algorithm=NSGAIII,
        algorithm_kwargs={"divisions_outer": 7},
        population_size=100,
        n_generations=17,
        n_restarts=2,
        random_seed=4,
    )
    bench = SolverBenchmark(
        expected_returns=_MU, covariance_matrix=_COV, model=MarkowitzLongOnly
    )
    result = bench.run([config])[0]
    assert limits == [4, 4]
    assert result.metrics["initial_population_size"] == 8
    assert result.metrics["requested_evaluations"] == 34
    assert result.metrics["actual_evaluations"] >= 34
    assert result.metrics["actual_evaluations"] % 8 == 0


def test_gwasfga_reports_population_after_initialization_rounds_odd_size():
    from dataclasses import replace

    config = replace(
        _quick_gwasfga_config(), population_size=21, n_generations=20, random_seed=1
    )
    bench = SolverBenchmark(
        expected_returns=_MU, covariance_matrix=_COV, model=MarkowitzLongOnly
    )
    result = bench.run([config])[0]
    assert result.metrics["initial_population_size"] == 20
    assert result.metrics["actual_evaluations"] == 20


@pytest.mark.parametrize("forecast", [0.0001, -0.0001])
@pytest.mark.parametrize("with_reference", [False, True])
def test_cost_aware_hypervolume_reference_bounds_low_net_returns(
    forecast, with_reference
):
    import numpy as np
    from platypus import Solution
    from investment_tool.domain.trading_costs import TradingCosts
    from investment_tool.strategies.optimization_strategies.solver_comparison.comparison_metrics import (
        hypervolume,
    )

    mu = np.full(2, forecast)
    cov = np.eye(2) * 0.01
    costs = TradingCosts(0.001, np.zeros(2))
    problem = BalancedPortfoliosRiskContributions(mu, cov, 1, 2, trading_costs=costs)
    solution = Solution(problem)
    solution.variables[:] = [0.5, 0.5]
    solution.evaluate()
    bench = SolverBenchmark(
        pd.Series(mu),
        pd.DataFrame(cov),
        BalancedPortfoliosRiskContributions,
        model_kwargs={"l_assets": 1, "u_assets": 2, "trading_costs": costs},
        reference_front=[solution] if with_reference else None,
    )
    reference = bench._compute_reference_point([], mu, cov)
    assert np.all(reference > np.array(solution.objectives[:2]))
    assert hypervolume([solution], reference) > 0
