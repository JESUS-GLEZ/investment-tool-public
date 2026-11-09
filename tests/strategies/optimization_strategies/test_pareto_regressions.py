"""Regression coverage for feasible Pareto extraction and search history."""

import random
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from platypus import (
    NSGAII,
    Problem,
    Real,
    Solution,
    nondominated_prune,
    nondominated_sort,
)

from investment_tool.strategies.optimization_strategies.custom_algorithm.gwasfga import (
    GWASFGA,
)
from investment_tool.strategies.optimization_strategies.custom_algorithm.portfolio_sampling import (
    ArchivingEvaluator,
    PortfolioGenerator,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.platypus_optimization import (
    PlatypusOptimizationStrategy,
)
from investment_tool.strategies.optimization_strategies.utils import (
    merge_nondominated,
    plot_solution,
    plot_solutions_with_markowitz_front,
)
from investment_tool.strategies.selection_strategies.risk_aversion_selection import (
    RiskAversionSelection,
)


def candidate(problem, objectives, weights=None, violation=0):
    sol = Solution(problem)
    sol.variables[:] = weights if weights is not None else [objectives[0]]
    sol.objectives[:] = objectives
    sol.constraint_violation = violation
    sol.feasible = violation == 0
    sol.evaluated = True
    return sol


def test_merge_filters_before_pruning_and_preserves_third_objective():
    problem = Problem(1, 3, 1)
    a = candidate(problem, [1, -2, 0.8])
    b = candidate(problem, [2, -1, 0.2])  # dominated only in the 2-D projection
    dominated = candidate(problem, [3, 0, 0.9])
    infeasible = candidate(problem, [0, -3, 0.1], violation=0.1)
    invalid = candidate(problem, [float("nan"), -4, 0.1])
    result = merge_nondominated([[infeasible, dominated, a, b, a, invalid]], 20)
    assert result == [a, b]
    assert len(merge_nondominated([[a, b, dominated]], 1)) == 1
    assert merge_nondominated([[infeasible]], 20) == []
    assert merge_nondominated([[a]], 0) == []


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("capacity", [1, 8, 1000])
def test_first_front_extraction_preserves_full_sort_order_and_pruning(
    dimensions, capacity
):
    problem = Problem(1, dimensions, 1)
    rng = np.random.default_rng(41)
    unique = [
        candidate(problem, point.tolist(), weights=[i / 1000])
        for i, point in enumerate(rng.uniform(size=(120, dimensions)))
    ]
    nondominated_sort(unique)
    expected = nondominated_prune([s for s in unique if s.rank == 0], capacity)
    # Rounded-weight deduplication must still keep the first representative,
    # even when a later duplicate's objectives would otherwise dominate it.
    duplicate = candidate(problem, [-1.0] * dimensions, weights=[1e-10])
    actual = merge_nondominated([unique[:60], unique[60:] + [duplicate]], capacity)
    assert actual == expected
    assert all(s.rank == 0 for s in actual)


def test_sparse_merge_keeps_rounded_equality_support_and_dimension():
    problem = Problem(6, 2)
    first = candidate(problem, [1, -1], [0.4, 0, 0, 0, 0, 0.6])
    duplicate = candidate(problem, [0, -2], [0.4, -0.0, 1e-10, 0, 0, 0.6])
    other_support = candidate(problem, [2, -2], [0, 0.4, 0, 0, 0, 0.6])
    other_dimension = candidate(Problem(7, 2), [3, -3], [0.4, 0, 0, 0, 0, 0.6, 0])
    assert merge_nondominated(
        [[first, duplicate, other_support, other_dimension]], 10
    ) == [first, other_support, other_dimension]


def test_sparse_seed_front_deduplication_has_bounded_temporary_memory():
    # Candidate vectors already exist in the greedy cache. Measure the merge's
    # additional memory: dense rounded keys formerly retained ~10 MB of zeros.
    problem = Problem(1000, 2)
    seeds = []
    for i in range(300):
        weights = [0.0] * 1000
        weights[0], weights[i + 1], weights[-1] = 0.2, 0.3, 0.5
        seeds.append(candidate(problem, [i / 1000, -i / 1000], weights))
    tracemalloc.start()
    try:
        actual = merge_nondominated([seeds], len(seeds))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == seeds
    assert peak < 4 * 1024**2


def test_sampler_uses_sparse_model_and_platypus_evaluation():
    np.random.seed(13)
    problem = BalancedPortfoliosRiskContributions(
        np.ones(8),
        np.eye(8),
        l_assets=2,
        u_assets=4,
    )
    algo = GWASFGA(problem, population_size=10)
    algo.step()
    assert algo.nfe == 10
    for sol in algo.population:
        assert 2 <= np.count_nonzero(sol.variables) <= 4
        assert sum(sol.variables) == pytest.approx(1)
        assert sol.evaluated
        assert sol.constraint_violation == pytest.approx(sum(sol.constraints))
    generic = Problem(1, 1)
    generic.types[:] = Real(0, 1)
    assert 0 <= PortfolioGenerator().generate(generic).variables[0] <= 1


@pytest.mark.parametrize("algorithm_class", [GWASFGA, NSGAII])
def test_archive_keeps_discarded_candidates_and_snapshots(algorithm_class):
    problem = Problem(1, 2)
    problem.types[:] = Real(0, 1)
    problem.function = lambda x: [x[0], 1 - x[0]]
    algo = algorithm_class(problem, population_size=4)
    if algorithm_class is NSGAII:
        recorder = ArchivingEvaluator(algo.evaluator)
        algo.evaluator = recorder
        archive = recorder.archive
    else:
        archive = algo.pareto_archive
    early = Solution(problem)
    early.variables[:] = [0.1]
    algo.evaluate_all([early])
    later = Solution(problem)
    later.variables[:] = [0.9]
    algo.evaluate_all([later])
    # Simulate environmental selection dropping a nondominated point.
    algo.population = [later]
    assert len(archive) == 2
    early.objectives[:] = [100, 100]
    assert sorted(s.objectives[0] for s in archive) == [0.1, 0.9]
    # Repeated evaluation must not inflate the archive.
    algo.evaluate_all([later])
    assert len(archive) == 2


@pytest.mark.parametrize("algorithm_class", [GWASFGA, NSGAII])
def test_strategy_returns_feasible_nondominated_history(algorithm_class):
    random.seed(4)
    np.random.seed(4)
    strategy = PlatypusOptimizationStrategy(
        pd.Series([0.01, 0.02, 0.03]),
        pd.DataFrame(np.eye(3)),
        0.5,
        algorithm_class,
        BalancedPortfoliosRiskContributions,
        population_size=10,
        n_generations=30,
        n_restarts=2,
        model_kwargs={"l_assets": 1, "u_assets": 3},
        include_markowitz_front=False,
    )
    front, archives = strategy._run_multi_start_optimization(
        strategy.expected_returns.values,
        strategy.covariance_matrix.values,
    )
    assert front
    assert all(s.constraint_violation == 0 for s in front)
    assert merge_nondominated([front], 10) == front
    assert len(archives) == 2


@pytest.mark.parametrize("dimensions", [2, 3])
def test_plot_dimensions_and_selection(tmp_path, monkeypatch, dimensions):
    problem = Problem(1, dimensions)
    sol = candidate(problem, [1, -2, 0.3][:dimensions])
    figures = []
    original_close = plt.close

    def capture_close(fig=None):
        figures.append(plt.gcf() if fig is None else fig)
        original_close(fig)

    monkeypatch.setattr(plt, "close", capture_close)
    path = plot_solution([sol], "front", tmp_path, selected_solution=sol)
    assert path and (tmp_path / "plots/front.png").exists()
    fig = figures[-1]
    assert fig.axes[0].name == ("3d" if dimensions == 3 else "rectilinear")
    if dimensions == 3:
        assert fig.axes[0].get_zlabel() == "Risk-contribution HHI"
    plot_solutions_with_markowitz_front([sol], [], "overlay", tmp_path)
    fig = figures[-1]
    assert len(fig.axes) == (2 if dimensions == 3 else 1)
    if dimensions == 3:
        assert "Projection" in fig.axes[0].get_title()


def test_positive_risk_aversion_prefers_higher_return_at_equal_variance():
    problem = Problem(2, 2)
    low = candidate(problem, [0.1, -0.01], [1, 0])
    high = candidate(problem, [0.1, -0.02], [0, 1])
    result = RiskAversionSelection(2).select([low, high], pd.Index(["A", "B"]))
    np.testing.assert_array_equal(result.values, [0, 1])


def test_merge_evaluates_missing_constraint_metadata():
    problem = Problem(1, 2, 1)
    problem.types[:] = Real(0, 1)
    problem.constraints[:] = "<=0"
    problem.function = lambda x: ([x[0], -x[0]], [x[0] - 0.5])
    sol = Solution(problem)
    sol.variables[:] = [0.9]
    problem.evaluate(sol)  # Mimics deterministic solver conversion.
    assert merge_nondominated([[sol]], 10) == []
    assert sol.constraint_violation == pytest.approx(0.4)
