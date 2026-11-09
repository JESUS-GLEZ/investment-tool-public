import numpy as np
import pandas as pd
import pytest
import random
from platypus import NSGAII
from unittest.mock import patch

from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    PortfolioVariator,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.multi_greedy import (
    multi_greedy_seeds,
)
from investment_tool.strategies.optimization_strategies.seeded_optimization import (
    MultiGreedyOptimizationStrategy,
)


def make_optimizer(constrained=False):
    s = MultiGreedyOptimizationStrategy(
        expected_returns=pd.Series([0.01, 0.02, 0.03, 0.04]),
        covariance_matrix=pd.DataFrame(np.diag([0.01, 0.04, 0.09, 0.16])),
        risk_aversion=0.5,
        algorithm=NSGAII,
        model=BalancedPortfoliosRiskContributions,
        model_kwargs={"l_assets": 2, "u_assets": 3, "hhi_as_constraint": constrained},
        algorithm_kwargs={"variator": PortfolioVariator(min_assets=2, max_assets=3)},
        population_size=20,
        n_generations=60,
        n_restarts=3,
        include_markowitz_front=False,
    )
    s.search_seconds = 2
    s.seed_max_subsets = 20
    s.refinement_supports = 0
    s._plot_optimization_results = lambda *args: None
    return s


@pytest.mark.parametrize("constrained", [False, True])
def test_current_model_objectives_and_cardinality_preserved(constrained):
    s = make_optimizer(constrained)
    result = s.optimize()
    assert result.front
    assert all(len(x.objectives) == (2 if constrained else 3) for x in result.front)
    assert all(
        x.feasible and 2 <= np.count_nonzero(np.array(x.variables) > 1e-4) <= 3
        for x in result.front
    )
    assert max(len(a) for a in result.all_archives) <= 20


def test_new_period_recomputes_seeds_and_objectives():
    s = make_optimizer()
    first = s.optimize()
    s.expected_returns = -s.expected_returns
    # Deliberately wrong-dimensional prior holdings must not seed a new universe.
    s.seed_weights = [np.ones(9) / 9]
    second = s.optimize()
    assert all(x.objectives[1] < 0 for x in first.front)
    assert all(x.objectives[1] > 0 for x in second.front)
    for x in second.front:
        assert (
            abs(x.objectives[1] + np.dot(x.variables, s.expected_returns.values))
            < 1e-12
        )


def test_seed_deadline_stops_before_subset_solves():
    p = BalancedPortfoliosRiskContributions(
        np.array([0.01, 0.02, 0.03]), np.eye(3), 2, 3
    )
    front, n = multi_greedy_seeds(p, max_subsets=10000, stop=lambda: True)
    assert front == [] and n == 0


def test_seed_phase_stops_before_total_search_deadline(monkeypatch):
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    optimizer = make_optimizer()
    optimizer.search_seconds = 10

    class CheckedDeadline(Exception):
        pass

    def inspect_seed_deadline(*args, **kwargs):
        stop = kwargs["stop"]
        now = module.time.monotonic()
        with patch.object(module.time, "monotonic", return_value=now + 1):
            assert not stop()
        with patch.object(module.time, "monotonic", return_value=now + 5):
            assert stop()  # 40% of ten seconds, with time left for NSGA-II.
        raise CheckedDeadline

    monkeypatch.setattr(module, "multi_greedy_seeds", inspect_seed_deadline)
    with pytest.raises(CheckedDeadline):
        optimizer.optimize()


def test_feasible_seeds_survive_when_refinement_has_no_budget(monkeypatch):
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    original = module.multi_greedy_seeds

    def expired(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(module.time, "monotonic", lambda: 1e20)
        return result

    monkeypatch.setattr(module, "multi_greedy_seeds", expired)
    result = make_optimizer().optimize()
    assert result.front and all(x.feasible for x in result.front)
    assert result.search_diagnostics["time_limited"]
    assert all(
        d["termination"] == "wall_time" for d in result.search_diagnostics["restarts"]
    )


def test_partial_seed_path_timeout_is_reported_before_overall_deadline(monkeypatch):
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    original = module.multi_greedy_seeds

    def first_path_timed_out(*args, **kwargs):
        result = original(*args, **kwargs)
        # The first path reached its shorter deadline. The second exhausted its
        # candidates before the overall seed deadline, which is still infinite.
        kwargs["diagnostics"]["paths"][0]["time_limited"] = True
        kwargs["diagnostics"]["paths"][0]["termination"] = "wall_time"
        return result

    monkeypatch.setattr(module, "multi_greedy_seeds", first_path_timed_out)
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    result = optimizer.optimize()
    assert result.search_diagnostics["seed_time_limited"]
    assert result.search_diagnostics["time_limited"]
    assert all(
        d["termination"] == "evaluation_budget"
        for d in result.search_diagnostics["restarts"]
    )


def test_independent_restarts_merge_feasible_fronts_and_report_budgets():
    from platypus import ParetoDominance

    optimizer = make_optimizer()
    optimizer.search_seconds = None
    result = optimizer.optimize()
    runs = result.search_diagnostics["restarts"]
    assert len(result.all_archives) == len(runs) == 3
    assert len({d["seed"] for d in runs}) == 3
    assert all(d["nfe"] == 60 and d["termination"] == "evaluation_budget" for d in runs)
    assert not result.search_diagnostics["time_limited"]
    assert all(s.feasible for s in result.front)
    dominance = ParetoDominance()
    assert all(dominance.compare(a, b) == 0 for a in result.front for b in result.front)
    for previous, current in zip(runs, runs[1:]):
        assert (
            current["merged_front"]["max_expected_return"]
            >= previous["merged_front"]["max_expected_return"]
        )
        assert (
            current["merged_front"]["min_variance"]
            <= previous["merged_front"]["min_variance"]
        )


def test_initial_portfolios_and_actual_injection_are_auditable():
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    result = optimizer.optimize()
    diagnostics = result.search_diagnostics
    discovery = diagnostics["seed_discovery"]
    assert [p["policy"] for p in discovery["paths"]] == [
        "utility",
        "cardinality_sharpe",
    ]
    assert discovery["attempted_subsets"] == diagnostics["seed_subsets"] <= 20
    seeds = diagnostics["initial_seed_portfolios"]
    assert len(seeds) == diagnostics["seed_front"]["size"]
    indices = [r["initial_seed_indices"] for r in diagnostics["restarts"]]
    assert indices[0] != indices[1]
    for injected in indices:
        assert len(injected) == min(len(seeds), optimizer.population_size // 2)
        for index in injected:
            assert set(seeds[index]) <= set(map(str, optimizer.expected_returns.index))
            assert sum(seeds[index].values()) == pytest.approx(1)
    assert not diagnostics["time_limited"]


def test_fixed_work_reproduces_front_despite_external_rng_state_and_restores_rngs():
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    first = optimizer.optimize()
    random.seed(918)
    np.random.seed(271)
    for _ in range(5):
        random.random()
    np.random.random(13)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    second = optimizer.optimize()
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert [list(s.variables) for s in first.front] == [
        list(s.variables) for s in second.front
    ]
    assert [list(s.objectives) for s in first.front] == [
        list(s.objectives) for s in second.front
    ]


def test_rng_state_is_restored_when_restart_fails(monkeypatch):
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    optimizer = make_optimizer()
    optimizer.search_seconds = None
    python_state, numpy_state = random.getstate(), np.random.get_state()

    def fail(*args, **kwargs):
        random.random()
        np.random.random()
        raise RuntimeError("failed restart")

    monkeypatch.setattr(module, "NSGAII", fail)
    with pytest.raises(RuntimeError, match="failed restart"):
        optimizer.optimize()
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


def test_mv_reference_failure_does_not_discard_feasible_portfolio(monkeypatch):
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    optimizer = make_optimizer()
    optimizer.include_markowitz_front = True
    monkeypatch.setattr(
        module,
        "convex_markowitz_front",
        lambda *_: (_ for _ in ()).throw(RuntimeError("optimal_inaccurate")),
    )
    result = optimizer.optimize()
    assert result.front
    assert result.markowitz_front is None


@pytest.mark.parametrize("kind", ["optimization", "combined", "sentiment_ewm"])
def test_execution_and_comparison_factory_uses_new_profile(kind):
    from types import SimpleNamespace

    from investment_tool.app.execution_coordinator import _build_strategy
    from investment_tool.constants import StrategyType
    from tests.strategies.orchestrators.test_investment_strategy_optimization import (
        _make_instance,
        _make_stock,
    )

    model_data = SimpleNamespace(
        instance=_make_instance(),
        stocks=[_make_stock(s) for s in ["A", "B", "C", "D"]],
        news=None,
    )
    strategy = _build_strategy(
        StrategyType(kind), model_data, None, min_assets=2, max_assets=3
    )
    mu = pd.Series([0.01, 0.02, 0.03, 0.04])
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.09, 0.16]))
    optimizer = strategy._build_optimizer(mu, cov)
    assert isinstance(optimizer, MultiGreedyOptimizationStrategy)
    assert optimizer.seed_max_subsets == 360000
    assert optimizer.refinement_supports == 80
    assert optimizer.search_seconds is None
    assert optimizer.seed_time_fraction == 0.4
    assert optimizer.n_restarts == 6
    assert optimizer.n_generations == 6000
    assert optimizer.population_size == 300
    assert optimizer.seed_weights is None


def test_all_nondominated_seed_tradeoffs_survive_when_search_has_no_work(monkeypatch):
    from platypus import Solution
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    optimizer = make_optimizer(True)
    optimizer.expected_returns = pd.Series([0.02, 0.04])
    optimizer.covariance_matrix = pd.DataFrame(np.eye(2))
    optimizer.model_kwargs = {
        "l_assets": 2,
        "u_assets": 2,
        "hhi_as_constraint": True,
        "hhi_bound": 0.6,
    }
    optimizer.population_size = 2
    optimizer.n_generations = 6
    optimizer.search_seconds = None
    optimizer.refinement_supports = 0
    problem = optimizer.model(
        optimizer.expected_returns.values,
        optimizer.covariance_matrix.values,
        **optimizer.model_kwargs,
    )
    seeds = []
    for w in np.linspace(0.476, 0.5, 6):
        solution = Solution(problem)
        solution.variables[:] = [w, 1 - w]
        solution.evaluate()
        assert solution.feasible
        seeds.append(solution)
    monkeypatch.setattr(module, "multi_greedy_seeds", lambda *a, **k: (seeds, 6))
    monkeypatch.setattr(
        module,
        "NSGAII",
        lambda *a, **k: type("Finished", (), {"nfe": 6, "result": []})(),
    )
    result = optimizer.optimize()
    assert len(result.front) == 6
    assert {tuple(s.variables) for s in result.front} == {
        tuple(s.variables) for s in seeds
    }
    assert all(len(a) <= 2 for a in result.all_archives)


def test_refinement_progress_is_monotonic_and_rng_independent():
    optimizer = make_optimizer()
    optimizer.search_seconds = None
    optimizer.refinement_supports = 3
    updates = []
    first = optimizer.optimize(
        progress_callback=lambda fraction, message: updates.append(fraction)
    )
    assert updates == sorted(updates)
    assert first.search_diagnostics["refinement"]["subproblems"] > 0
    random.seed(719)
    np.random.seed(914)
    second = optimizer.optimize()
    np.testing.assert_array_equal(
        [s.variables[:] for s in first.front], [s.variables[:] for s in second.front]
    )


def test_bounded_archive_preserves_sharpe_and_high_return_risk_anchors():
    from platypus import Solution
    from investment_tool.strategies.optimization_strategies.seeded_optimization import (
        _merge_search_archive,
    )

    problem = BalancedPortfoliosRiskContributions(
        np.array([0.01, 0.02]),
        np.diag([1.0, 4.0]),
        2,
        2,
        hhi_as_constraint=True,
        hhi_bound=0.6,
    )
    candidates = []
    for w in 2 / 3 + np.linspace(-0.02, 0.02, 101):
        s = Solution(problem)
        s.variables[:] = [w, 1 - w]
        s.evaluate()
        assert s.feasible
        candidates.append(s)
    archive = _merge_search_archive([candidates], 10)
    assert len(archive) == 10

    def score(s):
        return -s.objectives[1] / np.sqrt(s.objectives[0])

    assert max(map(score, archive)) == max(map(score, candidates))
    low = -min(candidates, key=lambda s: s.objectives[0]).objectives[1]
    high = max(-s.objectives[1] for s in candidates)
    target = low + 0.75 * (high - low)
    assert min(s.objectives[0] for s in archive if -s.objectives[1] >= target) == min(
        s.objectives[0] for s in candidates if -s.objectives[1] >= target
    )


def test_generation_discoveries_survive_per_restart_plot_pruning(monkeypatch):
    from platypus import Solution
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    optimizer = make_optimizer(True)
    optimizer.expected_returns = pd.Series([0.02, 0.04])
    optimizer.covariance_matrix = pd.DataFrame(np.eye(2))
    optimizer.model_kwargs = {
        "l_assets": 2,
        "u_assets": 2,
        "hhi_as_constraint": True,
        "hhi_bound": 0.6,
    }
    optimizer.population_size = 1
    optimizer.n_generations = 2
    optimizer.n_restarts = 2
    optimizer.refinement_supports = 0
    optimizer.search_seconds = None
    problem = optimizer.model(
        optimizer.expected_returns.values,
        optimizer.covariance_matrix.values,
        **optimizer.model_kwargs,
    )
    seed, discovery = Solution(problem), Solution(problem)
    seed.variables[:] = [0.5, 0.5]
    discovery.variables[:] = [0.476, 0.524]
    seed.evaluate()
    discovery.evaluate()
    monkeypatch.setattr(module, "multi_greedy_seeds", lambda *a, **k: ([seed], 1))
    original_merge = module.merge_nondominated

    def prune_plot_archive(fronts, target_size):
        # Reproduce crowding's loss of a valuable interior discovery.
        if target_size == 1:
            return [seed]
        return original_merge(fronts, target_size=target_size)

    monkeypatch.setattr(module, "merge_nondominated", prune_plot_archive)

    class Algorithm:
        nfe = 0
        result = []

        def __init__(self, *a, **k):
            pass

        def step(self):
            self.nfe += 1
            self.result = [discovery] if self.nfe == 1 else []

    monkeypatch.setattr(module, "NSGAII", Algorithm)
    result = optimizer.optimize()
    assert all(a == [seed] for a in result.all_archives)
    assert {tuple(s.variables) for s in result.front} == {
        tuple(seed.variables),
        tuple(discovery.variables),
    }


def test_archive_sorting_cannot_change_live_nsga_rank_or_crowding(monkeypatch):
    from platypus import Solution
    from investment_tool.strategies.optimization_strategies import (
        seeded_optimization as module,
    )

    optimizer = make_optimizer()
    optimizer.search_seconds = None
    optimizer.refinement_supports = 0
    populations = []

    class CheckedAlgorithm:
        nfe = 0

        def __init__(self, problem, **kwargs):
            solution = Solution(problem)
            solution.variables[:] = [6 / 11, 3 / 11, 2 / 11, 0]
            solution.evaluate()
            solution.rank = 71
            solution.crowding_distance = 93
            self.result = [solution]
            populations.append(self.result)

        def step(self):
            assert self.result[0].rank == 71
            assert self.result[0].crowding_distance == 93
            self.nfe += 20

    monkeypatch.setattr(module, "NSGAII", CheckedAlgorithm)
    result = optimizer.optimize()
    assert result.front
    assert all(
        pop[0].rank == 71 and pop[0].crowding_distance == 93 for pop in populations
    )
