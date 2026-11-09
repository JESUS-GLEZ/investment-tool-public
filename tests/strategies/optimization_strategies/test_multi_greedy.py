"""Analytical RP checks, bounded search, and fresh seed injection."""

import itertools
import random
import weakref

import numpy as np
import pytest
from platypus import NSGAII, NSGAIII, SPEA2, EpsNSGAII, Solution

from investment_tool.constants import PortfolioBalance
from investment_tool.domain.trading_costs import TradingCosts
from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    PortfolioVariator,
)
from investment_tool.strategies.optimization_strategies.custom_algorithm.gwasfga import (
    GWASFGA,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.multi_greedy import (
    SeededPortfolioGenerator,
    multi_greedy_seeds,
    pooled_multi_greedy_seeds,
    risk_parity_weights,
)


def problem():
    return BalancedPortfoliosRiskContributions(
        np.array([0.01, 0.02, 0.03, 0.04]),
        np.diag([0.01, 0.04, 0.09, 0.16]),
        2,
        3,
        hhi_as_constraint=True,
    )


def test_diagonal_rp_is_inverse_volatility():
    cov = np.diag([0.01, 0.04, 0.09])
    expected = 1 / np.sqrt(np.diag(cov))
    expected /= expected.sum()
    np.testing.assert_allclose(risk_parity_weights(cov), expected, atol=1e-6)


def test_search_explores_new_subsets_and_respects_model():
    p = problem()
    front, count = multi_greedy_seeds(p, max_subsets=100, beam_width=20)
    assert count == sum(len(list(itertools.combinations(range(4), k))) for k in (2, 3))
    assert front
    for s in front:
        assert s.feasible
        w = np.array(s.variables)
        active = w > p.activity_tol
        assert 2 <= active.sum() <= 3
        g = w * (p.cov @ w)
        g /= g.sum()
        np.testing.assert_allclose(g[active], 1 / active.sum(), atol=1e-5)


def test_budget_and_repeatability():
    f, n = multi_greedy_seeds(problem(), max_subsets=3)
    f2, n2 = multi_greedy_seeds(problem(), max_subsets=3)
    assert n == n2 == 3
    assert [list(s.variables) for s in f] == [list(s.variables) for s in f2]


def test_incompatible_hhi_filters_seeds():
    p = problem()
    p._hhi_bound = 0.1
    front, count = multi_greedy_seeds(p, max_subsets=20)
    assert not front
    assert count <= 20


@pytest.mark.parametrize("balance", list(PortfolioBalance))
@pytest.mark.parametrize("constrained_hhi", [False, True])
def test_sparse_seed_cache_preserves_dense_front_without_retaining_candidates(
    monkeypatch, balance, constrained_hhi
):
    """Keep all evaluated evidence, but only a transient dense candidate in search."""
    from investment_tool.strategies.optimization_strategies import multi_greedy
    from investment_tool.strategies.optimization_strategies.utils import (
        merge_nondominated,
    )

    n = 24
    held = np.zeros(n)
    held[:5] = 0.2
    p = BalancedPortfoliosRiskContributions(
        np.linspace(0.01, 0.1, n),
        np.diag(np.linspace(0.02, 0.04, n)),
        5,
        10,
        balance=balance,
        hhi_as_constraint=constrained_hhi,
        trading_costs=TradingCosts(rate=0.0025, current_weights=held),
    )
    live = weakref.WeakSet()
    evaluations = []
    in_search = []

    class TrackedSolution(Solution):
        def __init__(self, problem):
            super().__init__(problem)
            live.add(self)

        def evaluate(self):
            super().evaluate()
            evaluations.append(
                (
                    tuple(self.variables),
                    tuple(self.objectives),
                    tuple(self.constraints),
                    self.feasible,
                )
            )

    monkeypatch.setattr(multi_greedy, "Solution", TrackedSolution)
    actual, count = multi_greedy.multi_greedy_seeds(
        p,
        max_subsets=400,
        beam_width=2,
        progress=lambda *_: in_search.append(len(live)),
    )
    assert count == 400
    # Hybrid can reject a support before constructing a full candidate.
    assert 1 < len(evaluations) <= 401
    if balance != PortfolioBalance.HYBRID:
        assert len(evaluations) == 401  # Each support once, plus the incumbent.
    assert in_search and max(in_search) <= 2  # Incumbent + transient candidate.
    # The legacy merge placed all attempted supports first, then the incumbent.
    expected = []
    for weights, objectives, constraints, feasible in evaluations[1:] + evaluations[:1]:
        if feasible:
            solution = Solution(p)
            solution.variables[:] = weights
            solution.objectives[:] = objectives
            solution.constraints[:] = constraints
            solution.constraint_violation = 0.0
            solution.feasible = True
            solution.evaluated = True
            expected.append(solution)
    expected = merge_nondominated([expected], target_size=400)

    def coordinates(front):
        return [
            (tuple(s.variables), tuple(s.objectives), tuple(s.constraints))
            for s in front
        ]

    assert coordinates(actual) == coordinates(expected)


@pytest.mark.parametrize(
    "algorithm,kwargs",
    [
        (NSGAII, {}),
        (NSGAIII, {"divisions_outer": 19}),
        (SPEA2, {}),
        (EpsNSGAII, {"epsilons": [0.001, 0.001]}),
        (GWASFGA, {}),
    ],
)
def test_all_solvers_receive_fresh_counted_seeds(algorithm, kwargs):
    original = problem()
    seeds, _ = multi_greedy_seeds(original, max_subsets=20)
    before = [list(s.variables) for s in seeds]
    fresh = problem()
    algo = algorithm(
        fresh,
        population_size=20,
        generator=SeededPortfolioGenerator(seeds, 10),
        variator=PortfolioVariator(min_assets=2, max_assets=3),
        **kwargs,
    )
    algo.step()  # initialize only, without variation
    assert algo.nfe == algo.population_size
    assert all(s.problem is fresh for s in algo.population)
    assert any(list(s.variables) == before[0] for s in algo.population)
    assert [list(s.variables) for s in seeds] == before


def discovery_problem():
    return BalancedPortfoliosRiskContributions(
        np.linspace(0.01, 0.1, 8),
        np.diag(np.linspace(0.01, 0.1, 8)),
        2,
        3,
        hhi_as_constraint=True,
        hhi_bound=0.6,
    )


@pytest.mark.parametrize("policy", ["utility", "cardinality_sharpe"])
def test_discovery_continues_to_all_reachable_subsets(policy):
    # The previous preferred-beam search stopped after 52 on this instance,
    # despite having discovered feasible supports with unvisited neighbors.
    p = discovery_problem()
    diagnostics = {}
    front, count = multi_greedy_seeds(
        p,
        max_subsets=500,
        beam_width=1,
        discovery=policy,
        diagnostics=diagnostics,
    )
    assert count == 84  # C(8,2) + C(8,3), exhaustively enumerated.
    if policy == "utility":
        assert diagnostics["continuations"] > 0
    assert diagnostics["termination"] == "discovered_candidates_exhausted"
    expected = []
    for size in (2, 3):
        for indices in itertools.combinations(range(8), size):
            solution = Solution(p)
            weights = np.zeros(8)
            indices = list(indices)
            weights[indices] = 1 / np.sqrt(np.diag(p.cov)[indices])
            weights /= weights.sum()
            solution.variables[:] = weights.tolist()
            solution.evaluate()
            if solution.feasible:
                expected.append(solution)
    for s in front:
        assert s.feasible
        assert any(
            np.allclose(s.variables[:], e.variables[:], atol=1e-7) for e in expected
        )
        # Exact-RP HHI ties differ by floating-point roundoff. Compare material
        # dominance rather than identical counts after strict sorting.
        assert not any(
            np.all(np.array(e.objectives) <= np.array(s.objectives) + 1e-9)
            and np.any(np.array(e.objectives) < np.array(s.objectives) - 1e-9)
            for e in expected
        )


def test_continuation_honors_budget_and_cancellation(monkeypatch):
    from investment_tool.strategies.optimization_strategies import (
        multi_greedy as module,
    )

    solves = []
    original = module.risk_parity_weights

    def solve(cov):
        solves.append(cov.shape)
        return original(cov)

    monkeypatch.setattr(module, "risk_parity_weights", solve)
    _, count = multi_greedy_seeds(discovery_problem(), max_subsets=60, beam_width=1)
    assert count == len(solves) == 60  # Beyond the previous 52-subset stall.
    solves.clear()
    _, count = multi_greedy_seeds(
        discovery_problem(),
        max_subsets=500,
        beam_width=1,
        stop=lambda: len(solves) >= 60,
    )
    assert count == len(solves) == 60
    solves.clear()
    front, count = pooled_multi_greedy_seeds(
        discovery_problem(),
        max_subsets=500,
        stop=lambda: True,
    )
    assert count == 0 and front == [] and not solves


@pytest.mark.parametrize("budget", [1, 3, 120])
def test_pooled_search_shares_one_budget_and_retains_both_path_fronts(budget):
    from investment_tool.strategies.optimization_strategies.utils import (
        merge_nondominated,
    )

    p = discovery_problem()
    diagnostics = {}
    front, count = pooled_multi_greedy_seeds(
        p,
        max_subsets=budget,
        beam_width=1,
        diagnostics=diagnostics,
    )
    separate = []
    for policy, cap in [
        ("utility", (budget + 1) // 2),
        ("cardinality_sharpe", budget // 2),
    ]:
        if cap:
            separate.append(
                multi_greedy_seeds(
                    p,
                    max_subsets=cap,
                    beam_width=1,
                    discovery=policy,
                )[0]
            )
    expected = merge_nondominated(separate, target_size=sum(map(len, separate)))
    assert count == budget
    assert count == sum(path["attempted_subsets"] for path in diagnostics["paths"])
    assert {tuple(s.variables) for s in front} == {tuple(s.variables) for s in expected}
    assert all(s.feasible for s in front)


def test_finite_seed_allowance_leaves_time_for_second_discovery_path(monkeypatch):
    from investment_tool.strategies.optimization_strategies import (
        multi_greedy as module,
    )

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    policies = []

    def search(problem, *, stop, discovery, diagnostics, **kwargs):
        policies.append(discovery)
        if discovery == "utility":
            clock[0] = 4.9
            assert not stop()
            clock[0] = 5
            assert stop()
            diagnostics["termination"] = "stop_requested"
        else:
            assert not stop()
            clock[0] = 6
            assert not stop()
            diagnostics["termination"] = "discovered_candidates_exhausted"
        return [], 0

    monkeypatch.setattr(module, "multi_greedy_seeds", search)
    diagnostics = {}
    pooled_multi_greedy_seeds(
        discovery_problem(), max_subsets=10, deadline=10, diagnostics=diagnostics
    )
    assert policies == ["utility", "cardinality_sharpe"]
    assert clock[0] < 10
    assert [p["time_limited"] for p in diagnostics["paths"]] == [True, False]
    assert diagnostics["paths"][0]["termination"] == "wall_time"


@pytest.mark.parametrize("count", [0, 1, 20, 235, 254])
def test_rotating_initial_portfolios_preserve_rngs_and_source_seeds(count):
    p, fresh = problem(), problem()
    p = BalancedPortfoliosRiskContributions(p.mu, p.cov, 2, 3)
    fresh = BalancedPortfoliosRiskContributions(fresh.mu, fresh.cov, 2, 3)
    seeds = []
    weights = 1 / np.sqrt(np.diag(p.cov)[:2])
    weights /= weights.sum()
    for i in range(count):
        epsilon = (i - count / 2) * 1e-6
        s = Solution(p)
        s.variables[:] = [weights[0] + epsilon, weights[1] - epsilon, 0, 0]
        s.evaluate()
        assert s.feasible
        seeds.append(s)
    before = [tuple(s.variables) for s in seeds]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    seen = set()
    for restart in range(6):
        generator = SeededPortfolioGenerator(seeds, 150, offset=restart)
        for _ in range(generator.limit):
            candidate = generator.generate(fresh)
            assert candidate.problem is fresh
            seen.add(tuple(candidate.variables))
            candidate.evaluate()
            assert candidate.feasible
        assert generator.injected_indices == generator.seed_indices
    assert seen == set(before)
    assert [tuple(s.variables) for s in seeds] == before
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


def test_offset_zero_preserves_original_spacing_and_random_tail():
    from investment_tool.strategies.optimization_strategies.custom_algorithm.portfolio_sampling import (
        PortfolioGenerator,
    )
    from investment_tool.strategies.optimization_strategies.seeded_optimization import (
        _seeded_randomness,
    )

    p = problem()
    seeds, _ = multi_greedy_seeds(p, max_subsets=20)
    generator = SeededPortfolioGenerator(seeds, 2)
    for i in range(2):
        expected = seeds[int(i * (len(seeds) - 1))]
        assert generator.generate(p).variables[:] == expected.variables[:]
    with _seeded_randomness(37):
        actual = generator.generate(p)
    with _seeded_randomness(37):
        expected = PortfolioGenerator().generate(p)
    assert actual.variables[:] == expected.variables[:]
