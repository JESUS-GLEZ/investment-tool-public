from types import SimpleNamespace

import numpy as np
from platypus import Solution
import pytest

from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.support_refinement import (
    refine_supports,
)


def seed_problem(constrained=False):
    problem = BalancedPortfoliosRiskContributions(
        np.full(5, 0.03),
        np.diag([1.0, 4.0, 9.0, 16.0, 25.0]),
        2,
        5,
        hhi_as_constraint=constrained,
        hhi_bound=0.4,
    )
    seed = Solution(problem)
    seed.variables[:] = [6 / 11, 3 / 11, 2 / 11, 0, 0]
    seed.evaluate()
    assert seed.feasible
    return problem, seed


@pytest.mark.parametrize("constrained", [False, True])
def test_refinement_improves_variance_without_changing_return_or_constraints(
    constrained,
):
    problem, seed = seed_problem(constrained)
    original = list(seed.variables)
    refined, diagnostic = refine_supports(problem, [seed], max_supports=1)
    assert refined and diagnostic["supports"] == 1
    assert not diagnostic["time_limited"]
    best = min(refined, key=lambda s: s.objectives[0])
    assert best.objectives[0] < 0.95 * seed.objectives[0]
    for solution in refined:
        weights = np.asarray(solution.variables)
        assert solution.feasible and len(solution.objectives) == (
            2 if constrained else 3
        )
        assert set(np.flatnonzero(weights)) == {0, 1, 2}
        assert sum(weights) == pytest.approx(1)
        assert weights @ problem.mu == pytest.approx(0.03)
        g = weights * (problem.cov @ weights) / (weights @ problem.cov @ weights)
        assert min(g[:3]) >= 0.18 - 1e-12
        assert max(g[:3]) <= 0.55 + 1e-12
        if constrained:
            assert g @ g <= 0.4
    assert list(seed.variables) == original
    second, _ = refine_supports(problem, [seed], max_supports=1)
    np.testing.assert_array_equal(
        [s.variables[:] for s in refined], [s.variables[:] for s in second]
    )


def test_deadline_and_disabled_refinement_do_no_subproblem_work():
    problem, seed = seed_problem()
    for budget, stop in [(0, None), (1, lambda: True)]:
        refined, diagnostic = refine_supports(
            problem, [seed], max_supports=budget, stop=stop
        )
        assert not refined and diagnostic["subproblems"] == 0
        assert diagnostic["time_limited"] == (budget != 0)


@pytest.mark.parametrize(
    "success,weights",
    [
        (False, [0.5, 0.3, 0.2]),
        (True, [1.0, 0.0, 0.0]),
        (True, [float("nan"), 0.3, 0.7]),
    ],
)
def test_unsuccessful_or_invalid_numerical_results_cannot_enter_archive(
    monkeypatch, success, weights
):
    problem, seed = seed_problem()
    monkeypatch.setattr(
        "investment_tool.strategies.optimization_strategies.support_refinement.minimize",
        lambda *a, **k: SimpleNamespace(success=success, x=np.array(weights), nit=1),
    )
    refined, diagnostic = refine_supports(problem, [seed], max_supports=1)
    assert not refined and diagnostic["accepted"] == 0
    assert seed.feasible


def test_global_return_target_gets_a_feasible_refined_candidate():
    problem, seed = seed_problem()
    problem.mu[:] = [0.01, 0.03, 0.06, 0.08, 0.09]
    seed.evaluate()
    refined, _ = refine_supports(problem, [seed], max_supports=1)
    pool = [seed] + refined
    low = -min(pool, key=lambda s: (s.objectives[0], s.objectives[1])).objectives[1]
    high = max(-s.objectives[1] for s in pool)
    target = low + 0.75 * (high - low)
    assert any(target <= -s.objectives[1] <= target + 1e-8 for s in refined)


def test_numerical_failure_preserves_existing_candidate(monkeypatch):
    problem, seed = seed_problem()

    def fail(*a, **k):
        raise np.linalg.LinAlgError("singular subproblem")

    monkeypatch.setattr(
        "investment_tool.strategies.optimization_strategies.support_refinement.minimize",
        fail,
    )
    refined, diagnostic = refine_supports(problem, [seed], max_supports=1)
    assert not refined and seed.feasible
    assert diagnostic["numerical_failures"] == diagnostic["subproblems"] > 0
