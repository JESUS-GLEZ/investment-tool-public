import numpy as np
import pytest
from platypus import Solution

from investment_tool.constants import PortfolioBalance
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.multi_greedy import (
    multi_greedy_seeds,
)
from investment_tool.strategies.optimization_strategies.support_refinement import (
    refine_supports,
)
from investment_tool.strategies.selection_strategies.return_target_selection import (
    ReturnTargetSelection,
)


def problem(balance=PortfolioBalance.CAPITAL_WEIGHTS):
    return BalancedPortfoliosRiskContributions(
        np.array([0.01, 0.03, 0.06, 0.08, 0.09]),
        np.diag([1.0, 4.0, 9.0, 16.0, 25.0]),
        2,
        5,
        balance=balance,
    )


def solution(p, w):
    s = Solution(p)
    s.variables[:] = w
    s.evaluate()
    return s


def test_capital_bounds_and_hhi_are_not_risk_bounds_or_exact_equal_weights():
    p = problem()
    s = solution(p, [0.5, 0.3, 0.2, 0, 0])
    assert s.feasible
    assert s.objectives[2] == pytest.approx(0.38)
    risk_solution = solution(
        problem(PortfolioBalance.RISK_CONTRIBUTIONS), [1 / 3, 1 / 3, 1 / 3, 0, 0]
    )
    assert not risk_solution.feasible
    assert risk_solution.objectives[2] != pytest.approx(1 / 3)
    assert not solution(p, [0.6, 0.2, 0.2, 0, 0]).feasible
    assert not solution(p, [0.7, 0.15, 0.15, 0, 0]).feasible
    # Five active assets cannot be disguised as an allowed four-asset portfolio.
    p.u_assets = 4
    assert not solution(p, [0.2] * 5).feasible


def test_capital_seed_search_and_refinement_honor_the_same_model():
    p = problem()
    seeds, count = multi_greedy_seeds(p, max_subsets=30, beam_width=2)
    assert seeds and count > 0
    for s in seeds:
        w = np.asarray(s.variables)
        assert s.feasible
        np.testing.assert_allclose(w[w > 0], 1 / np.count_nonzero(w))
    seed = solution(p, [1 / 3, 1 / 3, 1 / 3, 0, 0])
    refined, diagnostics = refine_supports(p, [seed], max_supports=1)
    assert refined and diagnostics["supports"] == 1
    assert min(s.objectives[0] for s in refined) < seed.objectives[0]
    for s in refined:
        w = np.asarray(s.variables)
        assert s.feasible and 2 <= np.count_nonzero(w) <= 5
        assert min(w[w > 0]) >= 0.18 - 1e-12 and max(w) <= 0.55 + 1e-12
        assert s.objectives[2] == pytest.approx(w @ w)
    selector = ReturnTargetSelection()
    import pandas as pd

    selector.select(refined, pd.Index(list("ABCDE")))
    assert "selected_capital_weight_hhi" in selector.last_diagnostics
    assert "selected_risk_contribution_hhi" not in selector.last_diagnostics


def test_balance_changes_candidate_front_cache_key(tmp_path):
    from investment_tool.strategies.optimization_strategies.front_cache import (
        CandidateFrontCache,
    )
    from tests.strategies.optimization_strategies.test_seeded_optimization import (
        make_optimizer,
    )

    cache = CandidateFrontCache(tmp_path, "optimization")
    optimizer = make_optimizer()
    key = cache._key(optimizer)
    optimizer.model_kwargs["balance"] = PortfolioBalance.CAPITAL_WEIGHTS
    assert cache._key(optimizer) != key
