import numpy as np
import pandas as pd
from platypus import Solution
import pytest

from investment_tool.constants import PortfolioBalance
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.multi_greedy import (
    multi_greedy_seeds,
    risk_parity_weights,
)
from investment_tool.strategies.optimization_strategies.support_refinement import (
    refine_supports,
)
from investment_tool.strategies.selection_strategies.return_target_selection import (
    ReturnTargetSelection,
)
from investment_tool.strategies.selection_strategies.return_with_risk_budget_selection import (
    ReturnWithRiskBudgetSelection,
)


def problem(balance=PortfolioBalance.HYBRID, constrained=False):
    return BalancedPortfoliosRiskContributions(
        np.array([0.01, 0.03, 0.06, 0.08, 0.09]),
        np.diag([1.0, 4.0, 16.0, 25.0, 36.0]),
        2,
        5,
        balance=balance,
        hhi_as_constraint=constrained,
        hhi_bound=0.4,
    )


def solution(p, weights):
    s = Solution(p)
    s.variables[:] = weights
    s.evaluate()
    return s


def assert_both_bounds(p, s):
    w = np.asarray(s.variables)
    active = w > p.activity_tol
    rc = w * (p.cov @ w) / (w @ p.cov @ w)
    assert s.feasible and 2 <= active.sum() <= 5
    assert w.sum() == pytest.approx(1)
    for quantity in (w[active], rc[active]):
        assert quantity.min() >= 0.18 - 1e-12
        assert quantity.max() <= 0.55 + 1e-12
    if len(s.objectives) == 3:
        assert s.objectives[2] == pytest.approx(rc @ rc)
    else:
        assert rc @ rc <= p._hhi_bound + 1e-12


@pytest.mark.parametrize("constrained", [False, True])
def test_hybrid_rejects_either_violation_and_preserves_risk_hhi(constrained):
    p = problem(constrained=constrained)
    rp = [4 / 7, 2 / 7, 1 / 7, 0, 0]
    equal = [1 / 3, 1 / 3, 1 / 3, 0, 0]
    assert solution(problem(PortfolioBalance.RISK_CONTRIBUTIONS), rp).feasible
    assert solution(problem(PortfolioBalance.CAPITAL_WEIGHTS), equal).feasible
    assert not solution(p, rp).feasible  # Risk parity violates capital bounds.
    assert not solution(p, equal).feasible  # Equal weights violate risk bounds.
    good = solution(p, [0.5, 0.3, 0.2, 0, 0])
    assert_both_bounds(p, good)
    assert len(good.objectives) == (2 if constrained else 3)
    # Both zero weight and cardinality are treated exactly as in the base model.
    p.u_assets = 4
    assert not solution(p, [0.2] * 5).feasible


@pytest.mark.parametrize("constrained", [False, True])
def test_hybrid_seed_blends_and_refinement_enforce_both_constraints(constrained):
    p = problem(constrained=constrained)
    rp = risk_parity_weights(p.cov[:3, :3])
    assert rp is not None
    assert not solution(p, [*rp, 0, 0]).feasible
    seeds, count = multi_greedy_seeds(p, max_subsets=30, beam_width=2)
    assert seeds and count <= 30
    assert any(set(np.flatnonzero(s.variables)) == {0, 1, 2} for s in seeds)
    for s in seeds:
        assert_both_bounds(p, s)
    seed = solution(p, [0.5, 0.3, 0.2, 0, 0])
    refined, diagnostics = refine_supports(p, [seed], max_supports=1)
    assert refined and diagnostics["supports"] == 1
    assert min(s.objectives[0] for s in refined) < seed.objectives[0]
    for s in refined:
        assert_both_bounds(p, s)


@pytest.mark.parametrize(
    "selector_type", [ReturnTargetSelection, ReturnWithRiskBudgetSelection]
)
@pytest.mark.parametrize("constrained", [False, True])
def test_hybrid_selection_reports_both_balance_quantities(selector_type, constrained):
    p = problem(constrained=constrained)
    seed = solution(p, [0.5, 0.3, 0.2, 0, 0])
    selector = selector_type()
    selector.select([seed], pd.Index(list("ABCDE")))
    d = selector.last_diagnostics
    assert d["selected_capital_weight_hhi"] == pytest.approx(0.38)
    w = np.asarray(seed.variables)
    rc = w * (p.cov @ w) / (w @ p.cov @ w)
    assert d["selected_risk_contribution_hhi"] == pytest.approx(rc @ rc)
    for quantity in ("capital_weight", "risk_contribution"):
        assert d[f"selected_{quantity}_min"] >= 0.18 - 1e-12
        assert d[f"selected_{quantity}_max"] <= 0.55 + 1e-12


@pytest.mark.parametrize(
    "selector_type", [ReturnTargetSelection, ReturnWithRiskBudgetSelection]
)
def test_capital_selection_does_not_label_weight_hhi_as_risk_hhi(selector_type):
    p = problem(PortfolioBalance.CAPITAL_WEIGHTS)
    seed = solution(p, [0.5, 0.3, 0.2, 0, 0])
    selector = selector_type()
    selector.select([seed], pd.Index(list("ABCDE")))
    assert selector.last_diagnostics["selected_capital_weight_hhi"] == pytest.approx(
        0.38
    )
    assert "selected_risk_contribution_hhi" not in selector.last_diagnostics


def test_hybrid_does_not_relax_bounds_when_no_support_is_feasible():
    p = BalancedPortfoliosRiskContributions(
        np.array([0.01, 0.02]),
        np.diag([1.0, 1e8]),
        2,
        2,
        balance=PortfolioBalance.HYBRID,
    )
    seeds, count = multi_greedy_seeds(p, max_subsets=10, beam_width=2)
    # With 45–55% capital allocations, the high-volatility asset necessarily
    # supplies almost all risk. Reject the support rather than relaxing bounds.
    assert not seeds and count == 1


def test_refinement_cannot_accept_a_solver_result_violating_capital_bounds(monkeypatch):
    from types import SimpleNamespace

    p = problem()
    seed = solution(p, [0.5, 0.3, 0.2, 0, 0])
    monkeypatch.setattr(
        "investment_tool.strategies.optimization_strategies.support_refinement.minimize",
        lambda *a, **kw: SimpleNamespace(
            success=True, x=np.array([4 / 7, 2 / 7, 1 / 7]), nit=1
        ),
    )
    refined, diagnostic = refine_supports(p, [seed], max_supports=1)
    assert not refined and diagnostic["accepted"] == 0
    assert seed.feasible


def test_hybrid_evolutionary_front_and_cache_are_separate_and_feasible(tmp_path):
    from investment_tool.strategies.optimization_strategies.front_cache import (
        CandidateFrontCache,
    )
    from tests.strategies.optimization_strategies.test_seeded_optimization import (
        make_optimizer,
    )

    cache = CandidateFrontCache(tmp_path, "optimization")
    optimizer = make_optimizer()
    original_key = cache._key(optimizer)
    optimizer.model_kwargs["balance"] = PortfolioBalance.CAPITAL_WEIGHTS
    capital_key = cache._key(optimizer)
    optimizer.model_kwargs["balance"] = PortfolioBalance.HYBRID
    assert cache._key(optimizer) not in (original_key, capital_key)
    optimizer.search_seconds = None
    optimizer.refinement_supports = 2
    result = optimizer.optimize()
    assert result.front
    for s in result.front:
        w = np.asarray(s.variables)
        rc = w * (s.problem.cov @ w) / (w @ s.problem.cov @ w)
        lower, upper = s.problem.balance_bounds
        for quantity in (w[w > 1e-4], rc[w > 1e-4]):
            assert quantity.min() >= lower - 1e-12
            assert quantity.max() <= upper + 1e-12
        assert s.feasible
