"""One cost-aware return definition across models, solvers and reference curves."""

import numpy as np
import pytest
from platypus import Solution

from investment_tool.constants import PortfolioBalance
from investment_tool.domain.trading_costs import TradingCosts, expected_net_return
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.optimization_strategies.scalarized_solver import (
    ScalarizedSubproblemSolver,
)
from investment_tool.strategies.optimization_strategies.convex_frontier import (
    convex_markowitz_front,
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


@pytest.mark.parametrize("balance", list(PortfolioBalance))
def test_balanced_models_keep_risk_and_constraints_and_use_net_return(balance):
    mu, cov = np.array([0.02, 0.025]), np.diag([0.04, 0.04])
    costs = TradingCosts(0.01, np.array([0.5, 0.5]))
    gross = BalancedPortfoliosRiskContributions(mu, cov, 1, 2, balance=balance)
    net = BalancedPortfoliosRiskContributions(
        mu, cov, 1, 2, balance=balance, trading_costs=costs
    )
    for weights in ([0.5, 0.5], [0.6, 0.4]):
        a, b = Solution(gross), Solution(net)
        a.variables[:] = weights
        b.variables[:] = weights
        a.evaluate()
        b.evaluate()
        assert list(a.constraints) == list(b.constraints)
        assert a.objectives[0] == b.objectives[0] and a.objectives[2] == b.objectives[2]
        assert -b.objectives[1] == pytest.approx(
            expected_net_return(mu, np.array(b.variables), costs)
        )


def test_markowitz_and_scalarized_solver_agree_on_net_coordinates():
    mu, cov = np.array([0.02, 0.025]), np.eye(2)
    costs = TradingCosts(0.01, np.array([0.8, 0.2]))
    weights = np.array([0.4, 0.6])
    model = MarkowitzLongOnly(mu, cov, costs)
    solution = Solution(model)
    solution.variables[:] = weights.tolist()
    solution.evaluate()
    solver = ScalarizedSubproblemSolver(
        mu, cov, 1, 2, hhi_as_constraint=True, trading_costs=costs
    )
    np.testing.assert_allclose(solution.objectives, solver.raw_objectives(weights))


def test_cost_aware_convex_reference_keeps_incumbent_for_small_return_improvement():
    mu, cov = np.array([0.02, 0.025]), np.diag([0.01, 0.04])
    costs = TradingCosts(0.01, np.array([0.8, 0.2]))
    front = convex_markowitz_front(mu, cov, 5, costs)
    assert front
    for solution in front:
        np.testing.assert_allclose(solution.variables, [0.8, 0.2], atol=2e-5)
        assert -solution.objectives[1] == pytest.approx(
            expected_net_return(mu, np.array(solution.variables), costs)
        )


def test_seeded_search_refinement_and_selector_use_identical_net_returns():
    from tests.strategies.optimization_strategies.test_seeded_optimization import (
        make_optimizer,
    )

    optimizer = make_optimizer()
    optimizer.search_seconds = None
    optimizer.n_restarts = 1
    optimizer.trading_costs = TradingCosts(0.01, np.array([2 / 3, 1 / 3, 0.0, 0.0]))
    result = optimizer.optimize()
    assert result.front
    for solution in result.front:
        assert -solution.objectives[1] == pytest.approx(
            expected_net_return(
                optimizer.expected_returns.to_numpy(),
                np.array(solution.variables),
                optimizer.trading_costs,
            )
        )
    selector = ReturnTargetSelection(0.75)
    selector.select(result.front, optimizer.expected_returns.index)
    assert (
        selector.last_diagnostics["selected_expected_return"]
        >= selector.last_diagnostics["target_expected_return"]
    )
    problem = result.front[0].problem
    seeds, _ = multi_greedy_seeds(problem, max_subsets=10)
    # A feasible held portfolio is retained rather than discarded as gross-dominated.
    assert any(np.allclose(s.variables, [2 / 3, 1 / 3, 0, 0]) for s in seeds)
    refined, diagnostic = refine_supports(problem, seeds, max_supports=1)
    assert diagnostic["subproblems"] > 0 and refined
    for solution in refined:
        assert -solution.objectives[1] == pytest.approx(
            expected_net_return(
                problem.mu, np.array(solution.variables), optimizer.trading_costs
            )
        )


def test_cost_aware_convex_reference_spans_varying_net_returns():
    mu, cov = np.array([0.01, 0.06, 0.03]), np.diag([0.01, 0.04, 0.02])
    costs = TradingCosts(0.001, np.array([0.7, 0.1, 0.1]), 0.05)
    front = convex_markowitz_front(mu, cov, 8, costs)
    assert len(front) >= 6
    returns = [-s.objectives[1] for s in front]
    assert max(returns) > min(returns) + 0.02
    assert max(returns) == pytest.approx(0.06 - 0.00175, abs=1e-7)
    for solution in front:
        assert solution.feasible
        assert -solution.objectives[1] == pytest.approx(
            expected_net_return(mu, np.array(solution.variables), costs)
        )


def test_orchestrator_refreshes_costs_from_bound_holdings_and_records_selection():
    import pandas as pd
    from tests.domain.test_trading_costs import state
    from tests.strategies.orchestrators.test_base_investment_strategy import (
        ConcreteStrategy,
    )
    from tests.strategies.optimization_strategies.test_seeded_optimization import (
        make_optimizer,
    )

    portfolio, stocks, _ = state()
    strategy = ConcreteStrategy(portfolio.instance, stocks)
    strategy.bind_portfolio(portfolio)
    optimizer = make_optimizer()
    optimizer.expected_returns = pd.Series([0.01, 0.02], index=["B", "A"])
    optimizer.covariance_matrix = pd.DataFrame(np.diag([0.04, 0.01]))
    optimizer.model_kwargs.update(l_assets=1, u_assets=2)
    optimizer.search_seconds = None
    optimizer.n_restarts = 1
    result = strategy.optimize_portfolios(optimizer)
    np.testing.assert_allclose(optimizer.trading_costs.current_weights, [0.2, 0.8])
    strategy.last_optimization_diagnostics = result.search_diagnostics
    selected = pd.Series([0.5, 0.5], index=["A", "B"])
    strategy.store_optimal_weights(selected)
    diagnostics = strategy.last_optimization_diagnostics
    assert diagnostics["cost_assets"] == ["B", "A"]
    assert diagnostics["selected_estimated_cost_fraction"] == pytest.approx(0.0006)
    assert diagnostics["selected_net_expected_return"] == pytest.approx(0.015 - 0.0006)
    assert diagnostics["selected_variance"] == pytest.approx(0.0125)
    assert diagnostics["selected_volatility"] == pytest.approx(np.sqrt(0.0125))
    # The next decision must mark actual holdings again rather than reuse a target.
    portfolio.positions = {"A": 5.0}
    portfolio.cash = 250
    strategy.optimize_portfolios(optimizer)
    np.testing.assert_allclose(optimizer.trading_costs.current_weights, [0.0, 0.8])
