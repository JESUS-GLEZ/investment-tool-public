import numpy as np

from investment_tool.strategies.optimization_strategies.convex_frontier import (
    convex_markowitz_front,
)


def test_analytic_two_asset_frontier_endpoints():
    mu = np.array([0.01, 0.03])
    cov = np.diag([0.01, 0.04])
    front = convex_markowitz_front(mu, cov, 21)
    low = min(front, key=lambda s: s.objectives[0])
    high = min(front, key=lambda s: s.objectives[1])
    np.testing.assert_allclose(low.variables, [0.8, 0.2], atol=1e-5)
    np.testing.assert_allclose(high.variables, [0, 1], atol=1e-5)
    for s in front:
        w = np.array(s.variables)
        assert s.feasible and abs(w.sum() - 1) < 1e-9 and min(w) >= 0
        np.testing.assert_allclose(s.objectives, [w @ cov @ w, -mu @ w], atol=1e-10)


def test_equal_returns_collapses_to_gmv():
    front = convex_markowitz_front(np.array([0.02, 0.02]), np.diag([0.01, 0.04]), 5)
    for s in front:
        np.testing.assert_allclose(s.variables, [0.8, 0.2], atol=1e-5)
