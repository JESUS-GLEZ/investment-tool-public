"""Solver benchmarks and investment strategies share their risk estimator."""


def test_comparison_uses_strategy_covariance_without_truncating_old_history():
    from datetime import datetime

    import numpy as np
    import pandas as pd

    from investment_tool.app.solver_comparison_coordinator import _build_problem_inputs
    from investment_tool.constants import StockFields
    from investment_tool.domain.model_data.entities.stock import Stock
    from investment_tool.domain.model_data.model_data import ModelData
    from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
        AveragePredictionStrategy,
    )
    from tests.strategies.orchestrators.test_base_investment_strategy import (
        ConcreteStrategy,
        _make_instance,
        _make_stock,
    )

    data = _make_stock("OLD", n_periods=60).data.copy()
    data[StockFields.CLOSE] *= 1 + 0.05 * np.sin(np.arange(60))
    md = ModelData()
    md.stocks = [
        Stock("OLD", data),
        Stock("NEW", data.iloc[-13:]),
        Stock("IPO", data.iloc[-3:]),
    ]
    mu, cov = _build_problem_inputs(md)
    strategy = ConcreteStrategy(_make_instance(datetime(2027, 1, 1)), md.stocks)
    pd.testing.assert_frame_equal(cov, strategy.build_covariance_matrix())
    pd.testing.assert_series_equal(
        mu,
        pd.Series(AveragePredictionStrategy(md.stocks[:2]).predict_returns()),
        check_names=False,
    )
    assert cov.attrs["covariance_diagnostics"]["observed_returns"] == {
        "OLD": 59,
        "NEW": 12,
    }


def test_covariance_regularization_change_invalidates_candidate_front(tmp_path):
    import numpy as np
    import pandas as pd

    from investment_tool.strategies.optimization_strategies.front_cache import (
        CandidateFrontCache,
    )
    from investment_tool.utils import build_exp_weighted_covariance_matrix
    from tests.strategies.optimization_strategies.test_seeded_optimization import (
        make_optimizer,
    )

    frame = pd.DataFrame(np.random.default_rng(7).normal(0, 0.04, (24, 4)))
    first, second = make_optimizer(), make_optimizer()
    first.covariance_matrix = build_exp_weighted_covariance_matrix(frame, shrinkage=0.1)
    second.covariance_matrix = build_exp_weighted_covariance_matrix(
        frame, shrinkage=0.2
    )
    cache = CandidateFrontCache(tmp_path, "optimization")
    assert cache._key(first) != cache._key(second)
