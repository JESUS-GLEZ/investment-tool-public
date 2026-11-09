"""The frontend risk-free rate must reach execution and comparison reports."""

from api.adapters import build_exec_params, build_strategy_comparison_params
from api.schemas import ExecutionRequest, StrategyComparisonRequest


def test_frontend_annual_risk_free_rate_is_not_discarded(tmp_path, monkeypatch):
    monkeypatch.setattr("api.adapters._OUTPUT_BASE", tmp_path)
    execution = build_exec_params(
        ExecutionRequest(risk_free_rate=0.045), "metric-config"
    )
    comparison = build_strategy_comparison_params(
        StrategyComparisonRequest(risk_free_rate=0.045), "metric-config"
    )
    assert execution.cfg.model.risk_free_rate == 0.045
    assert comparison.cfg.model.risk_free_rate == 0.045


def test_optimization_selection_reaches_comparison_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr("api.adapters._OUTPUT_BASE", tmp_path)
    comparison = build_strategy_comparison_params(
        StrategyComparisonRequest(
            optimization_selection="return_with_risk_budget",
            optimization_volatility_increase=0.25,
        ),
        "selection-config",
    )
    assert comparison.optimization_selection == "return_with_risk_budget"
    assert comparison.optimization_volatility_increase == 0.25


def test_old_comparison_requests_keep_sharpe_selection():
    assert StrategyComparisonRequest().optimization_selection == "sharpe"
