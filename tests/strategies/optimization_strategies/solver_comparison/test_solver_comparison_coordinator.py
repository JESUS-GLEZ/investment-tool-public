"""Unit tests for solver_comparison_coordinator (app layer)."""

from __future__ import annotations

import pytest

from investment_tool.app.solver_comparison_coordinator import (
    SolverComparisonParameters,
    _build_solver_configs,
    get_available_solvers,
)
from investment_tool.constants import DataSource, Frequency


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_params(**overrides: object) -> SolverComparisonParameters:
    from pathlib import Path

    defaults = dict(
        data_source=DataSource.LOCAL,
        frequency=Frequency.MONTHLY,
        output_directory=Path("/tmp/test"),
        population_size=20,
        n_generations=200,
        n_restarts=1,
        min_assets=2,
        num_assets=3,
        solvers=["GWASFGA", "NSGA-II"],
    )
    defaults.update(overrides)
    return SolverComparisonParameters(**defaults)  # type: ignore[arg-type]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestGetAvailableSolvers:
    def test_returns_list(self) -> None:
        result = get_available_solvers()
        assert isinstance(result, list)
        assert len(result) >= 4

    def test_contains_expected(self) -> None:
        result = get_available_solvers()
        for expected in ["GWASFGA", "NSGA-II", "NSGA-III", "SPEA2"]:
            assert expected in result


class TestBuildSolverConfigs:
    def test_builds_requested_solvers(self) -> None:
        params = _make_params(solvers=["GWASFGA", "NSGA-II"])
        configs = _build_solver_configs(params)
        assert len(configs) == 2
        labels = {c.label for c in configs}
        assert labels == {"GWASFGA", "NSGA-II"}

    def test_respects_population_size(self) -> None:
        params = _make_params(population_size=42, solvers=["NSGA-II"])
        configs = _build_solver_configs(params)
        assert configs[0].population_size == 42

    def test_respects_n_generations(self) -> None:
        params = _make_params(n_generations=999, solvers=["NSGA-II"])
        configs = _build_solver_configs(params)
        assert configs[0].n_generations == 999

    def test_unknown_solver_raises(self) -> None:
        params = _make_params(solvers=["NONEXISTENT"])
        with pytest.raises(ValueError, match="Unknown solver"):
            _build_solver_configs(params)

    def test_case_insensitive(self) -> None:
        params = _make_params(solvers=["gwasfga", "nsga-ii"])
        configs = _build_solver_configs(params)
        assert len(configs) == 2


class TestSolverComparisonParameters:
    def test_defaults(self) -> None:
        from pathlib import Path

        p = SolverComparisonParameters(
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=Path("/tmp/test"),
        )
        assert p.population_size == 300
        assert p.n_generations == 500_000
        assert p.n_restarts == 5
        assert p.min_assets == 2
        assert p.num_assets == 5
        assert p.include_markowitz_reference is True
        assert len(p.solvers) == 5

    def test_custom(self) -> None:
        from pathlib import Path

        p = SolverComparisonParameters(
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            output_directory=Path("/tmp/test"),
            solvers=["SPEA2"],
            population_size=50,
        )
        assert p.solvers == ["SPEA2"]
        assert p.population_size == 50


@pytest.mark.parametrize("cost_rate", [0.0, 0.001])
def test_seeded_coordinator_persists_baselines_and_times(
    monkeypatch, tmp_path, cost_rate
):
    import numpy as np
    import pandas as pd
    from api.result_presenter import _add_solver_figures
    from api.sanitize import sanitize
    from investment_tool.app import solver_comparison_coordinator as module
    from datetime import date, datetime
    from investment_tool.domain.model_data.entities.instance import Instance
    from investment_tool.domain.model_data.entities.period import Period
    from investment_tool.domain.model_data.entities.portfolio import Portfolio

    mu = pd.Series([0.01, 0.02, 0.03, 0.04])
    cov = pd.DataFrame(np.diag([0.01, 0.04, 0.09, 0.16]))
    model_data = module.ModelData()
    model_data.stocks = []
    period = Period(
        datetime(2025, 1, 1),
        pd.DataFrame(),
        decision_time=datetime.fromisoformat("2025-01-01T05:00:00Z"),
    )
    model_data.instance = Instance(
        date(2023, 1, 1), date(2024, 12, 31), Frequency.MONTHLY, period, None, period
    )
    model_data.portfolio = Portfolio(model_data.instance, 1000, cost_rate)
    monkeypatch.setattr(
        module.ModelDataFactory,
        "create_model_data",
        lambda **kwargs: model_data,
    )
    monkeypatch.setattr(module, "_build_problem_inputs", lambda data: (mu, cov))
    params = _make_params(
        output_directory=tmp_path,
        solvers=["NSGA-II"],
        n_generations=40,
        seed_max_subsets=15,
    )
    result = module.execute_solver_comparison(params)
    assert bool(result["trading_costs"]) == bool(cost_rate)
    for solution in result["results"][0].front:
        assert -solution.objectives[1] == pytest.approx(
            np.dot(solution.variables, mu) - cost_rate
        )
    assert result["reference_front"] and result["seed_front"]
    assert result["seed_diagnostics"]["subsets_evaluated"] <= 15
    assert (
        result["results"][0].metrics["total_time_with_seeds_s"]
        >= result["results"][0].wall_time_seconds
    )
    serialized = sanitize(_add_solver_figures(result))
    assert len(serialized["figures"][0]["data"]) == 3
    assert serialized["seed_front"][0]["feasible"]
    assert "problem" not in serialized["seed_front"][0]


def test_nsgaiii_population_matches_request_for_two_objectives():
    import numpy as np
    from investment_tool.strategies.optimization_strategies.model.bprc import (
        BalancedPortfoliosRiskContributions,
    )

    config = _build_solver_configs(
        _make_params(solvers=["NSGA-III"], population_size=300)
    )[0]
    problem = BalancedPortfoliosRiskContributions(
        np.array([0.01, 0.02, 0.03]), np.eye(3), 2, 3, hhi_as_constraint=True
    )
    algorithm = config.algorithm(
        problem, population_size=config.population_size, **config.algorithm_kwargs
    )
    assert algorithm.population_size == 300
