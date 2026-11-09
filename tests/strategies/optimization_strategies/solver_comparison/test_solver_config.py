"""Unit tests for SolverConfig dataclass."""

from __future__ import annotations

from platypus import NSGAII

from investment_tool.strategies.optimization_strategies.solver_comparison.solver_config import (
    SolverConfig,
)


class TestSolverConfig:
    """Test SolverConfig construction and immutability."""

    def test_basic_construction(self) -> None:
        cfg = SolverConfig(label="NSGA-II", algorithm=NSGAII)
        assert cfg.label == "NSGA-II"
        assert cfg.algorithm is NSGAII
        assert cfg.algorithm_kwargs == {}
        assert cfg.population_size == 100
        assert cfg.n_generations == 10_000
        assert cfg.n_restarts == 3

    def test_custom_params(self) -> None:
        cfg = SolverConfig(
            label="Custom",
            algorithm=NSGAII,
            algorithm_kwargs={"foo": "bar"},
            population_size=200,
            n_generations=5000,
            n_restarts=5,
        )
        assert cfg.population_size == 200
        assert cfg.n_generations == 5000
        assert cfg.n_restarts == 5
        assert cfg.algorithm_kwargs == {"foo": "bar"}

    def test_frozen(self) -> None:
        cfg = SolverConfig(label="Test", algorithm=NSGAII)
        try:
            cfg.label = "changed"  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except AttributeError:
            pass
