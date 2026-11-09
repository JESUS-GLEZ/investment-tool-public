"""Translate requests once and dispatch to existing application coordinators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

if TYPE_CHECKING:
    from investment_tool.app.execution_coordinator import Parameters
    from investment_tool.app.strategy_comparison_coordinator import (
        StrategyComparisonParameters,
    )
    from investment_tool.app.solver_comparison_coordinator import (
        SolverComparisonParameters,
    )
    from investment_tool.app.prediction_comparison_coordinator import (
        PredictionComparisonParameters,
    )
    from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
        ProgressCallback,
    )

    RunParameters: TypeAlias = (
        Parameters
        | StrategyComparisonParameters
        | SolverComparisonParameters
        | PredictionComparisonParameters
    )


@dataclass
class PreparedRun:
    run_type: str
    parameters: RunParameters
    coordinator: Callable[..., dict[str, Any]]

    def execute(
        self,
        progress: ProgressCallback | None = None,
        partial_sink: dict[str, Any] | None = None,
        partial_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.run_type in {"strategy_comparison", "solver_comparison"}:
            kwargs["partial_sink"] = partial_sink
        if self.run_type == "strategy_comparison":
            kwargs["partial_callback"] = partial_callback
        if progress is None:
            return self.coordinator(self.parameters, **kwargs)
        return self.coordinator(self.parameters, progress, **kwargs)


def prepare_run(run_type: str, payload: dict[str, Any], run_id: str) -> PreparedRun:
    # Lazy imports keep coordinator dependencies out of API startup/submission.
    from api import adapters, schemas

    if run_type == "execution":
        from investment_tool.app.execution_coordinator import execute

        return PreparedRun(
            run_type,
            adapters.build_exec_params(schemas.ExecutionRequest(**payload), run_id),
            execute,
        )
    if run_type == "strategy_comparison":
        from investment_tool.app.strategy_comparison_coordinator import (
            compare_strategies,
        )

        return PreparedRun(
            run_type,
            adapters.build_strategy_comparison_params(
                schemas.StrategyComparisonRequest(**payload), run_id
            ),
            compare_strategies,
        )
    if run_type == "solver_comparison":
        from investment_tool.app.solver_comparison_coordinator import (
            execute_solver_comparison,
        )

        return PreparedRun(
            run_type,
            adapters.build_solver_comparison_params(
                schemas.SolverComparisonRequest(**payload), run_id
            ),
            execute_solver_comparison,
        )
    if run_type == "prediction_comparison":
        from investment_tool.app.prediction_comparison_coordinator import (
            execute_prediction_comparison,
        )

        return PreparedRun(
            run_type,
            adapters.build_prediction_comparison_params(
                schemas.PredictionComparisonRequest(**payload), run_id
            ),
            execute_prediction_comparison,
        )
    raise ValueError(f"Unknown run_type: {run_type!r}")
