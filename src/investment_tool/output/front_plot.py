"""The single Standard frontier view, generated after portfolio selection."""

from pathlib import Path
from platypus import Solution
from investment_tool.strategies.optimization_strategies.utils import (
    plot_solutions_with_markowitz_front,
)


def plot_final_front(
    front: list[Solution],
    reference: list[Solution] | None,
    directory: Path | None,
    date_label: str | None,
    selected: Solution | None,
    reference_objectives: list[list[float]] | None = None,
) -> None:
    if selected is None:
        return
    plot_solutions_with_markowitz_front(
        front,
        reference or [],
        "pareto_front_selected",
        directory,
        date_label,
        selected_solution=selected,
        reference_objectives=reference_objectives,
    )
