"""Model-aware sampling and persistent feasible Pareto archives."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from platypus import Archive, Evaluator, RandomGenerator, Solution


class PortfolioGenerator(RandomGenerator):
    """Use a model's portfolio sampler, falling back for generic problems."""

    def generate(self, problem: Any) -> Solution:
        sampler = getattr(problem, "random", None)
        if not callable(sampler):
            return super().generate(problem)
        solution = sampler()
        # Model.evaluate alone does not populate Platypus feasibility metadata.
        # Let the algorithm evaluate through Problem.__call__ and count the NFE.
        solution.evaluated = False
        return solution


class FeasibleParetoArchive(Archive):
    """Keep finite feasible nondominated snapshots, without objective duplicates.

    Unbounded during a run so earlier nondominated discoveries are not lost.
    The strategy applies its output-size limit only after merging restarts.
    """

    def add(self, solution: Solution) -> bool:
        if (
            not solution.evaluated
            or solution.constraint_violation != 0
            or not np.all(np.isfinite(solution.objectives))
        ):
            return False
        if any(list(s.objectives) == list(solution.objectives) for s in self):
            return False
        # Preserve the model identity without copying its covariance for each point.
        snapshot = deepcopy(solution, {id(solution.problem): solution.problem})
        return bool(super().add(snapshot))


class ArchivingEvaluator(Evaluator):
    """Record every evaluated candidate before environmental selection."""

    def __init__(self, evaluator: Evaluator) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.archive = FeasibleParetoArchive()

    def evaluate_all(self, jobs: Any, **kwargs: Any) -> list[Any]:
        results = list(self.evaluator.evaluate_all(jobs, **kwargs))
        for result in results:
            self.archive.add(result.solution)
        return results
