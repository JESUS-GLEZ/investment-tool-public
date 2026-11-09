"""Bounded Multi-Greedy + NSGA-II optimization for investment strategies."""

from __future__ import annotations

import logging
import hashlib
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np
from platypus import NSGAII, Solution, nondominated_truncate

from investment_tool.defaults import STRATEGY_OPTIMIZER
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.optimization_strategies.convex_frontier import (
    convex_markowitz_front,
)
from investment_tool.strategies.optimization_strategies.multi_greedy import (
    SeededPortfolioGenerator,
    pooled_multi_greedy_seeds as multi_greedy_seeds,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)
from investment_tool.strategies.optimization_strategies.platypus_optimization import (
    PlatypusOptimizationStrategy,
)
from investment_tool.strategies.optimization_strategies.utils import merge_nondominated
from investment_tool.strategies.optimization_strategies.support_refinement import (
    refine_supports,
)

logger = logging.getLogger(__name__)


@contextmanager
def _seeded_randomness(seed: int) -> Iterator[None]:
    """Isolate the RNGs used by Platypus and the portfolio variator/sampler.

    Production workers use separate prefork processes. This context must not
    be shared by concurrent optimization threads in the same process.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _front_summary(front: list[Solution], labels: list[str]) -> dict[str, Any]:
    if not front:
        return {"size": 0}
    best = max(
        front,
        key=lambda s: -float(s.objectives[1])
        / np.sqrt(max(float(s.objectives[0]), 1e-12)),
    )
    return {
        "size": len(front),
        "min_variance": min(float(s.objectives[0]) for s in front),
        "max_expected_return": max(-float(s.objectives[1]) for s in front),
        "sharpe_objectives": [float(v) for v in best.objectives],
        "sharpe_weights": {
            labels[i]: float(w) for i, w in enumerate(best.variables) if w > 0
        },
    }


def _merge_search_archive(
    fronts: list[list[Solution]], capacity: int
) -> list[Solution]:
    """Protect useful portfolio tradeoffs before pruning a rank-zero archive.

    Crowding alone can evict the best Sharpe or a low-risk high-return point.
    Protect extrema and several return levels before pruning remaining
    three-objective tradeoffs. Anchors do not depend on the UI selector.
    """
    front: list[Solution] = merge_nondominated(
        fronts, target_size=sum(map(len, fronts))
    )
    if len(front) <= capacity:
        return front
    anchors: list[Solution] = []

    def retain(solution: Solution) -> None:
        if len(anchors) < capacity and not any(solution is s for s in anchors):
            anchors.append(solution)

    for objective in range(len(front[0].objectives)):
        retain(min(front, key=lambda s: s.objectives[objective]))
    retain(
        max(
            front, key=lambda s: -s.objectives[1] / np.sqrt(max(s.objectives[0], 1e-16))
        )
    )
    low = -min(front, key=lambda s: (s.objectives[0], s.objectives[1])).objectives[1]
    high = max(-s.objectives[1] for s in front)
    for fraction in [0.25, 0.5, 0.6, 0.75, 0.85, 0.9]:
        target = low + fraction * (high - low)
        retain(
            min(
                (s for s in front if -s.objectives[1] >= target),
                key=lambda s: (s.objectives[0], s.objectives[1]),
            )
        )
    remaining = [s for s in front if not any(s is a for a in anchors)]
    return anchors + list(nondominated_truncate(remaining, capacity - len(anchors)))


class MultiGreedyOptimizationStrategy(PlatypusOptimizationStrategy):
    """Fresh current-window RP seeds, independent NSGA-II runs, merged front.

    Inherits diagnostic plotting and selection integration. BPRC objectives and
    constraints come from the calling investment strategy unchanged. Previous
    period weights are deliberately not reused: asset universes may change.
    ``n_generations`` is the evaluation ceiling per restart. Seeds are generated
    once; each restart gets the original refinement time allowance. A finite
    allowance A with seed fraction f and R restarts permits A*(f + R*(1-f))
    search seconds overall; it is not one total allowance shared by all restarts.
    Set ``search_seconds=None`` for fixed subset/evaluation budgets without a
    wall-clock cutoff. Fixed seeds alone cannot reproduce time-truncated runs.
    Seed and generation boundaries may slightly overrun the wall-time limit.
    When refinement is enabled, 10% of each restart time allowance is reserved
    for the shared fixed-support phase. It samples small local fronts and a
    common return grid, preserving the calling model and validating candidates.
    The global archive retains all seed tradeoffs initially and keeps generation
    discoveries before the compact per-restart plot archives are pruned.
    Reference construction and plotting are outside search time.
    """

    search_version = "pooled-greedy-rotating-initial-seeds-v6"
    search_seconds: float | None = STRATEGY_OPTIMIZER.search_seconds
    random_seed = STRATEGY_OPTIMIZER.random_seed
    seed_time_fraction = STRATEGY_OPTIMIZER.seed_time_fraction
    seed_max_subsets = STRATEGY_OPTIMIZER.seed_max_subsets
    seed_beam_width = STRATEGY_OPTIMIZER.seed_beam_width
    refinement_supports = STRATEGY_OPTIMIZER.refinement_supports

    def optimize(
        self, progress_callback: ProgressCallback | None = None
    ) -> OptimizationResult:
        started = time.monotonic()
        if self.n_restarts < 1 or self.n_generations < 1:
            raise ValueError("Restarts and evaluation budgets must be positive")
        if self.search_seconds is not None and self.search_seconds <= 0:
            raise ValueError("Search seconds must be positive or None")
        if not 0 < self.seed_time_fraction < 1:
            raise ValueError("Seed time fraction must be between zero and one")
        if self.refinement_supports < 0:
            raise ValueError("Refinement support budget must be nonnegative")
        allowance = self.search_seconds
        total_seconds = (
            allowance
            * (
                self.seed_time_fraction
                + self.n_restarts * (1 - self.seed_time_fraction)
            )
            if allowance is not None
            else None
        )
        deadline = (
            started + total_seconds if total_seconds is not None else float("inf")
        )
        seed_deadline = (
            started + allowance * self.seed_time_fraction
            if allowance is not None
            else float("inf")
        )
        problem = self.model(
            mu=self.expected_returns.to_numpy(),
            cov=self.covariance_matrix.to_numpy(),
            **self.model_kwargs,
            **self.cost_kwargs,
        )

        def report(fraction: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(
                    min(0.99, (time.monotonic() - started) / total_seconds)
                    if total_seconds is not None
                    else min(0.99, fraction),
                    message,
                )

        report(0.0, "Generating current-window Multi-Greedy balanced-portfolio seeds")
        discovery: dict[str, Any] = {}
        seeds, attempted = multi_greedy_seeds(
            problem,
            max_subsets=self.seed_max_subsets,
            beam_width=self.seed_beam_width,
            progress=lambda fraction, message: report(0.25 * fraction, message),
            stop=lambda: time.monotonic() >= seed_deadline,
            deadline=seed_deadline,
            diagnostics=discovery,
        )
        seed_seconds = time.monotonic() - started
        seed_time_limited = time.monotonic() >= seed_deadline or any(
            path["time_limited"] for path in discovery.get("paths", [])
        )
        kwargs = dict(self.algorithm_kwargs)
        kwargs.pop("generator", None)
        kwargs.pop("divisions_outer", None)
        archives: list[list[Solution]] = []
        diagnostics: list[dict[str, Any]] = []
        labels = [str(symbol) for symbol in self.expected_returns.index]
        ordered_seeds = sorted(seeds, key=lambda s: s.objectives[0])
        # Bounded per-run archives; final selection receives the rank-zero union
        # (capacity at least n_restarts*population_size and the seed-front size).
        archive_size = max(len(seeds), self.population_size * self.n_restarts)
        front = _merge_search_archive([seeds], archive_size)
        for restart in range(self.n_restarts):
            seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.random_seed}:{self.date_label}:{restart}".encode()
                ).digest()[:4],
                "big",
            )
            restart_started = time.monotonic()
            restart_deadline = min(
                deadline,
                restart_started
                + allowance
                * (1 - self.seed_time_fraction)
                * (0.9 if self.refinement_supports else 1)
                if allowance is not None
                else float("inf"),
            )
            run_front = merge_nondominated([seeds], target_size=self.population_size)
            pending: list[Solution] = []
            last_report = restart_started
            generator = SeededPortfolioGenerator(
                ordered_seeds, self.population_size // 2, offset=restart
            )
            with _seeded_randomness(seed):
                algorithm = NSGAII(
                    problem,
                    population_size=self.population_size,
                    generator=generator,
                    **kwargs,
                )
                while (
                    time.monotonic() < restart_deadline
                    and algorithm.nfe < self.n_generations
                ):
                    algorithm.step()
                    # Retain discoveries before pruning the plot archive.
                    # Batch ten generations to bound memory and merge cost.
                    candidates = deepcopy(
                        list(algorithm.result), memo={id(problem): problem}
                    )
                    pending.extend(candidates)
                    if len(pending) >= 10 * self.population_size:
                        front = _merge_search_archive([front, pending], archive_size)
                        pending = []
                    run_front = merge_nondominated(
                        [run_front, candidates],
                        target_size=self.population_size,
                    )
                    if time.monotonic() - last_report >= 1:
                        report(
                            0.25
                            + 0.6
                            * (restart + algorithm.nfe / self.n_generations)
                            / self.n_restarts,
                            f"NSGA-II restart {restart + 1}/{self.n_restarts}: {algorithm.nfe}/{self.n_generations} evaluations",
                        )
                        last_report = time.monotonic()
            archives.append(run_front)
            front = _merge_search_archive([front, pending, run_front], archive_size)
            diagnostics.append(
                {
                    "restart": restart + 1,
                    "seed": seed,
                    "nfe": algorithm.nfe,
                    "initial_seed_indices": generator.injected_indices,
                    "elapsed_seconds": time.monotonic() - restart_started,
                    "termination": "evaluation_budget"
                    if algorithm.nfe >= self.n_generations
                    else "wall_time",
                    "run_front": _front_summary(run_front, labels),
                    "merged_front": _front_summary(front, labels),
                }
            )
            report(
                0.25 + 0.6 * (restart + 1) / self.n_restarts,
                f"NSGA-II restart {restart + 1}/{self.n_restarts} complete",
            )
        pre_refinement_summary = _front_summary(front, labels)
        refined, refinement = refine_supports(
            problem,
            seeds + front,
            max_supports=self.refinement_supports,
            stop=lambda: time.monotonic() >= deadline,
            progress=lambda fraction, message: report(0.9 + 0.09 * fraction, message),
        )
        front = _merge_search_archive([front, refined], archive_size)
        elapsed = time.monotonic() - started
        if not front:
            raise RuntimeError(
                "No feasible portfolio found within the optimization budget"
            )
        # Explicit guard supplements the model's risk-contribution constraints.
        if any(
            not problem.l_assets
            <= np.count_nonzero(np.asarray(s.variables) > problem.activity_tol)
            <= problem.u_assets
            for s in front
        ):
            raise RuntimeError(
                "Optimizer produced a portfolio outside the cardinality bounds"
            )
        logger.info(
            "Multi-Greedy + NSGA-II: subsets=%d seeds=%d seed_seconds=%.3f nfe=%d search_seconds=%.3f front=%d",
            attempted,
            len(seeds),
            seed_seconds,
            sum(d["nfe"] for d in diagnostics),
            elapsed,
            len(front),
        )
        reference = None
        if self.include_markowitz_front:
            try:
                reference = convex_markowitz_front(
                    problem.mu, problem.cov, **self.cost_kwargs
                )
            except Exception as exc:
                # The unconstrained MV curve is diagnostic only. A numerical
                # failure must not discard the feasible BPRC portfolio.
                logger.warning("Skipping MV reference: %s", exc)
        # Return compact feasible archives; no old-period objectives enter this result.
        result = OptimizationResult(
            front=front,
            all_archives=archives,
            markowitz_front=reference,
            search_diagnostics={
                "search_version": self.search_version,
                "archive_capacity": archive_size,
                "base_seed": self.random_seed,
                "configured_restarts": self.n_restarts,
                "population_size": self.population_size,
                "seed_max_subsets": self.seed_max_subsets,
                "seed_beam_width": self.seed_beam_width,
                "seed_discovery": discovery,
                # Sparse weights once, with actual injected indices per restart;
                # no new artifact or dense per-generation portfolio dump.
                "initial_seed_portfolios": [
                    {labels[i]: float(w) for i, w in enumerate(s.variables) if w > 0}
                    for s in ordered_seeds
                ],
                "seed_front": _front_summary(seeds, labels),
                "pre_refinement_front": pre_refinement_summary,
                "final_front": _front_summary(front, labels),
                "refinement": refinement,
                "search_time_limit_seconds": total_seconds,
                "date_label": self.date_label,
                "seed_subsets": attempted,
                "seed_seconds": seed_seconds,
                "seed_time_limited": seed_time_limited,
                "evaluation_budget_per_restart": self.n_generations,
                "time_limited": seed_time_limited
                or any(d["termination"] == "wall_time" for d in diagnostics)
                or refinement["time_limited"],
                "restarts": diagnostics,
            },
        )
        try:
            self._plot_optimization_results(front, archives, reference)
        except Exception:
            logger.warning(
                "Could not save optimization diagnostic plots", exc_info=True
            )
        if progress_callback:
            progress_callback(1.0, "Multi-Greedy + NSGA-II complete")
        return result
