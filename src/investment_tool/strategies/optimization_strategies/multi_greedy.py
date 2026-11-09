"""Bounded Multi-Greedy risk-parity subset search (Cesarone et al., 2018).

An adaptation of sections 4–5: retain a beam for each normalized mean–variance
utility, explore add/drop/swap neighborhoods, and cache the unique RP portfolio
of each subset. It is a heuristic, not a certificate of BPRC optimality.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import islice
import time
from typing import Any, Literal

import numpy as np
from platypus import Archive, Solution
from scipy.optimize import minimize

from investment_tool.constants import PortfolioBalance
from investment_tool.strategies.optimization_strategies.custom_algorithm.portfolio_sampling import (
    PortfolioGenerator,
)
from investment_tool.strategies.optimization_strategies.utils import merge_nondominated

DiscoveryPolicy = Literal["utility", "cardinality_sharpe"]


@dataclass(frozen=True, slots=True)
class _Seed:
    """Retain evaluated coordinates on a support, rather than N dense weights."""

    weights: tuple[float, ...]
    objectives: tuple[float, ...]
    constraints: tuple[float, ...]

    @classmethod
    def pack(cls, solution: Solution, subset: tuple[int, ...]) -> _Seed:
        return cls(
            tuple(solution.variables[i] for i in subset),
            tuple(solution.objectives),
            tuple(solution.constraints),
        )

    def unpack(self, problem: Any, subset: tuple[int, ...]) -> Solution:
        solution = Solution(problem)
        solution.variables[:] = [0.0] * problem.nvars
        for i, weight in zip(subset, self.weights):
            solution.variables[i] = weight
        solution.objectives[:] = self.objectives
        solution.constraints[:] = self.constraints
        solution.constraint_violation = 0.0
        solution.feasible = True
        solution.evaluated = True
        return solution


def risk_parity_weights(cov: np.ndarray) -> np.ndarray | None:
    """Solve the positive log-barrier RP problem, then normalize weights."""
    scale = float(np.mean(np.diag(cov)))
    if scale <= 0:
        return None
    matrix = cov / scale
    result = minimize(
        lambda x: float(0.5 * x @ matrix @ x - np.log(x).sum()),
        1 / np.sqrt(np.maximum(np.diag(matrix), 1e-12)),
        jac=lambda x: matrix @ x - 1 / x,
        bounds=[(1e-10, None)] * len(cov),
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-9},
    )
    result_weights = np.asarray(result.x, dtype=float)
    w: np.ndarray = result_weights / result_weights.sum()
    contributions = w * (cov @ w)
    if not result.success or not np.all(np.isfinite(w)) or contributions.sum() <= 0:
        return None
    if np.max(np.abs(contributions / contributions.sum() - 1 / len(w))) > 1e-5:
        return None
    return w


def _neighbors(
    subset: tuple[int, ...], n: int, lower: int, upper: int
) -> Iterator[tuple[int, ...]]:
    members = set(subset)
    outside = sorted(set(range(n)) - members)
    if len(subset) < upper:
        for j in outside:
            yield tuple(sorted((*subset, j)))
    if len(subset) > lower:
        for i in subset:
            yield tuple(j for j in subset if j != i)
    # Swaps are essential at the cardinality boundaries.
    for i in subset:
        for j in outside:
            yield tuple(sorted((members - {i}) | {j}))


def _hybrid_seed_weights(
    cov: np.ndarray,
    risk_parity: np.ndarray | None,
    lower: float,
    upper: float,
) -> np.ndarray | None:
    """Search a small deterministic RP/equal-weight segment on one support.

    RP alone may violate capital bounds even when another allocation on the
    support satisfies both. This is seed discovery, not a feasibility proof;
    NSGA-II and constrained refinement still explore other allocations.
    """
    equal = np.full(len(cov), 1 / len(cov))
    candidates = (
        [equal]
        if risk_parity is None
        else [(1 - t) * risk_parity + t * equal for t in np.linspace(0, 1, 17)]
    )
    feasible: list[tuple[float, np.ndarray]] = []
    for w in candidates:
        variance = float(w @ cov @ w)
        if variance <= 0 or not np.all(np.isfinite(w)):
            continue
        rc = w * (cov @ w) / variance
        if (
            np.all(w >= lower - 1e-12)
            and np.all(w <= upper + 1e-12)
            and np.all(rc >= lower - 1e-12)
            and np.all(rc <= upper + 1e-12)
        ):
            feasible.append((float(rc @ rc), w))
    return min(feasible, key=lambda item: item[0])[1] if feasible else None


def multi_greedy_seeds(
    problem: Any,
    *,
    max_subsets: int = 2000,
    beam_width: int = 5,
    progress: Callable[[float, str], None] | None = None,
    stop: Callable[[], bool] | None = None,
    discovery: DiscoveryPolicy = "utility",
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[Solution], int]:
    """Return feasible nondominated seeds and attempted subset count.

    Budget includes failed RP solves. Deterministic exploration uses fixed
    random starts plus return/variance rankings; every candidate is validated
    against the actual BPRC evaluator (including activity tolerance and HHI).
    """
    if max_subsets < 1 or beam_width < 1:
        raise ValueError("Subset budget and beam width must be positive")
    if discovery not in ("utility", "cardinality_sharpe"):
        raise ValueError("Unknown greedy discovery policy")
    n = problem.nvars
    lower, upper = problem.l_assets, problem.u_assets
    cache: dict[tuple[int, ...], _Seed | None] = {}
    mu, cov = problem.mu, problem.cov
    rng = np.random.default_rng(0)

    def visit(subset: tuple[int, ...]) -> None:
        if (
            subset in cache
            or len(cache) >= max_subsets
            or (stop is not None and stop())
        ):
            return
        cache[subset] = None
        idx = list(subset)
        weights = (
            np.full(len(idx), 1 / len(idx))
            if getattr(problem, "balance", PortfolioBalance.RISK_CONTRIBUTIONS)
            == PortfolioBalance.CAPITAL_WEIGHTS
            else risk_parity_weights(cov[np.ix_(idx, idx)])
        )
        if getattr(problem, "balance", None) == PortfolioBalance.HYBRID:
            weights = _hybrid_seed_weights(
                cov[np.ix_(idx, idx)], weights, *problem.balance_bounds
            )
        if weights is not None:
            solution = Solution(problem)
            w = np.zeros(n)
            w[idx] = weights
            solution.variables[:] = w.tolist()
            solution.evaluate()
            if solution.feasible and np.all(np.isfinite(solution.objectives)):
                cache[subset] = _Seed.pack(solution, subset)
        if progress is not None and len(cache) % 25 == 0:
            progress(
                len(cache) / max_subsets,
                f"Multi-Greedy: {len(cache)}/{max_subsets} subsets",
            )

    costs = getattr(problem, "trading_costs", None)
    # Singleton net-return ranking differs from gross return by 2*c*held_weight
    # plus a shared constant. Preserve an executable incumbent as another seed.
    ranked_mu = mu if costs is None else mu + 2 * costs.rate * costs.current_weights
    incumbent: list[Solution] = []
    if costs is not None and costs.current_weights.sum() > 0:
        held = Solution(problem)
        held.variables[:] = (
            costs.current_weights / costs.current_weights.sum()
        ).tolist()
        held.evaluate()
        if held.feasible and np.all(np.isfinite(held.objectives)):
            incumbent.append(held)
    rankings = [np.argsort(-ranked_mu), np.argsort(np.diag(cov))]
    for k in range(lower, upper + 1):
        for order in rankings:
            visit(tuple(sorted(order[:k].tolist())))
        for _ in range(beam_width):
            visit(tuple(sorted(rng.choice(n, k, replace=False).tolist())))

    expanded: set[tuple[int, ...]] = set()
    variance_scale = max(float(np.max(np.diag(cov))), 1e-12)
    return_scale = max(
        float(np.ptp(mu)), 2 * costs.rate if costs is not None else 0, 1e-12
    )
    continuations = 0

    def score(key: tuple[int, ...], weight: float) -> tuple[float, tuple[int, ...]]:
        solution = cache[key]
        assert solution is not None
        utility = (
            weight * solution.objectives[0] / variance_scale
            + (1 - weight) * solution.objectives[1] / return_scale
        )
        return utility, key

    def sharpe_score(key: tuple[int, ...]) -> tuple[float, tuple[int, ...]]:
        solution = cache[key]
        assert solution is not None
        return (
            solution.objectives[1] / np.sqrt(max(solution.objectives[0], 1e-30)),
            key,
        )

    while len(cache) < max_subsets and not (stop is not None and stop()):
        feasible = [key for key, value in cache.items() if value is not None]
        if not feasible:
            break
        selected: set[tuple[int, ...]] = set()
        for risk_weight in np.linspace(0, 1, 11):
            if discovery == "utility":
                selected.update(
                    sorted(feasible, key=lambda key: score(key, risk_weight))[
                        :beam_width
                    ]
                )
            else:
                for cardinality in range(lower, upper + 1):
                    available = [
                        key
                        for key in feasible
                        if len(key) == cardinality and key not in expanded
                    ]
                    selected.update(
                        sorted(available, key=lambda key: score(key, risk_weight))[
                            :beam_width
                        ]
                    )
        if discovery == "cardinality_sharpe":
            for cardinality in range(lower, upper + 1):
                available = [
                    key
                    for key in feasible
                    if len(key) == cardinality and key not in expanded
                ]
                selected.update(
                    sorted(
                        available,
                        key=sharpe_score,
                    )[:beam_width]
                )
        pending = sorted(selected - expanded)
        if not pending:
            # The preferred beam may be exhausted while other discovered
            # feasible supports still have unexplored neighborhoods.
            unexpanded = [key for key in feasible if key not in expanded]
            for risk_weight in np.linspace(0, 1, 11):
                selected.update(
                    sorted(unexpanded, key=lambda key: score(key, risk_weight))[
                        :beam_width
                    ]
                )
            pending = sorted(selected - expanded)
            if not pending:
                break
            continuations += 1
        expanded.update(pending)
        # Round-robin neighborhoods prevent one beam member spending the budget.
        streams = [_neighbors(key, n, lower, upper) for key in pending]
        while (
            streams and len(cache) < max_subsets and not (stop is not None and stop())
        ):
            remaining = []
            for stream in streams:
                candidate = list(islice(stream, 1))
                if candidate:
                    visit(candidate[0])
                    remaining.append(stream)
            streams = remaining
    # Match merge_nondominated's insertion order and rounded-vector deduplication,
    # but stream dense candidates through its same Platypus rank-zero archive.
    # Materializing the full cache here would recreate the seeding memory peak.
    archive = Archive()
    seen: set[tuple[tuple[int, float], ...]] = set()

    def retain(subset: tuple[int, ...], seed: _Seed) -> None:
        rounded = np.round(np.asarray(seed.weights, dtype=float), 8)
        key = tuple((i, float(w)) for i, w in zip(subset, rounded) if w != 0)
        if key not in seen:
            seen.add(key)
            archive.add(seed.unpack(problem, subset))

    for subset, seed in cache.items():
        if seed is not None:
            retain(subset, seed)
    for solution in incumbent:
        subset = tuple(int(i) for i in np.flatnonzero(solution.variables))
        retain(subset, _Seed.pack(solution, subset))
    front = merge_nondominated([list(archive)], target_size=max_subsets)
    if diagnostics is not None:
        diagnostics.update(
            policy=discovery,
            max_subsets=max_subsets,
            attempted_subsets=len(cache),
            continuations=continuations,
            front_size=len(front),
            termination=(
                "subset_budget"
                if len(cache) >= max_subsets
                else "stop_requested"
                if stop is not None and stop()
                else "discovered_candidates_exhausted"
            ),
        )
    return front, len(cache)


def pooled_multi_greedy_seeds(
    problem: Any,
    *,
    max_subsets: int,
    beam_width: int = 5,
    progress: Callable[[float, str], None] | None = None,
    stop: Callable[[], bool] | None = None,
    deadline: float = float("inf"),
    diagnostics: dict[str, Any] | None = None,
) -> tuple[list[Solution], int]:
    """Pool two complementary discovery paths within one total attempt cap.

    Each path has its own deterministic starts and subset cache, matching the
    independent library construction. A subset revisited by the other path
    counts again. Split work equally (odd remainder to utility); finite time
    is shared so the first path cannot consume the second path's allowance.
    Never prune the union to the GA population size.
    """
    if max_subsets < 1 or beam_width < 1:
        raise ValueError("Subset budget and beam width must be positive")
    started = time.monotonic()
    midpoint = started + (deadline - started) / 2
    fronts: list[list[Solution]] = []
    paths: list[dict[str, Any]] = []
    attempted = allocated = 0
    budgets: tuple[tuple[DiscoveryPolicy, int], ...] = (
        ("utility", (max_subsets + 1) // 2),
        ("cardinality_sharpe", max_subsets // 2),
    )
    for index, (policy, budget) in enumerate(budgets):
        if not budget:
            continue
        path_deadline = midpoint if index == 0 else deadline
        path_diagnostics: dict[str, Any] = {}

        def report(fraction: float, message: str) -> None:
            if progress is not None:
                progress(
                    (allocated + budget * fraction) / max_subsets,
                    f"{policy}: {message}",
                )

        front, count = multi_greedy_seeds(
            problem,
            max_subsets=budget,
            beam_width=beam_width,
            discovery=policy,
            diagnostics=path_diagnostics,
            progress=report,
            stop=lambda: time.monotonic() >= path_deadline
            or (stop is not None and stop()),
        )
        path_diagnostics["time_limited"] = (
            path_diagnostics.get("termination") == "stop_requested"
            and time.monotonic() >= path_deadline
        )
        if path_diagnostics["time_limited"]:
            path_diagnostics["termination"] = "wall_time"
        fronts.append(front)
        paths.append(path_diagnostics)
        attempted += count
        allocated += budget
    pooled = merge_nondominated(fronts, target_size=sum(map(len, fronts)))
    if diagnostics is not None:
        diagnostics.update(
            paths=paths,
            attempted_subsets=attempted,
            front_size=len(pooled),
            accounting="sum_of_path_attempts_including_overlap",
        )
    return pooled, attempted


class SeededPortfolioGenerator(PortfolioGenerator):
    """Inject a bounded seed prefix, then use portfolio-aware random sampling."""

    def __init__(self, seeds: list[Solution], limit: int, *, offset: int = 0) -> None:
        self.weights = [list(s.variables) for s in seeds]
        self.limit = max(0, min(limit, len(seeds)))
        self.index = 0
        self.seed_indices = [
            (int(i * (len(seeds) - 1) / max(1, self.limit - 1)) + offset) % len(seeds)
            for i in range(self.limit)
        ]
        self.injected_indices: list[int] = []

    def generate(self, problem: Any) -> Solution:
        if self.index < self.limit:
            # Spread the injected seeds across the risk-ordered seed front.
            index = self.seed_indices[self.index]
            solution = Solution(problem)
            solution.variables[:] = self.weights[index]
            self.index += 1
            self.injected_indices.append(index)
            return solution
        return super().generate(problem)
