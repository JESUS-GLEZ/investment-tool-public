"""Deterministic weight refinement on a bounded set of small asset supports.

Subset discovery remains Multi-Greedy/NSGA-II's job. These local constrained
solves fill mean–variance gaps without changing the BPRC objectives or claiming
global optimality. Every result is reevaluated by the actual portfolio model.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import time

import numpy as np
from platypus import Solution
from scipy.optimize import minimize

from investment_tool.constants import PortfolioBalance
from investment_tool.domain.trading_costs import (
    expected_net_return,
    net_return_gradient,
)

from .model.bprc import BalancedPortfoliosRiskContributions


def _support(solution: Solution, tolerance: float) -> tuple[int, ...]:
    return tuple(np.flatnonzero(np.asarray(solution.variables) > tolerance).tolist())


def _representatives(
    problem: BalancedPortfoliosRiskContributions,
    candidates: list[Solution],
    budget: int,
) -> list[Solution]:
    """Cover return levels and each cardinality, with deterministic tie breaks."""
    groups = [
        [s for s in candidates if len(_support(s, problem.activity_tol)) == k]
        for k in range(problem.l_assets, problem.u_assets + 1)
    ]
    priorities: list[list[Solution]] = []
    global_low = (
        -min(candidates, key=lambda s: (s.objectives[0], s.objectives[1])).objectives[1]
        if candidates
        else 0.0
    )
    global_high = max((-s.objectives[1] for s in candidates), default=0.0)
    for group in groups:
        if not group:
            continue
        ordered = sorted(group, key=lambda s: (tuple(s.objectives), tuple(s.variables)))
        low, high = (
            min(-s.objectives[1] for s in group),
            max(-s.objectives[1] for s in group),
        )
        queue = [
            min(ordered, key=lambda s: (s.objectives[0], s.objectives[1])),
            min(ordered, key=lambda s: s.objectives[1]),
            max(
                ordered,
                key=lambda s: -s.objectives[1] / np.sqrt(max(s.objectives[0], 1e-16)),
            ),
        ]
        # Prioritise supports near shared high-return targets, including nearby
        # lower-return candidates whose weights may reach those targets.
        for fraction in [0.25, 0.5, 0.6, 0.75, 0.85, 0.9]:
            target = global_low + fraction * (global_high - global_low)
            eligible = [
                s
                for s in ordered
                if -s.objectives[1] >= target - 0.05 * (global_high - global_low)
            ]
            if eligible:
                queue.append(min(eligible, key=lambda s: s.objectives[0]))
        for target in np.linspace(low, high, 17):
            queue.append(
                min(
                    (s for s in ordered if -s.objectives[1] >= target),
                    key=lambda s: s.objectives[0],
                )
            )
        queue.extend(ordered)
        priorities.append(queue)
    selected: dict[tuple[int, ...], Solution] = {}
    # Round-robin avoids spending all supports on one cardinality.
    while priorities and len(selected) < budget:
        remaining = []
        for queue in priorities:
            while queue and _support(queue[0], problem.activity_tol) in selected:
                queue.pop(0)
            if queue and len(selected) < budget:
                solution = queue.pop(0)
                selected[_support(solution, problem.activity_tol)] = solution
            if queue:
                remaining.append(queue)
        priorities = remaining
    return list(selected.values())


def refine_supports(
    problem: BalancedPortfoliosRiskContributions,
    candidates: list[Solution],
    *,
    max_supports: int,
    stop: Callable[[], bool] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[Solution], dict[str, Any]]:
    """Sample fixed-support fronts using variance/return and HHI epsilon bounds.

    No new constraint is added to the investment model. Temporary return/HHI
    bounds only select points to sample; the final archive retains all three
    original objectives. Fixed work and deterministic starts make this phase
    reproducible. A failed numerical solve never discards existing candidates.
    """
    if max_supports < 0:
        raise ValueError("Refinement support budget must be nonnegative")
    started = time.monotonic()
    selected = (
        _representatives(problem, candidates, max_supports) if max_supports else []
    )
    accepted: list[Solution] = []
    attempts = iterations = completed = passes = numerical_failures = 0
    lower, upper = problem.balance_bounds
    # Stay inside the model's exact feasibility tolerance after simplex repair.
    margin = min(1e-9, max(0.0, upper - lower) / 100)

    endpoint_returns: dict[
        tuple[tuple[int, ...], float | None], tuple[float, float]
    ] = {}
    global_targets: np.ndarray = np.array([])
    for phase in range(2):
        if phase == 1 and accepted:
            pool = candidates + accepted
            low_return = -min(
                pool, key=lambda s: (s.objectives[0], s.objectives[1])
            ).objectives[1]
            high_return = max(-s.objectives[1] for s in pool)
            global_targets = low_return + np.linspace(0.025, 0.975, 39) * (
                high_return - low_return
            )
        for seed in selected:
            if stop is not None and stop():
                break
            indices = np.asarray(_support(seed, problem.activity_tol))
            cov = problem.cov[np.ix_(indices, indices)]
            mu = problem.mu[indices]
            costs = (
                problem.trading_costs.restrict(indices)
                if problem.trading_costs is not None
                else None
            )
            initial = np.asarray(seed.variables)[indices]
            scale = max(float(np.mean(np.diag(cov))), 1e-16)
            return_scale = max(
                float(np.ptp(mu)), 2 * costs.rate if costs is not None else 0, 1e-8
            )
            k = len(indices)

            def contributions(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                if problem.balance == PortfolioBalance.CAPITAL_WEIGHTS:
                    return w.copy(), np.eye(k)
                sw = cov @ w
                variance = max(float(w @ sw), 1e-30)
                g = w * sw / variance
                jac = (np.diag(sw) + w[:, None] * cov - np.outer(g, 2 * sw)) / variance
                return g, jac

            constraints: list[dict[str, Any]] = [
                {
                    "type": "eq",
                    "fun": lambda w: float(w.sum() - 1),
                    "jac": lambda w: np.ones(k),
                },
                {
                    "type": "ineq",
                    "fun": lambda w: contributions(w)[0] - lower - margin,
                    "jac": lambda w: contributions(w)[1],
                },
                {
                    "type": "ineq",
                    "fun": lambda w: upper - margin - contributions(w)[0],
                    "jac": lambda w: -contributions(w)[1],
                },
            ]

            def solve(
                target: float | None, hhi_cap: float | None, maximize: bool = False
            ) -> Solution | None:
                nonlocal attempts, iterations, numerical_failures
                if stop is not None and stop():
                    return None
                extra = list(constraints)
                if target is not None:
                    extra.append(
                        {
                            "type": "ineq",
                            "fun": lambda w: float(
                                (expected_net_return(mu, w, costs) - target - 1e-10)
                                / return_scale
                            ),
                            "jac": lambda w: net_return_gradient(mu, w, costs)
                            / return_scale,
                        }
                    )
                if hhi_cap is not None:
                    safe_cap = hhi_cap - min(1e-9, max(0.0, hhi_cap - 1 / k) / 100)
                    extra.append(
                        {
                            "type": "ineq",
                            "fun": lambda w: float(
                                safe_cap - contributions(w)[0] @ contributions(w)[0]
                            ),
                            "jac": lambda w: -2
                            * contributions(w)[0]
                            @ contributions(w)[1],
                        }
                    )
                attempts += 1
                try:
                    result = minimize(
                        (lambda w: -expected_net_return(mu, w, costs) / return_scale)
                        if maximize
                        else (lambda w: float(w @ cov @ w / scale)),
                        initial,
                        jac=(
                            lambda w: -net_return_gradient(mu, w, costs) / return_scale
                        )
                        if maximize
                        else (lambda w: 2 * cov @ w / scale),
                        constraints=extra,
                        bounds=(
                            [(lower + margin, upper - margin)] * k
                            if problem.balance == PortfolioBalance.HYBRID
                            else [(problem.activity_tol * 1.01, 1.0)] * k
                        ),
                        method="SLSQP",
                        options={"ftol": 1e-11, "maxiter": 100},
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                    numerical_failures += 1
                    return None
                iterations += int(result.nit)
                if not result.success or not np.all(np.isfinite(result.x)):
                    return None
                solution = Solution(problem)
                weights = np.zeros(problem.nvars)
                weights[indices] = result.x
                solution.variables[:] = weights.tolist()
                solution.evaluate()
                if not solution.feasible or _support(
                    solution, problem.activity_tol
                ) != tuple(indices):
                    return None
                if not np.all(np.isfinite(solution.objectives)):
                    return None
                if target is not None and -solution.objectives[1] < target:
                    return None
                if hhi_cap is not None:
                    evaluated = np.asarray(solution.variables)[indices]
                    g, _ = contributions(evaluated)
                    if float(g @ g) > hhi_cap + 1e-10:
                        return None
                accepted.append(solution)
                return solution

            # Conservative HHI slice and the full original feasible support.
            balanced_cap = 1 / k + 0.25 * max(0.0, upper - 1 / k)
            caps = (
                [balanced_cap, None]
                if not problem._hhi_as_constraint
                else [problem._hhi_bound]
            )
            for cap in caps:
                key = (tuple(indices), cap)
                if phase == 0:
                    low = solve(None, cap)
                    high = solve(None, cap, maximize=True)
                    if low is None or high is None:
                        continue
                    endpoint_returns[key] = (-low.objectives[1], -high.objectives[1])
                    targets = np.linspace(-low.objectives[1], -high.objectives[1], 9)[
                        1:-1
                    ]
                else:
                    low_mu, high_mu = endpoint_returns.get(
                        key, (float("inf"), float("-inf"))
                    )
                    targets = global_targets[
                        (global_targets > low_mu + 1e-9)
                        & (global_targets < high_mu - 1e-9)
                    ]
                for target in targets:
                    solve(float(target), cap)
            completed += int(phase == 0)
            passes += 1
            if progress is not None:
                progress(
                    passes / max(1, 2 * len(selected)),
                    f"Refining weights: {completed}/{len(selected)} asset subsets",
                )
    return accepted, {
        "support_budget": max_supports,
        "supports": completed,
        "subproblems": attempts,
        "iterations": iterations,
        "accepted": len(accepted),
        "numerical_failures": numerical_failures,
        "seconds": time.monotonic() - started,
        "time_limited": passes < 2 * len(selected),
    }
