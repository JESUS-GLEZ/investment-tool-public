"""Global WASF-GA evolutionary multi-objective optimisation algorithm.

Implements the algorithm from Saborido, Ruiz & Luque (2017) using
Wierzbicki's Achievement Scalarizing Function (ASF) with dual reference
points (utopian and nadir) for building fronts.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from platypus import (
    AbstractGeneticAlgorithm,
    Direction,
    Generator,
    PlatypusConfig,
    PlatypusError,
    Solution,
    random_weights,
)
from platypus._math import POSITIVE_INFINITY
from platypus._tools import only_keys_for, remove_keys

from .portfolio_sampling import FeasibleParetoArchive, PortfolioGenerator


def asf(
    solution: Solution,
    ref_point: Sequence[float],
    weights: Sequence[float],
    norm: Sequence[float] | None = None,
    rho: float = 1e-3,
    min_weight: float = 1e-6,
) -> float:
    """Wierzbicki Achievement Scalarizing Function (ASF).

    Computes ``max_i { μ_i (f_i - q_i) } + ρ Σ_i μ_i (f_i - q_i)``
    where ``μ_i = 1 / u_i`` (inverse of the supplied weight).

    Parameters
    ----------
    solution : Solution
        Platypus solution with minimisation objectives.
    ref_point : Sequence[float]
        Reference point ``q``.
    weights : Sequence[float]
        Weight vector ``u`` in ``(0, 1]^k``; inverted internally.
    norm : Sequence[float] | None, optional
        Normalisation range ``z_nad - z_ideal``.  Defaults to ones.
    rho : float, optional
        Augmentation coefficient, by default 1e-3.
    min_weight : float, optional
        Floor for weight components, by default 1e-6.

    Returns
    -------
    float
        Scalarised objective value (lower is better).
    """
    objs = np.asarray(solution.objectives, dtype=float)
    ref = np.asarray(ref_point, dtype=float)
    u = np.asarray(weights, dtype=float)

    # guard against zeros in norm and weights
    if norm is None:
        norm_arr = np.ones_like(objs)
    else:
        norm_arr = np.asarray(norm, dtype=float)

    norm_arr = np.where(np.abs(norm_arr) < 1e-12, 1.0, norm_arr)
    u = np.maximum(u, min_weight)

    # μ = 1/u (strictly positive)
    mu = 1.0 / u

    diff = (objs - ref) / norm_arr
    tcheb_term = float(np.max(mu * diff))
    reg_term = float(rho * np.sum(mu * diff))
    return tcheb_term + reg_term


class GWASFGA(AbstractGeneticAlgorithm):
    """
    Global WASF-GA (Saborido, Ruiz, Luque 2017).

    Key differences vs. MOEA/D-like variants:
      - Uses Wierzbicki ASF with strictly positive weights μ and both reference points simultaneously:
        * half the weights with utopian z^u (ideal - ε)
        * half the weights with a slightly worsened nadir z_nad^ε (nadir + ε)
      - Builds *fronts by weight vectors*: in each front, for every weight, select the unique solution with the
        lowest ASF (no duplicates allowed within a front); remove selected from the pool and continue.
      - Next generation is filled by concatenating fronts; if the last front would overflow, keep those with
        the smallest ASF values inside that front.
      - Feasible-first constraint handling: classify feasible solutions into fronts first. If the new population
        is not full, fill the remainder with infeasible solutions sorted by total constraint violation.
    """

    # ------------- Construction -------------

    def __init__(
        self,
        problem: Any,
        population_size: int = 100,
        generator: Generator | None = None,
        variator: Any | None = None,
        weight_generator: Callable[..., Sequence[Sequence[float]]] = random_weights,
        scalarizing_function: Callable[..., float] = asf,
        *,
        rho: float = 1e-3,
        eps_scale: float = 0.01,
        **kwargs: Any,
    ) -> None:
        # remove population_size from kwargs because AbstractGeneticAlgorithm already took it
        super().__init__(
            problem,
            population_size,
            generator if generator is not None else PortfolioGenerator(),
            **remove_keys(kwargs, "population_size"),
        )

        if any(d != Direction.MINIMIZE for d in problem.directions):
            raise PlatypusError(
                "GWASFGA currently only supports minimization problems."
            )

        self.population_size = population_size
        self.variator = variator
        self.weight_generator = weight_generator
        self.scalarizing_function = scalarizing_function
        self.weight_generator_kwargs = only_keys_for(kwargs, weight_generator)

        # if using platypus.random_weights, ensure it receives the intended population size
        if (
            weight_generator == random_weights
            and "population_size" not in self.weight_generator_kwargs
        ):
            self.weight_generator_kwargs["population_size"] = population_size

        # Hyperparameters for ASF and reference expansion
        self.rho: float = rho
        self.eps_scale: float = eps_scale

        # Internal state — set in initialize()
        self.weights: np.ndarray = np.empty((0, 0))  # u-vectors (we invert internally)
        self.ideal_point: list[float] = []
        self.nadir_point: list[float] = []
        self.ideal_point_adjusted: np.ndarray = np.array([])
        self.nadir_point_adjusted: np.ndarray = np.array([])
        self.norm_range: np.ndarray = np.array([])

        # bookkeeping
        self.generation: int = 0
        self.pareto_archive = FeasibleParetoArchive()

        # Seed solutions injected externally before run() triggers initialize()
        self._seed_solutions: list[Solution] | None = None

        if self.scalarizing_function is None:
            raise PlatypusError("GWASFGA requires a scalarizing_function (e.g., asf).")

    # ------------- Helpers -------------

    def _update_ideal(self, solution: Solution) -> None:
        """Track the running per-objective minimum."""
        for j in range(self.problem.nobjs):
            self.ideal_point[j] = min(
                self.ideal_point[j], float(solution.objectives[j])
            )

    def _update_nadir(self, solution: Solution) -> None:
        """Track the running per-objective maximum."""
        for j in range(self.problem.nobjs):
            self.nadir_point[j] = max(
                self.nadir_point[j], float(solution.objectives[j])
            )

    def _recompute_references(self) -> None:
        """Recompute ε-expanded ideal/nadir references and normalization range."""
        nadir = np.asarray(self.nadir_point, dtype=float)
        ideal = np.asarray(self.ideal_point, dtype=float)

        eps = self.eps_scale * np.abs(nadir - ideal)
        self.nadir_point_adjusted = nadir + eps  # ε-nadir
        self.ideal_point_adjusted = ideal - eps  # ε-utopian (ideal improved)
        self.norm_range = self.nadir_point_adjusted - self.ideal_point_adjusted
        self.norm_range = np.where(
            np.abs(self.norm_range) < 1e-12, 1.0, self.norm_range
        )

    def _asf_value(self, sol: Solution, w: np.ndarray, ref: np.ndarray) -> float:
        return float(
            self.scalarizing_function(
                sol, ref_point=ref, weights=w, norm=self.norm_range, rho=self.rho
            )
        )

    def _build_fronts(self, pool: list[Solution]) -> list[list[tuple[Solution, float]]]:
        """
        Build fronts as in Global WASF-GA:
          - For each front, iterate all weight vectors.
          - For even j use nadir-ε, for odd j use utopian.
          - For each weight, select the best (lowest ASF) *unique* solution among the remaining candidates.
          - After finishing a front, remove selected solutions from the remaining pool and repeat.
        Returns a list of fronts, each as a list of (solution, asf_value_for_its_selected_weight).
        """
        if not pool:
            return []

        remaining_indices: list[int] = list(range(len(pool)))
        fronts: list[list[tuple[Solution, float]]] = []

        # ensure weights array exists
        W = self.weights
        num_w = W.shape[0]

        while remaining_indices:
            used_in_front: set[int] = set()
            front: list[tuple[Solution, float]] = []

            # iterate weights in a fixed order to match paper (odd -> utopian, even -> nadir)
            for j in range(num_w):
                # choose reference depending on parity
                ref = (
                    self.ideal_point_adjusted
                    if (j % 2 == 1)
                    else self.nadir_point_adjusted
                )
                w = W[j]

                best_idx: int | None = None
                best_val: float = math.inf

                for idx in remaining_indices:
                    if idx in used_in_front:
                        continue
                    cand = pool[idx]
                    val = self._asf_value(cand, w, ref)
                    if val < best_val:
                        best_val = val
                        best_idx = idx

                if best_idx is not None:
                    used_in_front.add(best_idx)
                    front.append((pool[best_idx], best_val))

            fronts.append(front)
            # remove selected from remaining
            remaining_indices = [i for i in remaining_indices if i not in used_in_front]

        return fronts

    @staticmethod
    def _is_feasible(sol: Solution) -> bool:
        return float(getattr(sol, "constraint_violation", 0.0) or 0.0) <= 0.0

    # ------------- Algorithm API -------------

    def evaluate_all(self, solutions: list[Solution]) -> None:
        super().evaluate_all(solutions)
        self.pareto_archive.extend(solutions)

    def step(self) -> None:
        super().step()
        # AbstractGeneticAlgorithm otherwise exposes only the current population.
        self.result = list(self.pareto_archive)

    def initialize(self) -> None:
        """Build weights, initial population, and initial references.

        If ``self._seed_solutions`` is not ``None``, the provided pre-built
        solutions are injected into the tail of the initial population,
        replacing randomly generated individuals.  This allows warm-starting
        the optimizer from a previous rebalancing period.
        """
        # 1) Generate weight vectors (u in (0,1]^k). Ensure an even count to split utopian/nadir.
        weights_list: Sequence[Sequence[float]] = self.weight_generator(
            self.problem.nobjs, **self.weight_generator_kwargs
        )
        W = np.asarray(weights_list, dtype=float)
        if W.ndim != 2 or W.shape[1] != self.problem.nobjs:
            raise PlatypusError("Weight generator returned an invalid shape.")
        # ensure strictly positive components
        W = np.maximum(W, 1e-8)

        # ensure even number of weights
        if W.shape[0] % 2 == 1:
            W = W[:-1, :]
        self.weights = W
        self.population_size = int(W.shape[0])

        # 2) Initial population (random)
        self.population = [
            self.generator.generate(self.problem) for _ in range(self.population_size)
        ]

        # 2b) Inject seed solutions if provided (warm-start for backtesting)
        if self._seed_solutions:
            n_seed = min(len(self._seed_solutions), self.population_size // 2)
            for i in range(n_seed):
                self.population[-(i + 1)] = self._seed_solutions[i]

        self.evaluate_all(self.population)

        # 3) Initialize ideal/nadir from the initial population (parents only, as in the paper)
        self.ideal_point = [POSITIVE_INFINITY] * self.problem.nobjs
        self.nadir_point = [-POSITIVE_INFINITY] * self.problem.nobjs
        for sol in self.population:
            self._update_ideal(sol)
            self._update_nadir(sol)

        # 4) Build ε-expanded references and normalization
        self._recompute_references()

        # 5) Default variator if none provided
        if self.variator is None:
            self.variator = PlatypusConfig.default_variator(self.problem)

        self.generation = 0

    def iterate(self) -> None:
        """
        One generational iteration (Global WASF-GA):
          - Create |P| offspring via crossover/mutation.
          - Combine Z = P ∪ Q, update references from Z, recompute normalization.
          - Classify FEASIBLE(Z) into GWASF fronts; fill new population from fronts.
          - If not full, fill with INFEASIBLE(Z) sorted by total constraint violation (smallest first).
        """
        # --- Offspring creation ---
        children: list[Solution] = []
        if self.variator is None:
            raise RuntimeError("Variator must be set before iteration")
        arity = getattr(self.variator, "arity", 2)
        while len(children) < self.population_size:
            parents = random.sample(self.population, k=min(arity, len(self.population)))
            offspring = self.variator.evolve(parents)
            if offspring is None:
                continue
            if isinstance(offspring, list):
                children.extend(offspring)
            else:
                children.append(offspring)
        children = children[: self.population_size]
        self.evaluate_all(children)

        # --- Combine and update references based on parents+offspring ---
        combined = self.population + children
        # reset trackers
        self.ideal_point = [POSITIVE_INFINITY] * self.problem.nobjs
        self.nadir_point = [-POSITIVE_INFINITY] * self.problem.nobjs
        for sol in combined:
            self._update_ideal(sol)
            self._update_nadir(sol)
        self._recompute_references()

        # --- Constraint handling: feasible first ---
        feasible = [s for s in combined if self._is_feasible(s)]
        infeasible = [s for s in combined if not self._is_feasible(s)]

        new_population: list[Solution] = []

        # Build fronts for feasibles
        fronts = self._build_fronts(feasible)
        for front in fronts:
            if not front:
                continue
            # if adding the whole front fits, add all
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend([s for (s, _) in front])
            else:
                remaining = self.population_size - len(new_population)
                # choose those with smallest ASF values in this front
                front_sorted = sorted(front, key=lambda t: t[1])
                new_population.extend([s for (s, _) in front_sorted[:remaining]])
                break

        # If still room, fill with infeasible by lowest constraint violation
        if len(new_population) < self.population_size and infeasible:
            infeasible_sorted = sorted(
                infeasible,
                key=lambda s: float(getattr(s, "constraint_violation", 0.0) or 0.0),
            )
            to_take = self.population_size - len(new_population)
            new_population.extend(infeasible_sorted[:to_take])

        # Final safeguard: if still short (shouldn't happen), pad with random evaluated children
        if len(new_population) < self.population_size:
            filler = (children + feasible + infeasible)[
                : self.population_size - len(new_population)
            ]
            new_population.extend(filler)

        # --- Next generation ---
        self.population = new_population
        self.generation += 1
