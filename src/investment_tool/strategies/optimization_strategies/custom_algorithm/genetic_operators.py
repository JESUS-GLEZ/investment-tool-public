"""Portfolio-aware genetic operators for evolutionary optimisation.

Provides:

* :func:`repair_portfolio` — project any weight vector onto the simplex
  (long-only, fully-invested).
* :class:`PortfolioVariator` — custom crossover + mutation that respects
  portfolio structure (cardinality control, weight perturbation, simplex
  repair).
"""

from __future__ import annotations

import numpy as np
from platypus import Solution, Variator

from investment_tool.defaults import GENETIC, OPTIMIZER


def repair_portfolio(
    x: np.ndarray,
    dust_tol: float = OPTIMIZER.dust_tol,
) -> np.ndarray:
    """Project a weight vector onto the probability simplex.

    Zeros out dust positions (below *dust_tol*) and renormalises so
    weights sum to 1.  If all weights are zero the fallback is a
    uniform allocation.

    Parameters
    ----------
    x : np.ndarray
        Raw weight vector (may be non-negative but not sum to 1).
    dust_tol : float, optional
        Threshold below which a weight is zeroed, by default 1e-3.

    Returns
    -------
    np.ndarray
        Repaired weight vector summing to 1.
    """
    x[x < dust_tol] = 0.0
    s = x.sum()
    return x / s if s > 0 else np.full_like(x, 1.0 / x.size)


class PortfolioVariator(Variator):
    """Custom crossover + mutation for portfolio weight vectors.

    **Crossover** (union-average):
    - Common active assets → average of parent weights.
    - Assets active in only one parent → copied to respective child.

    **Mutation** (two-stage):
    - *Weight perturbation* (probability ``perturb_prob``): applies small
      Gaussian noise to active weights — the primary, gentle mutation
      that refines promising solutions.
    - *Cardinality change* (probability ``cardinality_prob``): randomly
      adds or drops entire positions — a structural mutation applied
      less frequently to explore different asset subsets.

    Both children are repaired to the simplex after each operation.

    Parameters
    ----------
    min_assets : int, optional
        Minimum number of active positions, by default 1.
    max_assets : int | None, optional
        Maximum number of active positions.  ``None`` means no upper
        bound (capped at the number of available assets).
    mutation_prob : float, optional
        Overall probability that *any* mutation is applied, by default 0.5.
    perturb_prob : float, optional
        Probability of weight-perturbation mutation (conditioned on
        mutation being applied), by default 0.7.
    cardinality_prob : float, optional
        Probability of cardinality mutation (conditioned on mutation
        being applied), by default 0.15.
        Note: the remaining probability (1 - perturb - cardinality)
        applies *both* mutations in sequence.
    noise_std : float, optional
        Standard deviation for Gaussian weight perturbation, by default 0.05.
    """

    def __init__(
        self,
        min_assets: int = 1,
        max_assets: int | None = None,
        mutation_prob: float = GENETIC.mutation_prob,
        perturb_prob: float = GENETIC.perturb_prob,
        cardinality_prob: float = GENETIC.cardinality_prob,
        noise_std: float = GENETIC.noise_std,
    ) -> None:
        super().__init__(2)  # arity: expects 2 parents
        self.min_assets = max(1, int(min_assets))
        self.max_assets = max_assets
        self.mutation_prob = float(mutation_prob)
        self.perturb_prob = float(perturb_prob)
        self.cardinality_prob = float(cardinality_prob)
        self.noise_std = float(noise_std)

    def _crossover(
        self, p1: np.ndarray, p2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Union-average crossover producing two children.

        Parameters
        ----------
        p1, p2 : np.ndarray
            Parent weight vectors.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Two child weight vectors (not yet repaired).
        """
        idx1 = np.flatnonzero(p1 > 0)
        idx2 = np.flatnonzero(p2 > 0)
        common = np.intersect1d(idx1, idx2)
        only1 = np.setdiff1d(idx1, idx2)
        only2 = np.setdiff1d(idx2, idx1)

        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p2)
        if common.size:
            avg = 0.5 * (p1[common] + p2[common])
            c1[common] = avg
            c2[common] = avg
        if only1.size:
            c1[only1] = p1[only1]
        if only2.size:
            c2[only2] = p2[only2]
        return c1, c2

    def _perturb_weights(self, x: np.ndarray) -> np.ndarray:
        """Weight-perturbation mutation: apply Gaussian noise to active weights.

        A gentle mutation that refines promising solutions without changing
        the set of active assets.  Noise is added only to positions that
        are already active, then clipped to stay non-negative.

        Parameters
        ----------
        x : np.ndarray
            Weight vector (may be pre-crossover child).

        Returns
        -------
        np.ndarray
            Perturbed weight vector (not yet repaired).
        """
        active = np.flatnonzero(x > 0)
        if active.size == 0:
            return x
        x_perturbed = x.copy()
        noise = np.random.normal(0, self.noise_std, size=active.size)
        x_perturbed[active] = np.clip(x_perturbed[active] + noise, 0.0, None)
        return x_perturbed

    def _mutate_cardinality(self, x: np.ndarray) -> np.ndarray:
        """Cardinality mutation: randomly add or drop active positions.

        Parameters
        ----------
        x : np.ndarray
            Weight vector (may be pre-crossover child).

        Returns
        -------
        np.ndarray
            Mutated weight vector (not yet repaired).
        """
        n = x.size
        k_now = int(np.count_nonzero(x > 0))
        k_min = self.min_assets
        k_max = n if self.max_assets is None else min(self.max_assets, n)
        k_new = np.random.randint(k_min, k_max + 1)
        active = np.flatnonzero(x > 0)
        x_mutated = x.copy()

        if k_new > k_now:
            # Add new random positions
            inactive = np.setdiff1d(np.arange(n), active)
            if inactive.size:
                add_count = min(k_new - k_now, inactive.size)
                add_idx = np.random.choice(inactive, size=add_count, replace=False)
                x_mutated[add_idx] = np.random.rand(add_idx.size)
        elif k_new < k_now:
            # Drop random active positions
            drop_idx = np.random.choice(active, size=(k_now - k_new), replace=False)
            x_mutated[drop_idx] = 0.0

        return x_mutated

    def _mutate(self, x: np.ndarray) -> np.ndarray:
        """Two-stage mutation dispatcher.

        With probability ``mutation_prob``, applies one of:
        - Weight perturbation only (probability ``perturb_prob``)
        - Cardinality change only (probability ``cardinality_prob``)
        - Both in sequence (remaining probability)

        Parameters
        ----------
        x : np.ndarray
            Weight vector.

        Returns
        -------
        np.ndarray
            Mutated weight vector (not yet repaired).
        """
        if np.random.rand() > self.mutation_prob:
            return x

        roll = np.random.rand()
        if roll < self.perturb_prob:
            # Gentle weight perturbation only (most common)
            return self._perturb_weights(x)
        elif roll < self.perturb_prob + self.cardinality_prob:
            # Structural cardinality change only (less common)
            return self._mutate_cardinality(x)
        else:
            # Both: cardinality change then perturbation (rare)
            x = self._mutate_cardinality(x)
            return self._perturb_weights(x)

    def evolve(self, parents: list[Solution]) -> list[Solution]:
        """Produce two offspring from two parents.

        Parameters
        ----------
        parents : list[Solution]
            Exactly two parent solutions.

        Returns
        -------
        list[Solution]
            Two child solutions with repaired (simplex) weights.
        """
        p1, p2 = parents
        x1 = np.asarray(p1.variables, dtype=float)
        x2 = np.asarray(p2.variables, dtype=float)

        c1, c2 = self._crossover(x1, x2)
        c1 = self._mutate(c1)
        c2 = self._mutate(c2)
        c1 = repair_portfolio(c1)
        c2 = repair_portfolio(c2)

        s1 = Solution(p1.problem)
        s1.variables[:] = c1.tolist()
        s2 = Solution(p2.problem)
        s2.variables[:] = c2.tolist()
        return [s1, s2]
