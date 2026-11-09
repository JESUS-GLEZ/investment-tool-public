"""Optimisation utilities: risk contributions, Pareto merging, matrix
repair, plotting, and metrics.

Provides:

* :func:`normalized_risk_contributions` — shared risk-contribution calc.
* :func:`merge_nondominated` — deduplicate and prune Pareto fronts.
* :func:`ensure_psd_matrix` / :func:`clean_matrix` — covariance repair.
* :func:`sharpe_like` — Sharpe-like ratio for Platypus solutions.
* :func:`plot_solution` / :func:`plot_solutions_with_markowitz_front` —
  diagnostic Pareto-front plots.
* :func:`compute_markowitz_front` — standalone Markowitz efficient-frontier
  computation usable by any optimiser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from platypus import Solution, nondominated, nondominated_prune, crowding_distance

from investment_tool.utils import nearest_psd
from investment_tool.domain.trading_costs import TradingCosts

logger = logging.getLogger(__name__)


def return_label(solutions: list[Solution]) -> str:
    return (
        "Estimated Net Return"
        if any(getattr(s.problem, "trading_costs", None) is not None for s in solutions)
        else "Expected Return"
    )


# ── Risk-contribution utilities ──────────────────────────────────────────────


def normalized_risk_contributions(
    w: np.ndarray,
    cov: np.ndarray,
    activity_tol: float = 1e-4,
) -> np.ndarray:
    """Compute normalised risk contributions ``g_i(w)``.

    ``g_i(w) = w_i (Σ w)_i / (w' Σ w)`` — the fraction of total
    portfolio variance attributable to asset *i*.

    Parameters
    ----------
    w : np.ndarray
        Portfolio weight vector on the simplex.
    cov : np.ndarray
        Covariance matrix, shape ``(N, N)``.
    activity_tol : float, optional
        Threshold to consider a weight active, by default 1e-4.

    Returns
    -------
    np.ndarray
        Normalised risk contributions summing to 1.
    """
    Sw = cov @ w
    var = float(w @ Sw)
    active = w > activity_tol
    k = int(active.sum())

    if var <= 1e-16 or k == 0:
        # Degenerate: distribute uniformly among active, else uniform overall
        if k > 0:
            gi = np.zeros_like(w)
            gi[active] = 1.0 / k
            return gi
        else:
            return np.full_like(w, 1.0 / w.size)

    gi = np.zeros_like(w)
    gi[active] = (w[active] * Sw[active]) / var
    # numerical tidy-up: small negatives to 0
    gi = np.where(gi < 0, 0.0, gi)
    # re-normalize tiny drift
    s = gi.sum()
    if s > 0:
        gi /= s
    return gi


# ── Pareto-front merging ─────────────────────────────────────────────────────


def merge_nondominated(
    lists_of_solutions: list[list[Solution]],
    target_size: int,
    dedup_decimals: int = 8,
) -> list[Solution]:
    """Merge finite feasible rank-zero solutions, then thin by crowding distance.

    Returns an empty list when no feasible candidate exists; infeasible or
    dominated candidates never fill unused capacity.

    Parameters
    ----------
    lists_of_solutions : list[list[Solution]]
        Archives from each restart / phase.
    target_size : int
        Maximum number of solutions to keep after pruning.
    dedup_decimals : int, optional
        Rounding precision for deduplication (default 8).

    Returns
    -------
    list[Solution]
        Pruned non-dominated set.
    """
    if target_size < 0:
        raise ValueError("target_size must be non-negative")
    if target_size == 0:
        return []

    seen: set[tuple[int, tuple[tuple[int, float], ...]]] = set()
    merged: list[Solution] = []
    for sols in lists_of_solutions:
        for s in sols:
            # Some deterministic solvers call model.evaluate directly, which
            # leaves Platypus constraint metadata unset until Solution.evaluate.
            if not s.evaluated:
                s.evaluate()
            if s.constraint_violation != 0 or not np.all(np.isfinite(s.objectives)):
                continue
            rounded = np.round(np.asarray(s.variables, dtype=float), dedup_decimals)
            # Keep the same rounded-vector equality without retaining one Python
            # scalar per zero weight for every candidate in a large seed pool.
            # Dimension distinguishes differently sized, zero-padded portfolios.
            key = (
                len(rounded),
                tuple((int(i), float(rounded[i])) for i in np.flatnonzero(rounded)),
            )
            if key not in seen:
                seen.add(key)
                merged.append(s)

    if not merged:
        return []

    # Only the first front is needed. Ranking every dominated seed is quadratic
    # in the full subset budget; Archive extraction compares against its current
    # front. It is already rank zero: assign rank/crowding metadata directly
    # instead of extracting the same front a second time.
    front = nondominated(merged)
    for solution in front:
        solution.rank = 0
    crowding_distance(front)
    pruned: list[Solution] = list(nondominated_prune(front, target_size))
    return pruned


# ── Matrix utilities ─────────────────────────────────────────────────────────


def clean_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN and Inf values in a covariance matrix.

    Diagonal NaN entries are set to 0.01 (a small positive variance);
    off-diagonal NaN entries are set to 0.

    Parameters
    ----------
    matrix : pd.DataFrame
        Covariance matrix to clean.

    Returns
    -------
    pd.DataFrame
        Cleaned matrix with no NaN or Inf values.
    """
    cleaned = matrix.replace([np.inf, -np.inf], np.nan)
    mask_nan = cleaned.isna()
    if mask_nan.any().any():
        cleaned = cleaned.fillna(0.0)
        for i in range(len(cleaned)):
            if mask_nan.iloc[i, i]:
                cleaned.iloc[i, i] = 0.01
    return cleaned


def ensure_psd_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Ensure a covariance matrix is positive semi-definite.

    Cleans the input, applies a nearest-PSD approximation, and adds a
    small epsilon to the diagonal.  Falls back to a diagonal matrix on
    failure.

    Parameters
    ----------
    matrix : pd.DataFrame
        Covariance matrix to process.

    Returns
    -------
    pd.DataFrame
        Adjusted PSD covariance matrix.
    """
    logger.info("Ensuring covariance matrix is PSD")
    matrix = clean_matrix(matrix)

    try:
        adjusted = nearest_psd(matrix)
        eps = 1e-6
        adjusted_matrix = adjusted + np.eye(matrix.shape[0]) * eps
        min_eig = float(np.min(np.linalg.eigvals(adjusted_matrix)))
        if min_eig <= 0:
            logger.warning(
                "Matrix still not PSD (min eigenvalue: %.6e); increasing epsilon",
                min_eig,
            )
            eps = max(abs(min_eig) * 2, 1e-3)
            adjusted_matrix = adjusted + np.eye(matrix.shape[0]) * eps

        return pd.DataFrame(adjusted_matrix, index=matrix.index, columns=matrix.columns)
    except (LinAlgError, ValueError) as e:
        logger.error("Error processing PSD matrix: %s", e)
        logger.warning("Falling back to diagonal covariance matrix")
        diag = np.diag(np.maximum(np.diag(matrix.values), 1e-4))
        return pd.DataFrame(diag, index=matrix.index, columns=matrix.columns)


# ── Metrics ──────────────────────────────────────────────────────────────────


def sharpe_like(sol: Solution, eps: float = 1e-12) -> float:
    """Compute a Sharpe-like ratio from a Platypus solution.

    Assumes objectives are ``[variance, -expected_return, ...]``.

    Parameters
    ----------
    sol : Solution
        Solution with at least two objectives.
    eps : float, optional
        Floor for variance to avoid division by zero, by default 1e-12.

    Returns
    -------
    float
        ``expected_return / sqrt(variance)``.
    """
    var = max(float(sol.objectives[0]), eps)
    exp_ret = -float(sol.objectives[1])
    return float(exp_ret / np.sqrt(var))


# ── Plotting helpers ─────────────────────────────────────────────────────────


def _resolve_plot_path(
    file_name: str,
    output_directory: Path,
    date_label: str | None,
) -> Path:
    """Build the full save path for a plot file.

    Parameters
    ----------
    file_name : str
        Base filename (without extension).
    output_directory : Path
        Run output directory; ``<dir>/plots/`` is used.
    date_label : str | None
        Optional date string appended for uniqueness.

    Returns
    -------
    Path
        Full path to the ``.png`` file (parent directory created).
    """
    plots_dir = Path(output_directory) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{date_label}" if date_label else ""
    return plots_dir / f"{file_name}{suffix}.png"


def _concentration_label(solutions: list[Solution]) -> str:
    from investment_tool.constants import PortfolioBalance

    balance = getattr(solutions[0].problem, "balance", None) if solutions else None
    return (
        "Capital-weight HHI"
        if balance == PortfolioBalance.CAPITAL_WEIGHTS
        else "Risk-contribution HHI"
    )


def plot_solution(
    solutions: list[Solution],
    file_name: str,
    output_directory: Path | None = None,
    date_label: str | None = None,
    selected_solution: Solution | None = None,
) -> str | None:
    """Save a 2-D frontier or a 3-D variance/return/HHI scatter plot.

    Parameters
    ----------
    solutions : list[Solution]
        Solutions with ``.objectives = [variance, -expected_return, ...]``.
    file_name : str
        Base name of the file (without extension).
    output_directory : Path | None
        Run output directory.
    date_label : str | None
        Date string appended for uniqueness across backtesting periods.
    selected_solution : Solution | None
        The solution chosen by the selection strategy.  When provided,
        it is highlighted as a distinct marker on the plot.

    Returns
    -------
    str | None
        Absolute path of the saved plot, or ``None`` if no solutions
        or no output directory.
    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D

    if not solutions:
        logger.warning("No solutions to plot for '%s'", file_name)
        return None

    if output_directory is None:
        logger.debug("No output directory — skipping plot '%s'", file_name)
        return None

    save_path = _resolve_plot_path(file_name, output_directory, date_label)

    three_objectives = len(solutions[0].objectives) == 3
    fig = plt.figure()
    variance = [s.objectives[0] for s in solutions]
    expected_return = [-s.objectives[1] for s in solutions]
    ax: Axes
    if three_objectives:
        ax_3d = cast(Axes3D, fig.add_subplot(111, projection="3d"))
        ax_3d.scatter(
            variance,
            expected_return,
            [s.objectives[2] for s in solutions],
            c="#1f77b4",
            alpha=0.6,
            label="Pareto solutions",
        )
        if selected_solution is not None:
            ax_3d.scatter(
                [selected_solution.objectives[0]],
                [-selected_solution.objectives[1]],
                [selected_solution.objectives[2]],
                c="#ff7f0e",
                s=120,
                marker="*",
                label="Selected portfolio",
            )
        ax_3d.set_zlabel(_concentration_label(solutions))
        ax_3d.set_title("Pareto Solutions: Variance, Return and HHI")
        ax = ax_3d
    else:
        ax = fig.add_subplot(111)
        ax.scatter(
            variance,
            expected_return,
            c="#1f77b4",
            alpha=0.6,
            label="Pareto solutions",
        )
        if selected_solution is not None:
            ax.scatter(
                [selected_solution.objectives[0]],
                [-selected_solution.objectives[1]],
                c="#ff7f0e",
                s=120,
                marker="*",
                label="Selected portfolio",
            )
        ax.set_title(f"Pareto Front: Variance vs. {return_label(solutions)}")
    if selected_solution is not None:
        ax.legend()
    ax.set_xlabel("Portfolio Variance")
    ax.set_ylabel(return_label(solutions))
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved optimization plot: %s", save_path)
    return str(save_path)


def plot_solutions_with_markowitz_front(
    solutions: list[Solution],
    markowitz_front: list[Solution],
    file_name: str,
    output_directory: Path | None = None,
    date_label: str | None = None,
    selected_solution: Solution | None = None,
    reference_objectives: list[list[float]] | None = None,
) -> str | None:
    """Plot solutions overlaid on the Markowitz efficient frontier.

    Parameters
    ----------
    solutions : list[Solution]
        Solutions from the main optimisation.
    markowitz_front : list[Solution]
        Non-dominated Markowitz frontier solutions.
    file_name : str
        Base name of the file (without extension).
    output_directory : Path | None
        Run output directory.
    date_label : str | None
        Date string appended for uniqueness across backtesting periods.
    selected_solution : Solution | None
        The solution chosen by the selection strategy.  When provided,
        it is highlighted as a distinct marker on the plot.

    Returns
    -------
    str | None
        Absolute path of the saved plot, or ``None`` if both inputs
        are empty or no output directory.
    """
    import matplotlib.pyplot as plt

    if output_directory is None:
        logger.debug("No output directory — skipping plot '%s'", file_name)
        return None

    save_path = _resolve_plot_path(file_name, output_directory, date_label)

    x_sol = [float(s.objectives[0]) for s in solutions]
    y_sol = [float(-s.objectives[1]) for s in solutions]

    curve = (
        reference_objectives
        if reference_objectives is not None
        else [list(map(float, s.objectives)) for s in markowitz_front]
    )
    x_m = [float(point[0]) for point in curve]
    y_m = [float(-point[1]) for point in curve]

    # Sort Markowitz frontier by variance for a clean line
    if x_m:
        order = np.argsort(x_m)
        x_m_sorted = [x_m[i] for i in order]
        y_m_sorted = [y_m[i] for i in order]
    else:
        x_m_sorted, y_m_sorted = [], []

    plt.figure()
    three_objectives = bool(solutions) and len(solutions[0].objectives) == 3
    scatter = plt.scatter(
        x_sol,
        y_sol,
        c=[s.objectives[2] for s in solutions] if three_objectives else "#1f77b4",
        alpha=0.7,
        label="Current solutions",
        zorder=2,
    )
    if three_objectives:
        plt.colorbar(scatter, label=_concentration_label(solutions))
    if x_m_sorted:
        plt.plot(
            x_m_sorted,
            y_m_sorted,
            color="#d62728",
            linewidth=2.0,
            label="Markowitz efficient frontier",
            zorder=1,
        )
        plt.scatter(x_m, y_m, c="#d62728", s=15, alpha=0.8, zorder=1)
    else:
        logger.warning("Markowitz frontier is empty; plotting only current solutions")

    if selected_solution is not None:
        plt.scatter(
            [selected_solution.objectives[0]],
            [-selected_solution.objectives[1]],
            c="#ff7f0e",
            s=120,
            marker="*",
            zorder=3,
            label="Selected portfolio",
        )

    plt.xlabel("Portfolio Variance")
    plt.ylabel(return_label(solutions))
    plt.title(
        "Variance–Return Projection (color: HHI) vs. Markowitz"
        if three_objectives
        else "Solutions vs. Markowitz Efficient Frontier"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=300)
    plt.close()
    logger.info("Saved optimization plot: %s", save_path)
    return str(save_path)


# ── Markowitz frontier computation ──────────────────────────────────────────


def compute_markowitz_front(
    mu: np.ndarray,
    cov: np.ndarray,
    n_evaluations: int = 20_000,
    population_size: int = 300,
    n_restarts: int = 3,
    trading_costs: "TradingCosts | None" = None,
) -> list[Solution]:
    """Compute the Markowitz long-only efficient frontier.

    Standalone function usable by any optimiser strategy.  Runs a
    lightweight NSGA-II on the 2-objective unconstrained Markowitz
    problem (minimise variance, maximise expected return) and returns
    the merged non-dominated front.

    Parameters
    ----------
    mu : np.ndarray
        Vector of expected returns, shape ``(n_assets,)``.
    cov : np.ndarray
        Covariance matrix, shape ``(n_assets, n_assets)``.
    n_evaluations : int, optional
        Function evaluations per restart, by default 20 000.
    population_size : int, optional
        GA population size, by default 300.
    n_restarts : int, optional
        Independent multi-start runs, by default 3.

    Returns
    -------
    list[Solution]
        Non-dominated Markowitz frontier solutions.

    Raises
    ------
    RuntimeError
        If no solutions are found after all restarts.
    """
    # Import here to avoid a circular import at module level; these are
    # only needed when this function is actually called.
    from platypus import NSGAII  # noqa: PLC0415

    from investment_tool.strategies.optimization_strategies.model.markowitz import (  # noqa: PLC0415
        MarkowitzLongOnly,
    )

    logger.info(
        "Computing Markowitz frontier benchmark (pop=%d, evals=%d, restarts=%d)",
        population_size,
        n_evaluations,
        n_restarts,
    )

    archives: list[list[Solution]] = []
    for _ in range(n_restarts):
        problem = MarkowitzLongOnly(mu=mu, cov=cov, trading_costs=trading_costs)
        algorithm = NSGAII(problem, population_size=population_size)
        algorithm.run(n_evaluations)
        archives.append(list(algorithm.result))

    front = merge_nondominated(archives, target_size=population_size)
    if not front:
        raise RuntimeError("No solutions found when computing Markowitz frontier.")

    return front
