"""Comparison plots for solver benchmarking.

Generates overlaid Pareto fronts, metric bar charts, and radar charts
to visually compare solver performance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from platypus import Solution

    from investment_tool.strategies.optimization_strategies.solver_comparison.solver_result import (
        SolverResult,
    )

logger = logging.getLogger(__name__)

# Colour palette (colourblind-friendly, up to 8 solvers)
_COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _resolve_path(output_dir: Path | None, name: str) -> Path:
    """Build save path under ``<output_dir>/solver_comparison/``."""
    base = (
        Path(output_dir) / "solver_comparison"
        if output_dir
        else Path.cwd() / "solver_comparison"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def plot_overlaid_fronts(
    results: list[SolverResult],
    reference_front: list[Solution] | None = None,
    output_directory: Path | None = None,
    seed_front: list[Solution] | None = None,
) -> str | None:
    """Plot all solver Pareto fronts on the same axes.

    Parameters
    ----------
    results : list[SolverResult]
        Results from :meth:`SolverBenchmark.run`.
    reference_front : list[Solution] | None
        Optional Markowitz frontier to overlay as baseline.
    output_directory : Path | None
        Output directory for the plot.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` on failure.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    save_path = _resolve_path(output_directory, "pareto_fronts_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 7))

    for i, result in enumerate(results):
        colour = _COLOURS[i % len(_COLOURS)]
        x = [float(s.objectives[0]) for s in result.front]
        y = [float(-s.objectives[1]) for s in result.front]
        ax.scatter(x, y, c=colour, alpha=0.6, s=20, label=result.config.label)

    if reference_front:
        x_ref = [float(s.objectives[0]) for s in reference_front]
        y_ref = [float(-s.objectives[1]) for s in reference_front]
        order = np.argsort(x_ref)
        ax.plot(
            [x_ref[i] for i in order],
            [y_ref[i] for i in order],
            color="black",
            linewidth=2,
            linestyle="--",
            label="MV reference (long-only, no BPRC constraints)",
            alpha=0.7,
        )

    if seed_front:
        ax.scatter(
            [s.objectives[0] for s in seed_front],
            [-s.objectives[1] for s in seed_front],
            marker="x",
            color="#d946ef",
            s=35,
            label="Multi-Greedy RP seeds",
            zorder=5,
        )

    ax.set_xlabel("Portfolio Variance")
    from investment_tool.strategies.optimization_strategies.utils import return_label

    ax.set_ylabel(return_label([s for result in results for s in result.front]))
    ax.set_title("Solver Comparison — Pareto Fronts")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved overlaid fronts plot: %s", save_path)
    return str(save_path)


def plot_metric_bars(
    results: list[SolverResult],
    output_directory: Path | None = None,
) -> str | None:
    """Side-by-side bar chart comparing all quality metrics.

    Parameters
    ----------
    results : list[SolverResult]
        Results from :meth:`SolverBenchmark.run`.
    output_directory : Path | None
        Output directory for the plot.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` on failure.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    save_path = _resolve_path(output_directory, "metric_bars.png")

    labels = [r.config.label for r in results]
    metric_names = [
        "hypervolume",
        "best_sharpe",
        "median_sharpe",
        "spread",
        "cardinality",
        "wall_time",
    ]
    display_names = [
        "Hypervolume ↑",
        "Best Sharpe ↑",
        "Median Sharpe ↑",
        "Spread ↑",
        "Cardinality ↑",
        "Wall Time (s) ↓",
    ]

    n_solvers = len(results)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for idx, (metric, display) in enumerate(zip(metric_names, display_names)):
        ax = axes_flat[idx]
        values = []
        for r in results:
            if metric == "wall_time":
                values.append(r.wall_time_seconds)
            else:
                values.append(r.metrics.get(metric, 0.0))

        colours = [_COLOURS[i % len(_COLOURS)] for i in range(n_solvers)]
        bars = ax.bar(labels, values, color=colours, alpha=0.8)
        ax.set_title(display, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=30)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            fmt = f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle("Solver Comparison — Quality Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved metric bars plot: %s", save_path)
    return str(save_path)


def plot_radar_chart(
    results: list[SolverResult],
    output_directory: Path | None = None,
) -> str | None:
    """Radar chart for multi-dimensional solver comparison.

    All metrics are normalised to [0, 1] (max across solvers = 1).
    Spacing and wall time are inverted so that *better = larger area*.

    Parameters
    ----------
    results : list[SolverResult]
        Results from :meth:`SolverBenchmark.run`.
    output_directory : Path | None
        Output directory for the plot.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` on failure.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    save_path = _resolve_path(output_directory, "radar_comparison.png")

    metric_keys = [
        "hypervolume",
        "best_sharpe",
        "median_sharpe",
        "spread",
        "cardinality",
    ]
    display_labels = [
        "Hypervolume",
        "Best Sharpe",
        "Median Sharpe",
        "Spread",
        "Cardinality",
    ]
    # Add wall time (inverted: lower is better → invert for radar)
    metric_keys.append("wall_time_inv")
    display_labels.append("Speed")

    n_metrics = len(metric_keys)

    # Gather raw values
    raw: list[list[float]] = []
    for r in results:
        row: list[float] = []
        for key in metric_keys:
            if key == "wall_time_inv":
                # Invert: 1/time so faster = higher
                t = max(r.wall_time_seconds, 1e-6)
                row.append(1.0 / t)
            else:
                row.append(r.metrics.get(key, 0.0))
        raw.append(row)

    raw_arr = np.array(raw)

    # Normalise each column to [0, 1]
    col_max = raw_arr.max(axis=0)
    col_max[col_max == 0] = 1.0
    normed = raw_arr / col_max

    # Radar angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles.append(angles[0])  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

    for i, r in enumerate(results):
        values = normed[i].tolist()
        values.append(values[0])  # close
        colour = _COLOURS[i % len(_COLOURS)]
        ax.plot(angles, values, linewidth=2, label=r.config.label, color=colour)
        ax.fill(angles, values, alpha=0.1, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_title("Solver Comparison — Radar", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved radar chart: %s", save_path)
    return str(save_path)
