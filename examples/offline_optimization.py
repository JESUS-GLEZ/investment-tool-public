"""Exercise the repository's real optimiser with explicit synthetic inputs.

Run from a source checkout; no credentials or network access are used.
This is a numerical demonstration, not a backtest or an investment recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from platypus import NSGAII

# Use this checkout's engine, including when the package is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (  # noqa: E402
    PortfolioVariator,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (  # noqa: E402
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.seeded_optimization import (  # noqa: E402
    MultiGreedyOptimizationStrategy,
)


def run_demo(output_dir: Path, seed: int = 17) -> dict:
    """Optimise, independently check the returned front, and export evidence."""
    labels = ["SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_E", "SYN_F"]
    mu = np.array([0.004, 0.007, 0.009, 0.012, 0.015, 0.019])
    volatility = np.array([0.025, 0.035, 0.045, 0.060, 0.080, 0.110])
    correlation = 0.2 * np.ones((6, 6)) + 0.8 * np.eye(6)
    cov = np.outer(volatility, volatility) * correlation
    model_settings = {
        "l_assets": 2,
        "u_assets": 3,
        "activity_tol": 1e-4,
        "constraint_slack": 0.1,
        "hhi_as_constraint": False,
        "balance": "risk_contributions",
    }
    search_settings = {
        "population_size": 40,
        "evaluations_per_restart": 160,
        "restarts": 2,
        "seed_max_subsets": 40,
        "seed_beam_width": 3,
        "refinement_supports": 0,
        "base_seed": seed,
        "time_limit_seconds": None,
    }
    scenario = {
        "kind": "synthetic_optimisation_inputs",
        "period": "illustrative monthly estimates",
        "assets": labels,
        "expected_returns": mu.tolist(),
        "covariance": cov.tolist(),
        "model": model_settings,
        "search": search_settings,
    }
    optimizer = MultiGreedyOptimizationStrategy(
        expected_returns=pd.Series(mu, index=labels),
        covariance_matrix=pd.DataFrame(cov, index=labels, columns=labels),
        risk_aversion=0.5,
        algorithm=NSGAII,
        model=BalancedPortfoliosRiskContributions,
        model_kwargs=model_settings,
        algorithm_kwargs={"variator": PortfolioVariator(min_assets=2, max_assets=3)},
        population_size=search_settings["population_size"],
        n_generations=search_settings["evaluations_per_restart"],
        n_restarts=search_settings["restarts"],
        include_markowitz_front=False,
        output_directory=None,
        date_label="synthetic-demo-v1",
    )
    optimizer.random_seed = seed
    optimizer.search_seconds = None
    optimizer.seed_max_subsets = search_settings["seed_max_subsets"]
    optimizer.seed_beam_width = search_settings["seed_beam_width"]
    optimizer.refinement_supports = 0
    result = optimizer.optimize()
    rows = []
    for solution in result.front:
        w = np.asarray(solution.variables, dtype=float)
        if not np.isfinite(w).all() or np.min(w) < 0:
            raise RuntimeError("Invalid long-only weights")
        if not np.isclose(w.sum(), 1.0, atol=1e-10, rtol=0):
            raise RuntimeError("Weights do not sum to one")
        active = w > model_settings["activity_tol"]
        if not 2 <= active.sum() <= 3:
            raise RuntimeError("Cardinality bound violated")
        variance = float(w @ cov @ w)
        if not np.isfinite(variance) or variance <= 0:
            raise RuntimeError("Portfolio variance must be finite and positive")
        risk_contributions = w * (cov @ w) / variance
        if np.any(risk_contributions[active] < 0.3 - 1e-9) or np.any(
            risk_contributions[active] > 0.55 + 1e-9
        ):
            raise RuntimeError("Risk-contribution bound violated")
        expected_return = float(mu @ w)
        hhi = float(risk_contributions @ risk_contributions)
        if not np.allclose(
            solution.objectives,
            [variance, -expected_return, hhi],
            atol=1e-10,
            rtol=1e-8,
        ):
            raise RuntimeError("Reported objectives do not reconcile")
        rows.append(
            {
                **dict(zip(labels, w.tolist())),
                "estimated_return": expected_return,
                "estimated_volatility": float(np.sqrt(variance)),
                "risk_contribution_hhi": hhi,
            }
        )
    if not rows:
        raise RuntimeError("The search returned no feasible candidate")
    front = pd.DataFrame(rows).sort_values(
        ["estimated_volatility", "estimated_return", "risk_contribution_hhi"]
    )
    objectives = front[
        ["estimated_volatility", "estimated_return", "risk_contribution_hhi"]
    ].to_numpy()
    objectives[:, 1] *= -1
    for i, point in enumerate(objectives):
        dominates = np.all(objectives <= point, axis=1) & np.any(
            objectives < point, axis=1
        )
        if dominates.any():
            raise RuntimeError(f"Candidate {i} is dominated in three objectives")
    # This illustrative selection uses zero risk-free return and model estimates.
    selected = front.loc[
        (front.estimated_return / front.estimated_volatility).idxmax()
    ].to_dict()
    encoded = json.dumps(scenario, sort_keys=True).encode()
    manifest = {
        "purpose": "Numerical demonstration; not a backtest or performance claim",
        "input_sha256": hashlib.sha256(encoded).hexdigest(),
        "feasible_candidates": len(front),
        "checks": [
            "finite long-only weights",
            "full investment",
            "2-3 active assets",
            "risk-contribution bounds",
            "independently reconstructed objectives",
            "three-objective nondominance",
        ],
        "selected_by": "maximum estimated return / volatility; zero risk-free rate",
        "selected": selected,
        "versions": {
            name: version(name)
            for name in ["numpy", "pandas", "scipy", "platypus-opt", "cvxpy"]
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [("inputs.json", scenario), ("manifest.json", manifest)]:
        (output_dir / name).write_text(
            json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    front.to_csv(output_dir / "feasible_front.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    points = ax.scatter(
        100 * front.estimated_volatility,
        100 * front.estimated_return,
        c=front.risk_contribution_hhi,
        cmap="viridis",
        s=34,
    )
    ax.scatter(
        100 * selected["estimated_volatility"],
        100 * selected["estimated_return"],
        marker="*",
        s=200,
        c="#db4c40",
        edgecolor="black",
        label="Illustrative selection",
    )
    ax.set(
        title="Synthetic portfolio optimisation: discovered candidates",
        xlabel="Estimated monthly volatility (%)",
        ylabel="Estimated monthly return (%)",
    )
    ax.legend(loc="lower right")
    fig.colorbar(points, ax=ax, label="Risk-contribution HHI")
    fig.text(
        0.5,
        -0.04,
        "Model estimates, not historical investment performance.",
        ha="center",
    )
    fig.savefig(output_dir / "front.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/output"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    manifest = run_demo(args.output_dir, args.seed)
    print(f"Validated {manifest['feasible_candidates']} synthetic candidates.")
    print(f"Outputs: {args.output_dir.resolve()}")
    print("Synthetic model estimates only; no historical performance claim.")


if __name__ == "__main__":
    main()
