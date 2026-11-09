#!/usr/bin/env python3
"""Command-line interface for solver comparison.

Benchmarks multiple evolutionary portfolio optimisation algorithms on
the same problem instance and generates comparison reports.

Examples
--------
Run with JSON parameters::

    $ python execute_solver_comparison.py --use-json params.json

Run with explicit solvers::

    $ python execute_solver_comparison.py \\
        --data-source local \\
        --input-path data/input/file.pkl \\
        --solvers GWASFGA NSGA-II SPEA2 \\
        --n-generations 5000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from investment_tool.app.solver_comparison_coordinator import (
    SolverComparisonParameters,
    execute_solver_comparison,
    get_available_solvers,
)
from investment_tool.config import load_config
from investment_tool.constants import DataSource, Frequency
from investment_tool.defaults import CONSTRUCTION, OPTIMIZER, STRATEGY_SELECTION

from common import (
    parse_arguments_with_json,
    parse_validation_level,
    resolve_input_path,
    resolve_output_dir,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for solver comparison.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    available = get_available_solvers()

    parser = argparse.ArgumentParser(
        description="Solver Comparison — benchmark portfolio optimisation algorithms"
    )

    parser.add_argument(
        "--data-source",
        type=str,
        default=STRATEGY_SELECTION.data_source,
        choices=[ds.value for ds in DataSource],
        help="Source of financial data (default: %(default)s)",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to input data file",
    )
    parser.add_argument(
        "--save-input-data",
        action="store_true",
        help="Whether to save input data locally",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default=STRATEGY_SELECTION.frequency,
        choices=[f.value for f in Frequency],
        help="Data frequency (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output files",
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["GWASFGA", "NSGA-II", "NSGA-III", "SPEA2"],
        help=f"Solvers to compare. Available: {', '.join(available)}",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=OPTIMIZER.pop_size,
        help="GA population size for all solvers (default: %(default)s)",
    )
    parser.add_argument(
        "--n-generations",
        type=int,
        default=OPTIMIZER.n_generations,
        help="Function evaluations per restart (default: %(default)s)",
    )
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=OPTIMIZER.n_restarts,
        help="Independent restarts per solver (default: %(default)s)",
    )
    parser.add_argument(
        "--min-assets",
        type=int,
        default=CONSTRUCTION.min_assets,
        help="Minimum active assets in portfolio (default: %(default)s)",
    )
    parser.add_argument(
        "--num-assets",
        type=int,
        default=CONSTRUCTION.max_assets,
        help="Maximum active assets in portfolio (default: %(default)s)",
    )
    parser.add_argument(
        "--include-markowitz-reference",
        action="store_true",
        default=True,
        help="Compute Markowitz frontier as reference (default: True)",
    )
    parser.add_argument(
        "--no-markowitz-reference",
        dest="include_markowitz_reference",
        action="store_false",
        help="Skip Markowitz reference computation",
    )
    parser.add_argument(
        "--validation-level",
        type=str,
        choices=["strict", "warn", "skip"],
        default="warn",
        help="Validation strictness (default: warn)",
    )
    parser.add_argument(
        "--use-json",
        type=str,
        help="Load parameters from a JSON file (path required)",
    )

    return parse_arguments_with_json(parser)


def main() -> int:
    """Execute the solver comparison.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    try:
        args = parse_arguments()

        base_dir = Path(__file__).resolve().parents[1]
        run_dir = resolve_output_dir(
            getattr(args, "output_dir", None), base_dir, prefix="solver_comparison"
        )

        setup_logging(run_dir, log_filename="solver_comparison.log")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("Platypus").setLevel(logging.WARNING)

        input_data_path = resolve_input_path(
            getattr(args, "input_path", None), base_dir
        )

        validation_level = parse_validation_level(
            getattr(args, "validation_level", "warn")
        )

        parameters = SolverComparisonParameters(
            data_source=DataSource(args.data_source),
            frequency=Frequency(args.frequency),
            output_directory=run_dir,
            input_data_path=input_data_path,
            save_input_data=getattr(args, "save_input_data", False),
            validation_level=validation_level,
            population_size=getattr(args, "population_size", OPTIMIZER.pop_size),
            n_generations=getattr(args, "n_generations", OPTIMIZER.n_generations),
            n_restarts=getattr(args, "n_restarts", OPTIMIZER.n_restarts),
            min_assets=getattr(args, "min_assets", CONSTRUCTION.min_assets),
            num_assets=getattr(args, "num_assets", CONSTRUCTION.max_assets),
            include_markowitz_reference=getattr(
                args, "include_markowitz_reference", True
            ),
            solvers=getattr(
                args, "solvers", ["GWASFGA", "NSGA-II", "NSGA-III", "SPEA2"]
            ),
            cfg=load_config(),
        )

        logger.info("Starting solver comparison run in %s", run_dir)
        result = execute_solver_comparison(parameters)

        logger.info(
            "Solver comparison completed. Files saved: %s",
            result.get("saved_files", []),
        )
        return 0

    except Exception as e:
        logger.error("Solver comparison failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
