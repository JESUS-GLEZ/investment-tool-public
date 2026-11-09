#!/usr/bin/env python3
"""Command-line interface for prediction strategy comparison.

Benchmarks multiple return prediction strategies via walk-forward
evaluation and generates comparison reports.

Examples
--------
Run with JSON parameters::

    $ python execute_prediction_comparison.py --use-json params.json

Run with explicit predictors::

    $ python execute_prediction_comparison.py \\
        --data-source local \\
        --input-path data/input/file.pkl \\
        --predictors RidgeVARX EWM-6 EWM-12 \\
        --n-eval-periods 18
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from investment_tool.app.prediction_comparison_coordinator import (
    PredictionComparisonParameters,
    execute_prediction_comparison,
    get_available_predictors,
)
from investment_tool.config import load_config
from investment_tool.constants import DataSource, Frequency
from investment_tool.defaults import STRATEGY_SELECTION

from common import (
    parse_arguments_with_json,
    parse_validation_level,
    resolve_input_path,
    resolve_output_dir,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for prediction comparison.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    available = get_available_predictors()

    parser = argparse.ArgumentParser(
        description=(
            "Prediction Comparison — benchmark return prediction strategies "
            "via walk-forward evaluation"
        )
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
        "--predictors",
        nargs="+",
        default=["RidgeVARX", "EWM-3", "EWM-6", "EWM-12"],
        help=f"Predictors to compare. Available: {', '.join(available)}",
    )
    parser.add_argument(
        "--n-eval-periods",
        type=int,
        default=12,
        help="Number of walk-forward evaluation steps (default: 12)",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=12,
        help="Minimum historical periods before first evaluation (default: 12)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top assets for hit-rate metric (default: 5)",
    )
    parser.add_argument(
        "--compute-sentiment",
        action="store_true",
        default=True,
        help="Compute sentiment scores for strategies that need them (default: True)",
    )
    parser.add_argument(
        "--no-sentiment",
        dest="compute_sentiment",
        action="store_false",
        help="Skip sentiment computation (RidgeVARX will use zero scores)",
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
    """Execute the prediction comparison.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    try:
        args = parse_arguments()

        base_dir = Path(__file__).resolve().parents[1]
        run_dir = resolve_output_dir(
            getattr(args, "output_dir", None),
            base_dir,
            prefix="prediction_comparison",
        )

        setup_logging(run_dir, log_filename="prediction_comparison.log")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)

        input_data_path = resolve_input_path(
            getattr(args, "input_path", None), base_dir
        )

        validation_level = parse_validation_level(
            getattr(args, "validation_level", "warn")
        )

        parameters = PredictionComparisonParameters(
            data_source=DataSource(args.data_source),
            frequency=Frequency(args.frequency),
            output_directory=run_dir,
            input_data_path=input_data_path,
            save_input_data=getattr(args, "save_input_data", False),
            validation_level=validation_level,
            n_eval_periods=getattr(args, "n_eval_periods", 12),
            min_history=getattr(args, "min_history", 12),
            top_k=getattr(args, "top_k", 5),
            predictors=getattr(
                args, "predictors", ["RidgeVARX", "EWM-3", "EWM-6", "EWM-12"]
            ),
            compute_sentiment=getattr(args, "compute_sentiment", True),
            cfg=load_config(),
        )

        logger.info("Starting prediction comparison run in %s", run_dir)
        result = execute_prediction_comparison(parameters)

        logger.info(
            "Prediction comparison completed. Files saved: %s",
            result.get("saved_files", []),
        )
        return 0

    except Exception as e:
        logger.error("Prediction comparison failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
