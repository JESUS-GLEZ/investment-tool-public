#!/usr/bin/env python3
"""CLI for running a side-by-side strategy comparison.

Loads market data once, backtests N strategies on independent copies,
and produces comparison metrics, overlay plots, and an HTML report.

Examples
--------
Run with JSON parameters::

    $ python execute_strategy_comparison.py --use-json params.json

Run specific strategies::

    $ python execute_strategy_comparison.py \\
        --strategies combined optimization \\
        --data-source local \\
        --input-path data/input/file.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from investment_tool.app.strategy_comparison_coordinator import (
    StrategyComparisonParameters,
    compare_strategies,
)
from investment_tool.config import load_config
from investment_tool.constants import DataSource, Frequency, StrategyType
from investment_tool.data.validation import ValidationLevel
from investment_tool.defaults import CONSTRUCTION, STRATEGY_SELECTION

from common import (
    parse_arguments_with_json,
    resolve_input_path,
    resolve_output_dir,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for strategy comparison.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Strategy Comparison CLI")

    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=[s.value for s in StrategyType],
        help="Strategies to compare (default: from JSON config)",
    )
    parser.add_argument(
        "--benchmark-asset",
        type=str,
        default="SPY",
        help="Asset symbol for buy-and-hold benchmark (default: SPY)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default=STRATEGY_SELECTION.data_source,
        choices=[ds.value for ds in DataSource],
    )
    parser.add_argument("--input-path", type=str, default=None)
    parser.add_argument(
        "--frequency",
        type=str,
        default=STRATEGY_SELECTION.frequency,
        choices=[f.value for f in Frequency],
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--save-input-data",
        action="store_true",
        help="Whether to save fetched input data locally (default: False)",
    )
    parser.add_argument(
        "--validation-level",
        type=str,
        default=ValidationLevel.WARN.value,
        choices=[v.value for v in ValidationLevel],
    )
    parser.add_argument(
        "--min-assets",
        type=int,
        default=CONSTRUCTION.min_assets,
        help="Minimum active assets in portfolio (default: %(default)s)",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=CONSTRUCTION.max_assets,
        help="Maximum active assets in portfolio (default: %(default)s)",
    )
    parser.add_argument(
        "--use-json",
        type=str,
        default=None,
        help="Load parameters from a JSON file (path required)",
    )

    args = parse_arguments_with_json(parser)

    # Final defaults
    if args.strategies is None:
        args.strategies = [StrategyType.COMBINED.value]

    return args


def main() -> None:
    """Entry point for strategy comparison."""

    args = parse_arguments()

    base_dir = Path(__file__).resolve().parents[1]
    output_dir = resolve_output_dir(args.output_dir, base_dir, prefix="comparison")

    setup_logging(output_dir, log_filename="strategy_comparison.log")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("Platypus").setLevel(logging.INFO)

    input_data_path = resolve_input_path(args.input_path, base_dir)

    strategy_types = [StrategyType(s) for s in args.strategies]

    params = StrategyComparisonParameters(
        strategy_types=strategy_types,
        data_source=DataSource(args.data_source),
        frequency=Frequency(args.frequency),
        output_directory=output_dir,
        input_data_path=input_data_path,
        save_input_data=args.save_input_data,
        benchmark_asset=args.benchmark_asset,
        min_assets=getattr(args, "min_assets", CONSTRUCTION.min_assets),
        max_assets=getattr(args, "max_assets", CONSTRUCTION.max_assets),
        cfg=load_config(),
    )

    logger.info(
        "Starting strategy comparison: %s  (benchmark asset: %s)",
        [s.value for s in strategy_types],
        args.benchmark_asset,
    )

    result = compare_strategies(params)

    logger.info("Comparison complete.  Output in: %s", output_dir)
    logger.info("Files saved: %d", len(result.get("saved_files", [])))

    metrics_df = result.get("metrics_df")
    if metrics_df is not None:
        print("\n" + "=" * 60)
        print("STRATEGY COMPARISON METRICS")
        print("=" * 60)
        print(metrics_df.to_string(float_format="%.4f"))
        print("=" * 60)


if __name__ == "__main__":
    main()
