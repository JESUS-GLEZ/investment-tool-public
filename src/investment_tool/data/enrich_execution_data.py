"""Add synchronized execution quotes without refetching history/news.

Usage: python -m investment_tool.data.enrich_execution_data INPUT.pkl OUTPUT.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.data.repositories._local_loader import LocalDataFetcher
from investment_tool.data.repositories.alpaca_repository import AlpacaRepository
from investment_tool.domain.market_time import (
    TIME_CONTRACT_VERSION,
    period_code,
    validate_bar_times,
)


def enrich_file(input_path: Path, output_path: Path) -> None:
    """Write a new pickle containing the original data plus execution quotes."""
    if output_path.exists():
        raise ValueError(
            "Output already exists; choose a new file to preserve your data"
        )
    if output_path.suffix != ".pkl":
        raise ValueError("Output must be a .pkl file")
    data = LocalDataFetcher.load_data(input_path)
    if data.get("metadata", {}).get("time_contract_version") != TIME_CONTRACT_VERSION:
        raise ValueError(
            "Upgrade legacy timestamps with upgrade_time_data before adding execution quotes"
        )
    frequency = Frequency(data["metadata"].get("frequency"))
    frames = [
        *data.get("stock_dict", {}).values(),
        *data.get("crypto_dict", {}).values(),
    ]
    dates = [
        pd.Timestamp(value)
        for frame in frames
        if not frame.empty
        for value in validate_bar_times(frame, frequency).index
    ]
    if not dates:
        raise ValueError("Input contains no historical bars")
    periods = pd.DatetimeIndex(dates).to_period(period_code(frequency))
    start = periods.min().start_time.date().isoformat()
    end = periods.max().end_time.date().isoformat()
    repository = AlpacaRepository(frequency)
    data["execution_dict"] = repository.fetch_execution_data(
        list(data.get("crypto_dict", {})), start, end
    )
    # Exclusive creation leaves the input and any existing output untouched.
    with output_path.open("xb") as destination:
        pickle.dump(data, destination)
    logging.getLogger(__name__).info(
        "Saved enriched %s data to %s", frequency, output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    enrich_file(args.input, args.output)


if __name__ == "__main__":
    main()
