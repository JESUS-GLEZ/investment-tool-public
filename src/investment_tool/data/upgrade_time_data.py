"""Explicit, non-destructive upgrade of this tool's legacy Alpaca snapshots."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.data.repositories._local_loader import LocalDataFetcher
from investment_tool.data.repositories.local_data_saver import LocalDataSaver
from investment_tool.domain.market_time import (
    TIME_CONTRACT_VERSION,
    normalize_bars,
    utc_instant,
)


def upgrade_legacy_alpaca(
    data: dict[str, Any],
    frequency: Frequency,
    *,
    start_date: str,
    end_date: str,
    retrieved_at: str,
) -> dict[str, Any]:
    """Convert only the explicitly declared old Alpaca convention.

    Legacy requests used UTC midnight bounds and subtracted one hour from
    monthly source labels. Preserve all bars/news; truncated bars are flagged,
    not relabelled as complete. Retrieval time must be supplied, not guessed.
    """
    if data.get("metadata", {}).get("time_contract_version") is not None:
        raise ValueError("Dataset is already versioned; refusing a legacy conversion")
    start = pd.Timestamp(start_date).tz_localize("UTC")
    end = pd.Timestamp(end_date).tz_localize("UTC")
    if start >= end:
        raise ValueError("Original request start must precede its end")
    fetched = utc_instant(retrieved_at)
    result = dict(data)
    for key, zone in (("stock_dict", "America/New_York"), ("crypto_dict", "UTC")):
        converted: dict[str, pd.DataFrame] = {}
        for symbol, frame in data.get(key, {}).items():
            source = frame.copy()
            if not source.empty:
                # The explicit legacy profile declares naive labels to be UTC.
                dates = pd.DatetimeIndex(pd.to_datetime(source.index))
                dates = (
                    dates.tz_localize("UTC")
                    if dates.tz is None
                    else dates.tz_convert("UTC")
                )
                if frequency == Frequency.MONTHLY:
                    dates += pd.Timedelta(hours=1)
                source.index = dates
            converted[symbol] = normalize_bars(
                source,
                frequency,
                market_timezone=zone,
                coverage_start=start,
                coverage_end=end,
                as_of=fetched,
            )
        result[key] = converted
    result["metadata"] = {
        **data.get("metadata", {}),
        "source": "Alpaca",
        "frequency": str(frequency),
        "time_contract_version": TIME_CONTRACT_VERSION,
        "legacy_conversion": "alpaca-minus-one-hour-monthly-v0",
        "start_date": start_date,
        "end_date": end_date,
        "retrieved_at": fetched.isoformat(),
        "availability_assumption": "completed calendar interval end; publication latency unknown",
    }
    return result


def save_upgraded_data(data: dict[str, Any], output: Path) -> None:
    """Write a new snapshot atomically, without overwriting an existing file."""
    if output.suffix not in (".pkl", ".json"):
        raise ValueError("Output must be .pkl or .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = stream.name
            if output.suffix == ".pkl":
                pickle.dump(data, stream)
            else:
                payload = {
                    key: {
                        s: LocalDataSaver._convert_dataframe_to_serializable(df)
                        for s, df in value.items()
                    }
                    if key in ("stock_dict", "crypto_dict")
                    else LocalDataSaver._make_json_serializable(value)
                    for key, value in data.items()
                }
                stream.write(
                    json.dumps(payload, default=LocalDataSaver._json_serial).encode()
                )
        # link fails if the target exists, including a concurrent creator.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy-alpaca", action="store_true", required=True)
    parser.add_argument(
        "--frequency", choices=[str(f) for f in Frequency], required=True
    )
    parser.add_argument(
        "--start-date", required=True, help="Original Alpaca request start date"
    )
    parser.add_argument(
        "--end-date", required=True, help="Original Alpaca request end date"
    )
    parser.add_argument(
        "--retrieved-at",
        required=True,
        help="Known retrieval instant, with timezone; a conservative earlier bound is acceptable",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must differ")
    upgraded = upgrade_legacy_alpaca(
        LocalDataFetcher.load_data(args.input),
        Frequency(args.frequency),
        start_date=args.start_date,
        end_date=args.end_date,
        retrieved_at=args.retrieved_at,
    )
    save_upgraded_data(upgraded, args.output)
    print(f"Saved versioned dataset to {args.output}; original file unchanged")


if __name__ == "__main__":
    main()
