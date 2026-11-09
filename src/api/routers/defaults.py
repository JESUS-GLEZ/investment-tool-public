"""Defaults & symbols router — read-only reference data.

Endpoints
---------
GET /defaults        Return all default parameter values (grouped by area).
GET /symbols         Return the available stock and crypto ticker symbols.
GET /data/files      List all available local .pkl dataset files with summary metadata.
GET /data/info       Return detailed metadata about one local .pkl dataset.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.adapters import _INPUT_BASE, _OUTPUT_BASE
from api.dataset_catalog import read_manifest, source_stamp, write_manifest
from ..local_data import available_datasets, resolve_dataset_path
from investment_tool.data.universes.loader import (
    DEFAULT_ASSET_CONFIG_FILE,
    list_asset_universes,
    load_asset_universe,
)
from investment_tool.defaults import (
    ALGORITHM,
    CONSTRUCTION,
    COVARIANCE,
    DECOMPOSITION,
    GENETIC,
    GENTLE_VARIATOR,
    INFRA,
    MARKOWITZ,
    MOMENTUM,
    OPTIMIZER,
    PORTFOLIO,
    PREDICTION,
    STRATEGY_SELECTION,
    STRATEGY_OPTIMIZER,
    VALIDATION,
)

router = APIRouter(tags=["reference"])
_CATALOG_BASE = _OUTPUT_BASE / ".dataset-catalog"


@router.get("/defaults")
def get_defaults() -> dict[str, Any]:
    """Return every default-parameter group as a flat JSON object."""
    return {
        "portfolio": asdict(PORTFOLIO),
        "strategy_selection": asdict(STRATEGY_SELECTION),
        "construction": asdict(CONSTRUCTION),
        "optimizer": asdict(OPTIMIZER),
        "strategy_optimizer": asdict(STRATEGY_OPTIMIZER),
        "genetic": asdict(GENETIC),
        "gentle_variator": asdict(GENTLE_VARIATOR),
        "decomposition": asdict(DECOMPOSITION),
        "algorithm": asdict(ALGORITHM),
        "markowitz": asdict(MARKOWITZ),
        "prediction": asdict(PREDICTION),
        "momentum": asdict(MOMENTUM),
        "validation": asdict(VALIDATION),
        "covariance": asdict(COVARIANCE),
        "infra": asdict(INFRA),
    }


@router.get("/asset-configs")
def get_asset_configs() -> list[dict[str, str | int]]:
    """List selectable asset universes and their stock/crypto counts."""
    return [universe.summary() for universe in list_asset_universes()]


@router.get("/symbols")
def get_symbols(file: str = DEFAULT_ASSET_CONFIG_FILE) -> dict[str, list[str]]:
    """Return symbols in a selected asset universe."""
    try:
        universe = load_asset_universe(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"stocks": universe.stocks, "crypto": universe.crypto}


@router.get("/data/files")
def list_local_data_files() -> list[dict[str, Any]]:
    """Return summary metadata for every .pkl file in ``/app/data/input/``.

    Each entry contains the filename and key fields parsed from the standard
    naming convention ``input_data_{freq}_{start}_to_{end}_{n}stocks_{date}.pkl``.
    Files are returned newest-first (by modification time).
    """
    data_dir = _INPUT_BASE
    pkls = available_datasets(data_dir)

    pattern = re.compile(
        r"input_data_(?P<freq>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2})_to_"
        r"(?P<end>\d{4}-\d{2}-\d{2})_(?P<n>\d+)stocks_"
    )

    result = []
    for name, path in pkls:
        m = pattern.search(name)
        result.append(
            {
                "frequency": m.group("freq") if m else None,
                "start_date": m.group("start") if m else None,
                "end_date": m.group("end") if m else None,
                "stock_count": int(m.group("n")) if m else None,
                **read_manifest(path, _CATALOG_BASE),
                "file": name,
            }
        )
    return result


@router.get("/data/info")
def get_local_data_info(
    file: str | None = Query(
        default=None, description="Filename within /app/data/input. Defaults to newest."
    ),
) -> dict[str, Any]:
    """Return detailed metadata extracted from a local .pkl dataset.

    Pass ``?file=<filename>`` to inspect a specific file, or omit it to use
    the most recently modified one.
    """
    data_dir = _INPUT_BASE

    if file is not None:
        try:
            pkl_path = resolve_dataset_path(data_dir, file)
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail="Local dataset not found in the input directory.",
            ) from exc
        selected_name = file
    else:
        pkls = available_datasets(data_dir)
        if not pkls:
            raise HTTPException(
                status_code=404,
                detail="No local .pkl dataset found. Mount a dataset or switch to alpaca.",
            )
        selected_name, pkl_path = pkls[0]

    try:
        stamp = source_stamp(pkl_path)
        with open(pkl_path, "rb") as f:
            data: dict[str, Any] = pickle.load(f)  # noqa: S301
        if not isinstance(data, dict):
            raise ValueError("Unsupported dataset structure")
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to load local dataset."
        ) from exc

    stock_dict: dict[str, Any] = data.get("stock_dict", {})
    crypto_dict: dict[str, Any] = data.get("crypto_dict", {})

    all_dates: list[Any] = []
    for df in list(stock_dict.values()) + list(crypto_dict.values()):
        try:
            if data.get("metadata", {}).get("time_contract_version") == 1:
                complete = df.loc[
                    df["is_complete"]
                    & (
                        pd.to_datetime(df["available_at"], utc=True)
                        <= pd.Timestamp.now(tz="UTC")
                    )
                ]
                if complete.empty:
                    continue
                all_dates.append(complete.index.min())
                all_dates.append(
                    pd.to_datetime(complete["period_end"]).max() - pd.Timedelta(days=1)
                )
            else:
                all_dates.append(df.index.min())
                all_dates.append(df.index.max())
        except Exception:
            pass

    start_date: str | None = str(min(all_dates))[:10] if all_dates else None
    end_date: str | None = str(max(all_dates))[:10] if all_dates else None

    metadata = data.get("metadata", {})
    frequency = metadata.get("frequency")
    if frequency not in {"daily", "weekly", "monthly"}:
        match = re.match(r"input_(?:data_)?(daily|weekly|monthly)_", selected_name)
        frequency = match.group(1) if match else None
    asset_config_file = metadata.get("asset_config_file")
    asset_config_name = None
    asset_config_available = None
    if isinstance(asset_config_file, str) and asset_config_file:
        try:
            universe = load_asset_universe(asset_config_file)
            asset_config_name = universe.name
            asset_config_available = True
        except ValueError:
            asset_config_available = False
    else:
        asset_config_file = None

    info: dict[str, Any] = {
        "file": selected_name,
        "frequency": frequency,
        "asset_config_file": asset_config_file,
        "asset_config_name": asset_config_name,
        "asset_config_available": asset_config_available,
        "stocks": sorted(stock_dict.keys()),
        "crypto": sorted(crypto_dict.keys()),
        "start_date": start_date,
        "end_date": end_date,
        "stock_count": len(stock_dict),
        "crypto_count": len(crypto_dict),
        "time_contract_version": metadata.get("time_contract_version")
        if type(metadata.get("time_contract_version")) is int
        else None,
    }

    def text_metadata(key: str) -> str | None:
        value = metadata.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    metadata_title = text_metadata("title")
    cached = read_manifest(pkl_path, _CATALOG_BASE)
    frequency_label = str(frequency).capitalize() if frequency else "Market"
    coverage = f"{start_date or '?'} to {end_date or '?'}"
    info.update(
        {
            "title": str(
                metadata_title
                or cached.get("title")
                or f"{frequency_label} history — {coverage}"
            ),
            "description": str(
                text_metadata("description")
                or cached.get("description")
                or "Saved market-data snapshot. Inspect its contents and coverage before choosing it for a run."
            ),
            "intended_use": text_metadata("intended_use"),
            "provider": text_metadata("source"),
            "retrieved_at": text_metadata("retrieved_at"),
            "inspected_at": datetime.now(timezone.utc).isoformat(),
            "size_bytes": stamp["bytes"],
            "features": {
                "prices": bool(stock_dict or crypto_dict),
                "execution_quotes": bool(data.get("execution_dict")),
                "news": bool(data.get("news_dict")),
                "cached_sentiment": metadata.get("sentiment_cached")
                if isinstance(metadata.get("sentiment_cached"), bool)
                else None,
            },
        }
    )
    write_manifest(pkl_path, info, stamp, _CATALOG_BASE)
    return info
