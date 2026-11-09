"""
Local data saver for persisting financial data.

This module provides the LocalDataSaver class for saving fetched
financial data to local JSON or pickle files for later use.

Unlike the fetcher modules, this is a **public** part of the
repositories package because saving is an output concern used by
multiple consumers (e.g. ModelDataFactory, LocalRepository).
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.config import AppConfig, get_config

logger = logging.getLogger(__name__)


class LocalDataSaver:
    """Saves investment data to local file storage.

    Supports saving to both JSON and pickle formats with automatic
    path generation based on configuration and data properties.
    """

    @staticmethod
    def save_data(
        data: dict[str, Any],
        frequency: str,
        save_format: str,
        cfg: AppConfig | None = None,
    ) -> Path:
        """Save investment data to a local file.

        Parameters
        ----------
        data : dict[str, Any]
            Data dictionary with stock_dict, crypto_dict, news_dict.
        frequency : str
            Data frequency for filename (e.g., 'daily', 'monthly').
        save_format : str
            Output format ('pickle' or 'json').
        cfg : AppConfig | None, optional
            Typed application configuration.  Defaults to ``get_config()``.

        Returns
        -------
        Path
            Path where data was saved.

        Raises
        ------
        ValueError
            If data is empty or save fails.
        """
        if not data:
            logger.warning("No input data to save locally")
            raise ValueError("Cannot save empty data")

        effective_cfg = cfg if cfg is not None else get_config()

        # Setup directory from config
        base_dir = Path(__file__).resolve().parents[4]
        data_dir = (base_dir / effective_cfg.paths.input_data_dir).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)

        stock_count = len(data.get("stock_dict", {}))
        today_str = datetime.now().date()
        start_date = effective_cfg.model.historical_start_date
        end_date = effective_cfg.model.historical_end_date
        if save_format == "pickle":
            filename = f"input_data_{frequency}_{start_date}_to_{end_date}_{stock_count}stocks_{today_str}.pkl"
        else:
            filename = f"input_data_{frequency}_{start_date}_to_{end_date}_{stock_count}stocks_{today_str}.json"
        filepath = data_dir / filename

        logger.info(f"Saving input data locally to: {filepath}")
        try:
            if save_format == "pickle":
                with open(filepath, "wb") as f:
                    pickle.dump(data, f)
            else:
                serializable_data: dict[str, Any] = {}
                for key, value in data.items():
                    if key in ["stock_dict", "crypto_dict"]:
                        serializable_data[key] = {
                            symbol: LocalDataSaver._convert_dataframe_to_serializable(
                                df
                            )
                            for symbol, df in value.items()
                        }
                    else:
                        serializable_data[key] = LocalDataSaver._make_json_serializable(
                            value
                        )
                with open(filepath, "w") as f:
                    json.dump(
                        serializable_data,
                        f,
                        indent=2,
                        default=LocalDataSaver._json_serial,
                    )

            logger.info(f"Input data successfully saved to: {filepath}")
            return Path(filepath)

        except Exception as e:
            logger.error(f"Failed to save input data locally: {e}")
            raise ValueError(f"Failed to save data: {e!s}")

    @staticmethod
    def _convert_dataframe_to_serializable(df: pd.DataFrame) -> dict[str, Any]:
        """Convert DataFrame to serializable 'split' format.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to convert.

        Returns
        -------
        dict[str, Any]
            Dictionary with 'index', 'columns', and 'data' keys.
        """
        result = df.to_dict("split")
        result["index"] = [
            val.isoformat() if isinstance(val, (pd.Timestamp, datetime)) else val
            for val in result["index"]
        ]
        return result

    @staticmethod
    def _make_json_serializable(obj: Any) -> Any:
        """Recursively convert non-serializable objects.

        Parameters
        ----------
        obj : Any
            Object to convert.

        Returns
        -------
        Any
            JSON-serializable version of the object.
        If a dict has a tuple as a key, build nested dictionaries using tuple elements as keys.

        Parameters
        ----------
        obj : Any
            The object to make serializable

        Returns
        -------
        Any
            A JSON serializable version of the object
        """
        if isinstance(obj, dict):
            new_dict: dict[str, Any] = {}
            for k, v in obj.items():
                serial_v = LocalDataSaver._make_json_serializable(v)
                # If key is a tuple, build nested dictionaries
                if isinstance(k, tuple):
                    nested = new_dict
                    for i, part in enumerate(k):
                        part = str(part)
                        # If this is the last part, assign the value
                        if i == len(k) - 1:
                            nested[part] = serial_v
                        else:
                            if part not in nested or not isinstance(
                                nested.get(part), dict
                            ):
                                nested[part] = {}
                            nested = nested[part]
                else:
                    # Ensure the key is of an allowed type (JSON only supports str keys)
                    new_key: str = str(k) if not isinstance(k, str) else k
                    new_dict[new_key] = serial_v
            return new_dict
        elif isinstance(obj, list):
            return [LocalDataSaver._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [LocalDataSaver._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    @staticmethod
    def _json_serial(obj: Any) -> Any:
        """Custom JSON serializer for objects not serializable by default.

        Used as the 'default' parameter in json.dump().

        Parameters
        ----------
        obj : Any
            Object to serialize.

        Returns
        -------
        Any
            JSON serializable representation.

        Raises
        ------
        TypeError
            If the object cannot be serialized.
        """
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )
