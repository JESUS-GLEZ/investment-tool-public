"""
Private local file loader for the Local repository.

This module provides the LocalDataFetcher class — the low-level I/O
logic for reading financial data from pickle/JSON files.  It is an
**implementation detail** of :class:`LocalRepository` and should not be
imported directly by other packages.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from investment_tool.config import PathsConfig, get_config

logger = logging.getLogger(__name__)


class LocalDataFetcher:
    """Fetches investment data from local file storage.

    Supports loading from both JSON and pickle formats, with automatic
    path resolution based on configuration.
    """

    @staticmethod
    def _resolve_path(
        file_path: Path,
        paths_config: PathsConfig | None = None,
    ) -> Path:
        """Resolve file path against configured input directory.

        Parameters
        ----------
        file_path : Path
            File path (absolute or relative).
        paths_config : PathsConfig | None, optional
            Typed path configuration.  Defaults to ``get_config().paths``.

        Returns
        -------
        Path
            Resolved absolute path.
        """
        if file_path.is_absolute() and file_path.exists():
            return Path(file_path)
        # Try as-is first
        if file_path.exists():
            return Path(file_path)
        cfg = paths_config if paths_config is not None else get_config().paths
        base_dir = Path(__file__).resolve().parents[4]
        candidate = (base_dir / cfg.input_data_dir / file_path).resolve()
        return Path(candidate)

    @staticmethod
    def load_data(file_path: Path | None) -> dict[str, Any]:
        """Load investment data from a local JSON or pickle file.

        Parameters
        ----------
        file_path : Path | None
            Path to data file (.json or .pkl).

        Returns
        -------
        dict[str, Any]
            Data dictionary with stock_dict, crypto_dict, and news_dict.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file format is invalid.
        """
        if file_path is None:
            raise ValueError("file_path cannot be None")

        resolved_path = LocalDataFetcher._resolve_path(file_path)
        logger.info(f"Loading input data from local file: {resolved_path}")
        if not resolved_path.exists():
            raise FileNotFoundError(f"Local data file not found: {resolved_path}")
        try:
            if resolved_path.suffix == ".pkl":
                with open(resolved_path, "rb") as f:
                    data: dict[str, Any] = pickle.load(f)
                return dict(data)

            with open(resolved_path, "r") as f:
                data = json.load(f)

            if "news_dict" in data:
                restored_news_dict: dict[str, dict[Any, Any]] = {}
                for symbol, periods in data["news_dict"].items():
                    restored_news_dict[symbol] = {}
                    for key1, nested in periods.items():
                        if isinstance(nested, dict):
                            for key2, articles_list in nested.items():
                                restored_news_dict[symbol][(key1, key2)] = articles_list
                        else:
                            restored_news_dict[symbol][key1] = nested
                data["news_dict"] = restored_news_dict

            if "stock_dict" in data:
                stock_dict = {}
                for symbol, records in data["stock_dict"].items():
                    if isinstance(records, dict) and all(
                        k in records for k in ("index", "columns", "data")
                    ):
                        df = pd.DataFrame(**records)
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            pass
                    else:  # fallback (if not saved in split format)
                        df = pd.DataFrame.from_records(records)
                        if "index" in df.columns:
                            df.set_index("index", inplace=True)
                        elif "date" in df.columns:
                            df.set_index("date", inplace=True)
                        df.index = pd.to_datetime(df.index)
                    stock_dict[symbol] = df
                data["stock_dict"] = stock_dict

            if "crypto_dict" in data:
                crypto_dict = {}
                for symbol, records in data["crypto_dict"].items():
                    if isinstance(records, dict) and all(
                        k in records for k in ("index", "columns", "data")
                    ):
                        df = pd.DataFrame(**records)
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            pass
                    else:
                        df = pd.DataFrame.from_records(records)
                        if "index" in df.columns:
                            df.set_index("index", inplace=True)
                        elif "date" in df.columns:
                            df.set_index("date", inplace=True)
                        df.index = pd.to_datetime(df.index)
                    crypto_dict[symbol] = df
                data["crypto_dict"] = crypto_dict

            if not all(
                key in data for key in ["stock_dict", "crypto_dict", "news_dict"]
            ):
                logger.warning("Missing expected keys in local data file")

            return dict(data)

        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in local data file: {resolved_path}")
        except Exception as e:
            logger.error(f"Error loading local data: {e}")
            raise ValueError(f"Failed to load data from {resolved_path}: {str(e)}")
