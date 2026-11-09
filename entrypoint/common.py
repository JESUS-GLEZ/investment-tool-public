"""Shared entrypoint utilities.

Provides common helpers used by all CLI entry points:

* :func:`load_json_params` — load and key-normalise a JSON parameter file.
* :func:`setup_logging` — configure root logger with console + file handlers.
* :func:`resolve_output_dir` — build a timestamped, unique run directory.
* :func:`resolve_input_path` — resolve a possibly-relative input path.
* :func:`parse_validation_level` — map a string to a ``ValidationLevel``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from collections.abc import Callable

from investment_tool.data.validation import ValidationLevel

logger = logging.getLogger(__name__)


# ── JSON parameter loading ──────────────────────────────────────────────────


def load_json_params(json_path: str | Path) -> dict[str, Any]:
    """Load parameters from a JSON file, converting kebab-case keys to snake_case.

    Parameters
    ----------
    json_path : str | Path
        Path to the JSON configuration file.

    Returns
    -------
    dict[str, Any]
        Loaded parameters with normalised keys.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the JSON is malformed.
    """
    json_path = Path(json_path)
    logger.info("Loading parameters from %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)
    return {k.replace("-", "_"): v for k, v in raw.items()}


# ── Argument precedence ─────────────────────────────────────────────────────


def parse_arguments_with_json(
    parser: argparse.ArgumentParser,
    load_params: Callable[[str | Path], dict[str, Any]] = load_json_params,
) -> argparse.Namespace:
    """Apply parser defaults, then JSON, then explicitly supplied CLI options.

    Parsing into an existing namespace fills missing defaults without overwriting
    JSON values. Explicit options still go through argparse's normal actions,
    type conversion and choice validation, even when equal to a parser default.
    """
    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.use_json:
        return args
    namespace = argparse.Namespace(**load_params(args.use_json))
    return parser.parse_args(argv, namespace=namespace)


# ── Logging ─────────────────────────────────────────────────────────────────


def setup_logging(
    run_dir: Path,
    log_filename: str = "run.log",
    *,
    console_level: int = logging.INFO,
) -> None:
    """Configure the root logger with console and file handlers.

    Parameters
    ----------
    run_dir : Path
        Directory where *log_filename* will be created.
    log_filename : str, optional
        Name of the log file (default ``"run.log"``).
    console_level : int, optional
        Logging level for the console handler (default ``INFO``).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(formatter)

    file_h = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)

    root.addHandler(console)
    root.addHandler(file_h)

    root.info("Logging initialized. Writing logs to: %s", log_path)


# ── Path helpers ────────────────────────────────────────────────────────────


def resolve_output_dir(
    output_dir_arg: str | None,
    base_dir: Path,
    prefix: str = "run",
) -> Path:
    """Build a timestamped output directory under *base_dir*.

    Parameters
    ----------
    output_dir_arg : str | None
        CLI/JSON ``--output-dir`` value.  Defaults to ``"data/output"``.
    base_dir : Path
        Project root (parent of ``entrypoint/``).
    prefix : str, optional
        Directory name prefix (default ``"run"``).

    Returns
    -------
    Path
        Created directory, e.g. ``<base>/data/output/run_20250101_120000``.
    """
    output_dir_arg = output_dir_arg or "data/output"
    output_path = Path(output_dir_arg)
    base_output = output_path if output_path.is_absolute() else base_dir / output_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_input_path(raw: str | None, base_dir: Path) -> Path | None:
    """Resolve a possibly-relative input file path.

    Parameters
    ----------
    raw : str | None
        Raw path from CLI/JSON, or ``None``.
    base_dir : Path
        Project root used to resolve relative paths.

    Returns
    -------
    Path | None
        Absolute path, or ``None`` if *raw* is falsy.
    """
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else base_dir / p


def parse_validation_level(level_str: str) -> ValidationLevel:
    """Map a string to a :class:`ValidationLevel`.

    Parameters
    ----------
    level_str : str
        One of ``"strict"``, ``"warn"``, or ``"skip"`` (case-insensitive).

    Returns
    -------
    ValidationLevel
        Corresponding enum member (defaults to ``WARN``).
    """
    mapping = {
        "strict": ValidationLevel.STRICT,
        "warn": ValidationLevel.WARN,
        "skip": ValidationLevel.SKIP,
    }
    return mapping.get(level_str.lower(), ValidationLevel.WARN)
