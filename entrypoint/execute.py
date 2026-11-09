#!/usr/bin/env python3
"""
Command-line interface for the Investment Tool.

This module provides the main entry point for running investment strategies,
backtesting, and portfolio optimization. It handles CLI argument parsing,
logging setup with file tee, and orchestrates the execution flow.

Examples
--------
Run with JSON parameters:
    $ python execute.py --use-json params.json

Run backtesting with local data:
    $ python execute.py --mode backtesting --data-source local --input-path data/input/file.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from io import TextIOWrapper
from logging.handlers import MemoryHandler
from pathlib import Path
from typing import Any, TextIO

from investment_tool.app.execution_coordinator import Parameters, execute
from investment_tool.config import load_config
from investment_tool.constants import DataSource, ExecutionMode, Frequency, StrategyType
from investment_tool.data.validation import ValidationLevel
from investment_tool.defaults import CONSTRUCTION, STRATEGY_SELECTION
from common import parse_arguments_with_json

logger = logging.getLogger(__name__)


class _Tee:
    """Stream multiplexer that writes to two streams simultaneously.

    Used to duplicate stdout/stderr output to both console and log file.

    Parameters
    ----------
    stream_a : TextIO
        Primary output stream (typically original stdout).
    stream_b : TextIO
        Secondary output stream (typically log file handle).
    """

    def __init__(self, stream_a: TextIO, stream_b: TextIO) -> None:
        self.a = stream_a
        self.b = stream_b

    def write(self, data: str) -> int:
        """Write data to both streams.

        Parameters
        ----------
        data : str
            String data to write.

        Returns
        -------
        int
            Number of characters written to primary stream.
        """
        written = 0
        try:
            written = self.a.write(data)
        except Exception:
            pass
        try:
            self.b.write(data)
        except Exception:
            pass
        return written

    def flush(self) -> None:
        """Flush both output streams."""
        try:
            self.a.flush()
        except Exception:
            pass
        try:
            self.b.flush()
        except Exception:
            pass


# Module-level state for stream redirection (prevents garbage collection)
_STD_TEE: _Tee | None = None
_STD_TEE_FILE: TextIOWrapper | None = None
_ORIG_STDOUT: TextIO | None = None
_ORIG_STDERR: TextIO | None = None
_EARLY_CONSOLE: logging.Handler | None = None
_EARLY_BUFFER: MemoryHandler | None = None


def load_json_params(json_path: str | Path | None = None) -> dict[str, Any]:
    """Load execution parameters from a JSON configuration file.

    Parameters
    ----------
    json_path : str | Path | None, optional
        Path to JSON file.

    Returns
    -------
    dict[str, Any]
        Parameters with keys converted from kebab-case to snake_case.

    Raises
    ------
    FileNotFoundError
        If the specified JSON file does not exist.
    json.JSONDecodeError
        If the JSON file contains invalid syntax.
    """
    if json_path is None:
        raise ValueError(
            "json_path is required. Pass an explicit path via --use-json <path>."
        )
    else:
        json_path = Path(json_path)

    logger.info(f"Loading parameters from {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            params: dict[str, Any] = json.load(f)
            # Convert kebab-case to snake_case for Python compatibility
            converted_params: dict[str, Any] = {}
            for key, value in params.items():
                converted_key = key.replace("-", "_")
                converted_params[converted_key] = value
            return converted_params
    except Exception as e:
        logger.error(f"Error loading JSON parameters: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the investment tool.

    Supports loading defaults from a JSON file and overriding with CLI flags.
    If no arguments are provided, uses argparse defaults sourced from
    ``defaults.py``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing data_source, mode, frequency, etc.
    """
    parser = argparse.ArgumentParser(description="Investment Tool CLI")

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
        help="Whether to save input data locally (default: False)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default=STRATEGY_SELECTION.mode,
        choices=[e.value for e in ExecutionMode],
        help="Execution mode (default: %(default)s)",
    )

    parser.add_argument(
        "--frequency",
        type=str,
        default=STRATEGY_SELECTION.frequency,
        choices=[f.value for f in Frequency],
        help="Data frequency (default: %(default)s)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=STRATEGY_SELECTION.strategy,
        choices=[s.value for s in StrategyType],
        help="Investment strategy to use (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for output files",
    )

    parser.add_argument(
        "--use-json",
        type=str,
        help="Load parameters from a JSON file (path required)",
    )

    # New logging-related controls
    parser.add_argument(
        "--platypus-logs",
        dest="platypus_logs",
        action="store_true",
        help="Enable Platypus library INFO logs (default: disabled)",
    )
    parser.set_defaults(platypus_logs=False)

    parser.add_argument(
        "--matplotlib-log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Logging level for matplotlib.* loggers (default: WARNING)",
    )

    parser.add_argument(
        "--validation-level",
        type=str,
        choices=["strict", "warn", "skip"],
        default="warn",
        help="Validation strictness: strict (fail on errors), warn (log and continue), skip (no validation)",
    )

    parser.add_argument(
        "--min-assets",
        type=int,
        default=CONSTRUCTION.min_assets,
        help="Minimum number of active assets in the portfolio (default: %(default)s)",
    )

    parser.add_argument(
        "--max-assets",
        type=int,
        default=CONSTRUCTION.max_assets,
        help="Maximum number of active assets in the portfolio (default: %(default)s)",
    )

    return parse_arguments_with_json(parser, load_json_params)


def _ensure_unique_dir(path: Path) -> Path:
    """Ensure directory path is unique by appending numeric suffix if needed.

    Parameters
    ----------
    path : Path
        Desired directory path.

    Returns
    -------
    Path
        Unique path: original if available, otherwise with ``_NNN`` suffix.
    """
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.parent / f"{path.name}_{counter:03d}"
        if not candidate.exists():
            return candidate
        counter += 1


def _setup_early_console_logging() -> None:
    """Set up early console logging so pre-setup logs keep the preferred appearance.

    Adds a console StreamHandler and a MemoryHandler to buffer early records. The
    buffer will be flushed into the file handler once full logging is configured.
    """
    global _EARLY_CONSOLE, _EARLY_BUFFER

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Only set up early handlers if none exist to avoid duplication in interactive
    # sessions where handlers might already be configured.
    if root_logger.handlers:
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    early_console = logging.StreamHandler(stream=sys.stdout)
    early_console.setLevel(logging.DEBUG)
    early_console.setFormatter(formatter)
    early_console.set_name("investment_tool_early_console")

    early_buffer = MemoryHandler(
        capacity=10000, flushLevel=logging.CRITICAL, target=None
    )
    early_buffer.setLevel(logging.DEBUG)
    early_buffer.setFormatter(formatter)
    early_buffer.set_name("investment_tool_early_buffer")

    root_logger.addHandler(early_console)
    root_logger.addHandler(early_buffer)

    _EARLY_CONSOLE = early_console
    _EARLY_BUFFER = early_buffer


def _restore_std_streams() -> None:
    """Restore original stdout/stderr and close tee file handle.

    Resets module-level stream references to None after restoration.
    Safe to call multiple times.
    """
    global _STD_TEE, _STD_TEE_FILE, _ORIG_STDOUT, _ORIG_STDERR

    if _ORIG_STDOUT is not None:
        sys.stdout = _ORIG_STDOUT
    if _ORIG_STDERR is not None:
        sys.stderr = _ORIG_STDERR

    if _STD_TEE_FILE is not None:
        try:
            _STD_TEE_FILE.flush()
            _STD_TEE_FILE.close()
        except Exception:
            pass

    _STD_TEE = None
    _STD_TEE_FILE = None
    _ORIG_STDOUT = None
    _ORIG_STDERR = None


def _setup_logging(run_dir: Path) -> None:
    """Configure logging to console and file, with stdout/stderr tee.

    Sets up root logger with DEBUG level, creates console and file handlers,
    flushes any buffered early logs, and redirects stdout/stderr to both
    console and the log file.

    Parameters
    ----------
    run_dir : Path
        Directory where ``run.log`` will be created.
    """
    global _STD_TEE, _STD_TEE_FILE, _EARLY_BUFFER, _ORIG_STDOUT, _ORIG_STDERR

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    root_logger = logging.getLogger()
    # Match previous configuration: DEBUG level and classic hyphen-separated format
    root_logger.setLevel(logging.DEBUG)

    # Keep reference to early buffer before clearing handlers so we can flush it later
    early_buffer = _EARLY_BUFFER

    # Remove only our own handlers to avoid clobbering external configuration
    for h in list(root_logger.handlers):
        if h in {_EARLY_CONSOLE, _EARLY_BUFFER} or h.get_name() in {
            "investment_tool_console",
            "investment_tool_file",
            "investment_tool_early_console",
            "investment_tool_early_buffer",
        }:
            root_logger.removeHandler(h)

    # Previous appearance: "% (asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Capture original stdout before any redirection
    if _ORIG_STDOUT is None:
        _ORIG_STDOUT = sys.stdout
    if _ORIG_STDERR is None:
        _ORIG_STDERR = sys.stderr

    console_handler = logging.StreamHandler(stream=_ORIG_STDOUT)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    console_handler.set_name("investment_tool_console")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.set_name("investment_tool_file")

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # If we buffered early records, flush them into the file now
    if early_buffer is not None:
        try:
            early_buffer.setTarget(file_handler)
            early_buffer.flush()
        except Exception:
            pass
        finally:
            _EARLY_BUFFER = None

    # Close any existing tee before re-opening
    _restore_std_streams()

    # Tee stdout/stderr using the module-level _Tee class so prints go
    # to both the console and run.log
    _STD_TEE_FILE = open(log_path, "a", encoding="utf-8")
    _STD_TEE = _Tee(_ORIG_STDOUT, _STD_TEE_FILE)
    sys.stdout = _STD_TEE  # type: ignore[assignment]
    sys.stderr = _STD_TEE  # type: ignore[assignment]

    root_logger.info(f"Logging initialized. Writing logs to: {log_path}")


def _configure_external_loggers(
    include_platypus_logs: bool, matplotlib_level_name: str
) -> None:
    """Configure log levels for third-party libraries.

    Parameters
    ----------
    include_platypus_logs : bool
        If True, set Platypus logger to INFO; otherwise WARNING.
    matplotlib_level_name : str
        Log level name for matplotlib loggers (e.g., "WARNING", "DEBUG").
    """
    # Matplotlib family
    try:
        level = getattr(logging, matplotlib_level_name.upper(), logging.WARNING)
        logging.getLogger("matplotlib").setLevel(level)
        logging.getLogger("matplotlib.font_manager").setLevel(level)
    except Exception:
        pass

    # Platypus GA library
    try:
        p_logger = logging.getLogger("Platypus")
        p_logger.setLevel(logging.INFO if include_platypus_logs else logging.WARNING)
    except Exception:
        pass


def main() -> int:
    """Execute the investment tool with CLI arguments.

    Parses arguments, sets up logging, runs the investment strategy or
    backtesting, and handles errors gracefully.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for error.
    """
    exit_code = 0
    try:
        # Ensure early logs (e.g., from load_json_params/parse_arguments) have the
        # classic appearance and show on console, then get flushed to the run log.
        _setup_early_console_logging()

        args = parse_arguments()

        data_source = DataSource(args.data_source)
        execution_mode = ExecutionMode(args.mode)
        frequency = Frequency(args.frequency)

        base_dir = Path(__file__).resolve().parents[1]
        # Fallback default if not provided via CLI/JSON
        output_dir_arg = args.output_dir or "data/output"
        base_output_dir = base_dir / output_dir_arg

        # Create a unique per-run directory under the base output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _ensure_unique_dir(base_output_dir / f"run_{timestamp}")
        run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging to console and run log file inside the run directory
        _setup_logging(run_dir)

        # Apply user-selected verbosity for third-party libraries
        _configure_external_loggers(
            include_platypus_logs=bool(getattr(args, "platypus_logs", False)),
            matplotlib_level_name=str(getattr(args, "matplotlib_log_level", "WARNING")),
        )

        input_data_path = None
        if args.input_path:
            raw_path = Path(args.input_path)
            input_data_path = (
                raw_path if raw_path.is_absolute() else base_dir / raw_path
            )

        # Parse validation level
        validation_level_str = getattr(args, "validation_level", "warn")
        validation_level_map = {
            "strict": ValidationLevel.STRICT,
            "warn": ValidationLevel.WARN,
            "skip": ValidationLevel.SKIP,
        }
        validation_level = validation_level_map.get(
            validation_level_str.lower(), ValidationLevel.WARN
        )

        # Parse strategy type
        strategy_str = getattr(args, "strategy", "combined")
        try:
            strategy_type = StrategyType(strategy_str.lower())
        except ValueError:
            logger.warning(
                f"Unknown strategy '{strategy_str}', falling back to 'combined'"
            )
            strategy_type = StrategyType.COMBINED

        parameters = Parameters(
            data_source=data_source,
            execution_mode=execution_mode,
            frequency=frequency,
            output_directory=run_dir,
            strategy_type=strategy_type,
            input_data_path=input_data_path,
            save_input_data=args.save_input_data,
            validation_level=validation_level,
            min_assets=getattr(args, "min_assets", CONSTRUCTION.min_assets),
            max_assets=getattr(args, "max_assets", CONSTRUCTION.max_assets),
            cfg=load_config(),
        )

        # Execute the investment tool
        logger.info(
            f"Starting execution with mode: {execution_mode}, frequency: {frequency}, output: {run_dir}"
        )
        result = execute(parameters)

        if result:
            logger.info(
                f"Execution completed successfully. Results saved to {result.get('output_dir', run_dir)}"
            )
        else:
            logger.info("Execution completed without results.")

        exit_code = 0

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        exit_code = 1
    finally:
        _restore_std_streams()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
