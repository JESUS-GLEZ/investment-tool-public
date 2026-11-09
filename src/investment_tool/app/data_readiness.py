"""Persist the existing validation report and enforce the backtest audit."""

import json
from pathlib import Path
from typing import Any

from investment_tool.exceptions import BacktestReadinessError


def require_ready(report: dict[str, Any], output_directory: Path | None = None) -> None:
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "data_validation_report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n"
        )
    readiness = report.get("readiness", {})
    if readiness.get("ready", True):
        return
    raise BacktestReadinessError(report)
