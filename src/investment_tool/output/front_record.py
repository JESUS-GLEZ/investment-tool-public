"""Compact exact candidate records, shared by audit output and front reuse."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from platypus import Solution
from investment_tool.output.policy import OutputMode
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)


def pack_front(front: list[Solution]) -> list[dict[str, Any]]:
    return [
        {
            "weights": {str(i): float(w) for i, w in enumerate(s.variables) if w != 0},
            "objectives": list(map(float, s.objectives)),
        }
        for s in front
    ]


def front_record(result: OptimizationResult, mode: OutputMode) -> dict[str, Any]:
    curve = result.reference_objectives
    if curve is None and result.markowitz_front is not None:
        curve = [list(map(float, s.objectives)) for s in result.markowitz_front]
    record: dict[str, Any] = {
        "format_version": 1,
        "output_mode": mode.value,
        "front": pack_front(result.front),
        "reference_objectives": curve,
        "diagnostics": result.search_diagnostics,
    }
    if mode == OutputMode.DIAGNOSTIC:
        record["archives"] = [pack_front(a) for a in result.all_archives]
        record["reference"] = (
            pack_front(result.markowitz_front)
            if result.markowitz_front is not None
            else None
        )
    return record


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            with gzip.GzipFile(
                fileobj=stream, mode="wb", mtime=0, filename=""
            ) as compressed:
                compressed.write(
                    json.dumps(record, allow_nan=False, separators=(",", ":")).encode()
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_record(path: Path) -> dict[str, Any]:
    # Read legacy uncompressed records too; old files are never rewritten.
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        record: dict[str, Any] = json.load(stream)
    return record
