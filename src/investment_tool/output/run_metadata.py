"""Small run manifest. Never serializes credentials or duplicates market data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from investment_tool.config import get_config
from investment_tool.data.universes.loader import ASSET_CONFIG_DIR, load_asset_universe


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def write_run_metadata(parameters: Any) -> None:
    cfg = parameters.cfg or get_config()
    universe = load_asset_universe(cfg.asset_config_file)
    universe_path = ASSET_CONFIG_DIR / universe.file
    if not universe_path.exists():
        universe_path = ASSET_CONFIG_DIR / "archive" / universe.file
    source_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    packages = {}
    for name in ("numpy", "pandas", "scipy", "platypus-opt"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    inputs = parameters.input_data_path
    payload = {
        "format_version": 1,
        "parameters": {
            f.name: _json_value(getattr(parameters, f.name))
            for f in fields(parameters)
            if f.name != "cfg"
        },
        "model_configuration": asdict(cfg.model),
        "asset_universe": _file_identity(universe_path),
        "input_dataset": _file_identity(inputs) if inputs is not None else None,
        "source_sha256": digest.hexdigest(),
        "package_versions": packages,
    }
    directory = parameters.output_directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def record_input_snapshot(directory: Path, snapshot_path: Path | None) -> None:
    """Identify the shared fetched snapshot without duplicating it."""
    if not isinstance(snapshot_path, Path):
        return
    path = directory / "run_manifest.json"
    payload = json.loads(path.read_text())
    payload["input_dataset"] = _file_identity(snapshot_path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
