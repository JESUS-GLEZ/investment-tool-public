"""Small, optional presentation manifests for operator-trusted datasets.

Catalog listing never deserializes datasets. Inspection refreshes its manifest;
source size/mtime detect stale entries. These are UI metadata, not validation or
cryptographic identities, and never alter the underlying pickle.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def source_stamp(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _metadata(info: dict[str, Any]) -> dict[str, Any]:
    """Ignore unsupported/malformed sidecar fields rather than trust JSON types."""
    strings = {
        "title",
        "description",
        "intended_use",
        "frequency",
        "provider",
        "retrieved_at",
        "inspected_at",
        "asset_config_file",
        "asset_config_name",
        "start_date",
        "end_date",
    }
    counts = {"stock_count", "crypto_count", "size_bytes", "time_contract_version"}
    clean: dict[str, Any] = {
        key: value
        for key, value in info.items()
        if (key in strings and (value is None or isinstance(value, str)))
        or (key in counts and type(value) is int and value >= 0)
        or (key == "asset_config_available" and isinstance(value, bool))
    }
    if isinstance(info.get("features"), dict):
        clean["features"] = {
            key: value
            for key, value in info["features"].items()
            if key in {"prices", "execution_quotes", "news", "cached_sentiment"}
            and (value is None or isinstance(value, bool))
        }
    return clean


def _manifest_path(path: Path, cache_dir: Path | None) -> Path:
    if cache_dir is None:
        return path.with_name(path.name + ".dataset.json")
    identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    return cache_dir / f"{identity}.json"


def read_manifest(path: Path, cache_dir: Path | None = None) -> dict[str, Any]:
    sidecar = _manifest_path(path, cache_dir)
    try:
        if sidecar.is_symlink() or sidecar.stat().st_size > 128_000:
            return {}
        payload = json.loads(sidecar.read_text())
        if (
            payload.get("format_version") != 1
            or payload.get("source") != source_stamp(path)
            or not isinstance(payload.get("info"), dict)
        ):
            return {}
        return _metadata(payload["info"])
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def write_manifest(
    path: Path,
    info: dict[str, Any],
    stamp: dict[str, int],
    cache_dir: Path | None = None,
) -> None:
    """Best-effort atomic metadata cache, including on read-only deployments."""
    temporary: str | None = None
    try:
        if source_stamp(path) != stamp:
            return
        sidecar = _manifest_path(path, cache_dir)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        if sidecar.is_symlink():
            return
        with tempfile.NamedTemporaryFile(
            mode="w", dir=sidecar.parent, prefix=".dataset-", delete=False
        ) as stream:
            temporary = stream.name
            json.dump(
                {
                    "format_version": 1,
                    "source": stamp,
                    "info": info,
                },
                stream,
            )
        os.replace(temporary, sidecar)
    except OSError:
        pass
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
