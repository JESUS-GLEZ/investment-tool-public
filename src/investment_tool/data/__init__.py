"""Data retrieval, validation, assembly and asset universe handling.

Public convenience imports load on demand so configuration can read manifests
without initializing repositories or importing the model factory recursively.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from investment_tool.data.assembly import ModelDataFactory
    from investment_tool.data.repositories import (
        BaseDataRepository,
        DataResult,
        AlpacaRepository,
        LocalRepository,
        LocalDataSaver,
        RepositoryFactory,
        get_repository,
    )

__all__ = [
    "BaseDataRepository",
    "DataResult",
    "AlpacaRepository",
    "LocalRepository",
    "LocalDataSaver",
    "RepositoryFactory",
    "get_repository",
    "ModelDataFactory",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "assembly" if name == "ModelDataFactory" else "repositories"
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
