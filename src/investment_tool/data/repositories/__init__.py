"""
Repository pattern implementation for data access.

This package provides a clean abstraction for data retrieval from various
sources using the Repository pattern, enabling:
- Consistent interface for all data sources
- Easy testing with mock repositories
- Flexible data source switching
- Clear separation of concerns

The low-level data fetchers (Alpaca client, local file loader) are
**private implementation details** of the respective repositories
(_alpaca_client.py, _local_loader.py) and should not be imported
directly.  LocalDataSaver is public because saving is an output
concern used by multiple consumers.
"""

from investment_tool.data.repositories.base_repository import (
    BaseDataRepository,
    DataResult,
)
from investment_tool.data.repositories.alpaca_repository import AlpacaRepository
from investment_tool.data.repositories.cached_repository import CachedRepository
from investment_tool.data.repositories.local_repository import LocalRepository
from investment_tool.data.repositories.local_data_saver import LocalDataSaver
from investment_tool.data.repositories.repository_factory import (
    RepositoryFactory,
    get_repository,
)

__all__ = [
    "BaseDataRepository",
    "CachedRepository",
    "DataResult",
    "AlpacaRepository",
    "LocalRepository",
    "LocalDataSaver",
    "RepositoryFactory",
    "get_repository",
]
