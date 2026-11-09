"""
Repository factory for creating data repositories.

This module provides a factory for creating the appropriate repository
based on the data source configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Type

from investment_tool.constants import DataSource, Frequency
from investment_tool.data.repositories.base_repository import BaseDataRepository
from investment_tool.data.repositories.alpaca_repository import AlpacaRepository
from investment_tool.data.repositories.local_repository import LocalRepository
from investment_tool.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """Factory for creating data repositories.

    Provides a centralized way to create repository instances based on
    the data source type, implementing the Factory pattern.

    This class maintains a registry of available repository types and
    creates instances with appropriate configuration.
    """

    # Registry of data sources to repository classes
    _registry: dict[DataSource, Type[BaseDataRepository]] = {
        DataSource.ALPACA: AlpacaRepository,
        DataSource.LOCAL: LocalRepository,
    }

    @classmethod
    def register(
        cls,
        data_source: DataSource,
        repository_class: Type[BaseDataRepository],
    ) -> None:
        """Register a new repository type.

        Parameters
        ----------
        data_source : DataSource
            The data source this repository handles.
        repository_class : Type[BaseDataRepository]
            The repository class to register.
        """
        cls._registry[data_source] = repository_class
        logger.info(
            f"Registered repository {repository_class.__name__} "
            f"for data source {data_source}"
        )

    @classmethod
    def create(
        cls,
        data_source: DataSource,
        frequency: Frequency,
        file_path: Path | None = None,
        **kwargs: Any,
    ) -> BaseDataRepository:
        """Create a repository instance.

        Parameters
        ----------
        data_source : DataSource
            The data source type (ALPACA, LOCAL).
        frequency : Frequency
            Data frequency (DAILY, WEEKLY, MONTHLY).
        file_path : Path | None, optional
            Path to local file (required for LOCAL source).
        **kwargs : Any
            Additional arguments passed to the repository constructor.

        Returns
        -------
        BaseDataRepository
            The created repository instance.

        Raises
        ------
        ConfigurationError
            If the data source is not supported or required config is missing.

        Examples
        --------
        >>> repo = RepositoryFactory.create(DataSource.ALPACA, Frequency.DAILY)
        >>> repo = RepositoryFactory.create(
        ...     DataSource.LOCAL,
        ...     Frequency.MONTHLY,
        ...     file_path=Path("data/input/data.pkl")
        ... )
        """
        repository_class = cls._registry.get(data_source)

        if repository_class is None:
            raise ConfigurationError(
                f"Unsupported data source: {data_source}",
                details={
                    "data_source": str(data_source),
                    "supported": [str(ds) for ds in cls._registry.keys()],
                },
            )

        # Handle LOCAL source requiring file_path
        if data_source == DataSource.LOCAL:
            if file_path is None:
                raise ConfigurationError(
                    "file_path is required for LOCAL data source",
                    details={"data_source": str(data_source)},
                )
            logger.info(f"Creating LocalRepository with file: {file_path}")
            return repository_class(frequency=frequency, file_path=file_path, **kwargs)

        # Handle other sources
        logger.info(f"Creating {repository_class.__name__} for {data_source}")
        return repository_class(frequency=frequency, **kwargs)

    @classmethod
    def get_supported_sources(cls) -> list[DataSource]:
        """Get list of supported data sources.

        Returns
        -------
        list[DataSource]
            List of registered data sources.
        """
        return list(cls._registry.keys())


def get_repository(
    data_source: DataSource,
    frequency: Frequency,
    file_path: Path | None = None,
    **kwargs: Any,
) -> BaseDataRepository:
    """Convenience function to create a repository.

    This is a shorthand for RepositoryFactory.create().

    Parameters
    ----------
    data_source : DataSource
        The data source type.
    frequency : Frequency
        Data frequency.
    file_path : Path | None, optional
        Path to local file (for LOCAL source).
    **kwargs
        Additional repository arguments.

    Returns
    -------
    BaseDataRepository
        The created repository instance.

    Examples
    --------
    >>> from investment_tool.data.repositories import get_repository
    >>> repo = get_repository(DataSource.ALPACA, Frequency.MONTHLY)
    """
    return RepositoryFactory.create(
        data_source=data_source,
        frequency=frequency,
        file_path=file_path,
        **kwargs,
    )
