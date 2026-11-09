"""SQLAlchemy database engine and session management.

Provides a session factory backed by the connection URL from
:class:`~investment_tool.config.AppConfig`.  In tests, callers can
override the engine via :func:`init_engine`.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from investment_tool.config import get_config

# ---------------------------------------------------------------------------
# Module-level state (initialised lazily or explicitly via init_engine)
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def init_engine(url: str | None = None, **kwargs: object) -> Engine:
    """Create (or replace) the module-level engine and session factory.

    Parameters
    ----------
    url : str | None
        Database URL.  ``None`` reads from ``get_config().database.url``.
    **kwargs
        Extra keyword arguments forwarded to :func:`sqlalchemy.create_engine`.
    """
    global _engine, _SessionLocal  # noqa: PLW0603
    resolved_url = url or get_config().database.url
    _engine = create_engine(resolved_url, **kwargs)  # type: ignore[arg-type]
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_engine() -> Engine:
    """Return the current engine, initialising lazily if needed."""
    if _engine is None:
        init_engine()
    assert _engine is not None
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the current session factory, initialising lazily if needed."""
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session.

    Usage::

        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Tear down engine and session factory (for tests)."""
    global _engine, _SessionLocal  # noqa: PLW0603
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
