"""FastAPI application factory for the Investment Tool REST API.

Usage
-----
::

    from api.main import create_app

    app = create_app()

Run with uvicorn::

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from investment_tool.exceptions import InvestmentToolError

from api.db_store import DBRunStore
from api.browser_origins import browser_origins
from api.request_logging import install_query_redaction
from api.routers import auth, campaigns, comparisons, defaults, presets, runs, ws
from api.runtime_security import validate_production_auth
from api.sanitize import sanitize
from api.store import RunStore

logger = logging.getLogger(__name__)

# Path to the built React frontend (populated by Docker build)
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


def create_app(
    *,
    store: RunStore | DBRunStore | None = None,
    init_db: bool = False,
) -> FastAPI:
    """Build and return a configured :class:`FastAPI` application.

    Parameters
    ----------
    store : RunStore | DBRunStore | None
        Optional pre-built run store (useful for testing).  If *None*
        and *init_db* is ``True``, a :class:`DBRunStore` is created.
        Otherwise falls back to an in-memory :class:`RunStore`.
    init_db : bool
        If ``True``, initialise the SQLAlchemy engine from config on
        startup.  Set to ``False`` in tests that supply their own store.
    """
    install_query_redaction()
    app = FastAPI(
        title="Investment Tool API",
        version="0.2.0",
        description="REST API for portfolio optimisation, backtesting, and comparison.",
    )
    validate_production_auth()

    # -- Browser origins (shared by CORS and WebSocket authorization) --------
    origins = browser_origins()
    app.state.browser_origins = frozenset(origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Database init -------------------------------------------------------
    if init_db:
        from api.database import init_engine

        init_engine()

    # -- Run store -----------------------------------------------------------
    if store is not None:
        _store = store
    elif init_db:
        _store = DBRunStore()
    else:
        _store = RunStore()

    runs.set_store(_store)
    comparisons.set_store(_store)
    app.state.run_store = _store

    # -- Health check --------------------------------------------------------

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        """Liveness probe for load balancers and container orchestrators."""
        return {"status": "ok"}

    # -- Routers -------------------------------------------------------------
    app.include_router(auth.router)
    app.include_router(runs.router)
    app.include_router(comparisons.router)
    app.include_router(campaigns.router)
    app.include_router(defaults.router)
    app.include_router(presets.router)
    app.include_router(ws.router)

    # -- Exception handler ---------------------------------------------------

    @app.exception_handler(RequestValidationError)
    async def _request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        # Nonstandard JSON constants/overflow can reach validation as floats.
        # Preserve FastAPI's field errors while making their inputs JSON-safe.
        return JSONResponse(
            status_code=422,
            content={"detail": sanitize(jsonable_encoder(exc.errors()))},
        )

    @app.exception_handler(InvestmentToolError)
    async def _investment_tool_error_handler(
        request: Request,
        exc: InvestmentToolError,
    ) -> JSONResponse:
        """Map domain exceptions to structured JSON error responses."""
        body: dict[str, Any] = exc.to_http_response()
        return JSONResponse(status_code=exc.http_status_code, content=body)

    # -- Static frontend (production only) -----------------------------------
    # In production the Docker image contains built React assets in
    # frontend/dist.  Serve them as a catch-all so the SPA works.
    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True))

    return app


# Convenience module-level app for ``uvicorn api.main:app``
app: FastAPI = create_app(init_db=True)
