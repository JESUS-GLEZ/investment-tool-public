"""WebSocket router — real-time progress streaming.

Endpoint
--------
WS /ws/progress/{run_id}   Stream progress updates for a run.

Authentication is done via a ``token`` query parameter (WebSocket
connections cannot send custom headers in browsers).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from sqlalchemy.orm import Session, sessionmaker

from api.auth import get_user_from_token
from api.database import get_session_factory
from api.db_store import DBRunStore
from api.progress import channel_for
from api.store import RunStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


def _authorize_progress(
    websocket: WebSocket,
    run_id: str,
    factory: Annotated[sessionmaker[Session], Depends(get_session_factory)],
    token: str = Query(default=""),
) -> None:
    """Check the user and run before accepting or subscribing to Redis."""
    denied = WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in websocket.app.state.browser_origins:
        raise denied
    if not token:
        raise denied
    # Release the short-lived DB session before entering the streaming loop.
    with factory() as db:
        try:
            user = get_user_from_token(token, db)
        except HTTPException as exc:
            raise denied from exc
        store: RunStore | DBRunStore = websocket.app.state.run_store
        run = (
            store.get_for_user(run_id, user.id, db=db)
            if isinstance(store, DBRunStore)
            else store.get(run_id)
        )
        if run is None:
            raise denied


@router.websocket("/ws/progress/{run_id}")
async def ws_progress(
    websocket: WebSocket,
    run_id: str,
    _authorized: Annotated[None, Depends(_authorize_progress)],
) -> None:
    """Stream progress updates for *run_id* via WebSocket.

    The client connects with a JWT access token in the query string::

        ws://host/ws/progress/<run_id>?token=<jwt>

    Messages are JSON objects with ``fraction`` (0-1) and ``message``.
    The connection is closed after a ``completed`` or ``failed`` sentinel.

    Missing/invalid tokens, deleted users and unavailable runs are rejected
    before acceptance. Database-backed runs require the authenticated owner;
    the in-memory development store retains its shared run visibility.
    """
    await websocket.accept()

    # -- Subscribe to Redis pub/sub channel ----------------------------------
    try:
        import redis.asyncio as aioredis

        from investment_tool.config import get_config

        redis_url = get_config().redis.url
        redis_client = aioredis.from_url(redis_url)  # type: ignore[attr-defined]
        pubsub = redis_client.pubsub()
        ch = channel_for(run_id)
        await pubsub.subscribe(ch)

        try:
            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if msg is not None and msg["type"] == "message":
                    data = msg["data"]
                    # Redis returns bytes; decode to str
                    if isinstance(data, bytes):
                        data = data.decode()
                    await websocket.send_text(data)

                    # Close after sentinel
                    try:
                        parsed = json.loads(data)
                        frac = parsed.get("fraction", -1)
                        message = parsed.get("message", "")
                        if frac >= 1.0 and (
                            message.startswith("completed")
                            or message.startswith("failed")
                            or message.startswith("cancelled")
                        ):
                            await websocket.close()
                            break
                    except (json.JSONDecodeError, AttributeError):
                        pass
                else:
                    # No message yet — yield control briefly
                    await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            logger.debug("Client disconnected from progress/%s", run_id)
        finally:
            await pubsub.unsubscribe(ch)
            await pubsub.close()
            await redis_client.close()

    except Exception:
        logger.exception("WebSocket error for run %s", run_id)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
