"""Redis-backed progress publishing and subscribing.

The Celery worker publishes progress updates to a Redis pub/sub channel.
The WebSocket endpoint subscribes to the same channel and relays
messages to connected clients.

Channel name convention::

    run:{run_id}:progress

Message format (JSON)::

    {"fraction": 0.42, "message": "Optimising portfolio …"}

A final sentinel is published when the run completes or fails::

    {"fraction": 1.0, "message": "completed"}   # or "failed: <error>"
"""

from __future__ import annotations

import json
import logging

import redis as _redis

from investment_tool.config import get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel helpers
# ---------------------------------------------------------------------------


def channel_for(run_id: str) -> str:
    """Return the Redis pub/sub channel name for *run_id*."""
    return f"run:{run_id}:progress"


# ---------------------------------------------------------------------------
# Publisher (used inside Celery worker)
# ---------------------------------------------------------------------------


def make_progress_publisher(
    run_id: str,
    redis_url: str | None = None,
) -> tuple[_redis.Redis, "ProgressCallback"]:  # type: ignore[type-arg]
    """Create a Redis client and a progress callback that publishes to it.

    Returns
    -------
    (redis_client, callback)
        A tuple of the Redis connection (caller must close it) and a
        ``ProgressCallback``-compatible callable.
    """
    url = redis_url or get_config().redis.url
    client: _redis.Redis = _redis.from_url(url)  # type: ignore[type-arg]
    ch = channel_for(run_id)

    def _publish(fraction: float, message: str) -> None:
        payload = json.dumps({"fraction": fraction, "message": message})
        try:
            client.publish(ch, payload)
        except Exception:
            logger.debug("Failed to publish progress for %s", run_id, exc_info=True)

    return client, _publish


def publish_sentinel(
    run_id: str,
    *,
    status: str,
    error: str | None = None,
    redis_url: str | None = None,
) -> None:
    """Publish a final message indicating the run is done.

    Parameters
    ----------
    status : str
        ``"completed"`` or ``"failed"``.
    error : str | None
        Error description when *status* is ``"failed"``.
    """
    url = redis_url or get_config().redis.url
    client: _redis.Redis = _redis.from_url(url)  # type: ignore[type-arg]
    ch = channel_for(run_id)
    msg = status if error is None else f"{status}: {error}"
    payload = json.dumps({"fraction": 1.0, "message": msg})
    try:
        client.publish(ch, payload)
    finally:
        client.close()


# Re-export for convenience
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (  # noqa: E402
    ProgressCallback,
)

__all__ = [
    "ProgressCallback",
    "channel_for",
    "make_progress_publisher",
    "publish_sentinel",
]
