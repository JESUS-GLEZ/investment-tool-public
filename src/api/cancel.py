"""Cooperative cancellation via Redis flags.

When a user requests cancellation the API sets a Redis key
``run:{run_id}:cancel``.  The worker's progress callback checks this key
on every progress tick and raises :class:`RunCancelledError` if it is set,
giving the coordinator a chance to flush partial results before exiting.

Key TTL is set to 24 hours so stale flags don't linger forever.
"""

from __future__ import annotations

import logging

import redis as _redis

from investment_tool.config import get_config

logger = logging.getLogger(__name__)

_CANCEL_TTL = 86400  # 24 hours


class RunCancelledError(BaseException):
    """Raised inside the worker when a user-requested cancellation is detected."""


def _key(run_id: str) -> str:
    return f"run:{run_id}:cancel"


def request_cancel(run_id: str) -> None:
    """Set the cancellation flag for *run_id* in Redis."""
    cfg = get_config()
    client = _redis.from_url(cfg.redis.url, decode_responses=True)
    try:
        client.setex(_key(run_id), _CANCEL_TTL, "1")
        logger.info("Cancellation requested for run %s", run_id)
    finally:
        client.close()


def is_cancel_requested(run_id: str, client: _redis.Redis) -> bool:  # type: ignore[type-arg]
    """Return True if cancellation has been requested for *run_id*."""
    try:
        return bool(client.exists(_key(run_id)))
    except Exception:
        return False


def clear_cancel(run_id: str, client: _redis.Redis) -> None:  # type: ignore[type-arg]
    """Remove the cancellation flag (called after the worker has handled it)."""
    try:
        client.delete(_key(run_id))
    except Exception:
        pass
