"""Omit query values from routine Uvicorn request/connection logs."""

import logging
from typing import Any


def _without_query(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("/"):
        return value.partition("?")[0]
    return value


class RequestQueryFilter(logging.Filter):
    """Retain server log fields while removing query strings from path args."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(_without_query(value) for value in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                key: _without_query(value) for key, value in record.args.items()
            }
        return True


def install_query_redaction() -> None:
    """Install once per server logger without changing its handlers or levels."""
    for name in ("uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, RequestQueryFilter) for f in logger.filters):
            logger.addFilter(RequestQueryFilter())
