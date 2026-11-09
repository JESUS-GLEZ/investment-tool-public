"""Exercise routine server formatters with owned synthetic query values."""

import io
import logging
from pathlib import Path
import re
from unittest.mock import patch

import pytest
from uvicorn.logging import AccessFormatter, DefaultFormatter
from uvicorn.protocols.utils import get_path_with_query_string

from api.main import create_app
from api.request_logging import RequestQueryFilter
from api.store import RunStore


def _emit(name, formatter, message, *args):
    create_app(store=RunStore())
    logger = logging.getLogger(name)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(formatter)
    level = logger.level
    logger.setLevel(logging.INFO)
    try:
        with (
            patch.object(logger, "handlers", [handler]),
            patch.object(logger, "propagate", False),
        ):
            logger.info(message, *args)
    finally:
        logger.setLevel(level)
    return output.getvalue()


@pytest.mark.parametrize("outcome", ["[accepted]", "403", "400"])
def test_websocket_connection_logs_keep_status_without_query(outcome):
    scope = {
        "path": "/ws/progress/owned-run",
        "root_path": "",
        "query_string": b"token=owned-synthetic-token&secret=owned-other-value",
    }
    path = get_path_with_query_string(scope)
    line = _emit(
        "uvicorn.error",
        DefaultFormatter(fmt="%(message)s", use_colors=False),
        '%s - "WebSocket %s" ' + outcome,
        "127.0.0.1:1234",
        path,
    )
    assert line == f'127.0.0.1:1234 - "WebSocket /ws/progress/owned-run" {outcome}\n'
    assert "owned-synthetic-token" not in line
    assert "owned-other-value" not in line


def test_real_access_formatter_retains_http_status_and_method():
    line = _emit(
        "uvicorn.access",
        AccessFormatter(fmt="%(request_line)s %(status_code)s", use_colors=False),
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:1234",
        "GET",
        "/data/info?file=owned-private-name.pkl",
        "1.1",
        404,
    )
    assert "GET /data/info HTTP/1.1" in line
    assert "404" in line
    assert "owned-private-name.pkl" not in line


def test_mapping_log_arguments_keep_non_path_fields():
    line = _emit(
        "uvicorn.error",
        DefaultFormatter(fmt="%(message)s", use_colors=False),
        "%(outcome)s %(target)s",
        {"outcome": "accepted", "target": "/ws/progress/run?token=owned-test-token"},
    )
    assert line == "accepted /ws/progress/run\n"


@pytest.mark.parametrize("path", ["/healthz", "/runs/owned%3Frun", "/ws/progress/run"])
def test_query_free_and_encoded_question_mark_paths_are_preserved(path):
    line = _emit(
        "uvicorn.error",
        DefaultFormatter(fmt="%(message)s", use_colors=False),
        "Path %s",
        path,
    )
    assert line == f"Path {path}\n"


def test_factory_does_not_accumulate_filters_or_change_logger_configuration():
    names = ["uvicorn.error", "uvicorn.access"]
    before = [
        (logging.getLogger(n).level, list(logging.getLogger(n).handlers)) for n in names
    ]
    create_app(store=RunStore())
    create_app(store=RunStore())
    for name, (level, handlers) in zip(names, before, strict=True):
        logger = logging.getLogger(name)
        assert logger.level == level
        assert logger.handlers == handlers
        assert sum(isinstance(f, RequestQueryFilter) for f in logger.filters) == 1


def test_nginx_progress_logs_exclude_query_and_header_fields():
    config = (Path(__file__).resolve().parents[2] / "nginx/nginx.conf").read_text()
    format_body = re.search(r"log_format progress_safe (.*?);", config, re.S).group(1)
    assert "$uri " in format_body
    assert "$request_method" in format_body and "$status" in format_body
    assert not any(
        variable in format_body
        for variable in [
            "$request'",
            '$request"',
            "$request_uri",
            "$args",
            "$query_string",
            "$http_",
        ]
    )
    location = re.search(r"location /ws/ \{(.*?)\n    \}", config, re.S).group(1)
    assert "access_log /var/log/nginx/access.log progress_safe;" in location
    assert "error_log /dev/null;" in location
    assert "proxy_pass         $api_upstream;" in location
    assert config.count("error_log /dev/null;") == 1
