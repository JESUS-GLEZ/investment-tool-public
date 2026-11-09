"""Tests for the exception-handler middleware in the API.

Verifies that :class:`InvestmentToolError` subclasses are caught by the
FastAPI exception handler and returned as structured JSON with the
correct HTTP status code.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.auth import get_current_user
from api.main import create_app
from api.models import UserRow
from api.schemas import RunStatus
from api.store import RunStore
from investment_tool.exceptions import (
    ConfigurationError,
    DataRetrievalError,
    DataValidationError,
    InsufficientDataError,
    InvestmentToolError,
    OptimizationError,
)

_FAKE_USER = UserRow(id=1, email="test@example.com", password_hash="hashed")


def _override_current_user() -> UserRow:
    return _FAKE_USER


@pytest.fixture()
def store() -> RunStore:
    return RunStore()


def _wait_for_status(
    store: RunStore,
    run_id: str,
    *statuses: RunStatus,
    timeout: float = 1.0,
) -> Any:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        run = store.get(run_id)
        if run is not None and run.status in statuses:
            return run
        time.sleep(0.01)
    raise AssertionError(f"Run {run_id} did not reach one of {statuses!r} in time")


@pytest.fixture()
def client(store: RunStore) -> TestClient:
    app = create_app(store=store)
    app.dependency_overrides[get_current_user] = _override_current_user
    return TestClient(app)


class TestExceptionHandler:
    """The exception handler should catch InvestmentToolError and return JSON."""

    def _post_run_with_exc(
        self,
        client: TestClient,
        exc: InvestmentToolError,
    ) -> Any:
        """POST /runs where the coordinator raises *exc*."""
        with patch(
            "investment_tool.app.execution_coordinator.execute",
            side_effect=exc,
        ):
            return client.post("/runs", json={})

    def test_base_error_marks_run_failed(
        self,
        client: TestClient,
        store: RunStore,
    ) -> None:
        """InvestmentToolError itself should be caught inside the endpoint
        and the run marked as FAILED (returned as 202)."""
        resp = self._post_run_with_exc(client, InvestmentToolError("oops"))
        assert resp.status_code == 202
        failed_run = _wait_for_status(store, resp.json()["id"], RunStatus.FAILED)
        assert failed_run.error_message is not None
        assert "oops" in failed_run.error_message

    def test_domain_exception_via_direct_raise(self, client: TestClient) -> None:
        """Exceptions that escape the try/except (e.g. in adapters) are
        caught by the FastAPI exception handler."""
        with patch(
            "api.adapters.build_exec_params",
            side_effect=ConfigurationError("bad config"),
        ):
            resp = client.post("/runs", json={})
        assert resp.status_code == 500  # ConfigurationError defaults to 500
        data = resp.json()
        assert data["error"] == "ConfigurationError"
        assert "bad config" in data["message"]

    def test_data_validation_error_422(self, client: TestClient) -> None:
        with patch(
            "api.adapters.build_exec_params",
            side_effect=DataValidationError("bad data"),
        ):
            resp = client.post("/runs", json={})
        assert resp.status_code == 422
        assert resp.json()["error"] == "DataValidationError"

    def test_data_retrieval_error_502(self, client: TestClient) -> None:
        with patch(
            "api.adapters.build_exec_params",
            side_effect=DataRetrievalError("alpaca", "timeout"),
        ):
            resp = client.post("/runs", json={})
        assert resp.status_code == 502

    def test_insufficient_data_error_422(self, client: TestClient) -> None:
        with patch(
            "api.adapters.build_exec_params",
            side_effect=InsufficientDataError(required=30, actual=5),
        ):
            resp = client.post("/runs", json={})
        assert resp.status_code == 422
        data = resp.json()
        assert data["details"]["required"] == 30
        assert data["details"]["actual"] == 5

    def test_optimization_error_500(self, client: TestClient) -> None:
        with patch(
            "api.adapters.build_exec_params",
            side_effect=OptimizationError("no convergence"),
        ):
            resp = client.post("/runs", json={})
        assert resp.status_code == 500
        assert resp.json()["error"] == "OptimizationError"
