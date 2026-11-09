"""Tests for API endpoints using FastAPI TestClient.

All coordinator functions are mocked to avoid real computation.
Auth is bypassed by overriding the ``get_current_user`` dependency.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.auth import get_current_user
from api.main import create_app
from api.models import UserRow
from api.schemas import RunStatus, RunType
from api.store import RunStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_USER = UserRow(id=1, email="test@example.com", password_hash="hashed")


def _override_current_user() -> UserRow:
    return _FAKE_USER


@pytest.fixture()
def store() -> RunStore:
    return RunStore()


@pytest.fixture()
def client(store: RunStore) -> TestClient:
    app = create_app(store=store)
    app.dependency_overrides[get_current_user] = _override_current_user
    return TestClient(app)


# A fake result dict that looks like a sanitized coordinator output.
FAKE_RESULT: dict[str, Any] = {
    "portfolio_value": 1234.56,
    "weights": {"AAPL": 0.5, "MSFT": 0.5},
}


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


# ---------------------------------------------------------------------------
# POST /runs — single execution
# ---------------------------------------------------------------------------


class TestCreateRun:
    @pytest.mark.parametrize(
        ("endpoint", "payload"),
        [
            ("/runs", {}),
            ("/comparisons/strategy", {}),
            ("/comparisons/solver", {}),
            ("/comparisons/prediction", {}),
        ],
    )
    def test_local_requests_require_input_file(
        self,
        client: TestClient,
        endpoint: str,
        payload: dict[str, Any],
    ) -> None:
        resp = client.post(endpoint, json={"data_source": "local", **payload})

        assert resp.status_code == 422
        assert "input_file is required" in str(resp.json()["detail"])

    @patch(
        "api.routers.runs.execute",
        return_value=FAKE_RESULT,
        create=True,
    )
    def test_create_run_success(
        self,
        mock_execute: Any,
        client: TestClient,
    ) -> None:
        """POST /runs with defaults should return 202 and a queued/running/completed run."""
        with patch(
            "investment_tool.app.execution_coordinator.execute",
            return_value=FAKE_RESULT,
        ):
            resp = client.post("/runs", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert data["run_type"] == RunType.EXECUTION
        assert data["status"] in (
            RunStatus.QUEUED,
            RunStatus.RUNNING,
            RunStatus.COMPLETED,
            RunStatus.FAILED,
        )
        assert "id" in data

    def test_create_run_with_explicit_params(self, client: TestClient) -> None:
        """POST /runs with explicit params should not fail on validation."""
        with patch(
            "investment_tool.app.execution_coordinator.execute",
            return_value=FAKE_RESULT,
        ):
            resp = client.post(
                "/runs",
                json={
                    "data_source": "alpaca",
                    "execution_mode": "backtesting",
                    "frequency": "monthly",
                    "strategy_type": "combined",
                    "min_assets": 3,
                    "max_assets": 8,
                },
            )
        assert resp.status_code == 202

    def test_create_run_coordinator_failure_marks_failed(
        self,
        client: TestClient,
        store: RunStore,
    ) -> None:
        """If the coordinator raises, the run should be marked FAILED."""
        with patch(
            "investment_tool.app.execution_coordinator.execute",
            side_effect=RuntimeError("kaboom"),
        ):
            resp = client.post("/runs", json={})
        assert resp.status_code == 202
        data = resp.json()
        failed_run = _wait_for_status(store, data["id"], RunStatus.FAILED)
        assert failed_run.error_message is not None
        assert "kaboom" in failed_run.error_message

    def test_create_run_returns_before_in_memory_execution_finishes(
        self,
        client: TestClient,
        store: RunStore,
    ) -> None:
        release_execute = threading.Event()

        def _blocking_execute(_: Any) -> dict[str, Any]:
            release_execute.wait(timeout=1.0)
            return FAKE_RESULT

        started_at = time.perf_counter()
        with patch(
            "investment_tool.app.execution_coordinator.execute",
            side_effect=_blocking_execute,
        ):
            resp = client.post("/runs", json={})
        elapsed = time.perf_counter() - started_at

        assert resp.status_code == 202
        assert elapsed < 0.3

        run_id = resp.json()["id"]
        running_run = _wait_for_status(
            store, run_id, RunStatus.RUNNING, RunStatus.COMPLETED
        )
        assert running_run.status in (RunStatus.RUNNING, RunStatus.COMPLETED)

        release_execute.set()
        completed_run = _wait_for_status(store, run_id, RunStatus.COMPLETED)
        assert completed_run.result == FAKE_RESULT

    def test_create_run_preserves_local_request_payload(
        self,
        client: TestClient,
        store: RunStore,
    ) -> None:
        payload = {
            "data_source": "local",
            "input_file": "synthetic_monthly_prices.pkl",
            "historical_start_date": "2022-12-31",
            "historical_end_date": "2026-05-01",
        }

        with patch("api.adapters.build_exec_params", return_value=object()):
            with patch(
                "investment_tool.app.execution_coordinator.execute",
                return_value=FAKE_RESULT,
            ):
                resp = client.post("/runs", json=payload)

        assert resp.status_code == 202
        detail = store.get(resp.json()["id"])
        assert detail is not None
        assert detail.request_payload["input_file"] == payload["input_file"]
        assert (
            detail.request_payload["historical_start_date"]
            == payload["historical_start_date"]
        )
        assert (
            detail.request_payload["historical_end_date"]
            == payload["historical_end_date"]
        )

    def test_create_run_invalid_enum_returns_422(self, client: TestClient) -> None:
        """Unknown enum value should trigger Pydantic validation error (422)."""
        resp = client.post("/runs", json={"data_source": "nonexistent"})
        assert resp.status_code == 422

    def test_create_run_unauthenticated_returns_401(self, store: RunStore) -> None:
        """Without auth, POST /runs should return 401."""
        app = create_app(store=store)
        # Do NOT override get_current_user
        unauthenticated_client = TestClient(app)
        resp = unauthenticated_client.post("/runs", json={})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_list_empty(self, client: TestClient) -> None:
        resp = client.get("/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_after_create(self, client: TestClient) -> None:
        with patch(
            "investment_tool.app.execution_coordinator.execute",
            return_value=FAKE_RESULT,
        ):
            client.post("/runs", json={})
            client.post("/runs", json={})
        resp = client.get("/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        # Summary should NOT include result payload
        for item in data:
            assert "result" not in item


# ---------------------------------------------------------------------------
# GET /runs/{id}
# ---------------------------------------------------------------------------


class TestGetRun:
    def test_get_existing(self, client: TestClient) -> None:
        with patch(
            "investment_tool.app.execution_coordinator.execute",
            return_value=FAKE_RESULT,
        ):
            create_resp = client.post("/runs", json={})
        run_id = create_resp.json()["id"]
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == run_id
        # Detail should include result
        assert "result" in data

    def test_get_nonexistent_returns_404(self, client: TestClient) -> None:
        resp = client.get("/runs/does_not_exist")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /comparisons/*
# ---------------------------------------------------------------------------


class TestComparisons:
    def test_strategy_comparison(self, client: TestClient) -> None:
        with patch(
            "investment_tool.app.strategy_comparison_coordinator.compare_strategies",
            return_value=FAKE_RESULT,
        ):
            resp = client.post("/comparisons/strategy", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert data["run_type"] == RunType.STRATEGY_COMPARISON

    def test_solver_comparison(self, client: TestClient) -> None:
        with patch(
            "investment_tool.app.solver_comparison_coordinator.execute_solver_comparison",
            return_value=FAKE_RESULT,
        ):
            resp = client.post("/comparisons/solver", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert data["run_type"] == RunType.SOLVER_COMPARISON

    def test_prediction_comparison(self, client: TestClient) -> None:
        with patch(
            "investment_tool.app.prediction_comparison_coordinator.execute_prediction_comparison",
            return_value=FAKE_RESULT,
        ):
            resp = client.post("/comparisons/prediction", json={})
        assert resp.status_code == 202
        data = resp.json()
        assert data["run_type"] == RunType.PREDICTION_COMPARISON

    def test_strategy_comparison_failure(
        self, client: TestClient, store: RunStore
    ) -> None:
        with patch(
            "investment_tool.app.strategy_comparison_coordinator.compare_strategies",
            side_effect=RuntimeError("fail"),
        ):
            resp = client.post("/comparisons/strategy", json={})
        assert resp.status_code == 202
        failed_run = _wait_for_status(store, resp.json()["id"], RunStatus.FAILED)
        assert failed_run.error_message is not None
        assert "fail" in failed_run.error_message

    def test_strategy_comparison_validation_error(self, client: TestClient) -> None:
        """Empty strategy_types list should trigger Pydantic 422."""
        resp = client.post(
            "/comparisons/strategy",
            json={"strategy_types": []},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_get_defaults(self, client: TestClient) -> None:
        resp = client.get("/defaults")
        assert resp.status_code == 200
        data = resp.json()
        # Spot-check some keys
        assert "portfolio" in data
        assert "optimizer" in data
        assert "construction" in data
        assert data["construction"]["min_assets"] == 2
        assert data["optimizer"]["pop_size"] == 300
        from dataclasses import asdict
        from investment_tool.defaults import STRATEGY_OPTIMIZER

        assert data["strategy_optimizer"] == asdict(STRATEGY_OPTIMIZER)

    def test_defaults_all_groups_present(self, client: TestClient) -> None:
        resp = client.get("/defaults")
        data = resp.json()
        expected_keys = {
            "portfolio",
            "strategy_selection",
            "construction",
            "optimizer",
            "strategy_optimizer",
            "genetic",
            "gentle_variator",
            "decomposition",
            "algorithm",
            "markowitz",
            "prediction",
            "momentum",
            "validation",
            "covariance",
            "infra",
        }
        assert set(data.keys()) == expected_keys


# ---------------------------------------------------------------------------
# GET /symbols
# ---------------------------------------------------------------------------


class TestSymbols:
    def test_get_symbols(self, client: TestClient) -> None:
        resp = client.get("/symbols")
        assert resp.status_code == 200
        data = resp.json()
        assert "stocks" in data
        assert "crypto" in data
        assert isinstance(data["stocks"], list)
        assert "AAPL" in data["stocks"]
        assert "BTC/USD" in data["crypto"]

    def test_selected_manifest_changes_symbols(
        self, client: TestClient, tmp_path, monkeypatch
    ) -> None:
        import json

        from investment_tool.data.universes import loader

        monkeypatch.setattr(loader, "ASSET_CONFIG_DIR", tmp_path)
        active = {
            "name": "Active",
            "groups": {"stocks": ["AAPL"]},
            "crypto_symbols": [],
        }
        alternate = {
            "name": "Alternate",
            "groups": {"stocks": ["SPY", "GLD"]},
            "crypto_symbols": ["BTC/USD"],
        }
        (tmp_path / "active.json").write_text(json.dumps(active))
        (tmp_path / "alternate.json").write_text(json.dumps(alternate))
        configs = client.get("/asset-configs")
        assert configs.status_code == 200
        assert {item["file"] for item in configs.json()} == {
            "active.json",
            "alternate.json",
        }
        response = client.get("/symbols", params={"file": "alternate.json"})
        assert response.status_code == 200
        assert set(response.json()["stocks"]) == {"SPY", "GLD"}
        assert response.json()["crypto"] == ["BTC/USD"]

    def test_archived_manifest_resolves_without_listing(
        self, client: TestClient, tmp_path, monkeypatch
    ) -> None:
        import json

        from investment_tool.data.universes import loader

        monkeypatch.setattr(loader, "ASSET_CONFIG_DIR", tmp_path)
        archive = tmp_path / "archive"
        archive.mkdir()
        (archive / "archived.json").write_text(
            json.dumps(
                {
                    "name": "Archived",
                    "groups": {"stocks": ["GLD"]},
                    "crypto_symbols": ["BTC/USD"],
                }
            )
        )
        assert client.get("/asset-configs").json() == []
        response = client.get("/symbols", params={"file": "archived.json"})
        assert response.status_code == 200
        assert response.json() == {"stocks": ["GLD"], "crypto": ["BTC/USD"]}

    def test_unsafe_manifest_is_rejected(self, client: TestClient) -> None:
        response = client.get("/symbols", params={"file": "../asset_universe.json"})
        assert response.status_code == 400
