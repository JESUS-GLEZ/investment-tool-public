"""Tests for :mod:`api.routers.ws` — WebSocket progress endpoint.

Uses FastAPI's TestClient WebSocket support. Redis pub/sub and
authentication are mocked.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from starlette.websockets import WebSocketDisconnect

from api.auth import create_access_token, create_refresh_token
from api.database import get_db, get_session_factory
from api.db_store import DBRunStore
from api.main import create_app
from api.models import Base, UserRow
from api.schemas import RunType
from api.store import RunStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app_client():
    """Two synthetic users, one owned run, and a one-connection SQLite pool."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        db.add_all(
            [
                UserRow(id=1, email="owner@example.test", password_hash="unused"),
                UserRow(id=2, email="other@example.test", password_hash="unused"),
            ]
        )
        db.commit()
        run = DBRunStore().create(RunType.EXECUTION, {}, db=db, user_id=1)
    app = create_app(store=DBRunStore())
    app.state.owned_run_id = run.id
    app.state.test_engine = engine
    app.dependency_overrides[get_session_factory] = lambda: factory

    def override_db():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    with patch(
        "api.auth._jwt_config",
        return_value=("owned-synthetic-secret-" * 4, "HS256", 30, 60),
    ):
        yield TestClient(app)
    engine.dispose()


def _owned_url(client: TestClient, token: str = "valid.test.token") -> str:
    return f"/ws/progress/{client.app.state.owned_run_id}?token={token}"


def _valid_token_payload() -> dict[str, Any]:
    return {"sub": "1", "type": "access", "exp": 9999999999}


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------


class TestWSAuth:
    """WebSocket authentication via query-string token."""

    def test_reject_missing_token(self, app_client: TestClient) -> None:
        """Connection without token should be rejected."""
        with pytest.raises(Exception):
            with app_client.websocket_connect("/ws/progress/run123"):
                pass  # pragma: no cover

    def test_reject_empty_token(self, app_client: TestClient) -> None:
        """Connection with empty token should be rejected."""
        with pytest.raises(Exception):
            with app_client.websocket_connect("/ws/progress/run123?token="):
                pass  # pragma: no cover

    def test_reject_invalid_token(self, app_client: TestClient) -> None:
        """Connection with garbage token should be rejected."""
        with pytest.raises(Exception):
            with app_client.websocket_connect(
                "/ws/progress/run123?token=invalid.jwt.here"
            ):
                pass  # pragma: no cover

    def test_reject_refresh_token(self, app_client: TestClient) -> None:
        """Refresh tokens should not be accepted for WS connections."""
        with patch("api.auth.decode_token") as mock_decode:
            mock_decode.return_value = {"sub": "1", "type": "refresh"}
            with pytest.raises(Exception):
                with app_client.websocket_connect(
                    "/ws/progress/run123?token=some.refresh.token"
                ):
                    pass  # pragma: no cover


# ---------------------------------------------------------------------------
# Message relay tests
# ---------------------------------------------------------------------------


class _FakePubSub:
    """Async fake for redis.asyncio pubsub."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = list(messages)
        self._idx = 0

    async def subscribe(self, channel: str) -> None:
        pass

    async def get_message(
        self, ignore_subscribe_messages: bool = True, timeout: float = 1.0
    ) -> dict[str, Any] | None:
        if self._idx < len(self._messages):
            msg = self._messages[self._idx]
            self._idx += 1
            return msg
        return None

    async def unsubscribe(self, channel: str) -> None:
        pass

    async def close(self) -> None:
        pass


class _FakeRedisAsync:
    """Async fake for redis.asyncio client."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._pubsub = _FakePubSub(messages)

    def pubsub(self) -> _FakePubSub:
        return self._pubsub

    async def close(self) -> None:
        pass


class TestWSMessageRelay:
    """Test that messages from Redis pub/sub are relayed to the WS client."""

    def _make_redis_messages(
        self, payloads: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build fake Redis pub/sub message dicts."""
        return [{"type": "message", "data": json.dumps(p).encode()} for p in payloads]

    def test_relay_progress_and_close_on_sentinel(self, app_client: TestClient) -> None:
        """Messages should be relayed; connection closes after sentinel."""
        progress_msg = {"fraction": 0.5, "message": "Optimising..."}
        sentinel_msg = {"fraction": 1.0, "message": "completed"}
        messages = self._make_redis_messages([progress_msg, sentinel_msg])

        fake_redis = _FakeRedisAsync(messages)

        with (
            patch("api.auth.decode_token") as mock_decode,
            patch("redis.asyncio.from_url", return_value=fake_redis),
        ):
            mock_decode.return_value = _valid_token_payload()

            with app_client.websocket_connect(_owned_url(app_client)) as ws:
                # Should receive the progress message
                data1 = ws.receive_text()
                parsed1 = json.loads(data1)
                assert parsed1["fraction"] == 0.5
                assert parsed1["message"] == "Optimising..."

                # Should receive the sentinel
                data2 = ws.receive_text()
                parsed2 = json.loads(data2)
                assert parsed2["fraction"] == 1.0
                assert parsed2["message"] == "completed"

    def test_relay_failure_sentinel(self, app_client: TestClient) -> None:
        """A failure sentinel should also close the connection."""
        sentinel_msg = {"fraction": 1.0, "message": "failed: timeout"}
        messages = self._make_redis_messages([sentinel_msg])
        fake_redis = _FakeRedisAsync(messages)

        with (
            patch("api.auth.decode_token") as mock_decode,
            patch("redis.asyncio.from_url", return_value=fake_redis),
        ):
            mock_decode.return_value = _valid_token_payload()

            with app_client.websocket_connect(_owned_url(app_client)) as ws:
                data = ws.receive_text()
                parsed = json.loads(data)
                assert parsed["message"] == "failed: timeout"

    def test_multiple_progress_messages(self, app_client: TestClient) -> None:
        """Multiple progress updates should all be relayed."""
        payloads = [
            {"fraction": 0.2, "message": "Loading data..."},
            {"fraction": 0.5, "message": "Optimising..."},
            {"fraction": 0.8, "message": "Backtesting..."},
            {"fraction": 1.0, "message": "completed"},
        ]
        messages = self._make_redis_messages(payloads)
        fake_redis = _FakeRedisAsync(messages)

        with (
            patch("api.auth.decode_token") as mock_decode,
            patch("redis.asyncio.from_url", return_value=fake_redis),
        ):
            mock_decode.return_value = _valid_token_payload()

            with app_client.websocket_connect(_owned_url(app_client)) as ws:
                received = []
                for _ in range(4):
                    data = ws.receive_text()
                    received.append(json.loads(data))

                assert len(received) == 4
                assert received[0]["fraction"] == 0.2
                assert received[-1]["message"] == "completed"


@pytest.mark.parametrize(
    "case", ["foreign", "deleted", "missing_run", "refresh", "bad"]
)
def test_denied_connections_never_open_redis(app_client: TestClient, case: str):
    subject = "2" if case == "foreign" else "99" if case == "deleted" else "1"
    token = create_access_token({"sub": subject})
    if case == "refresh":
        token = create_refresh_token({"sub": "1"})
    if case == "bad":
        token = "invalid.jwt.value"
    url = _owned_url(app_client, token)
    if case == "missing_run":
        url = "/ws/progress/missing-run?token=" + token
    with patch("redis.asyncio.from_url") as redis:
        with pytest.raises(WebSocketDisconnect) as denied:
            with app_client.websocket_connect(url):
                pytest.fail("Denied socket was accepted")
        assert denied.value.code == 1008
        redis.assert_not_called()


def test_owner_stream_releases_database_before_subscribing(app_client: TestClient):
    token = create_access_token({"sub": "1"})
    fake = _FakeRedisAsync(
        [{"type": "message", "data": b'{"fraction":1.0,"message":"completed"}'}]
    )

    def connect(*args, **kwargs):
        assert app_client.app.state.test_engine.pool.checkedout() == 0
        return fake

    with patch("redis.asyncio.from_url", side_effect=connect) as redis:
        with app_client.websocket_connect(_owned_url(app_client, token)) as ws:
            assert ws.receive_json() == {"fraction": 1.0, "message": "completed"}
        redis.assert_called_once()


def test_in_memory_store_keeps_shared_visibility_but_requires_existing_run(
    app_client: TestClient,
):
    store = RunStore()
    run = store.create(RunType.EXECUTION, {})
    app_client.app.state.run_store = store
    token = create_access_token({"sub": "2"})
    fake = _FakeRedisAsync(
        [{"type": "message", "data": b'{"fraction":1.0,"message":"completed"}'}]
    )
    with patch("redis.asyncio.from_url", return_value=fake):
        with app_client.websocket_connect(f"/ws/progress/{run.id}?token={token}") as ws:
            assert ws.receive_json()["message"] == "completed"
    with patch("redis.asyncio.from_url") as redis:
        with pytest.raises(WebSocketDisconnect):
            with app_client.websocket_connect("/ws/progress/missing?token=" + token):
                pytest.fail("Missing run was accepted")
        redis.assert_not_called()


@pytest.mark.parametrize("origin", ["https://untrusted.example.test", "null", ""])
def test_browser_origin_is_rejected_before_account_db_or_redis(app_client, origin):
    factory = MagicMock()
    app_client.app.dependency_overrides[get_session_factory] = lambda: factory
    with patch("api.routers.ws.get_user_from_token") as account, patch(
        "redis.asyncio.from_url"
    ) as redis:
        with pytest.raises(WebSocketDisconnect) as denied:
            with app_client.websocket_connect(
                _owned_url(app_client), headers={"Origin": origin}
            ):
                pytest.fail("Untrusted browser origin was accepted")
        assert denied.value.code == 1008
        factory.assert_not_called()
        account.assert_not_called()
        redis.assert_not_called()


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://localhost:8443",
        "https://127.0.0.1:8443",
        "https://localhost",
    ],
)
def test_supported_browser_origin_can_stream_owned_progress(app_client, origin):
    token = create_access_token({"sub": "1"})
    fake = _FakeRedisAsync(
        [{"type": "message", "data": b'{"fraction":1.0,"message":"completed"}'}]
    )
    with patch("redis.asyncio.from_url", return_value=fake):
        with app_client.websocket_connect(
            _owned_url(app_client, token), headers={"Origin": origin}
        ) as ws:
            assert ws.receive_json()["message"] == "completed"


def test_configured_browser_origin_can_stream_owned_progress(app_client, monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://CUSTOM.example.test/")
    app = create_app(store=app_client.app.state.run_store)
    app.dependency_overrides.update(app_client.app.dependency_overrides)
    app.state.owned_run_id = app_client.app.state.owned_run_id
    client = TestClient(app)
    token = create_access_token({"sub": "1"})
    fake = _FakeRedisAsync(
        [{"type": "message", "data": b'{"fraction":1.0,"message":"completed"}'}]
    )
    with patch("redis.asyncio.from_url", return_value=fake):
        with client.websocket_connect(
            _owned_url(client, token), headers={"Origin": "https://custom.example.test"}
        ) as ws:
            assert ws.receive_json()["message"] == "completed"
