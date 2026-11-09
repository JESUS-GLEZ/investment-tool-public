"""Tests for :mod:`api.auth` — JWT authentication and :mod:`api.routers.auth`.

Uses an in-memory SQLite database to test auth endpoints end-to-end.
"""

from __future__ import annotations

from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from api.database import get_db
from api.main import create_app
from api.models import Base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():  # type: ignore[no-untyped-def]
    """Create an in-memory SQLite engine with shared connection pool.

    ``StaticPool`` ensures every connection sees the same in-memory
    database, which avoids "no such table" errors when FastAPI's
    thread-pool opens new connections.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):  # type: ignore[no-untyped-def]
    """Return a session bound to the test engine."""
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_engine) -> TestClient:  # type: ignore[no-untyped-def]
    """TestClient wired to an in-memory SQLite via dependency override."""
    app = create_app()

    # Override get_db so the app uses our test session factory
    factory = sessionmaker(bind=db_engine, autoflush=False, expire_on_commit=False)

    def _override_get_db() -> Generator[Session, None, None]:
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests — password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_and_verify(self) -> None:
        hashed = hash_password("mysecretpassword")
        assert verify_password("mysecretpassword", hashed)
        assert not verify_password("wrongpassword", hashed)

    def test_different_hashes_for_same_password(self) -> None:
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2  # bcrypt uses random salt


# ---------------------------------------------------------------------------
# Unit tests — JWT tokens
# ---------------------------------------------------------------------------


class TestJWTTokens:
    def test_create_and_decode_access_token(self) -> None:
        token = create_access_token({"sub": "42"})
        payload = decode_token(token)
        assert payload["sub"] == "42"
        assert payload["type"] == "access"

    def test_create_and_decode_refresh_token(self) -> None:
        token = create_refresh_token({"sub": "7"})
        payload = decode_token(token)
        assert payload["sub"] == "7"
        assert payload["type"] == "refresh"

    def test_invalid_token_raises(self) -> None:
        from jwt import InvalidTokenError

        with pytest.raises(InvalidTokenError):
            decode_token("not.a.valid.jwt")


# ---------------------------------------------------------------------------
# Integration tests — auth endpoints
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_success(self, client: TestClient) -> None:
        resp = client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "strongpass123"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_register_duplicate_email(self, client: TestClient) -> None:
        payload = {"email": "dup@example.com", "password": "strongpass123"}
        client.post("/auth/register", json=payload)
        resp = client.post("/auth/register", json=payload)
        assert resp.status_code == 409

    def test_register_short_password(self, client: TestClient) -> None:
        resp = client.post(
            "/auth/register",
            json={"email": "short@example.com", "password": "abc"},
        )
        assert resp.status_code == 422

    def test_register_missing_email(self, client: TestClient) -> None:
        resp = client.post(
            "/auth/register",
            json={"password": "strongpass123"},
        )
        assert resp.status_code == 422


class TestLogin:
    def test_login_success(self, client: TestClient) -> None:
        client.post(
            "/auth/register",
            json={"email": "bob@example.com", "password": "validpass123"},
        )
        resp = client.post(
            "/auth/login",
            json={"email": "bob@example.com", "password": "validpass123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_login_wrong_password(self, client: TestClient) -> None:
        client.post(
            "/auth/register",
            json={"email": "carol@example.com", "password": "correctpass1"},
        )
        resp = client.post(
            "/auth/login",
            json={"email": "carol@example.com", "password": "wrongpass"},
        )
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client: TestClient) -> None:
        resp = client.post(
            "/auth/login",
            json={"email": "nobody@example.com", "password": "whatever1"},
        )
        assert resp.status_code == 401


class TestRefresh:
    def test_refresh_success(self, client: TestClient) -> None:
        reg = client.post(
            "/auth/register",
            json={"email": "dave@example.com", "password": "goodpass1234"},
        )
        refresh_token = reg.json()["refresh_token"]
        resp = client.post(
            "/auth/refresh",
            json={"refresh_token": refresh_token},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data

    def test_refresh_with_access_token_fails(self, client: TestClient) -> None:
        reg = client.post(
            "/auth/register",
            json={"email": "eve@example.com", "password": "goodpass1234"},
        )
        access_token = reg.json()["access_token"]
        resp = client.post(
            "/auth/refresh",
            json={"refresh_token": access_token},
        )
        assert resp.status_code == 401

    def test_refresh_invalid_token(self, client: TestClient) -> None:
        resp = client.post(
            "/auth/refresh",
            json={"refresh_token": "garbage.token.here"},
        )
        assert resp.status_code == 401
