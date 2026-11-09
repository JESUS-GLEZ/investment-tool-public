"""Real JWT verification across HTTP, refresh and WebSocket entry points."""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.websockets import WebSocketDisconnect

from api import auth, main
from api.database import get_db, get_session_factory
from api.main import create_app
from api.models import Base, UserRow
from api.schemas import RunType
from api.store import RunStore

# Public test-only key. Tokens captured from python-jose 3.5.0 before migration;
# expiry is 2100-01-01. Never use these fixtures as application credentials.
SECRET = "jwt-migration-fixture-key-32-bytes-minimum"
LEGACY_ACCESS = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxIiwidHlwZSI6ImFjY2VzcyIsImV4cCI6NDEwMjQ0NDgwMH0."
    "SjG2e2BAFXeYfQR9je8RrgozsnfUm3qGSZlMcfZ3m1U"
)
LEGACY_REFRESH = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJzdWIiOiIxIiwidHlwZSI6InJlZnJlc2giLCJleHAiOjQxMDI0NDQ4MDB9."
    "An8d-VSe_6xpg8_el7o1On6Ae3t1YTuMaTR2DCUjU1M"
)


@pytest.fixture(autouse=True)
def jwt_config(monkeypatch):
    monkeypatch.setattr(auth, "_jwt_config", lambda: (SECRET, "HS256", 30, 10080))


@pytest.fixture
def client(tmp_path, monkeypatch):
    # The fixture endpoint is added after create_app; a local frontend build
    # must not install a catch-all mount ahead of that test-only route.
    monkeypatch.setattr(main, "_FRONTEND_DIST", tmp_path / "absent-frontend")
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        db.add(UserRow(id=1, email="jwt-fixture@example.com", password_hash="unused"))
        db.commit()
    store = RunStore()
    run = store.create(RunType.EXECUTION, {})
    app = create_app(store=store)
    app.state.migration_run_id = run.id

    def database():
        with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    app.dependency_overrides[get_session_factory] = lambda: factory

    @app.get("/jwt-fixture/protected")
    def protected(user=Depends(auth.get_current_user)):
        return {"id": user.id}

    with TestClient(app) as test_client:
        yield test_client
    engine.dispose()


def protected(client, token):
    return client.get(
        "/jwt-fixture/protected", headers={"Authorization": f"Bearer {token}"}
    )


def test_previously_issued_tokens_remain_usable(client):
    assert protected(client, LEGACY_ACCESS).json() == {"id": 1}
    response = client.post("/auth/refresh", json={"refresh_token": LEGACY_REFRESH})
    assert response.status_code == 200
    pair = response.json()
    assert auth.decode_token(pair["access_token"])["type"] == "access"
    assert auth.decode_token(pair["refresh_token"])["type"] == "refresh"
    assert protected(client, pair["access_token"]).status_code == 200


@pytest.mark.parametrize(
    "create,kind,minutes",
    [
        (auth.create_access_token, "access", 30),
        (auth.create_refresh_token, "refresh", 10080),
    ],
)
def test_token_claims_and_expiry_are_preserved(create, kind, minutes):
    before = datetime.now(timezone.utc).timestamp()
    payload = auth.decode_token(create({"sub": "1"}))
    after = datetime.now(timezone.utc).timestamp()
    assert payload["sub"] == "1" and payload["type"] == kind
    assert before + minutes * 60 - 1 <= payload["exp"] <= after + minutes * 60
    short = auth.decode_token(create({"sub": "1"}, expires_delta=timedelta(minutes=2)))
    assert before + 119 <= short["exp"] <= datetime.now(timezone.utc).timestamp() + 120


def invalid_token(case, kind):
    now = datetime.now(timezone.utc)
    payload = {"sub": "1", "type": kind, "exp": now + timedelta(hours=1)}
    key, algorithm = SECRET, "HS256"
    if case == "expired":
        payload["exp"] = now - timedelta(seconds=5)
    elif case == "wrong-signature":
        key = "another-test-only-secret-at-least-32-bytes"
    elif case == "wrong-algorithm":
        key, algorithm = SECRET * 2, "HS384"
    elif case == "unsigned":
        key, algorithm = "", "none"
    elif case == "future-nbf":
        payload["nbf"] = now + timedelta(hours=1)
    elif case == "invalid-exp":
        payload["exp"] = []
    elif case == "invalid-sub":
        payload["sub"] = [1]
    elif case == "malformed":
        return "not.a.valid.jwt"
    return jwt.encode(payload, key, algorithm=algorithm)


@pytest.mark.parametrize(
    "case",
    [
        "expired",
        "wrong-signature",
        "wrong-algorithm",
        "unsigned",
        "future-nbf",
        "invalid-exp",
        "invalid-sub",
        "malformed",
    ],
)
def test_invalid_tokens_fail_http_refresh_and_websocket(client, monkeypatch, case):
    access = invalid_token(case, "access")
    response = protected(client, access)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert (
        client.post(
            "/auth/refresh", json={"refresh_token": invalid_token(case, "refresh")}
        ).status_code
        == 401
    )
    redis_connect = MagicMock(
        side_effect=AssertionError("Rejected token must not reach Redis")
    )
    monkeypatch.setattr("redis.asyncio.from_url", redis_connect)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/progress/fixture?token={access}"):
            pass
    assert exc.value.code == 1008
    redis_connect.assert_not_called()


def test_refresh_tokens_cannot_authorize_access_or_websocket(client):
    assert protected(client, LEGACY_REFRESH).status_code == 401
    assert (
        client.post("/auth/refresh", json={"refresh_token": LEGACY_ACCESS}).status_code
        == 401
    )
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(f"/ws/progress/fixture?token={LEGACY_REFRESH}"):
            pass
    assert exc.value.code == 1008


def test_legacy_access_token_can_open_real_websocket_authentication(
    client, monkeypatch
):
    pubsub = MagicMock()
    for name in ("subscribe", "unsubscribe", "close"):
        setattr(pubsub, name, AsyncMock())
    pubsub.get_message = AsyncMock(
        return_value={
            "type": "message",
            "data": b'{"fraction":1,"message":"completed"}',
        }
    )
    redis = MagicMock()
    redis.pubsub.return_value = pubsub
    redis.close = AsyncMock()
    monkeypatch.setattr("redis.asyncio.from_url", lambda *args: redis)
    with client.websocket_connect(
        f"/ws/progress/{client.app.state.migration_run_id}?token={LEGACY_ACCESS}"
    ) as websocket:
        assert websocket.receive_json()["message"] == "completed"
    pubsub.subscribe.assert_awaited_once()


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def pem_private(key):
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


@pytest.mark.parametrize(
    "algorithm", ["HS256", "HS384", "HS512", "RS256", "RS384", "RS512", "ES256"]
)
def test_configured_algorithms_keep_signing_key_round_trip(
    monkeypatch, rsa_key, algorithm
):
    if algorithm.startswith("RS"):
        key = pem_private(rsa_key)
    elif algorithm == "ES256":
        key = pem_private(ec.generate_private_key(ec.SECP256R1()))
    else:
        key = SECRET * 2
    monkeypatch.setattr(auth, "_jwt_config", lambda: (key, algorithm, 30, 10080))
    token = auth.create_access_token({"sub": "1"})
    assert auth.decode_token(token)["sub"] == "1"
    if algorithm.startswith("RS"):
        public = (
            rsa_key.public_key()
            .public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            .decode()
        )
        monkeypatch.setattr(auth, "_jwt_config", lambda: (public, algorithm, 30, 10080))
        assert auth.decode_token(token)["sub"] == "1"


def test_der_public_key_hmac_forgery_is_rejected(monkeypatch, rsa_key):
    public_key = rsa_key.public_key()
    der = public_key.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pem = public_key.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()

    # Build an attacker signature independently of PyJWT's key guards.
    def b64(data):
        return base64.urlsafe_b64encode(data).rstrip(b"=")

    signing_input = b".".join(
        [
            b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()),
            b64(json.dumps({"sub": "1", "type": "access", "exp": 4102444800}).encode()),
        ]
    )
    forged = (
        signing_input
        + b"."
        + b64(hmac.new(der, signing_input, hashlib.sha256).digest())
    ).decode()
    monkeypatch.setattr(auth, "_jwt_config", lambda: (pem, "RS256", 30, 10080))
    with pytest.raises(jwt.InvalidAlgorithmError):
        auth.decode_token(forged)
    # Even a misconfigured HS256/DER combination must fail closed.
    monkeypatch.setattr(auth, "_jwt_config", lambda: (der, "HS256", 30, 10080))
    with pytest.raises(jwt.InvalidKeyError):
        auth.decode_token(forged)
