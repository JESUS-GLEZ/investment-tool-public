"""Exercise explicit origin policy without browser/network/provider access."""

from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from api.browser_origins import browser_origins
from api.main import create_app
from api.store import RunStore


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://localhost",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "https://localhost:8443",
        "https://127.0.0.1:8443",
        "https://127.0.0.1",
    ],
)
def test_local_cors_preflights_keep_authorization_header_support(monkeypatch, origin):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    client = TestClient(create_app(store=RunStore()))
    response = client.options(
        "/runs",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


@pytest.mark.parametrize("origin", ["https://untrusted.example.test", "null"])
def test_untrusted_cors_preflights_are_not_allowed(monkeypatch, origin):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    client = TestClient(create_app(store=RunStore()))
    response = client.options(
        "/runs",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_custom_origins_normalize_and_replace_local_defaults(monkeypatch):
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "https://EXAMPLE.test:443/,http://[::1]:5173,https://example.test",
    )
    assert browser_origins() == ("https://example.test", "http://[::1]:5173")
    client = TestClient(create_app(store=RunStore()))
    for origin, expected in [
        ("https://example.test", 200),
        ("http://localhost:5173", 400),
    ]:
        response = client.options(
            "/runs", headers={"Origin": origin, "Access-Control-Request-Method": "GET"}
        )
        assert response.status_code == expected


@pytest.mark.parametrize(
    "configured",
    [
        "*",
        "https://*.example.test",
        "null",
        "ftp://example.test",
        "https://example.test/path",
        "https://user:owned-password@example.test",
        "https://example.test?token=owned",
        "https://example.test#fragment",
        "http://example.test:70000",
        "http://example.test:0",
        "https://faß.example",
        "https://é.example",
        "https://%65xample.test",
        "http://127.1",
        "http://0177.0.0.1",
        "http://0x7f000001",
        "http://[fe80::1%25eth0]",
        "https://exam\tple.test",
        ", ,",
    ],
)
def test_invalid_origin_configuration_fails_before_database_init(
    monkeypatch, configured
):
    monkeypatch.setenv("CORS_ORIGINS", configured)
    with patch("api.database.init_engine") as initialize:
        with pytest.raises(ValueError) as error:
            create_app(init_db=True)
    initialize.assert_not_called()
    assert "owned-password" not in str(error.value)
    assert "token=owned" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    ("configured", "serialized"),
    [
        ("https://xn--fa-hia.example", "https://xn--fa-hia.example"),
        ("http://[0:0:0:0:0:0:0:1]:5173", "http://[::1]:5173"),
        ("http://127.0.0.1:80", "http://127.0.0.1"),
    ],
)
def test_browser_host_spelling_and_ipv6_preflight(monkeypatch, configured, serialized):
    monkeypatch.setenv("CORS_ORIGINS", configured)
    assert browser_origins() == (serialized,)
    client = TestClient(create_app(store=RunStore()))
    response = client.options(
        "/runs", headers={"Origin": serialized, "Access-Control-Request-Method": "GET"}
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == serialized
    if "xn--fa-hia" in serialized:
        response = client.options(
            "/runs",
            headers={
                "Origin": "https://fass.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers
