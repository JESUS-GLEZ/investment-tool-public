"""Exercise production startup with owned configuration, never account secrets."""

from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import pytest

from api import runtime_security
from api.main import create_app
from api.store import RunStore
from investment_tool.config import AppConfig, JWTConfig


@pytest.fixture()
def production_config(monkeypatch):
    monkeypatch.setenv("ENV", "production")

    def configure(secret, algorithm="HS256"):
        config = AppConfig(jwt=JWTConfig(secret_key=secret, algorithm=algorithm))
        monkeypatch.setattr(runtime_security, "get_config", lambda: config)

    return configure


@pytest.mark.parametrize(
    "secret",
    [
        "",
        "  ",
        "CHANGE-ME-IN-PRODUCTION",
        "CHANGE_ME_generate_a_strong_secret",
        " change-me-in-production ",
    ],
)
def test_production_rejects_empty_and_documented_placeholder_keys(
    production_config, secret
):
    production_config(secret)
    with pytest.raises(RuntimeError, match="non-placeholder"):
        create_app(store=RunStore())


@pytest.mark.parametrize(
    "algorithm,minimum", [("HS256", 32), ("HS384", 48), ("HS512", 64)]
)
@pytest.mark.parametrize("offset", [-1, 0])
def test_hmac_minimum_is_algorithm_specific(
    production_config, algorithm, minimum, offset
):
    production_config("x" * (minimum + offset), algorithm)
    if offset < 0:
        with pytest.raises(RuntimeError, match=f"at least {minimum} bytes"):
            create_app(store=RunStore())
    else:
        assert create_app(store=RunStore()).title == "Investment Tool API"


def test_minimum_counts_utf8_bytes(production_config):
    production_config("é" * 16)
    assert create_app(store=RunStore()).title == "Investment Tool API"


def test_rejection_precedes_database_initialization_and_omits_key(production_config):
    owned_short_key = "owned-short-key"
    production_config(owned_short_key)
    with patch("api.database.init_engine") as initialize:
        with pytest.raises(RuntimeError) as error:
            create_app(init_db=True)
    initialize.assert_not_called()
    assert owned_short_key not in str(error.value)


@pytest.mark.parametrize("environment", ["", "development", "test"])
def test_non_production_startup_keeps_existing_behavior(
    production_config, monkeypatch, environment
):
    production_config("CHANGE-ME-IN-PRODUCTION")
    monkeypatch.setenv("ENV", environment)
    assert create_app(store=RunStore()).title == "Investment Tool API"


def test_production_environment_normalizes_case_and_space(
    production_config, monkeypatch
):
    production_config("CHANGE-ME-IN-PRODUCTION")
    monkeypatch.setenv("ENV", " Production ")
    with pytest.raises(RuntimeError, match="non-placeholder"):
        create_app(store=RunStore())


def test_asymmetric_config_does_not_use_hmac_length_policy(production_config):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    production_config(pem, "RS256")
    assert create_app(store=RunStore()).title == "Investment Tool API"
