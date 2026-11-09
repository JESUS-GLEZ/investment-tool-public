"""Reject documented development JWT defaults in production API mode."""

import os

from investment_tool.config import get_config

_JWT_PLACEHOLDERS = {
    "change-me-in-production",
    "change_me_generate_a_strong_secret",
}
_HMAC_KEY_BYTES = {"HS256": 32, "HS384": 48, "HS512": 64}


def validate_production_auth() -> None:
    """Fail before database initialization when production auth is unconfigured."""
    if os.environ.get("ENV", "").strip().casefold() != "production":
        return
    cfg = get_config().jwt
    secret = cfg.secret_key
    if not secret.strip() or secret.strip().casefold() in _JWT_PLACEHOLDERS:
        raise RuntimeError(
            "Production API requires an explicit non-placeholder JWT key"
        )
    minimum = _HMAC_KEY_BYTES.get(cfg.algorithm)
    if minimum is not None and len(secret.encode("utf-8")) < minimum:
        raise RuntimeError(
            f"Production {cfg.algorithm} JWT key requires at least {minimum} bytes"
        )
