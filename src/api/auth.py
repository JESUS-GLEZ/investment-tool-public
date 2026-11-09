"""JWT authentication utilities.

Provides password hashing, JWT token creation/verification, and a
FastAPI dependency for extracting the current user from a request.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt import PyJWTError
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from sqlalchemy.orm import Session

from api.database import get_db
from api.models import UserRow
from investment_tool.config import get_config

# ---------------------------------------------------------------------------
# Password hashing (bcrypt directly — passlib has compat issues with bcrypt 5)
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return ``True`` if *plain* matches *hashed*."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


def _jwt_config() -> tuple[str, str, int, int]:
    """Return (secret_key, algorithm, access_minutes, refresh_minutes)."""
    cfg = get_config().jwt
    return (
        cfg.secret_key,
        cfg.algorithm,
        cfg.access_token_expire_minutes,
        cfg.refresh_token_expire_minutes,
    )


def create_access_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT access token."""
    secret, algorithm, access_minutes, _ = _jwt_config()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=access_minutes)
    )
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, secret, algorithm=algorithm)


def create_refresh_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT refresh token."""
    secret, algorithm, _, refresh_minutes = _jwt_config()
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=refresh_minutes)
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, secret, algorithm=algorithm)


def decode_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT, raising :class:`jwt.PyJWTError` on failure."""
    secret, algorithm, _, _ = _jwt_config()
    verification_key: Any = secret
    if algorithm in {"RS256", "RS384", "RS512"}:
        key = jwt.get_algorithm_by_name(algorithm).prepare_key(secret)
        # python-jose accepted the configured signing key for verification too.
        verification_key = key.public_key() if isinstance(key, RSAPrivateKey) else key
    return jwt.decode(token, verification_key, algorithms=[algorithm])


# ---------------------------------------------------------------------------
# FastAPI dependency — current authenticated user
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserRow:
    """Decode the bearer token and return the corresponding :class:`UserRow`.

    Raises :class:`HTTPException` 401 if the token is invalid or the
    user no longer exists.
    """
    return get_user_from_token(token, db)


def get_user_from_token(token: str, db: Session) -> UserRow:
    """Validate an access token and require its user to still exist."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        if payload.get("type") != "access":
            raise credentials_exception
        sub: str | None = payload.get("sub")
        if sub is None:
            raise credentials_exception
        user_id = int(sub)
    except (PyJWTError, ValueError):
        raise credentials_exception

    user = db.query(UserRow).filter(UserRow.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user
