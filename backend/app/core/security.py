"""
app/core/security.py

JWT creation/decoding and password hashing.
All auth logic that isn't HTTP-specific lives here — routes and middleware
call these helpers rather than touching jose/passlib directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.core.exceptions import AuthenticationError, TokenExpiredError

# ── Password hashing ──────────────────────────────────────────────────────────

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return bcrypt hash of *plain*."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches *hashed*."""
    return _pwd_context.verify(plain, hashed)


# ── Token creation ────────────────────────────────────────────────────────────

def create_access_token(
    subject: str,
    role: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """
    Issue a short-lived access token.

    Args:
        subject:      Stable user identifier (UUID string).
        role:         ``"patient"`` | ``"clinician"`` | ``"admin"``.
        extra_claims: Optional additional claims merged into the payload.

    Returns:
        Signed JWT string.
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
        "jti": str(uuid.uuid4()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(subject: str) -> tuple[str, str]:
    """
    Issue a long-lived refresh token.

    Returns:
        ``(token_string, jti)`` — the *jti* must be stored in Redis so that
        logout can revoke it before natural expiry.
    """
    now = datetime.now(timezone.utc)
    jti = str(uuid.uuid4())
    payload: dict[str, Any] = {
        "sub": subject,
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        "jti": jti,
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, jti


# ── Token decoding ────────────────────────────────────────────────────────────

def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and verify *token*.

    Raises:
        TokenExpiredError:    JWT has passed its ``exp`` claim.
        AuthenticationError:  Any other verification failure (bad signature,
                              malformed token, wrong algorithm).
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Access token has expired") from exc
    except JWTError as exc:
        raise AuthenticationError(f"Invalid token: {exc}") from exc


def decode_access_token(token: str) -> dict[str, Any]:
    """
    Decode an access token and validate its *type* claim.

    Raises:
        AuthenticationError: If the token type is not ``"access"``.
    """
    payload = decode_token(token)
    if payload.get("type") != "access":
        raise AuthenticationError("Expected an access token")
    return payload


def decode_refresh_token(token: str) -> dict[str, Any]:
    """
    Decode a refresh token and validate its *type* claim.

    Raises:
        AuthenticationError: If the token type is not ``"refresh"``.
    """
    payload = decode_token(token)
    if payload.get("type") != "refresh":
        raise AuthenticationError("Expected a refresh token")
    return payload


# ── Token revocation helpers ──────────────────────────────────────────────────

def revocation_key(jti: str) -> str:
    """Redis key used to mark a token JTI as revoked."""
    return f"revoked_token:{jti}"