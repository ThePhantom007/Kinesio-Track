"""
Request and response schemas for authentication endpoints:
  POST /api/v1/auth/register
  POST /api/v1/auth/login
  POST /api/v1/auth/refresh
  POST /api/v1/auth/logout
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from pydantic import EmailStr, Field, field_validator

from app.models.user import UserRole
from app.schemas.base import AppBaseModel, AppResponseModel

# ── Validators ────────────────────────────────────────────────────────────────

_PASSWORD_MIN_LENGTH = 8
_PASSWORD_RE = re.compile(
    r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$"
)


def _validate_password(v: str) -> str:
    if len(v) < _PASSWORD_MIN_LENGTH:
        raise ValueError(f"Password must be at least {_PASSWORD_MIN_LENGTH} characters.")
    if not _PASSWORD_RE.match(v):
        raise ValueError(
            "Password must contain at least one uppercase letter, "
            "one lowercase letter, and one digit."
        )
    return v


# ── Requests ──────────────────────────────────────────────────────────────────

class RegisterRequest(AppBaseModel):
    """
    New account registration.
    Role defaults to 'patient'; clinicians must be created by an admin
    or supply a valid role in a trusted context.
    """

    email: EmailStr = Field(..., description="Primary login email address.")
    password: str = Field(..., min_length=8, description="Plain-text password (hashed server-side).")
    full_name: str = Field(..., min_length=2, max_length=256)
    phone: str | None = Field(
        None,
        pattern=r"^\+?[1-9]\d{6,14}$",
        description="E.164 format preferred, e.g. +919876543210.",
    )
    role: UserRole = Field(UserRole.PATIENT, description="Account role.")
    # Patient-specific fields — ignored if role != patient
    date_of_birth: str | None = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="ISO 8601 date string, e.g. '1990-04-15'.",
    )
    region: str | None = Field(None, max_length=128)

    @field_validator("password")
    @classmethod
    def strong_password(cls, v: str) -> str:
        return _validate_password(v)

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.lower().strip()


class LoginRequest(AppBaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.lower().strip()


class RefreshRequest(AppBaseModel):
    refresh_token: str = Field(..., min_length=1)


class LogoutRequest(AppBaseModel):
    """
    Client must supply the refresh token so its JTI can be added to the
    Redis revocation list.  The access token is revoked via its JTI extracted
    from request.state.user by the route handler.
    """

    refresh_token: str = Field(..., min_length=1)


# ── Responses ─────────────────────────────────────────────────────────────────

class TokenResponse(AppResponseModel):
    """Issued on successful login or token refresh."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Access token lifetime in seconds.")


class UserResponse(AppResponseModel):
    """
    Minimal public representation of the authenticated user.
    Returned alongside TokenResponse on login/register and from GET /auth/me.
    """

    id: UUID
    email: str
    full_name: str | None
    role: UserRole
    is_active: bool
    created_at: datetime


class AuthResponse(AppResponseModel):
    """Combined login/register response."""

    tokens: TokenResponse
    user: UserResponse


class MessageResponse(AppResponseModel):
    """Generic success message for endpoints that don't return a resource."""

    message: str