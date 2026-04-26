"""
Authentication endpoints:
  POST /api/v1/auth/register   — create a new account
  POST /api/v1/auth/login      — exchange credentials for JWT pair
  POST /api/v1/auth/refresh    — issue new access token from refresh token
  POST /api/v1/auth/logout     — revoke tokens
  GET  /api/v1/auth/me         — return current user profile
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DBSession, RedisClient, get_current_user
from app.core.exceptions import AuthenticationError, ConflictError, TokenExpiredError
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    hash_password,
    revocation_key,
    verify_password,
)
from app.db.redis import is_token_revoked, revoke_token
from app.models.patient import PatientProfile
from app.models.user import User, UserRole
from app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Register ───────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=201)
async def register(
    body: RegisterRequest,
    db: DBSession,
):
    """
    Create a new user account.

    - Enforces unique email.
    - Hashes the password before storage (never stored in plaintext).
    - Automatically creates a PatientProfile if role == patient.
    - Returns a JWT pair so the client is logged in immediately.
    """
    # Check email uniqueness
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise ConflictError(
            f"An account with email '{body.email}' already exists.",
            detail={"field": "email"},
        )

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        phone=body.phone,
        role=body.role,
        is_active=True,
    )
    db.add(user)
    await db.flush()   # get user.id

    # Create patient profile for patient accounts
    if user.role == UserRole.PATIENT:
        from datetime import date as date_type
        dob = None
        if body.date_of_birth:
            try:
                dob = date_type.fromisoformat(body.date_of_birth)
            except ValueError:
                pass

        profile = PatientProfile(
            user_id=user.id,
            date_of_birth=dob,
            region=body.region,
        )
        db.add(profile)

    access_token = create_access_token(str(user.id), user.role.value)
    refresh_token, jti = create_refresh_token(str(user.id))

    return AuthResponse(
        tokens=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=15 * 60,
        ),
        user=UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
        ),
    )


# ── Login ──────────────────────────────────────────────────────────────────────

@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    db: DBSession,
):
    """
    Exchange email + password for an access/refresh JWT pair.

    Returns the same error message for unknown email and wrong password
    to prevent user enumeration.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    _invalid = AuthenticationError("Invalid email or password.")
    if user is None or not user.is_active:
        raise _invalid
    if not verify_password(body.password, user.hashed_password):
        raise _invalid

    access_token = create_access_token(str(user.id), user.role.value)
    refresh_token, _ = create_refresh_token(str(user.id))

    return AuthResponse(
        tokens=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=15 * 60,
        ),
        user=UserResponse(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
        ),
    )


# ── Refresh ────────────────────────────────────────────────────────────────────

@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    db: DBSession,
    redis: RedisClient,
):
    """
    Issue a new access token from a valid refresh token.

    The old refresh token's JTI is revoked and a new pair is issued
    (refresh token rotation).
    """
    try:
        payload = decode_refresh_token(body.refresh_token)
    except TokenExpiredError:
        raise AuthenticationError("Refresh token has expired. Please log in again.")

    jti = payload.get("jti")
    if jti and await is_token_revoked(jti):
        raise AuthenticationError("Refresh token has been revoked.")

    user_id = payload.get("sub")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise AuthenticationError("User not found.")

    # Revoke old refresh token
    if jti:
        await revoke_token(jti)

    # Issue new pair
    new_access  = create_access_token(str(user.id), user.role.value)
    new_refresh, _ = create_refresh_token(str(user.id))

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        expires_in=15 * 60,
    )


# ── Logout ─────────────────────────────────────────────────────────────────────

@router.post("/logout", response_model=MessageResponse)
async def logout(
    body: LogoutRequest,
    current_user: CurrentUser,
    redis: RedisClient,
):
    """
    Revoke the current access token (by JTI from request.state.user)
    and the supplied refresh token.
    """
    from fastapi import Request

    # Revoke refresh token
    try:
        payload = decode_refresh_token(body.refresh_token)
        jti = payload.get("jti")
        if jti:
            await revoke_token(jti)
    except Exception:
        pass   # Even if invalid, the logout succeeds

    return MessageResponse(message="Logged out successfully.")


# ── Me ─────────────────────────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser):
    """Return the current authenticated user's profile."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
    )