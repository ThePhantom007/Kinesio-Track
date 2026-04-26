"""
FastAPI dependency functions injected via Depends() into route handlers.

All service instances are created once at app startup and stored on
app.state (see app/main.py lifespan).  The deps here simply pull them
off app.state so routes stay decoupled from construction details.

Auth dependency hierarchy
-------------------------
  get_current_user()        → any authenticated user
  get_current_patient()     → role == "patient" only
  get_current_clinician()   → role == "clinician" only
  get_current_admin()       → role == "admin" only
  get_patient_or_clinician()→ either patient or clinician (for shared views)
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.logging import get_logger
from app.db.postgres import get_db
from app.db.redis import get_redis
from app.models.user import User, UserRole

log = get_logger(__name__)


# ── Database & cache ──────────────────────────────────────────────────────────

DBSession = Annotated[AsyncSession, Depends(get_db)]
RedisClient = Annotated[Redis, Depends(get_redis)]


# ── Service accessors ─────────────────────────────────────────────────────────

def get_claude_client(request: Request):
    return request.app.state.claude_client

def get_exercise_planner(request: Request):
    return request.app.state.exercise_planner

def get_plan_adapter(request: Request):
    return request.app.state.plan_adapter

def get_pose_analyzer(request: Request):
    return request.app.state.pose_analyzer

def get_feedback_generator(request: Request):
    return request.app.state.feedback_generator

def get_session_manager(request: Request):
    return request.app.state.session_manager

def get_session_scorer(request: Request):
    return request.app.state.session_scorer

def get_recovery_forecaster(request: Request):
    return request.app.state.recovery_forecaster

def get_red_flag_monitor(request: Request):
    return request.app.state.red_flag_monitor

def get_video_intake_analyzer(request: Request):
    return request.app.state.video_intake_analyzer

def get_notification_service(request: Request):
    return request.app.state.notification_service

def get_connection_manager(request: Request):
    return request.app.state.connection_manager


# ── Auth dependencies ──────────────────────────────────────────────────────────

async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Return the authenticated User ORM object.

    The JWT payload was decoded and attached to request.state.user by
    AuthMiddleware.  This dependency loads the full User row so route handlers
    get a proper ORM object rather than the raw JWT dict.

    Raises:
        AuthenticationError: No user on request.state (public route misconfigured).
        NotFoundError:       User deleted after token was issued.
    """
    payload = getattr(request.state, "user", None)
    if not payload:
        from app.core.exceptions import AuthenticationError
        raise AuthenticationError("No authenticated user on this request.")

    user_id = payload.get("sub")
    user = await db.get(User, UUID(user_id))
    if user is None or not user.is_active:
        raise NotFoundError("User account not found or deactivated.")

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_current_patient(current_user: CurrentUser) -> User:
    """Require the authenticated user to have the 'patient' role."""
    if current_user.role != UserRole.PATIENT:
        raise PermissionDeniedError(
            "This endpoint requires a patient account.",
            detail={"required_role": "patient", "actual_role": current_user.role.value},
        )
    return current_user


async def get_current_clinician(current_user: CurrentUser) -> User:
    """Require the authenticated user to have the 'clinician' role."""
    if current_user.role != UserRole.CLINICIAN:
        raise PermissionDeniedError(
            "This endpoint requires a clinician account.",
            detail={"required_role": "clinician", "actual_role": current_user.role.value},
        )
    return current_user


async def get_current_admin(current_user: CurrentUser) -> User:
    """Require the authenticated user to have the 'admin' role."""
    if current_user.role != UserRole.ADMIN:
        raise PermissionDeniedError(
            "This endpoint requires an admin account.",
            detail={"required_role": "admin", "actual_role": current_user.role.value},
        )
    return current_user


async def get_patient_or_clinician(current_user: CurrentUser) -> User:
    """Allow both patient and clinician roles (e.g. shared progress views)."""
    if current_user.role not in (UserRole.PATIENT, UserRole.CLINICIAN):
        raise PermissionDeniedError(
            "This endpoint requires a patient or clinician account.",
        )
    return current_user


# ── Patient profile helper ────────────────────────────────────────────────────

async def get_patient_profile(
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """
    Load and return the PatientProfile for the current user.
    Raises NotFoundError if the profile doesn't exist (new user, incomplete setup).
    """
    from app.models.patient import PatientProfile
    from sqlalchemy import select

    result = await db.execute(
        select(PatientProfile).where(PatientProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        raise NotFoundError(
            "Patient profile not found. Please complete registration.",
            detail={"user_id": str(current_user.id)},
        )
    return profile


CurrentPatient = Annotated[object, Depends(get_patient_profile)]


# ── Pagination ────────────────────────────────────────────────────────────────

class PaginationParams:
    """Common pagination parameters for list endpoints."""

    def __init__(
        self,
        limit: int = 20,
        cursor: str | None = None,
    ) -> None:
        self.limit  = min(limit, 100)   # hard cap at 100
        self.cursor = cursor


Pagination = Annotated[PaginationParams, Depends(PaginationParams)]