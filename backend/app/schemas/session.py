"""
Schemas for exercise session endpoints:
  POST  /api/v1/sessions                — start a session
  PATCH /api/v1/sessions/{id}           — end a session
  GET   /api/v1/sessions/{id}/summary   — post-session detail
  GET   /api/v1/sessions                — session history
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field

from app.models.session import SessionStatus
from app.schemas.base import AppBaseModel, AppResponseModel
from app.schemas.plan import ExerciseSummary


# ── Requests ──────────────────────────────────────────────────────────────────

class SessionStartRequest(AppBaseModel):
    """
    Client submits this to begin a live exercise session.
    The server loads the active plan, determines the current exercise,
    caches landmark_rules in Redis, and returns the session_id + first
    exercise details so the client can start sending frames immediately.
    """

    plan_id: UUID
    exercise_id: UUID | None = Field(
        None,
        description=(
            "Specific exercise to perform. If omitted, the server selects "
            "the next due exercise in the current phase."
        ),
    )


class SessionEndRequest(AppBaseModel):
    """
    Submitted when the patient finishes or stops a session.
    Triggers the Celery post_session_analysis task.
    """

    post_session_pain: int = Field(
        ...,
        ge=1,
        le=10,
        description="Self-reported pain level immediately after the session.",
    )
    patient_notes: str | None = Field(
        None,
        max_length=2000,
        description="Optional free-text notes from the patient.",
    )
    completion_pct: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Client-reported completion fraction if known. "
            "If None, the server computes this from rep_metric rows."
        ),
    )


# ── Responses ─────────────────────────────────────────────────────────────────

class SessionStartResponse(AppResponseModel):
    """
    Returned immediately after POST /sessions.
    Contains everything the client needs to open the WebSocket and begin.
    """

    session_id: UUID
    websocket_url: str = Field(
        ...,
        description="Full WSS URL to connect to, e.g. wss://api.kinesiotrack.app/ws/session/{session_id}.",
    )
    first_exercise: ExerciseSummary
    total_exercises_today: int = Field(
        ...,
        description="Number of exercises in today's session plan.",
    )
    current_phase: int
    phase_name: str


class SessionMetrics(AppResponseModel):
    """Computed quality metrics for a completed session."""

    avg_quality_score: float | None = Field(None, description="Mean form quality (0–100).")
    completion_pct: float | None = Field(None, description="Fraction of prescribed work completed.")
    total_reps_completed: int | None
    total_sets_completed: int | None
    peak_rom_degrees: float | None = Field(None, description="Maximum ROM angle recorded during the session.")
    post_session_pain: int | None
    duration_seconds: int | None


class SessionSummaryResponse(AppResponseModel):
    """
    Full post-session summary.
    Returned by GET /sessions/{id}/summary after the Celery task has run.
    """

    id: UUID
    patient_id: UUID
    plan_id: UUID
    exercise_id: UUID | None
    status: SessionStatus
    metrics: SessionMetrics
    summary_text: str | None = Field(
        None,
        description="AI-generated plain-language session summary.",
    )
    plan_adapted: bool = Field(
        False,
        description="True if Claude adapted the exercise plan after this session.",
    )
    feedback_event_count: int = Field(0, description="Total correction messages sent during the session.")
    red_flag_count: int = Field(0, description="Number of red-flag events triggered during the session.")
    started_at: datetime | None
    ended_at: datetime | None
    created_at: datetime


class SessionListItem(AppResponseModel):
    """Condensed session row for history list endpoints."""

    id: UUID
    exercise_name: str | None
    status: SessionStatus
    avg_quality_score: float | None
    completion_pct: float | None
    post_session_pain: int | None
    duration_seconds: int | None
    plan_adapted: bool
    started_at: datetime | None
    ended_at: datetime | None


class SessionHistoryResponse(AppResponseModel):
    """Paginated session history."""

    sessions: list[SessionListItem]
    total: int
    next_cursor: str | None = None
    has_more: bool = False