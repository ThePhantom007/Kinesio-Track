"""
Exercise session lifecycle endpoints:
  POST  /api/v1/sessions              — start a session
  PATCH /api/v1/sessions/{id}         — end a session
  GET   /api/v1/sessions/{id}/summary — post-session detail
  GET   /api/v1/sessions              — session history (paginated)
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    CurrentUser,
    DBSession,
    Pagination,
    get_patient_profile,
    get_session_manager,
)
from app.core.config import settings
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.models.session import ExerciseSession, SessionStatus
from app.models.user import UserRole
from app.schemas.session import (
    SessionEndRequest,
    SessionHistoryResponse,
    SessionListItem,
    SessionMetrics,
    SessionStartRequest,
    SessionStartResponse,
    SessionSummaryResponse,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


# ── Start session ──────────────────────────────────────────────────────────────

@router.post("", response_model=SessionStartResponse, status_code=201)
async def start_session(
    body: SessionStartRequest,
    db: DBSession,
    current_user: CurrentUser,
    patient=Depends(get_patient_profile),
    session_mgr=Depends(get_session_manager),
):
    """
    Start a new exercise session.

    Creates the ExerciseSession row, primes Redis session state (landmark rules,
    counters), and returns the WebSocket URL + first exercise details.

    The client should immediately open the WebSocket connection using the
    returned ``websocket_url`` to begin sending landmark frames.
    """
    base_ws = getattr(settings, "BASE_WS_URL", "wss://api.kinesiotrack.app")
    return await session_mgr.start_session(
        db=db,
        patient_id=patient.id,
        plan_id=body.plan_id,
        exercise_id=body.exercise_id,
        base_ws_url=base_ws,
    )


# ── End session ────────────────────────────────────────────────────────────────

@router.patch("/{session_id}", response_model=SessionSummaryResponse)
async def end_session(
    session_id: UUID,
    body: SessionEndRequest,
    db: DBSession,
    current_user: CurrentUser,
    patient=Depends(get_patient_profile),
    session_mgr=Depends(get_session_manager),
):
    """
    End an active session and trigger post-session analysis.

    Steps performed:
      1. Validate the session belongs to the patient and is active.
      2. Write post_session_pain and patient_notes to the session row.
      3. Flush the Redis feedback buffer to Postgres feedback_events.
      4. Clean up all Redis session keys.
      5. Enqueue the Celery post_session_analysis task (async scoring + adaptation).

    The Celery task runs asynchronously; the summary returned here contains
    only the data written synchronously.  The full summary (with quality
    score and plan adaptation status) is available after the task completes
    via GET /sessions/{id}/summary.
    """
    session = await session_mgr.end_session(
        db=db,
        session_id=session_id,
        patient_id=patient.id,
        post_session_pain=body.post_session_pain,
        patient_notes=body.patient_notes,
        completion_pct=body.completion_pct,
    )

    # Enqueue async post-session analysis
    from app.workers.session_tasks import post_session_analysis
    post_session_analysis.delay(str(session_id))

    return _serialize_session_summary(session)


# ── Session summary ────────────────────────────────────────────────────────────

@router.get("/{session_id}/summary", response_model=SessionSummaryResponse)
async def get_session_summary(
    session_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """
    Return the full post-session summary.

    quality_score and plan_adapted are populated after the Celery task
    completes (typically 5–30 s after session end).  Poll this endpoint
    or use push notifications to know when it's ready.
    """
    session = await _load_session(db, session_id, current_user)
    return _serialize_session_summary(session)


# ── Session history ────────────────────────────────────────────────────────────

@router.get("", response_model=SessionHistoryResponse)
async def list_sessions(
    db: DBSession,
    current_user: CurrentUser,
    patient=Depends(get_patient_profile),
    plan_id: UUID | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(None, description="Opaque pagination cursor (session UUID)."),
):
    """
    Return paginated session history for the current patient.

    Results are ordered by started_at DESC (most recent first).
    Use the ``next_cursor`` from the response to fetch the next page.
    """
    query = (
        select(ExerciseSession)
        .where(ExerciseSession.patient_id == patient.id)
        .where(ExerciseSession.status == SessionStatus.COMPLETED)
        .order_by(ExerciseSession.started_at.desc())
        .limit(limit + 1)   # fetch one extra to detect next page
    )

    if plan_id:
        query = query.where(ExerciseSession.plan_id == plan_id)

    if cursor:
        # Cursor is the UUID of the last session on the previous page
        cursor_result = await db.execute(
            select(ExerciseSession.started_at).where(ExerciseSession.id == UUID(cursor))
        )
        cursor_ts = cursor_result.scalar_one_or_none()
        if cursor_ts:
            query = query.where(ExerciseSession.started_at < cursor_ts)

    result = await db.execute(query)
    sessions = result.scalars().all()

    has_more = len(sessions) > limit
    sessions = sessions[:limit]
    next_cursor = str(sessions[-1].id) if has_more and sessions else None

    from app.models.exercise import Exercise
    items = []
    for s in sessions:
        ex_name = None
        if s.exercise_id:
            ex_result = await db.execute(
                select(Exercise.name).where(Exercise.id == s.exercise_id)
            )
            ex_name = ex_result.scalar_one_or_none()

        items.append(SessionListItem(
            id=s.id,
            exercise_name=ex_name,
            status=s.status,
            avg_quality_score=s.avg_quality_score,
            completion_pct=s.completion_pct,
            post_session_pain=s.post_session_pain,
            duration_seconds=s.duration_seconds,
            plan_adapted=s.plan_adapted,
            started_at=s.started_at,
            ended_at=s.ended_at,
        ))

    return SessionHistoryResponse(
        sessions=items,
        total=len(items),
        next_cursor=next_cursor,
        has_more=has_more,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _load_session(
    db: DBSession,
    session_id: UUID,
    current_user: CurrentUser,
) -> ExerciseSession:
    result = await db.execute(
        select(ExerciseSession).where(ExerciseSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise NotFoundError(f"Session {session_id} not found.")

    if current_user.role == UserRole.PATIENT:
        from app.models.patient import PatientProfile
        from sqlalchemy import select as sa_select
        p = await db.execute(
            sa_select(PatientProfile).where(PatientProfile.user_id == current_user.id)
        )
        patient = p.scalar_one_or_none()
        if patient is None or session.patient_id != patient.id:
            raise PermissionDeniedError("You do not have access to this session.")

    return session


def _serialize_session_summary(session: ExerciseSession) -> SessionSummaryResponse:
    return SessionSummaryResponse(
        id=session.id,
        patient_id=session.patient_id,
        plan_id=session.plan_id,
        exercise_id=session.exercise_id,
        status=session.status,
        metrics=SessionMetrics(
            avg_quality_score=session.avg_quality_score,
            completion_pct=session.completion_pct,
            total_reps_completed=session.total_reps_completed,
            total_sets_completed=session.total_sets_completed,
            peak_rom_degrees=session.peak_rom_degrees,
            post_session_pain=session.post_session_pain,
            duration_seconds=session.duration_seconds,
        ),
        summary_text=session.summary_text,
        plan_adapted=session.plan_adapted,
        feedback_event_count=0,   # populated async by Celery task
        red_flag_count=0,
        started_at=session.started_at,
        ended_at=session.ended_at,
        created_at=session.created_at,
    )