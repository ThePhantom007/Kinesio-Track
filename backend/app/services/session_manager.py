"""
Manages the full lifecycle of an exercise session:
  - start_session():  create the DB row, cache landmark_rules in Redis,
                      return WebSocket URL and first exercise.
  - end_session():    validate the transition, flush Redis feedback buffer
                      to Postgres, enqueue Celery post-session analysis task.

Redis session state
-------------------
During a live session, per-frame data is buffered in Redis rather than
written to Postgres on every frame to keep the hot path free of DB round-trips.

Keys used (all scoped to session_id):
  session:state:{session_id}        HASH — status, patient_id, plan_id, exercise_id
  session:rules:{session_id}        STRING (JSON) — landmark_rules for current exercise
  session:feedback:{session_id}     LIST — feedback event dicts (flushed on session end)
  session:frame_count:{session_id}  STRING — total frames received
  session:rep_count:{session_id}    STRING — reps completed (incremented by WS handler)

All keys share a TTL of REDIS_SESSION_RULES_TTL (default 2 h) so orphaned
sessions from dropped connections are automatically cleaned up.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    NotFoundError,
    SessionAlreadyActiveError,
    SessionNotActiveError,
    SessionNotFoundError,
)
from app.core.logging import get_logger
from app.models.exercise import Exercise
from app.models.feedback_event import FeedbackEvent, FeedbackSeverity
from app.models.patient import PatientProfile
from app.models.phase import PlanPhase
from app.models.plan import ExercisePlan, PlanStatus
from app.models.session import ExerciseSession, SessionStatus
from app.schemas.session import SessionStartResponse
from app.schemas.plan import ExerciseSummary

log = get_logger(__name__)

# ── Redis key helpers ─────────────────────────────────────────────────────────

def _state_key(session_id: UUID) -> str:
    return f"session:state:{session_id}"

def _rules_key(session_id: UUID) -> str:
    return f"session:rules:{session_id}"

def _feedback_key(session_id: UUID) -> str:
    return f"session:feedback:{session_id}"

def _frame_count_key(session_id: UUID) -> str:
    return f"session:frame_count:{session_id}"

def _rep_count_key(session_id: UUID) -> str:
    return f"session:rep_count:{session_id}"


class SessionManagerService:

    def __init__(self, redis) -> None:
        self._redis = redis

    # ═══════════════════════════════════════════════════════════════════════════
    # Start
    # ═══════════════════════════════════════════════════════════════════════════

    async def start_session(
        self,
        *,
        db: AsyncSession,
        patient_id: UUID,
        plan_id: UUID,
        exercise_id: UUID | None = None,
        base_ws_url: str = "wss://api.kinesiotrack.app",
    ) -> SessionStartResponse:
        """
        Create an ExerciseSession row, warm Redis state, and return the
        WebSocket URL + first exercise detail.

        Raises:
            SessionAlreadyActiveError: Patient has an in-progress session.
            NotFoundError:             Plan or exercise not found.
        """
        await self._assert_no_active_session(db, patient_id)

        plan = await db.get(ExercisePlan, plan_id)
        if plan is None or plan.status != PlanStatus.ACTIVE:
            raise NotFoundError(f"Active plan {plan_id} not found.")

        exercise = await self._resolve_exercise(db, plan, exercise_id)

        # Create DB row
        session = ExerciseSession(
            patient_id=patient_id,
            plan_id=plan_id,
            exercise_id=exercise.id,
            status=SessionStatus.PENDING,
        )
        db.add(session)
        await db.flush()   # get session.id

        # Prime Redis state
        await self._prime_redis(session, plan, exercise)

        log.info(
            "session_started",
            session_id=str(session.id),
            patient_id=str(patient_id),
            exercise=exercise.slug,
        )

        phase_result = await db.execute(
            select(PlanPhase).where(PlanPhase.id == exercise.phase_id)
        )
        phase = phase_result.scalar_one()

        # Count today's exercises
        exercises_today = await self._count_phase_exercises(db, plan.id, plan.current_phase)

        return SessionStartResponse(
            session_id=session.id,
            websocket_url=f"{base_ws_url}/ws/session/{session.id}",
            first_exercise=ExerciseSummary(
                id=exercise.id,
                slug=exercise.slug,
                name=exercise.name,
                order_index=exercise.order_index,
                sets=exercise.sets,
                reps=exercise.reps,
                hold_seconds=exercise.hold_seconds,
                difficulty=exercise.difficulty,
                target_joints=exercise.target_joints,
            ),
            total_exercises_today=exercises_today,
            current_phase=plan.current_phase,
            phase_name=phase.name,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # End
    # ═══════════════════════════════════════════════════════════════════════════

    async def end_session(
        self,
        *,
        db: AsyncSession,
        session_id: UUID,
        patient_id: UUID,
        post_session_pain: int,
        patient_notes: str | None = None,
        completion_pct: float | None = None,
    ) -> ExerciseSession:
        """
        Transition session → COMPLETED, flush Redis feedback buffer to Postgres,
        and return the updated session ORM object for the Celery task to process.

        Raises:
            SessionNotFoundError:  Session not found or not owned by patient.
            SessionNotActiveError: Session is not in IN_PROGRESS or PENDING status.
        """
        session = await self._load_and_validate(db, session_id, patient_id)

        now = datetime.now(timezone.utc)
        session.status          = SessionStatus.COMPLETED
        session.ended_at        = now
        session.post_session_pain = post_session_pain
        session.patient_notes   = patient_notes
        if completion_pct is not None:
            session.completion_pct = completion_pct

        if session.started_at is None:
            session.started_at = now  # edge case: ended before first frame

        db.add(session)

        # Flush Redis feedback buffer → Postgres feedback_events
        await self._flush_feedback_buffer(db, session_id)

        # Capture frame count from Redis before cleaning up
        frame_count_raw = await self._redis.get(_frame_count_key(session_id))
        if frame_count_raw:
            log.info(
                "session_ended",
                session_id=str(session_id),
                frames_processed=int(frame_count_raw),
                pain=post_session_pain,
            )

        await self._cleanup_redis(session_id)
        await db.flush()

        return session

    async def mark_session_in_progress(
        self,
        *,
        db: AsyncSession,
        session_id: UUID,
    ) -> None:
        """
        Transition PENDING → IN_PROGRESS when the first landmark frame arrives.
        Called by the WebSocket handler on the first valid FRAME_DATA message.
        """
        session = await db.get(ExerciseSession, session_id)
        if session and session.status == SessionStatus.PENDING:
            session.status     = SessionStatus.IN_PROGRESS
            session.started_at = datetime.now(timezone.utc)
            db.add(session)
            await db.flush()

            await self._redis.hset(
                _state_key(session_id),
                "status",
                SessionStatus.IN_PROGRESS.value,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Redis helpers (called by WebSocket handler)
    # ═══════════════════════════════════════════════════════════════════════════

    async def get_landmark_rules(self, session_id: UUID) -> dict[str, Any] | None:
        """
        Return the cached landmark_rules dict for the current session.
        Returns None if the session key has expired (stale connection).
        """
        raw = await self._redis.get(_rules_key(session_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def get_red_flag_rules(self, session_id: UUID) -> list[dict[str, Any]]:
        """Return cached red-flag rules for the session's current exercise."""
        raw = await self._redis.hget(_state_key(session_id), "red_flag_rules")
        if not raw:
            return []
        return json.loads(raw)

    async def append_feedback_event(
        self,
        session_id: UUID,
        event: dict[str, Any],
    ) -> None:
        """
        Append a feedback event dict to the Redis buffer list.
        Serialises to JSON; flushed to Postgres on session end.
        """
        await self._redis.rpush(_feedback_key(session_id), json.dumps(event))

    async def increment_frame_count(self, session_id: UUID) -> int:
        """Increment and return the running frame count for this session."""
        return await self._redis.incr(_frame_count_key(session_id))

    async def increment_rep_count(self, session_id: UUID) -> int:
        """Increment and return the running rep count for this session."""
        return await self._redis.incr(_rep_count_key(session_id))

    async def get_session_state(self, session_id: UUID) -> dict[str, str] | None:
        """Return the full session state hash from Redis, or None if expired."""
        data = await self._redis.hgetall(_state_key(session_id))
        return data if data else None

    # ═══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ═══════════════════════════════════════════════════════════════════════════

    async def _assert_no_active_session(self, db: AsyncSession, patient_id: UUID) -> None:
        result = await db.execute(
            select(ExerciseSession).where(
                ExerciseSession.patient_id == patient_id,
                ExerciseSession.status.in_([SessionStatus.PENDING, SessionStatus.IN_PROGRESS]),
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            raise SessionAlreadyActiveError(
                f"Patient already has an active session: {existing.id}. "
                "End it before starting a new one.",
                detail={"active_session_id": str(existing.id)},
            )

    async def _resolve_exercise(
        self,
        db: AsyncSession,
        plan: ExercisePlan,
        exercise_id: UUID | None,
    ) -> Exercise:
        """Return the requested exercise, or the next due exercise in the current phase."""
        if exercise_id:
            exercise = await db.get(Exercise, exercise_id)
            if exercise is None:
                raise NotFoundError(f"Exercise {exercise_id} not found.")
            return exercise

        # Select first exercise of the current phase ordered by order_index
        result = await db.execute(
            select(Exercise)
            .join(PlanPhase, PlanPhase.id == Exercise.phase_id)
            .where(
                PlanPhase.plan_id == plan.id,
                PlanPhase.phase_number == plan.current_phase,
            )
            .order_by(Exercise.order_index)
            .limit(1)
        )
        exercise = result.scalar_one_or_none()
        if exercise is None:
            raise NotFoundError(
                f"No exercises found in phase {plan.current_phase} of plan {plan.id}."
            )
        return exercise

    async def _prime_redis(
        self,
        session: ExerciseSession,
        plan: ExercisePlan,
        exercise: Exercise,
    ) -> None:
        """Write all session state needed by the WebSocket handler into Redis."""
        ttl = settings.REDIS_SESSION_RULES_TTL
        sid = session.id

        # State hash
        await self._redis.hset(
            _state_key(sid),
            mapping={
                "status":        SessionStatus.PENDING.value,
                "patient_id":    str(session.patient_id),
                "plan_id":       str(session.plan_id),
                "exercise_id":   str(exercise.id),
                "exercise_slug": exercise.slug,
                "exercise_name": exercise.name,
                "difficulty":    exercise.difficulty or "beginner",
                "sets":          str(exercise.sets),
                "reps":          str(exercise.reps),
                "red_flag_rules": json.dumps(exercise.red_flags or []),
            },
        )
        await self._redis.expire(_state_key(sid), ttl)

        # Landmark rules (separate key — larger payload, frequently read)
        await self._redis.set(
            _rules_key(sid),
            json.dumps(exercise.landmark_rules),
            ex=ttl,
        )

        # Counters
        await self._redis.set(_frame_count_key(sid), 0, ex=ttl)
        await self._redis.set(_rep_count_key(sid), 0, ex=ttl)

    async def _flush_feedback_buffer(
        self,
        db: AsyncSession,
        session_id: UUID,
    ) -> None:
        """
        Read all feedback events from the Redis list and bulk-insert them
        into the feedback_events table.
        """
        raw_events = await self._redis.lrange(_feedback_key(session_id), 0, -1)
        if not raw_events:
            return

        events: list[FeedbackEvent] = []
        for raw in raw_events:
            try:
                data = json.loads(raw)
                events.append(FeedbackEvent(
                    session_id=session_id,
                    exercise_id=UUID(data["exercise_id"]) if data.get("exercise_id") else None,
                    occurred_at=datetime.fromisoformat(data["occurred_at"]),
                    frame_timestamp_ms=data.get("frame_timestamp_ms"),
                    severity=FeedbackSeverity(data.get("severity", "warning")),
                    error_type=data.get("error_type"),
                    affected_joint=data.get("affected_joint"),
                    actual_angle=data.get("actual_angle"),
                    expected_min_angle=data.get("expected_min_angle"),
                    expected_max_angle=data.get("expected_max_angle"),
                    deviation_degrees=data.get("deviation_degrees"),
                    form_score_at_event=data.get("form_score"),
                    message=data.get("message", ""),
                    from_cache=data.get("from_cache", False),
                    overlay_points=data.get("overlay_points"),
                ))
            except (KeyError, ValueError) as exc:
                log.warning("feedback_event_parse_error", error=str(exc))

        db.add_all(events)
        await db.flush()
        log.info("feedback_buffer_flushed", session_id=str(session_id), count=len(events))

    async def _cleanup_redis(self, session_id: UUID) -> None:
        """Delete all Redis keys for a completed or abandoned session."""
        keys = [
            _state_key(session_id),
            _rules_key(session_id),
            _feedback_key(session_id),
            _frame_count_key(session_id),
            _rep_count_key(session_id),
        ]
        await self._redis.delete(*keys)

    async def _count_phase_exercises(
        self,
        db: AsyncSession,
        plan_id: UUID,
        phase_number: int,
    ) -> int:
        result = await db.execute(
            select(Exercise)
            .join(PlanPhase, PlanPhase.id == Exercise.phase_id)
            .where(
                PlanPhase.plan_id == plan_id,
                PlanPhase.phase_number == phase_number,
            )
        )
        return len(result.scalars().all())

    async def _load_and_validate(
        self,
        db: AsyncSession,
        session_id: UUID,
        patient_id: UUID,
    ) -> ExerciseSession:
        result = await db.execute(
            select(ExerciseSession).where(
                ExerciseSession.id == session_id,
                ExerciseSession.patient_id == patient_id,
            )
        )
        session = result.scalar_one_or_none()
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} not found.")
        if session.status not in (SessionStatus.PENDING, SessionStatus.IN_PROGRESS):
            raise SessionNotActiveError(
                f"Session {session_id} is {session.status.value}, not active."
            )
        return session