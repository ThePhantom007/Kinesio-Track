"""
Detects clinical danger signals mid-session and orchestrates escalation.

Trigger sources (checked in priority order)
-------------------------------------------
  1. pose_analyzer result carries red_flag_triggered=True
     (exercise.red_flags condition matched on a single frame)
  2. Post-session pain report >= PAIN_RED_FLAG_THRESHOLD
     (checked by the Celery post_session_analysis task)
  3. Bilateral asymmetry > 25° sustained across POSE_VIOLATION_FRAME_COUNT frames
  4. ROM regression: peak_rom has declined > 15% vs previous session avg
     (checked by the Celery analytics task)

Escalation flow (triggered by the WebSocket handler for real-time cases)
------------------------------------------------------------------------
  1. This service is called with the trigger type and context.
  2. Calls claude_client.escalate_red_flag() to get severity + messages.
  3. Writes a RedFlagEvent row.
  4. Returns an immediate_action string to the WebSocket handler, which
     sends it to the patient as a RedFlagMessage.
  5. Enqueues a Celery notification_tasks.notify_clinician_red_flag task.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.claude_client import ClaudeClient
from app.ai.prompt_templates.red_flag import RedFlagContext
from app.core.config import settings
from app.core.logging import get_logger
from app.models.injury import Injury
from app.models.patient import PatientProfile
from app.models.plan import ExercisePlan
from app.models.red_flag import RedFlagEvent, RedFlagSeverity, RedFlagTrigger
from app.models.session import ExerciseSession
from app.services.pose_analyzer import FrameAnalysisResult

log = get_logger(__name__)


class RedFlagMonitorService:

    def __init__(self, claude_client: ClaudeClient) -> None:
        self._claude = claude_client

    # ═══════════════════════════════════════════════════════════════════════════
    # Real-time checks (called from WebSocket handler)
    # ═══════════════════════════════════════════════════════════════════════════

    async def check_frame_result(
        self,
        *,
        db: AsyncSession,
        session: ExerciseSession,
        analysis: FrameAnalysisResult,
        exercise_name: str,
        exercise_slug: str,
        patient: PatientProfile,
        injury: Injury | None,
        plan: ExercisePlan | None,
    ) -> RedFlagEvent | None:
        """
        Check a single frame analysis result for red-flag conditions.
        Called inline by the WebSocket handler — must be fast.

        Returns:
            RedFlagEvent if a red flag was triggered, else None.
        """
        if not analysis.red_flag_triggered:
            return None

        ctx = self._build_context(
            trigger_type=RedFlagTrigger.EXERCISE_RED_FLAG,
            trigger_context={
                "condition": analysis.red_flag_condition,
                "form_score": analysis.form_score,
                "joint_angles": analysis.joint_angles,
            },
            session=session,
            exercise_name=exercise_name,
            exercise_slug=exercise_slug,
            patient=patient,
            injury=injury,
            plan=plan,
        )
        return await self._escalate(db, ctx, session)

    async def check_pain_spike(
        self,
        *,
        db: AsyncSession,
        session: ExerciseSession,
        pain_score: int,
        previous_avg_pain: float,
        exercise_name: str,
        exercise_slug: str,
        patient: PatientProfile,
        injury: Injury | None,
        plan: ExercisePlan | None,
    ) -> RedFlagEvent | None:
        """
        Check whether the reported post-session pain score warrants escalation.
        Called by the Celery post_session_analysis task.
        """
        if pain_score < settings.PAIN_RED_FLAG_THRESHOLD:
            return None

        ctx = self._build_context(
            trigger_type=RedFlagTrigger.PAIN_SPIKE,
            trigger_context={
                "pain_score":       pain_score,
                "previous_avg_pain": round(previous_avg_pain, 1),
                "increase":         round(pain_score - previous_avg_pain, 1),
            },
            session=session,
            exercise_name=exercise_name,
            exercise_slug=exercise_slug,
            patient=patient,
            injury=injury,
            plan=plan,
            current_pain_score=pain_score,
            previous_avg_pain=previous_avg_pain,
        )
        return await self._escalate(db, ctx, session)

    async def check_rom_regression(
        self,
        *,
        db: AsyncSession,
        session: ExerciseSession,
        current_rom: float,
        previous_avg_rom: float,
        exercise_name: str,
        exercise_slug: str,
        patient: PatientProfile,
        injury: Injury | None,
        plan: ExercisePlan | None,
    ) -> RedFlagEvent | None:
        """
        Detect significant ROM regression across sessions.
        Called by the Celery analytics task.
        """
        if previous_avg_rom <= 0:
            return None
        regression_pct = (previous_avg_rom - current_rom) / previous_avg_rom * 100
        if regression_pct < 15.0:   # < 15% decline is within normal variation
            return None

        ctx = self._build_context(
            trigger_type=RedFlagTrigger.ROM_REGRESSION,
            trigger_context={
                "current_rom_deg":    round(current_rom, 1),
                "previous_avg_deg":   round(previous_avg_rom, 1),
                "regression_pct":     round(regression_pct, 1),
            },
            session=session,
            exercise_name=exercise_name,
            exercise_slug=exercise_slug,
            patient=patient,
            injury=injury,
            plan=plan,
        )
        return await self._escalate(db, ctx, session)

    async def check_bilateral_asymmetry(
        self,
        *,
        db: AsyncSession,
        session: ExerciseSession,
        asymmetry_degrees: float,
        joint_pair: str,
        exercise_name: str,
        exercise_slug: str,
        patient: PatientProfile,
        injury: Injury | None,
        plan: ExercisePlan | None,
    ) -> RedFlagEvent | None:
        """Escalate if bilateral asymmetry exceeds the severe threshold (25°)."""
        if asymmetry_degrees < 25.0:
            return None

        ctx = self._build_context(
            trigger_type=RedFlagTrigger.BILATERAL_ASYMMETRY,
            trigger_context={
                "joint_pair":          joint_pair,
                "asymmetry_degrees":   round(asymmetry_degrees, 1),
                "threshold_degrees":   25.0,
            },
            session=session,
            exercise_name=exercise_name,
            exercise_slug=exercise_slug,
            patient=patient,
            injury=injury,
            plan=plan,
        )
        return await self._escalate(db, ctx, session)

    # ═══════════════════════════════════════════════════════════════════════════
    # Shared escalation pipeline
    # ═══════════════════════════════════════════════════════════════════════════

    async def _escalate(
        self,
        db: AsyncSession,
        ctx: RedFlagContext,
        session: ExerciseSession,
    ) -> RedFlagEvent:
        """
        Call Claude, write the RedFlagEvent, and return it.
        The WebSocket handler and Celery tasks handle downstream notification.
        """
        log.warning(
            "red_flag_triggered",
            trigger=ctx.trigger_type.value,
            session_id=str(session.id),
            patient_id=str(ctx.patient_age),   # age only — no PII in logs
            exercise=ctx.exercise_slug,
        )

        ai_response = await self._claude.escalate_red_flag(
            ctx, db, patient_id=session.patient_id
        )

        severity_map = {
            "warn":       RedFlagSeverity.WARN,
            "stop":       RedFlagSeverity.STOP,
            "seek_care":  RedFlagSeverity.SEEK_CARE,
        }
        severity = severity_map.get(ai_response["severity"], RedFlagSeverity.STOP)

        event = RedFlagEvent(
            patient_id=session.patient_id,
            session_id=session.id,
            trigger_type=ctx.trigger_type,
            trigger_context=ctx.trigger_context,
            severity=severity,
            immediate_action=ai_response["immediate_action"],
            clinician_note=ai_response["clinician_note"],
            session_recommendation=ai_response.get("session_recommendation"),
            claude_raw_response=ai_response,
            clinician_notified_at=None,   # set by notification_tasks
        )
        db.add(event)
        await db.flush()

        log.info(
            "red_flag_event_created",
            event_id=str(event.id),
            severity=severity.value,
            action=ai_response["immediate_action"][:60],
        )
        return event

    # ── Context builder ────────────────────────────────────────────────────────

    def _build_context(
        self,
        *,
        trigger_type: RedFlagTrigger,
        trigger_context: dict[str, Any],
        session: ExerciseSession,
        exercise_name: str,
        exercise_slug: str,
        patient: PatientProfile,
        injury: Injury | None,
        plan: ExercisePlan | None,
        current_pain_score: int | None = None,
        previous_avg_pain: float | None = None,
    ) -> RedFlagContext:
        return RedFlagContext(
            trigger_type=trigger_type.value,
            trigger_context=trigger_context,
            exercise_name=exercise_name,
            exercise_slug=exercise_slug,
            current_pain_score=current_pain_score,
            previous_avg_pain=previous_avg_pain,
            age=patient.age,
            activity_level=patient.activity_level.value if patient.activity_level else None,
            body_part=injury.body_part.value if injury else "unknown",
            session_reps_completed=session.total_reps_completed or 0,
            session_quality_score=session.avg_quality_score,
            escalation_criteria=plan.escalation_criteria or [] if plan else [],
        )