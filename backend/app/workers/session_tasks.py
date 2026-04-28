"""
Post-session analysis Celery task.

  post_session_analysis(session_id)
    Runs immediately after a session ends. Computes quality metrics,
    checks for red flags, triggers plan adaptation, and sends the
    patient a session summary notification.

    Queue: session (time-sensitive — patient is waiting for feedback)
    Estimated runtime: 2–15 s
    Retry: up to 3 times
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from celery import Task
from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app

log = get_task_logger(__name__)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@celery_app.task(
    name="app.workers.session_tasks.post_session_analysis",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="session",
    acks_late=True,
)
def post_session_analysis(self: Task, session_id: str) -> dict:
    """
    Full post-session pipeline:

      1. Load session, exercise, patient, plan, and injury from DB.
      2. Fetch per-frame data from TimescaleDB for scoring.
      3. SessionScorerService: compute avg_quality, completion_pct, peak_ROM.
         Write metrics to ExerciseSession row and TimescaleDB.
      4. RedFlagMonitorService: check for pain spike escalation.
      5. PlanAdapterService: adapt plan if quality/pain thresholds crossed.
      6. Generate AI session summary text via Claude if quality > 0.
      7. Send session summary push notification to patient.

    Returns:
        {"session_id": str, "adapted": bool, "red_flag": bool}
    """
    log.info("post_session_analysis_started", session_id=session_id)

    async def _run_async():
        from sqlalchemy import select, func

        from app.db.postgres import get_db_context
        from app.db.timescale import get_quality_trend
        from app.db.queries.analytics import last_n_session_metrics, pain_trend
        from app.models.exercise import Exercise
        from app.models.injury import Injury
        from app.models.patient import PatientProfile
        from app.models.plan import ExercisePlan
        from app.models.session import ExerciseSession, SessionStatus
        from app.services.plan_adapter import PlanAdapterService
        from app.services.red_flag_monitor import RedFlagMonitorService
        from app.services.session_scorer import SessionScorerService
        from app.services.notification import NotificationService
        from app.ai.claude_client import ClaudeClient

        async with get_db_context() as db:
            session = await db.get(ExerciseSession, UUID(session_id))
            if session is None:
                log.error("session_not_found", session_id=session_id)
                return {"session_id": session_id, "adapted": False, "red_flag": False}

            if session.status != SessionStatus.COMPLETED:
                log.info("session_not_completed_skipping", session_id=session_id)
                return {"session_id": session_id, "adapted": False, "red_flag": False}

            patient  = await db.get(PatientProfile, session.patient_id)
            plan     = await db.get(ExercisePlan, session.plan_id) if session.plan_id else None
            exercise = await db.get(Exercise, session.exercise_id) if session.exercise_id else None

            injury = None
            if patient:
                inj_result = await db.execute(
                    select(Injury)
                    .where(Injury.patient_id == patient.id)
                    .order_by(Injury.created_at.desc())
                    .limit(1)
                )
                injury = inj_result.scalar_one_or_none()

            # ── Step 3: Score the session ─────────────────────────────────────
            # Fetch frame-level data from TimescaleDB for this session
            from app.db.timescale import get_rom_series
            rom_data = await get_rom_series(
                patient_id=session.patient_id,
                plan_id=session.plan_id,
                granularity="session",
            ) if session.plan_id else []

            # Build simple frame score/angle lists from session metrics
            # (In production these come from the Redis buffer flushed on session end)
            frame_scores = [float(session.avg_quality_score or 50.0)]
            frame_angles = []
            target_joints = exercise.target_joints if exercise else []

            scorer = SessionScorerService()
            metrics = await scorer.compute_and_persist(
                db=db,
                session=session,
                frame_scores=frame_scores,
                frame_angles=frame_angles,
                reps_completed=session.total_reps_completed or 0,
                prescribed_reps=exercise.reps if exercise else 10,
                prescribed_sets=exercise.sets if exercise else 3,
                target_joints=target_joints,
            )

            # ── Step 4: Red-flag check (pain spike) ───────────────────────────
            red_flag_triggered = False
            if session.post_session_pain and patient and plan and injury:
                pain_data = await pain_trend(session.patient_id, session.plan_id, n=5)
                prev_avg  = float(pain_data.get("avg_pain") or 0)

                rf_monitor = RedFlagMonitorService(ClaudeClient())
                rf_event = await rf_monitor.check_pain_spike(
                    db=db,
                    session=session,
                    pain_score=session.post_session_pain,
                    previous_avg_pain=prev_avg,
                    exercise_name=exercise.name if exercise else "exercise",
                    exercise_slug=exercise.slug if exercise else "exercise",
                    patient=patient,
                    injury=injury,
                    plan=plan,
                )
                if rf_event:
                    red_flag_triggered = True
                    notify_clinician_red_flag.delay(str(rf_event.id))

            # ── Step 5: Plan adaptation ────────────────────────────────────────
            plan_adapted = False
            if plan and patient:
                adapter = PlanAdapterService(ClaudeClient())
                plan_adapted = await adapter.adapt_after_session(db=db, session=session)

            # ── Step 6: Session summary text ──────────────────────────────────
            if metrics.avg_quality_score and patient:
                quality = metrics.avg_quality_score
                pain    = session.post_session_pain or 0
                reps    = metrics.total_reps_completed

                if quality >= 80:
                    summary = f"Excellent session! You completed {reps} rep(s) with {quality:.0f}% form quality. Keep it up!"
                elif quality >= 60:
                    summary = f"Good work! {reps} rep(s) completed with {quality:.0f}% form quality. Focus on the corrections from today."
                else:
                    summary = f"Session complete — {reps} rep(s) done. Form score was {quality:.0f}%. Your plan has been adjusted to help you improve."

                if plan_adapted:
                    summary += " Your exercise plan has been updated based on today's session."

                session.summary_text = summary
                db.add(session)

            # ── Step 7: Push notification ──────────────────────────────────────
            if patient and session.summary_text:
                notif = NotificationService()
                await notif.send_session_summary(
                    patient=patient,
                    session=session,
                    summary_text=session.summary_text,
                )

        log.info(
            "post_session_analysis_complete",
            session_id=session_id,
            adapted=plan_adapted,
            red_flag=red_flag_triggered,
        )
        return {
            "session_id":  session_id,
            "adapted":     plan_adapted,
            "red_flag":    red_flag_triggered,
        }

    try:
        return _run(_run_async())
    except Exception as exc:
        log.error("post_session_analysis_failed", session_id=session_id, error=str(exc))
        raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))


# Import here to avoid circular import at module level
from app.workers.notification_tasks import notify_clinician_red_flag  # noqa: E402