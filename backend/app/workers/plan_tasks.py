"""
Celery tasks for exercise plan management.

  generate_initial_plan_async(intake_id)
    Async plan generation path for large intake videos where the HTTP
    request would time out if Claude ran inline.  Called by the intake
    route when video analysis takes > 30 s.

    Queue: default
    Estimated runtime: 10–30 s (Claude call)

  adapt_plan_after_session(session_id)
    Standalone adaptation task.  Normally called via session_tasks, but
    can be enqueued directly for manual re-evaluation or backfilling.

    Queue: default
    Estimated runtime: 5–15 s
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


# ── generate_initial_plan_async ───────────────────────────────────────────────

@celery_app.task(
    name="app.workers.plan_tasks.generate_initial_plan_async",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
    queue="default",
    acks_late=True,
)
def generate_initial_plan_async(self: Task, injury_id: str, patient_id: str) -> dict:
    """
    Generate an exercise plan asynchronously after intake video processing
    has completed and baseline ROM is available.

    Called by the video_tasks worker after process_intake_video() if the
    patient doesn't have a plan yet.
    """
    log.info(
        "generate_initial_plan_async_started",
        injury_id=injury_id,
        patient_id=patient_id,
    )

    async def _run_async():
        from sqlalchemy import select

        from app.db.postgres import get_db_context
        from app.models.injury import Injury
        from app.models.patient import PatientProfile
        from app.models.plan import ExercisePlan, PlanStatus
        from app.services.exercise_planner import ExercisePlannerService
        from app.ai.claude_client import ClaudeClient

        async with get_db_context() as db:
            patient = await db.get(PatientProfile, UUID(patient_id))
            injury  = await db.get(Injury, UUID(injury_id))

            if patient is None or injury is None:
                log.error(
                    "generate_plan_missing_data",
                    patient_id=patient_id,
                    injury_id=injury_id,
                )
                return {"status": "failed", "reason": "patient or injury not found"}

            # Skip if a plan already exists for this injury
            existing_result = await db.execute(
                select(ExercisePlan).where(
                    ExercisePlan.injury_id == injury.id,
                    ExercisePlan.status == PlanStatus.ACTIVE,
                )
            )
            if existing_result.scalar_one_or_none():
                log.info("plan_already_exists_skipping", injury_id=injury_id)
                return {"status": "skipped", "reason": "plan already exists"}

            planner = ExercisePlannerService(ClaudeClient())
            response = await planner.create_plan_from_intake(
                db=db,
                patient=patient,
                injury=injury,
            )

            # Notify patient that their plan is ready
            from app.services.notification import NotificationService
            notif = NotificationService()
            await notif.send_to_patient(
                patient=patient,
                title="Your exercise plan is ready!",
                body="Tap to view your personalised physiotherapy programme.",
                data={"type": "plan_ready", "plan_id": str(response.plan_id)},
            )

        log.info(
            "generate_initial_plan_async_complete",
            plan_id=str(response.plan_id),
            phases=response.estimated_phases,
        )
        return {
            "status":   "done",
            "plan_id":  str(response.plan_id),
            "phases":   response.estimated_phases,
        }

    try:
        return _run(_run_async())
    except Exception as exc:
        log.error(
            "generate_initial_plan_async_failed",
            injury_id=injury_id,
            error=str(exc),
        )
        raise self.retry(exc=exc, countdown=60)


# ── adapt_plan_after_session ──────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.plan_tasks.adapt_plan_after_session",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
    queue="default",
    acks_late=True,
)
def adapt_plan_after_session(self: Task, session_id: str) -> dict:
    """
    Run plan adaptation for a completed session.  Can be called directly
    to trigger re-evaluation without going through post_session_analysis.
    """
    log.info("adapt_plan_task_started", session_id=session_id)

    async def _run_async():
        from app.db.postgres import get_db_context
        from app.models.session import ExerciseSession, SessionStatus
        from app.services.plan_adapter import PlanAdapterService
        from app.ai.claude_client import ClaudeClient

        async with get_db_context() as db:
            session = await db.get(ExerciseSession, UUID(session_id))
            if session is None or session.status != SessionStatus.COMPLETED:
                return {"session_id": session_id, "adapted": False}

            adapter = PlanAdapterService(ClaudeClient())
            adapted = await adapter.adapt_after_session(db=db, session=session)

        log.info("adapt_plan_task_complete", session_id=session_id, adapted=adapted)
        return {"session_id": session_id, "adapted": adapted}

    try:
        return _run(_run_async())
    except Exception as exc:
        log.error("adapt_plan_task_failed", session_id=session_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60)