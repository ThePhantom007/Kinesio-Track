"""
Celery tasks for all outbound notifications.

  notify_clinician_red_flag(red_flag_id)
    Send webhook + email alert to the assigned clinician immediately.

  send_daily_session_reminders()      [Beat — 08:00 UTC]
    Send push/email reminders to patients due for a session today.

  check_missed_sessions()             [Beat — 20:00 UTC]
    Alert patients who have missed 3+ consecutive days.

  send_weekly_progress_digest()       [Beat — Monday 09:00 UTC]
    Send each active patient a weekly progress summary.

  send_session_reminder(patient_id)
    Individual on-demand reminder (called by routes or other tasks).

  send_milestone(patient_id, label)
    Notify a patient of a recovery milestone.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app

log = get_task_logger(__name__)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── notify_clinician_red_flag ─────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.notification_tasks.notify_clinician_red_flag",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    queue="notifications",
    acks_late=True,
)
def notify_clinician_red_flag(self, red_flag_id: str) -> dict:
    """
    Send webhook + email alert to the clinician assigned to the patient
    who triggered the red flag.  Updates RedFlagEvent.clinician_notified_at.
    """
    log.info("notify_clinician_red_flag_started", red_flag_id=red_flag_id)

    async def _run_async():
        from datetime import datetime, timezone
        from sqlalchemy import select

        from app.db.postgres import get_db_context
        from app.models.clinician import ClinicianProfile
        from app.models.patient import PatientProfile
        from app.models.red_flag import RedFlagEvent
        from app.models.user import User
        from app.services.notification import NotificationService

        async with get_db_context() as db:
            event   = await db.get(RedFlagEvent, UUID(red_flag_id))
            if event is None:
                return {"status": "not_found"}

            patient   = await db.get(PatientProfile, event.patient_id)
            clinician = (
                await db.get(ClinicianProfile, patient.assigned_clinician_id)
                if patient and patient.assigned_clinician_id else None
            )
            patient_user = await db.get(User, patient.user_id) if patient else None
            patient_name = patient_user.full_name if patient_user else None

            notif = NotificationService()
            await notif.send_red_flag_alert(
                clinician=clinician,
                event=event,
                patient_name=patient_name,
            )

            event.clinician_notified_at = datetime.now(timezone.utc)
            event.notification_method   = "webhook+email" if (clinician and clinician.webhook_url) else "email"
            db.add(event)

        log.info("notify_clinician_red_flag_complete", red_flag_id=red_flag_id)
        return {"status": "sent", "red_flag_id": red_flag_id}

    try:
        return _run(_run_async())
    except Exception as exc:
        log.error("notify_clinician_red_flag_failed", red_flag_id=red_flag_id, error=str(exc))
        raise self.retry(exc=exc, countdown=30)


# ── send_daily_session_reminders ──────────────────────────────────────────────

@celery_app.task(
    name="app.workers.notification_tasks.send_daily_session_reminders",
    queue="notifications",
)
def send_daily_session_reminders() -> dict:
    """
    Send session reminders to all patients who have an active plan and have
    not completed a session in the past 24 hours.
    """
    log.info("send_daily_session_reminders_started")

    async def _run_async():
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import select, func, not_, exists

        from app.db.postgres import get_db_context
        from app.models.exercise import Exercise
        from app.models.patient import PatientProfile
        from app.models.phase import PlanPhase
        from app.models.plan import ExercisePlan, PlanStatus
        from app.models.session import ExerciseSession, SessionStatus
        from app.models.user import User
        from app.services.notification import NotificationService

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        sent_count = 0

        async with get_db_context() as db:
            # Patients with active plan who haven't sessioned in 24h
            recent_session_subq = (
                select(ExerciseSession.patient_id)
                .where(
                    ExerciseSession.status == SessionStatus.COMPLETED,
                    ExerciseSession.started_at >= cutoff,
                )
                .scalar_subquery()
            )
            result = await db.execute(
                select(PatientProfile)
                .join(ExercisePlan, ExercisePlan.id == PatientProfile.active_plan_id)
                .where(
                    ExercisePlan.status == PlanStatus.ACTIVE,
                    PatientProfile.id.not_in(recent_session_subq),
                )
            )
            patients = result.scalars().all()

            notif = NotificationService()
            for patient in patients:
                # Find the next exercise name
                exercise_name = "today's exercise"
                if patient.active_plan_id:
                    plan = await db.get(ExercisePlan, patient.active_plan_id)
                    if plan:
                        ex_result = await db.execute(
                            select(Exercise)
                            .join(PlanPhase, PlanPhase.id == Exercise.phase_id)
                            .where(
                                PlanPhase.plan_id == plan.id,
                                PlanPhase.phase_number == plan.current_phase,
                            )
                            .order_by(Exercise.order_index)
                            .limit(1)
                        )
                        ex = ex_result.scalar_one_or_none()
                        if ex:
                            exercise_name = ex.name

                await notif.send_session_reminder(
                    patient=patient,
                    exercise_name=exercise_name,
                )
                sent_count += 1

        log.info("send_daily_session_reminders_complete", sent=sent_count)
        return {"sent": sent_count}

    return _run(_run_async())


# ── check_missed_sessions ─────────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.notification_tasks.check_missed_sessions",
    queue="notifications",
)
def check_missed_sessions() -> dict:
    """
    Alert patients who have missed 3+ consecutive days with an active plan.
    """
    log.info("check_missed_sessions_started")

    async def _run_async():
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import select

        from app.db.postgres import get_db_context
        from app.models.patient import PatientProfile
        from app.models.plan import ExercisePlan, PlanStatus
        from app.models.session import ExerciseSession, SessionStatus
        from app.services.notification import NotificationService

        cutoff_3days = datetime.now(timezone.utc) - timedelta(days=3)
        alerted = 0

        async with get_db_context() as db:
            result = await db.execute(
                select(PatientProfile)
                .join(ExercisePlan, ExercisePlan.id == PatientProfile.active_plan_id)
                .where(ExercisePlan.status == PlanStatus.ACTIVE)
            )
            patients = result.scalars().all()

            notif = NotificationService()
            for patient in patients:
                last_session_result = await db.execute(
                    select(ExerciseSession.started_at)
                    .where(
                        ExerciseSession.patient_id == patient.id,
                        ExerciseSession.status == SessionStatus.COMPLETED,
                    )
                    .order_by(ExerciseSession.started_at.desc())
                    .limit(1)
                )
                last_session_at = last_session_result.scalar_one_or_none()

                if last_session_at is None or last_session_at < cutoff_3days:
                    days_missed = (
                        (datetime.now(timezone.utc) - last_session_at).days
                        if last_session_at else 7
                    )
                    await notif.send_missed_session_alert(
                        patient=patient,
                        days_missed=days_missed,
                    )
                    alerted += 1

        log.info("check_missed_sessions_complete", alerted=alerted)
        return {"alerted": alerted}

    return _run(_run_async())


# ── send_weekly_progress_digest ───────────────────────────────────────────────

@celery_app.task(
    name="app.workers.notification_tasks.send_weekly_progress_digest",
    queue="notifications",
)
def send_weekly_progress_digest() -> dict:
    """
    Send a weekly progress summary push/email to all active patients.
    """
    log.info("send_weekly_progress_digest_started")

    async def _run_async():
        from datetime import datetime, timedelta, timezone
        from sqlalchemy import select, func

        from app.db.postgres import get_db_context
        from app.models.patient import PatientProfile
        from app.models.plan import ExercisePlan, PlanStatus
        from app.models.session import ExerciseSession, SessionStatus
        from app.services.notification import NotificationService

        week_ago = datetime.now(timezone.utc) - timedelta(days=7)
        sent = 0

        async with get_db_context() as db:
            result = await db.execute(
                select(PatientProfile)
                .join(ExercisePlan, ExercisePlan.id == PatientProfile.active_plan_id)
                .where(ExercisePlan.status == PlanStatus.ACTIVE)
            )
            patients = result.scalars().all()

            notif = NotificationService()
            for patient in patients:
                stats_result = await db.execute(
                    select(
                        func.count(ExerciseSession.id).label("session_count"),
                        func.avg(ExerciseSession.avg_quality_score).label("avg_quality"),
                    )
                    .where(
                        ExerciseSession.patient_id == patient.id,
                        ExerciseSession.status == SessionStatus.COMPLETED,
                        ExerciseSession.started_at >= week_ago,
                    )
                )
                row = stats_result.one()
                count   = int(row.session_count or 0)
                quality = round(float(row.avg_quality or 0), 1)

                if count == 0:
                    continue

                body = (
                    f"This week: {count} session(s) completed, "
                    f"average form score {quality}%. Keep it up!"
                )
                await notif.send_milestone(patient=patient, label=body)
                sent += 1

        log.info("send_weekly_progress_digest_complete", sent=sent)
        return {"sent": sent}

    return _run(_run_async())


# ── On-demand helpers ─────────────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.notification_tasks.send_milestone",
    queue="notifications",
)
def send_milestone(patient_id: str, label: str) -> dict:
    """Send a milestone notification to a specific patient."""
    async def _run_async():
        from app.db.postgres import get_db_context
        from app.models.patient import PatientProfile
        from app.services.notification import NotificationService

        async with get_db_context() as db:
            patient = await db.get(PatientProfile, UUID(patient_id))
            if patient is None:
                return {"status": "not_found"}
            notif = NotificationService()
            await notif.send_milestone(patient=patient, label=label)
        return {"status": "sent"}

    return _run(_run_async())