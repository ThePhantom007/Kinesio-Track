"""
Celery tasks for scheduled analytics and maintenance.

  refresh_all_recovery_etas()        [Beat — 03:00 UTC daily]
    Recompute recovery forecasts for all active patients.

  refresh_timescale_aggregates()     [Beat — 03:30 UTC daily]
    Trigger TimescaleDB continuous aggregate refresh.

  check_claude_budget()              [Beat — 07:00 UTC daily]
    Alert if monthly Claude API spend exceeds 80% or 100% of budget.
"""

from __future__ import annotations

import asyncio

from celery.utils.log import get_task_logger

from app.workers.celery_app import celery_app

log = get_task_logger(__name__)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── refresh_all_recovery_etas ─────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.analytics_tasks.refresh_all_recovery_etas",
    queue="default",
)
def refresh_all_recovery_etas() -> dict:
    """
    Recompute recovery ETAs for all patients with an active plan and at
    least MIN_SESSIONS_FOR_ETA completed sessions.

    Results are written to a Redis cache keyed by patient_id so the
    dashboard can load them without a DB query.
    """
    log.info("refresh_all_recovery_etas_started")

    async def _run_async():
        from sqlalchemy import select

        from app.core.config import settings
        from app.db.postgres import get_db_context
        from app.db.redis import get_redis_client
        from app.models.patient import PatientProfile
        from app.models.plan import ExercisePlan, PlanStatus
        from app.services.recovery_forecaster import RecoveryForecasterService
        import json

        refreshed = 0
        skipped   = 0

        async with get_db_context() as db:
            result = await db.execute(
                select(PatientProfile)
                .join(ExercisePlan, ExercisePlan.id == PatientProfile.active_plan_id)
                .where(ExercisePlan.status == PlanStatus.ACTIVE)
            )
            patients = result.scalars().all()

            forecaster = RecoveryForecasterService()
            redis = get_redis_client()

            for patient in patients:
                plan = await db.get(ExercisePlan, patient.active_plan_id)
                if plan is None:
                    continue
                try:
                    forecast = await forecaster.forecast(
                        db=db,
                        patient_id=patient.id,
                        plan=plan,
                    )
                    if forecast.confidence in ("moderate", "high"):
                        cache_key = f"recovery_eta:{patient.id}"
                        await redis.setex(
                            cache_key,
                            86400,   # 24 h
                            json.dumps(forecast.model_dump(mode="json")),
                        )
                        refreshed += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    log.warning(
                        "eta_refresh_failed_for_patient",
                        patient_id=str(patient.id),
                        error=str(exc),
                    )
                    skipped += 1

        log.info(
            "refresh_all_recovery_etas_complete",
            refreshed=refreshed,
            skipped=skipped,
        )
        return {"refreshed": refreshed, "skipped": skipped}

    return _run(_run_async())


# ── refresh_timescale_aggregates ──────────────────────────────────────────────

@celery_app.task(
    name="app.workers.analytics_tasks.refresh_timescale_aggregates",
    queue="default",
)
def refresh_timescale_aggregates() -> dict:
    """
    Manually trigger a TimescaleDB continuous aggregate refresh to ensure
    daily ROM and quality score rollups are up to date before the ETA refresh.
    """
    log.info("refresh_timescale_aggregates_started")

    async def _run_async():
        from app.db.timescale import refresh_continuous_aggregates
        await refresh_continuous_aggregates()
        return {"status": "refreshed"}

    result = _run(_run_async())
    log.info("refresh_timescale_aggregates_complete")
    return result


# ── check_claude_budget ────────────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.analytics_tasks.check_claude_budget",
    queue="default",
)
def check_claude_budget() -> dict:
    """
    Check the current month's Claude API spend against the configured budget.
    Sends an admin alert notification if usage exceeds 80% or 100%.
    """
    log.info("check_claude_budget_started")

    async def _run_async():
        from datetime import datetime, timezone

        from app.ai.cost_tracker import CostTracker
        from app.core.config import settings
        from app.db.postgres import get_db_context

        async with get_db_context() as db:
            tracker  = CostTracker()
            summary  = await tracker.check_budget(db)

        pct     = summary["percent_used"]
        total   = summary["total_cost_usd"]
        budget  = summary["budget_usd"]
        month   = summary["month"]

        log.info(
            "budget_check",
            month=month,
            total_usd=total,
            budget_usd=budget,
            percent_used=pct,
        )

        if pct >= 100:
            log.error(
                "BUDGET_EXCEEDED",
                month=month,
                total_usd=total,
                budget_usd=budget,
            )
        elif pct >= 80:
            log.warning(
                "budget_80_percent_warning",
                month=month,
                total_usd=total,
                budget_usd=budget,
                percent_used=pct,
            )

        return summary

    return _run(_run_async())