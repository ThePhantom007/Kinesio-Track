"""
Progress and analytics endpoints:
  GET /api/v1/patients/{patient_id}/progress   — full dashboard payload
  GET /api/v1/patients/{patient_id}/progress/recovery-eta   — ETA only
  GET /api/v1/admin/ai-costs                   — token spend (admin)
"""

from __future__ import annotations

from datetime import date
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    CurrentUser,
    DBSession,
    get_current_admin,
    get_patient_or_clinician,
    get_recovery_forecaster,
)
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.db.queries.progress import (
    get_milestones,
    progress_summary,
    quality_score_series,
    rom_series_all_joints,
)
from app.db.queries.analytics import monthly_token_spend
from app.models.patient import PatientProfile
from app.models.plan import ExercisePlan, PlanStatus
from app.models.user import UserRole
from app.schemas.progress import (
    JointROMSeries,
    ProgressMilestone,
    ProgressResponse,
    QualityDataPoint,
    RecoveryForecast,
    ROMDataPoint,
)

router = APIRouter(tags=["progress"])


# ── Full progress dashboard ────────────────────────────────────────────────────

@router.get(
    "/patients/{patient_id}/progress",
    response_model=ProgressResponse,
)
async def get_progress(
    patient_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
    _access=Depends(get_patient_or_clinician),
    from_date: date | None = Query(None, alias="from"),
    to_date:   date | None = Query(None, alias="to"),
    joint:     str | None  = Query(None),
    granularity: str       = Query("session", pattern="^(session|daily|weekly)$"),
    forecaster=Depends(get_recovery_forecaster),
):
    """
    Return the full progress payload for the patient dashboard.

    Includes:
      - ROM time-series per joint
      - Quality score time-series
      - Recovery forecast (ETA + confidence)
      - Milestone events
      - Aggregate summary stats

    Access:
      - A patient may only query their own patient_id.
      - A clinician may query any of their assigned patients.
    """
    patient = await _load_and_authorise_patient(db, patient_id, current_user)

    # Find the active plan to scope all queries
    plan = await _active_plan(db, patient)
    plan_id   = plan.id   if plan else None
    plan_title = plan.title if plan else None

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    # --- ROM series ---
    rom_by_joint: dict = {}
    quality_points: list = []
    milestones_raw: list = []
    summary: dict = {}
    forecast: RecoveryForecast | None = None

    if plan_id:
        rom_by_joint   = await rom_series_all_joints(patient_id, plan_id, from_date, to_date, granularity)
        quality_raw    = await quality_score_series(patient_id, plan_id, from_date, to_date, granularity)
        milestones_raw = await get_milestones(patient_id, plan_id)
        summary        = await progress_summary(patient_id, plan_id, from_date, to_date)
        forecast       = await forecaster.forecast(db=db, patient_id=patient_id, plan=plan)

    # Serialise ROM series
    rom_series = []
    for jnt, points in rom_by_joint.items():
        data_points = [
            ROMDataPoint(
                timestamp=p["timestamp"],
                session_id=p.get("session_id"),
                joint=jnt,
                angle_deg=float(p["peak_angle_deg"] or 0),
                avg_angle_deg=float(p["avg_angle_deg"] or 0),
            )
            for p in points
        ]
        current_rom = data_points[-1].angle_deg if data_points else None
        baseline = (patient.baseline_rom or {}).get(jnt, {}).get("angle_deg")
        rom_series.append(JointROMSeries(
            joint=jnt,
            data_points=data_points,
            current_rom=current_rom,
            baseline_rom=float(baseline) if baseline else None,
        ))

    # Serialise quality series
    quality_series = [
        QualityDataPoint(
            timestamp=p["timestamp"],
            session_id=p.get("session_id"),
            quality_score=float(p["quality_score"] or 0),
            completion_pct=float(p["completion_pct"]) if p.get("completion_pct") else None,
            post_session_pain=p.get("post_session_pain"),
        )
        for p in (quality_raw if plan_id else [])
    ]

    # Serialise milestones
    milestones = [
        ProgressMilestone(
            milestone_type=m["milestone_type"],
            label=m["label"],
            achieved_at=m["achieved_at"],
            value=m.get("value"),
        )
        for m in milestones_raw
    ]

    return ProgressResponse(
        patient_id=patient_id,
        plan_id=plan_id,
        plan_title=plan_title,
        current_phase=plan.current_phase if plan else None,
        total_phases=len(plan.phases) if plan else None,
        sessions_completed=int(summary.get("sessions_completed") or 0),
        last_session_at=summary.get("last_session_at"),
        rom_series=rom_series,
        quality_series=quality_series,
        avg_quality_score=float(summary["avg_quality_score"]) if summary.get("avg_quality_score") else None,
        avg_pain_score=float(summary["avg_pain_score"]) if summary.get("avg_pain_score") else None,
        total_sessions_in_range=int(summary.get("sessions_completed") or 0),
        recovery_forecast=forecast,
        milestones=milestones,
        from_date=from_date,
        to_date=to_date,
        granularity=granularity,
        generated_at=now,
    )


# ── Recovery ETA only ──────────────────────────────────────────────────────────

@router.get(
    "/patients/{patient_id}/progress/recovery-eta",
    response_model=RecoveryForecast,
)
async def get_recovery_eta(
    patient_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
    _access=Depends(get_patient_or_clinician),
    forecaster=Depends(get_recovery_forecaster),
):
    """
    Return the recovery ETA forecast without the full progress payload.
    Useful for the app's home screen widget.
    """
    patient = await _load_and_authorise_patient(db, patient_id, current_user)
    plan = await _active_plan(db, patient)
    if plan is None:
        return RecoveryForecast(
            estimated_recovery_date=None,
            estimated_days_remaining=None,
            confidence="low",
            trend="plateauing",
            sessions_analysed=0,
        )
    return await forecaster.forecast(db=db, patient_id=patient_id, plan=plan)


# ── AI cost dashboard (admin) ──────────────────────────────────────────────────

@router.get("/admin/ai-costs")
async def get_ai_costs(
    db: DBSession,
    current_user: CurrentUser,
    _admin=Depends(get_current_admin),
    year:  int = Query(..., ge=2024, le=2100),
    month: int = Query(..., ge=1, le=12),
):
    """
    Return Claude API token spend breakdown for a given month.
    Admin only.
    """
    rows = await monthly_token_spend(year, month)
    total_cost = sum(float(r.get("cost_usd") or 0) for r in rows)
    from app.core.config import settings
    return {
        "year":           year,
        "month":          month,
        "total_cost_usd": round(total_cost, 4),
        "budget_usd":     settings.MONTHLY_TOKEN_BUDGET_USD,
        "by_call_type":   rows,
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _load_and_authorise_patient(
    db: AsyncSession,
    patient_id: UUID,
    current_user: CurrentUser,
) -> PatientProfile:
    result = await db.execute(
        select(PatientProfile).where(PatientProfile.id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        raise NotFoundError(f"Patient {patient_id} not found.")

    if current_user.role == UserRole.PATIENT:
        if patient.user_id != current_user.id:
            raise PermissionDeniedError("You can only view your own progress.")

    elif current_user.role == UserRole.CLINICIAN:
        from app.models.clinician import ClinicianProfile
        c_result = await db.execute(
            select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
        )
        clinician = c_result.scalar_one_or_none()
        if clinician is None or patient.assigned_clinician_id != clinician.id:
            raise PermissionDeniedError("You are not assigned to this patient.")

    return patient


async def _active_plan(db: AsyncSession, patient: PatientProfile) -> ExercisePlan | None:
    if not patient.active_plan_id:
        return None
    from sqlalchemy.orm import selectinload
    from app.models.phase import PlanPhase
    result = await db.execute(
        select(ExercisePlan)
        .options(selectinload(ExercisePlan.phases))
        .where(
            ExercisePlan.id == patient.active_plan_id,
            ExercisePlan.status == PlanStatus.ACTIVE,
        )
    )
    return result.scalar_one_or_none()