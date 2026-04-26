"""
Exercise plan endpoints:
  GET   /api/v1/plans/{plan_id}           — full plan with all phases and exercises
  GET   /api/v1/plans/{plan_id}/exercises — exercises for the current phase only
  PATCH /api/v1/plans/{plan_id}           — clinician override (creates new version)
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import (
    CurrentUser,
    DBSession,
    get_current_clinician,
    get_patient_profile,
)
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.models.exercise import Exercise
from app.models.patient import PatientProfile
from app.models.phase import PlanPhase
from app.models.plan import ExercisePlan, PlanStatus
from app.models.user import UserRole
from app.schemas.plan import (
    ExercisePlanResponse,
    ExercisePlanSummary,
    ExerciseResponse,
    PlanPatchRequest,
    PlanPhaseResponse,
)

router = APIRouter(prefix="/plans", tags=["plans"])


# ── GET full plan ──────────────────────────────────────────────────────────────

@router.get("/{plan_id}", response_model=ExercisePlanResponse)
async def get_plan(
    plan_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """
    Return the full exercise plan including all phases and exercises.

    Access rules:
      - Patient: can only access their own plans.
      - Clinician: can access plans for any of their assigned patients.
      - Admin: unrestricted.
    """
    plan = await _load_plan_with_phases(db, plan_id)
    await _assert_plan_access(plan, current_user, db)

    return _serialize_plan(plan)


# ── GET current phase exercises ────────────────────────────────────────────────

@router.get("/{plan_id}/exercises", response_model=list[ExerciseResponse])
async def get_current_exercises(
    plan_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """
    Return only the exercises for the plan's current phase.

    This is the primary endpoint called by the app before starting a session —
    it returns a lean list of exercises the patient needs to do today.
    """
    plan = await _load_plan_with_phases(db, plan_id)
    await _assert_plan_access(plan, current_user, db)

    current_phase = next(
        (p for p in plan.phases if p.phase_number == plan.current_phase),
        None,
    )
    if current_phase is None:
        return []

    return [_serialize_exercise(ex) for ex in current_phase.exercises]


# ── PATCH plan (clinician override) ───────────────────────────────────────────

@router.patch("/{plan_id}", response_model=ExercisePlanResponse)
async def patch_plan(
    plan_id: UUID,
    body: PlanPatchRequest,
    db: DBSession,
    current_user: CurrentUser,
    _clinician=Depends(get_current_clinician),
):
    """
    Apply a clinician override to the current plan.

    Creates a new plan version (increments plan.version, sets ai_generated=False).
    Clinician notes and exercise patches are applied atomically.

    Only clinicians can call this endpoint (enforced by get_current_clinician dep).
    """
    plan = await _load_plan_with_phases(db, plan_id)
    await _assert_plan_access(plan, current_user, db)

    # Update plan-level fields
    if body.title is not None:
        plan.title = body.title
    if body.contraindications is not None:
        plan.contraindications = body.contraindications
    if body.clinician_notes is not None:
        plan.clinician_notes = body.clinician_notes
    if body.recovery_target_days is not None:
        plan.recovery_target_days = body.recovery_target_days
    if body.current_phase is not None:
        plan.current_phase = body.current_phase

    plan.ai_generated = False
    plan.version += 1
    db.add(plan)

    # Apply per-exercise patches
    if body.exercise_patches:
        for exercise_id_str, patch in body.exercise_patches.items():
            ex_id = UUID(exercise_id_str)
            ex_result = await db.execute(
                select(Exercise).where(Exercise.id == ex_id)
            )
            exercise = ex_result.scalar_one_or_none()
            if exercise is None:
                continue

            patch_data = patch.model_dump(exclude_unset=True)
            for field, value in patch_data.items():
                if field == "landmark_rules" and value is not None:
                    value = {k: v.model_dump() for k, v in value.items()}
                elif field == "red_flags" and value is not None:
                    value = [rf.model_dump() for rf in value]
                setattr(exercise, field, value)
            db.add(exercise)

    await db.flush()

    # Reload to pick up changes
    plan = await _load_plan_with_phases(db, plan_id)
    return _serialize_plan(plan)


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _load_plan_with_phases(db: DBSession, plan_id: UUID) -> ExercisePlan:
    """Eagerly load plan → phases → exercises in one query."""
    result = await db.execute(
        select(ExercisePlan)
        .options(
            selectinload(ExercisePlan.phases).selectinload(PlanPhase.exercises)
        )
        .where(ExercisePlan.id == plan_id)
    )
    plan = result.scalar_one_or_none()
    if plan is None:
        raise NotFoundError(f"Plan {plan_id} not found.")
    return plan


async def _assert_plan_access(
    plan: ExercisePlan,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    """Raise PermissionDeniedError if the current user cannot access this plan."""
    if current_user.role == UserRole.ADMIN:
        return

    if current_user.role == UserRole.PATIENT:
        # Patient can only view their own plans
        result = await db.execute(
            select(PatientProfile).where(PatientProfile.user_id == current_user.id)
        )
        patient = result.scalar_one_or_none()
        if patient is None or plan.patient_id != patient.id:
            raise PermissionDeniedError("You do not have access to this plan.")
        return

    if current_user.role == UserRole.CLINICIAN:
        # Clinician must be assigned to the patient
        from app.models.clinician import ClinicianProfile
        result = await db.execute(
            select(PatientProfile).where(PatientProfile.id == plan.patient_id)
        )
        patient = result.scalar_one_or_none()
        if patient is None:
            raise PermissionDeniedError("Patient not found.")
        clinician_result = await db.execute(
            select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
        )
        clinician = clinician_result.scalar_one_or_none()
        if clinician is None or patient.assigned_clinician_id != clinician.id:
            raise PermissionDeniedError("You are not assigned to this patient.")


def _serialize_exercise(ex: Exercise) -> ExerciseResponse:
    from app.schemas.plan import JointRule, RedFlagRule
    return ExerciseResponse(
        id=ex.id,
        phase_id=ex.phase_id,
        slug=ex.slug,
        name=ex.name,
        order_index=ex.order_index,
        sets=ex.sets,
        reps=ex.reps,
        hold_seconds=ex.hold_seconds,
        rest_seconds=ex.rest_seconds,
        tempo=ex.tempo,
        target_joints=ex.target_joints,
        landmark_rules={
            k: JointRule(**v) for k, v in (ex.landmark_rules or {}).items()
        },
        red_flags=[RedFlagRule(**rf) for rf in (ex.red_flags or [])],
        patient_instructions=ex.patient_instructions,
        reference_video_url=ex.reference_video_url,
        difficulty=ex.difficulty,
        created_at=ex.created_at,
    )


def _serialize_phase(phase: PlanPhase) -> PlanPhaseResponse:
    return PlanPhaseResponse(
        id=phase.id,
        plan_id=phase.plan_id,
        phase_number=phase.phase_number,
        name=phase.name,
        goal=phase.goal,
        duration_days=phase.duration_days,
        progression_criteria=phase.progression_criteria,
        exercises=[_serialize_exercise(ex) for ex in phase.exercises],
        created_at=phase.created_at,
    )


def _serialize_plan(plan: ExercisePlan) -> ExercisePlanResponse:
    return ExercisePlanResponse(
        id=plan.id,
        patient_id=plan.patient_id,
        injury_id=plan.injury_id,
        title=plan.title,
        version=plan.version,
        status=plan.status,
        current_phase=plan.current_phase,
        recovery_target_days=plan.recovery_target_days,
        ai_generated=plan.ai_generated,
        contraindications=plan.contraindications,
        clinician_notes=plan.clinician_notes,
        phases=[_serialize_phase(p) for p in sorted(plan.phases, key=lambda p: p.phase_number)],
        created_at=plan.created_at,
        updated_at=plan.updated_at,
    )