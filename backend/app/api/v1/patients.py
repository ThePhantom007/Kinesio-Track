"""
Patient profile endpoints:
  GET   /api/v1/patients/me        — own profile (patient)
  PATCH /api/v1/patients/me        — update own profile (patient)
  GET   /api/v1/patients/{id}      — any patient profile (clinician/admin)
  GET   /api/v1/patients           — list all assigned patients (clinician)
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    CurrentUser,
    DBSession,
    get_current_clinician,
    get_patient_profile,
)
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.models.clinician import ClinicianProfile
from app.models.patient import PatientProfile
from app.models.session import ExerciseSession, SessionStatus
from app.models.user import User, UserRole
from app.schemas.patient import PatientResponse, PatientSummary, PatientUpdateRequest

router = APIRouter(prefix="/patients", tags=["patients"])


# ── Own profile (patient) ──────────────────────────────────────────────────────

@router.get("/me", response_model=PatientResponse)
async def get_my_profile(
    db: DBSession,
    current_user: CurrentUser,
    patient=Depends(get_patient_profile),
):
    """Return the authenticated patient's own profile."""
    return await _serialize_patient(db, patient, include_medical_notes=False)


@router.patch("/me", response_model=PatientResponse)
async def update_my_profile(
    body: PatientUpdateRequest,
    db: DBSession,
    current_user: CurrentUser,
    patient=Depends(get_patient_profile),
):
    """
    Update the authenticated patient's profile.
    Only fields included in the request body are modified (PATCH semantics).
    """
    update_data = body.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        if field == "full_name":
            # full_name lives on User, not PatientProfile
            current_user.full_name = value
            db.add(current_user)
        elif field == "phone":
            current_user.phone = value
            db.add(current_user)
        else:
            setattr(patient, field, value)

    db.add(patient)
    await db.flush()
    return await _serialize_patient(db, patient, include_medical_notes=False)


# ── Single patient (clinician / admin) ────────────────────────────────────────

@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """
    Return a patient profile.

    - Patients may only call /patients/me; this endpoint is for clinicians/admins.
    - Clinicians may only view patients assigned to them.
    """
    if current_user.role == UserRole.PATIENT:
        raise PermissionDeniedError(
            "Patients cannot view other patient profiles. Use GET /patients/me."
        )

    result = await db.execute(
        select(PatientProfile).where(PatientProfile.id == patient_id)
    )
    patient = result.scalar_one_or_none()
    if patient is None:
        raise NotFoundError(f"Patient {patient_id} not found.")

    if current_user.role == UserRole.CLINICIAN:
        await _assert_clinician_assigned(db, current_user, patient)

    return await _serialize_patient(db, patient, include_medical_notes=True)


# ── Patient list (clinician) ───────────────────────────────────────────────────

@router.get("", response_model=list[PatientSummary])
async def list_patients(
    db: DBSession,
    current_user: CurrentUser,
    _clinician=Depends(get_current_clinician),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Return all patients assigned to the authenticated clinician.
    Sorted by most recently active (last session date DESC).
    """
    clinician_result = await db.execute(
        select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
    )
    clinician = clinician_result.scalar_one_or_none()
    if clinician is None:
        raise NotFoundError("Clinician profile not found.")

    result = await db.execute(
        select(PatientProfile)
        .where(PatientProfile.assigned_clinician_id == clinician.id)
        .order_by(PatientProfile.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    patients = result.scalars().all()

    summaries = []
    for p in patients:
        last_session = await _last_session_at(db, p.id)
        total = await _total_sessions(db, p.id)
        user  = await db.get(User, p.user_id)
        summaries.append(PatientSummary(
            id=p.id,
            full_name=user.full_name if user else None,
            email=user.email if user else None,
            age=p.age,
            region=p.region,
            activity_level=p.activity_level,
            active_plan_id=p.active_plan_id,
            last_session_at=last_session,
            total_sessions=total,
            created_at=p.created_at,
        ))
    return summaries


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _assert_clinician_assigned(
    db: AsyncSession,
    current_user: CurrentUser,
    patient: PatientProfile,
) -> None:
    c_result = await db.execute(
        select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
    )
    clinician = c_result.scalar_one_or_none()
    if clinician is None or patient.assigned_clinician_id != clinician.id:
        raise PermissionDeniedError("You are not assigned to this patient.")


async def _serialize_patient(
    db: AsyncSession,
    patient: PatientProfile,
    include_medical_notes: bool,
) -> PatientResponse:
    user = await db.get(User, patient.user_id)
    return PatientResponse(
        id=patient.id,
        user_id=patient.user_id,
        full_name=user.full_name if user else None,
        email=user.email if user else None,
        date_of_birth=patient.date_of_birth,
        age=patient.age,
        gender=patient.gender,
        region=patient.region,
        activity_level=patient.activity_level,
        baseline_rom=patient.baseline_rom,
        mobility_notes=patient.mobility_notes,
        active_plan_id=patient.active_plan_id,
        assigned_clinician_id=patient.assigned_clinician_id,
        medical_notes=patient.medical_notes if include_medical_notes else None,
        created_at=patient.created_at,
        updated_at=patient.updated_at,
    )


async def _last_session_at(db: AsyncSession, patient_id: UUID):
    result = await db.execute(
        select(func.max(ExerciseSession.started_at))
        .where(
            ExerciseSession.patient_id == patient_id,
            ExerciseSession.status == SessionStatus.COMPLETED,
        )
    )
    return result.scalar_one_or_none()


async def _total_sessions(db: AsyncSession, patient_id: UUID) -> int:
    result = await db.execute(
        select(func.count(ExerciseSession.id))
        .where(
            ExerciseSession.patient_id == patient_id,
            ExerciseSession.status == SessionStatus.COMPLETED,
        )
    )
    return result.scalar_one_or_none() or 0