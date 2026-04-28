"""
Clinician-specific endpoints:
  GET   /api/v1/clinicians/me                         — own clinician profile
  GET   /api/v1/clinicians/alerts                     — red-flag queue
  PATCH /api/v1/clinicians/alerts/{red_flag_id}       — acknowledge alert
  POST  /api/v1/clinicians/patients/{patient_id}/assign   — assign patient
  DELETE /api/v1/clinicians/patients/{patient_id}/assign  — unassign patient
  PATCH /api/v1/clinicians/patients/{patient_id}/notes    — add medical notes
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DBSession, get_current_clinician
from app.core.exceptions import ConflictError, NotFoundError, PermissionDeniedError
from app.models.clinician import ClinicianPatient, ClinicianProfile
from app.models.patient import PatientProfile
from app.models.red_flag import RedFlagEvent
from app.models.user import User

router = APIRouter(prefix="/clinicians", tags=["clinicians"])


# ── Own profile ────────────────────────────────────────────────────────────────

@router.get("/me")
async def get_clinician_profile(
    db: DBSession,
    current_user: CurrentUser,
    _clin=Depends(get_current_clinician),
):
    """Return the authenticated clinician's profile."""
    result = await db.execute(
        select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        raise NotFoundError("Clinician profile not found.")
    return {
        "id":              str(profile.id),
        "user_id":         str(profile.user_id),
        "full_name":       current_user.full_name,
        "email":           current_user.email,
        "license_number":  profile.license_number,
        "specialty":       profile.specialty,
        "institution":     profile.institution,
        "webhook_url":     profile.webhook_url,
        "email_alerts_enabled": profile.email_alerts_enabled,
    }


# ── Red-flag alert queue ───────────────────────────────────────────────────────

@router.get("/alerts")
async def get_alert_queue(
    db: DBSession,
    current_user: CurrentUser,
    _clin=Depends(get_current_clinician),
    unacknowledged_only: bool = Query(True),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Return red-flag alerts for all of the clinician's assigned patients,
    ordered by severity and creation time.

    Use ``unacknowledged_only=false`` to view the full history.
    """
    clinician = await _get_clinician(db, current_user)

    # Get patient IDs for this clinician
    patient_result = await db.execute(
        select(PatientProfile.id).where(
            PatientProfile.assigned_clinician_id == clinician.id
        )
    )
    patient_ids = [r[0] for r in patient_result.all()]

    if not patient_ids:
        return []

    query = (
        select(RedFlagEvent)
        .where(RedFlagEvent.patient_id.in_(patient_ids))
        .order_by(
            # seek_care first, then stop, then warn
            RedFlagEvent.severity.desc(),
            RedFlagEvent.created_at.desc(),
        )
        .limit(limit)
    )
    if unacknowledged_only:
        query = query.where(RedFlagEvent.acknowledged_at.is_(None))

    result = await db.execute(query)
    events = result.scalars().all()

    alerts = []
    for ev in events:
        patient = await db.get(PatientProfile, ev.patient_id)
        user    = await db.get(User, patient.user_id) if patient else None
        alerts.append({
            "id":                   str(ev.id),
            "patient_id":           str(ev.patient_id),
            "patient_name":         user.full_name if user else None,
            "session_id":           str(ev.session_id) if ev.session_id else None,
            "trigger_type":         ev.trigger_type.value,
            "severity":             ev.severity.value,
            "immediate_action":     ev.immediate_action,
            "clinician_note":       ev.clinician_note,
            "session_recommendation": ev.session_recommendation,
            "acknowledged":         ev.acknowledged_at is not None,
            "acknowledged_at":      ev.acknowledged_at.isoformat() if ev.acknowledged_at else None,
            "created_at":           ev.created_at.isoformat(),
        })
    return alerts


@router.patch("/alerts/{red_flag_id}")
async def acknowledge_alert(
    red_flag_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
    _clin=Depends(get_current_clinician),
    notes: str | None = None,
):
    """
    Acknowledge a red-flag alert and optionally add a clinician response note.
    """
    clinician = await _get_clinician(db, current_user)

    result = await db.execute(
        select(RedFlagEvent).where(RedFlagEvent.id == red_flag_id)
    )
    event = result.scalar_one_or_none()
    if event is None:
        raise NotFoundError(f"Alert {red_flag_id} not found.")

    # Verify the clinician is assigned to this patient
    patient = await db.get(PatientProfile, event.patient_id)
    if patient is None or patient.assigned_clinician_id != clinician.id:
        raise PermissionDeniedError("You are not assigned to this patient.")

    event.acknowledged_by    = current_user.id
    event.acknowledged_at    = datetime.now(timezone.utc)
    event.clinician_response_notes = notes
    db.add(event)
    await db.flush()

    return {"message": "Alert acknowledged.", "red_flag_id": str(red_flag_id)}


# ── Patient assignment ─────────────────────────────────────────────────────────

@router.post("/patients/{patient_id}/assign")
async def assign_patient(
    patient_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
    _clin=Depends(get_current_clinician),
    notes: str | None = None,
):
    """
    Assign a patient to the authenticated clinician.

    Creates a ClinicianPatient join record and sets
    PatientProfile.assigned_clinician_id.
    """
    clinician = await _get_clinician(db, current_user)
    patient   = await _get_patient(db, patient_id)

    if patient.assigned_clinician_id and patient.assigned_clinician_id != clinician.id:
        raise ConflictError(
            "This patient is already assigned to another clinician. "
            "Ask the current clinician to unassign them first.",
            detail={"patient_id": str(patient_id)},
        )

    # Check for existing active assignment record
    existing = await db.execute(
        select(ClinicianPatient).where(
            and_(
                ClinicianPatient.clinician_id == clinician.id,
                ClinicianPatient.patient_id   == patient.id,
                ClinicianPatient.is_active    == True,
            )
        )
    )
    if existing.scalar_one_or_none():
        raise ConflictError("Patient is already assigned to you.")

    assignment = ClinicianPatient(
        clinician_id=clinician.id,
        patient_id=patient.id,
        is_active=True,
        notes=notes,
    )
    db.add(assignment)

    patient.assigned_clinician_id = clinician.id
    db.add(patient)
    await db.flush()

    return {"message": "Patient assigned successfully.", "patient_id": str(patient_id)}


@router.delete("/patients/{patient_id}/assign")
async def unassign_patient(
    patient_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
    _clin=Depends(get_current_clinician),
):
    """
    Unassign a patient from the authenticated clinician.
    The assignment record is soft-deleted (is_active=False) for audit purposes.
    """
    clinician = await _get_clinician(db, current_user)
    patient   = await _get_patient(db, patient_id)

    if patient.assigned_clinician_id != clinician.id:
        raise PermissionDeniedError("This patient is not assigned to you.")

    result = await db.execute(
        select(ClinicianPatient).where(
            and_(
                ClinicianPatient.clinician_id == clinician.id,
                ClinicianPatient.patient_id   == patient.id,
                ClinicianPatient.is_active    == True,
            )
        )
    )
    assignment = result.scalar_one_or_none()
    if assignment:
        assignment.is_active        = False
        assignment.unassigned_at    = datetime.now(timezone.utc)
        db.add(assignment)

    patient.assigned_clinician_id = None
    db.add(patient)
    await db.flush()

    return {"message": "Patient unassigned.", "patient_id": str(patient_id)}


# ── Medical notes ──────────────────────────────────────────────────────────────

@router.patch("/patients/{patient_id}/notes")
async def update_medical_notes(
    patient_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
    _clin=Depends(get_current_clinician),
    notes: str = "",
):
    """
    Update the clinician-authored medical notes for a patient.
    Notes are never shown to the patient directly.
    """
    clinician = await _get_clinician(db, current_user)
    patient   = await _get_patient(db, patient_id)

    if patient.assigned_clinician_id != clinician.id:
        raise PermissionDeniedError("You are not assigned to this patient.")

    patient.medical_notes = notes
    db.add(patient)
    await db.flush()

    return {"message": "Medical notes updated.", "patient_id": str(patient_id)}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _get_clinician(db: AsyncSession, current_user: CurrentUser) -> ClinicianProfile:
    result = await db.execute(
        select(ClinicianProfile).where(ClinicianProfile.user_id == current_user.id)
    )
    clinician = result.scalar_one_or_none()
    if clinician is None:
        raise NotFoundError("Clinician profile not found.")
    return clinician


async def _get_patient(db: AsyncSession, patient_id: UUID) -> PatientProfile:
    patient = await db.get(PatientProfile, patient_id)
    if patient is None:
        raise NotFoundError(f"Patient {patient_id} not found.")
    return patient