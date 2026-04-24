"""
Schemas for patient profile endpoints:
  GET  /api/v1/patients/me
  GET  /api/v1/patients/{id}          (clinician-scoped)
  PATCH /api/v1/patients/{id}
  GET  /api/v1/clinicians/{id}/patients  (list view)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from app.models.patient import ActivityLevel
from app.schemas.base import AppBaseModel, AppResponseModel


# ── Requests ──────────────────────────────────────────────────────────────────

class PatientUpdateRequest(AppBaseModel):
    """
    All fields are optional — PATCH semantics.
    Only fields explicitly provided are updated; omitted fields are left
    unchanged (use Pydantic model_fields_set in the route handler).
    """

    full_name: str | None = Field(None, min_length=2, max_length=256)
    phone: str | None = Field(None, pattern=r"^\+?[1-9]\d{6,14}$")
    date_of_birth: date | None = None
    gender: str | None = Field(None, max_length=32)
    region: str | None = Field(None, max_length=128)
    activity_level: ActivityLevel | None = None
    fcm_token: str | None = Field(
        None,
        max_length=512,
        description="Firebase Cloud Messaging device token for push notifications.",
    )
    web_push_subscription: dict[str, Any] | None = Field(
        None,
        description="Web Push API subscription object from the browser.",
    )


# ── Responses ─────────────────────────────────────────────────────────────────

class BaselineROMEntry(AppResponseModel):
    """One joint's baseline measurement from the intake video."""

    angle_deg: float
    timestamp: str | None = None   # ISO string from video frame


class PatientResponse(AppResponseModel):
    """Full patient profile — returned to the patient themselves or their clinician."""

    id: UUID
    user_id: UUID
    full_name: str | None
    email: str | None = None          # populated by joining User; omitted in some views
    date_of_birth: date | None
    age: int | None
    gender: str | None
    region: str | None
    activity_level: ActivityLevel | None
    baseline_rom: dict[str, BaselineROMEntry] | None = Field(
        None,
        description="Per-joint baseline ROM captured from the intake video.",
    )
    mobility_notes: str | None
    active_plan_id: UUID | None
    assigned_clinician_id: UUID | None
    medical_notes: str | None = Field(
        None,
        description="Clinician-authored notes. Omitted from patient-facing responses.",
    )
    created_at: datetime
    updated_at: datetime


class PatientSummary(AppResponseModel):
    """
    Condensed view used in clinician list endpoints.
    Does not include medical notes or detailed baseline ROM.
    """

    id: UUID
    full_name: str | None
    email: str | None = None
    age: int | None
    region: str | None
    activity_level: ActivityLevel | None
    active_plan_id: UUID | None
    last_session_at: datetime | None = None
    total_sessions: int = 0
    created_at: datetime