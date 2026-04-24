"""
Clinician profile and the many-to-many join table that records clinician–patient
assignments.  A clinician can be assigned to many patients; a patient can have
at most one assigned clinician at a time (enforced at the application layer via
PatientProfile.assigned_clinician_id).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, BaseModel

if TYPE_CHECKING:
    from app.models.patient import PatientProfile
    from app.models.user import User


# ── Join table ────────────────────────────────────────────────────────────────

class ClinicianPatient(Base):
    """
    Many-to-many assignment record.

    A row exists as long as the clinician is actively responsible for the
    patient.  When unassigned, is_active is set to False (soft delete) so
    the assignment history is preserved for audit purposes.
    """

    __tablename__ = "clinician_patients"
    __table_args__ = (
        UniqueConstraint(
            "clinician_id",
            "patient_id",
            name="uq_clinician_patient",
            comment="One active assignment per clinician–patient pair.",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    clinician_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clinician_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    unassigned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional context note recorded at time of assignment.",
    )


# ── Clinician profile ─────────────────────────────────────────────────────────

class ClinicianProfile(BaseModel):
    __tablename__ = "clinician_profiles"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    license_number: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        comment="Regulatory body license number — validated externally.",
    )
    specialty: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="E.g. 'sports physiotherapy', 'neurological rehabilitation'.",
    )
    institution: Mapped[str | None] = mapped_column(String(256), nullable=True)
    webhook_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment=(
            "Optional HTTPS endpoint that receives red-flag alert POSTs in real time. "
            "Must respond 200 within 5 s."
        ),
    )
    email_alerts_enabled: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Whether to send email on red-flag escalation.",
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    user: Mapped[User] = relationship(
        "User",
        back_populates="clinician_profile",
    )
    assigned_patients: Mapped[list[PatientProfile]] = relationship(
        "PatientProfile",
        back_populates="assigned_clinician",
        foreign_keys="PatientProfile.assigned_clinician_id",
    )