"""
Core authentication identity.  One row per registered account regardless of
role.  Role-specific profile data lives in PatientProfile or ClinicianProfile.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Enum, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.clinician import ClinicianProfile
    from app.models.patient import PatientProfile


class UserRole(str, enum.Enum):
    PATIENT = "patient"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class User(BaseModel):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(
        String(320),
        unique=True,
        nullable=False,
        index=True,
        comment="Normalised to lowercase on write.",
    )
    hashed_password: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        comment="bcrypt hash — never store plain text.",
    )
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role"),
        nullable=False,
        default=UserRole.PATIENT,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Soft-disable without deleting the account.",
    )
    full_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    patient_profile: Mapped[PatientProfile | None] = relationship(
        "PatientProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="select",
    )
    clinician_profile: Mapped[ClinicianProfile | None] = relationship(
        "ClinicianProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="select",
    )