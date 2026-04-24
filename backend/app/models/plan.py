"""
Exercise plan — the top-level treatment programme for one injury.

Every time the plan is adapted by Claude (post-session or clinician override)
a new row is created with an incremented ``version`` number.  The patient's
PatientProfile.active_plan_id always points to the current version.  Old
versions are retained for audit and for feeding adaptation context to Claude.
"""

from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.injury import Injury
    from app.models.patient import PatientProfile
    from app.models.phase import PlanPhase
    from app.models.session import ExerciseSession


class PlanStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    SUPERSEDED = "superseded"   # replaced by a newer version


class ExercisePlan(BaseModel):
    __tablename__ = "exercise_plans"

    # ── Ownership ──────────────────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    injury_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("injuries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Versioning ─────────────────────────────────────────────────────────────
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Increments on each AI adaptation or clinician override.",
    )
    parent_plan_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exercise_plans.id", ondelete="SET NULL"),
        nullable=True,
        comment="FK to the plan this version was adapted from.",
    )

    # ── Plan metadata ──────────────────────────────────────────────────────────
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[PlanStatus] = mapped_column(
        Enum(PlanStatus, name="plan_status"),
        nullable=False,
        default=PlanStatus.ACTIVE,
        index=True,
    )
    current_phase: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="1-based index of the phase the patient is currently working through.",
    )
    recovery_target_days: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="AI-estimated total duration of the programme in days.",
    )
    ai_generated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="False when the plan was created or fully overridden by a clinician.",
    )

    # ── Clinical constraints ───────────────────────────────────────────────────
    contraindications: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
        comment="Movements or exercises to avoid entirely.",
    )
    escalation_criteria: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "Conditions that should trigger a red-flag escalation, e.g. "
            "[{trigger: 'pain_score >= 8', action: 'stop_session'}]. "
            "Evaluated by red_flag_monitor service."
        ),
    )
    clinician_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Clinician-authored notes attached to this plan version.",
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    patient: Mapped[PatientProfile] = relationship(
        "PatientProfile",
        back_populates="plans",
        foreign_keys=[patient_id],
    )
    injury: Mapped[Injury] = relationship(
        "Injury",
        back_populates="plans",
    )
    phases: Mapped[list[PlanPhase]] = relationship(
        "PlanPhase",
        back_populates="plan",
        cascade="all, delete-orphan",
        order_by="PlanPhase.phase_number",
    )
    sessions: Mapped[list[ExerciseSession]] = relationship(
        "ExerciseSession",
        back_populates="plan",
        cascade="all, delete-orphan",
    )
    children: Mapped[list[ExercisePlan]] = relationship(
        "ExercisePlan",
        foreign_keys=[parent_plan_id],
        back_populates="parent",
    )
    parent: Mapped[ExercisePlan | None] = relationship(
        "ExercisePlan",
        remote_side="ExercisePlan.id",
        back_populates="children",
    )