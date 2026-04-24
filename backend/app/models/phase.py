"""
A plan phase groups exercises by rehabilitation stage.
Typical phases: Acute / Sub-acute / Strength / Return-to-function.

Exercises are ordered within a phase; the patient works through them
sequentially unless the clinician reorders them.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.exercise import Exercise
    from app.models.plan import ExercisePlan


class PlanPhase(BaseModel):
    __tablename__ = "plan_phases"

    plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exercise_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    phase_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1-based ordering within the plan.",
    )
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Display name shown to the patient, e.g. 'Phase 1 – Acute Recovery'.",
    )
    goal: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Plain-language description of what this phase aims to achieve.",
    )
    duration_days: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Expected duration before progressing to the next phase.",
    )
    progression_criteria: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment=(
            "Conditions required to advance to the next phase, e.g. "
            "'avg_quality_score >= 78 over 3 consecutive sessions'."
        ),
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    plan: Mapped[ExercisePlan] = relationship(
        "ExercisePlan",
        back_populates="phases",
    )
    exercises: Mapped[list[Exercise]] = relationship(
        "Exercise",
        back_populates="phase",
        cascade="all, delete-orphan",
        order_by="Exercise.order_index",
    )