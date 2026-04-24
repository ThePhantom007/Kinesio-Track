"""
An ExerciseSession represents one live exercise attempt by a patient.

Lifecycle
---------
  PENDING     — created by POST /sessions before the patient starts moving
  IN_PROGRESS — patient is connected to the WebSocket and sending frames
  COMPLETED   — session ended normally (patient or timeout)
  ABANDONED   — WebSocket disconnected without a formal end call

Metrics are written by the session_scorer service on transition to COMPLETED.
The Celery post_session_analysis task is enqueued at the same time.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.exercise import Exercise
    from app.models.feedback_event import FeedbackEvent
    from app.models.media import MediaFile
    from app.models.patient import PatientProfile
    from app.models.plan import ExercisePlan
    from app.models.red_flag import RedFlagEvent


class SessionStatus(str, enum.Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class ExerciseSession(BaseModel):
    __tablename__ = "exercise_sessions"

    # ── Ownership ──────────────────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exercise_plans.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    exercise_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exercises.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="The specific exercise being performed. NULL for multi-exercise sessions.",
    )

    # ── Lifecycle ──────────────────────────────────────────────────────────────
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, name="session_status"),
        nullable=False,
        default=SessionStatus.PENDING,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Set when the patient sends the first frame over WebSocket.",
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ── Outcome metrics (written by session_scorer on COMPLETED) ───────────────
    avg_quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Mean form quality score across all frames in the session (0–100).",
    )
    completion_pct: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Proportion of prescribed sets/reps completed (0.0–1.0).",
    )
    post_session_pain: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Self-reported pain level 1–10 submitted by patient at session end.",
    )
    total_reps_completed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_sets_completed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peak_rom_degrees: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Maximum range-of-motion angle recorded during this session.",
    )

    # ── Post-session summary ───────────────────────────────────────────────────
    summary_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="AI-generated plain-language session summary sent to the patient.",
    )
    plan_adapted: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        comment="True if the plan was modified by the plan_adapter after this session.",
    )
    patient_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional free-text notes entered by the patient after the session.",
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    patient: Mapped[PatientProfile] = relationship(
        "PatientProfile",
        back_populates="sessions",
    )
    plan: Mapped[ExercisePlan] = relationship(
        "ExercisePlan",
        back_populates="sessions",
    )
    exercise: Mapped[Exercise | None] = relationship(
        "Exercise",
        back_populates="sessions",
    )
    feedback_events: Mapped[list[FeedbackEvent]] = relationship(
        "FeedbackEvent",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="FeedbackEvent.occurred_at",
    )
    red_flag_events: Mapped[list[RedFlagEvent]] = relationship(
        "RedFlagEvent",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    media_files: Mapped[list[MediaFile]] = relationship(
        "MediaFile",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    # ── Computed helpers ───────────────────────────────────────────────────────
    @property
    def duration_seconds(self) -> int | None:
        if self.started_at and self.ended_at:
            return int((self.ended_at - self.started_at).total_seconds())
        return None