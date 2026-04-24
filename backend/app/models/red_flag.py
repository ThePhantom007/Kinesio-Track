"""
A RedFlagEvent is the audit record for a clinical escalation triggered during
a session.

Trigger sources
---------------
  - pose_analyzer:       bilateral asymmetry ratio > threshold
  - red_flag_monitor:    post_session_pain >= PAIN_RED_FLAG_THRESHOLD
  - red_flag_monitor:    ROM regression across consecutive sessions
  - red_flag_monitor:    exercise.red_flags condition matched mid-session

Escalation flow
---------------
  1. red_flag_monitor detects the trigger condition.
  2. Calls claude_client.escalate_red_flag() → receives severity + messages.
  3. Writes a RedFlagEvent row (this model).
  4. Sends patient-facing immediate_action message over the WebSocket.
  5. Notifies the assigned clinician via webhook / email (notification_tasks).
  6. Clinician acknowledges via PATCH /clinicians/{id}/alerts/{red_flag_id}.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.patient import PatientProfile
    from app.models.session import ExerciseSession


class RedFlagSeverity(str, enum.Enum):
    WARN = "warn"           # continue with caution; message sent
    STOP = "stop"           # stop current exercise; rest and reassess
    SEEK_CARE = "seek_care" # stop all activity; contact a clinician


class RedFlagTrigger(str, enum.Enum):
    PAIN_SPIKE = "pain_spike"
    ROM_REGRESSION = "rom_regression"
    COMPENSATION_PATTERN = "compensation_pattern"
    BILATERAL_ASYMMETRY = "bilateral_asymmetry"
    EXERCISE_RED_FLAG = "exercise_red_flag"     # matched a plan_exercise.red_flags rule
    CLINICIAN_MANUAL = "clinician_manual"       # clinician escalated manually


class RedFlagEvent(BaseModel):
    __tablename__ = "red_flag_events"

    # ── Ownership ──────────────────────────────────────────────────────────────
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patient_profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exercise_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="NULL if the red flag was raised outside of an active session.",
    )

    # ── Trigger ────────────────────────────────────────────────────────────────
    trigger_type: Mapped[RedFlagTrigger] = mapped_column(
        Enum(RedFlagTrigger, name="red_flag_trigger"),
        nullable=False,
        index=True,
    )
    trigger_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "Structured context describing what triggered the flag. "
            "Shape varies by trigger_type, e.g. for PAIN_SPIKE: "
            "{pain_score: 9, previous_avg: 3.4}."
        ),
    )

    # ── Claude output ──────────────────────────────────────────────────────────
    severity: Mapped[RedFlagSeverity] = mapped_column(
        Enum(RedFlagSeverity, name="red_flag_severity"),
        nullable=False,
        index=True,
    )
    immediate_action: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Patient-facing instruction sent immediately over the WebSocket or push notification.",
    )
    clinician_note: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Clinical context summary sent to the assigned clinician.",
    )
    session_recommendation: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="AI recommendation on whether to resume, modify, or discontinue the session.",
    )
    claude_raw_response: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Full validated Claude response stored for audit and retraining.",
    )

    # ── Clinician acknowledgement ──────────────────────────────────────────────
    acknowledged_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="user_id of the clinician who acknowledged this alert.",
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    clinician_response_notes: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Optional free-text notes added by the clinician when acknowledging.",
    )

    # ── Notification tracking ──────────────────────────────────────────────────
    clinician_notified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp when the clinician notification was dispatched.",
    )
    notification_method: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="webhook | email | push",
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    patient: Mapped[PatientProfile] = relationship("PatientProfile")
    session: Mapped[ExerciseSession | None] = relationship(
        "ExerciseSession",
        back_populates="red_flag_events",
    )

    # ── Computed helpers ───────────────────────────────────────────────────────
    @property
    def is_acknowledged(self) -> bool:
        return self.acknowledged_at is not None

    @property
    def requires_session_stop(self) -> bool:
        return self.severity in (RedFlagSeverity.STOP, RedFlagSeverity.SEEK_CARE)