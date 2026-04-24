"""
An individual exercise within a plan phase.

landmark_rules
--------------
JSONB column consumed by the pose_analyzer service.  It defines per-joint
acceptable angle ranges so the rules engine can flag deviations without
calling Claude on every frame.

Schema example:
{
  "left_knee":      {"min_angle": 80,  "max_angle": 120, "axis": "sagittal"},
  "right_knee":     {"min_angle": 80,  "max_angle": 120, "axis": "sagittal"},
  "lumbar_spine":   {"min_angle": 160, "max_angle": 180, "axis": "sagittal"},
  "left_ankle":     {"min_angle": 70,  "max_angle": 110, "axis": "sagittal"}
}

red_flags
---------
JSONB list of conditions that, if triggered mid-exercise, route to the
red_flag_monitor service immediately.

Schema example:
[
  {"condition": "left_knee.angle < 40",  "action": "stop",    "reason": "hyperflexion risk"},
  {"condition": "pain_score >= 8",        "action": "stop",    "reason": "acute pain spike"},
  {"condition": "bilateral_asymmetry > 25", "action": "warn", "reason": "compensation pattern"}
]
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.feedback_event import FeedbackEvent
    from app.models.phase import PlanPhase
    from app.models.session import ExerciseSession


class Exercise(BaseModel):
    __tablename__ = "exercises"

    phase_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("plan_phases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # ── Identity ───────────────────────────────────────────────────────────────
    slug: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Stable machine-readable identifier, e.g. 'seated-ankle-circles'.",
    )
    name: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Human-readable display name.",
    )
    order_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Display order within the phase (0-based ascending).",
    )

    # ── Prescription ───────────────────────────────────────────────────────────
    sets: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    reps: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    hold_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Duration to hold at end range. 0 means no hold required.",
    )
    rest_seconds: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=30,
        comment="Rest period between sets in seconds.",
    )
    tempo: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="Optional tempo notation, e.g. '2-1-2' (eccentric-hold-concentric).",
    )

    # ── Target anatomy ─────────────────────────────────────────────────────────
    target_joints: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
        default=list,
        comment="MediaPipe joint names this exercise primarily targets, e.g. ['left_ankle', 'right_ankle'].",
    )

    # ── Pose analysis rules ────────────────────────────────────────────────────
    landmark_rules: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Per-joint angle ranges consumed by the pose_analyzer rules engine. See module docstring.",
    )
    red_flags: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Mid-exercise conditions that trigger immediate red_flag_monitor escalation.",
    )

    # ── Patient-facing content ─────────────────────────────────────────────────
    patient_instructions: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        comment="Step-by-step instructions shown to the patient before and during the exercise.",
    )
    reference_video_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True,
        comment="URL of a demonstration video shown to the patient.",
    )
    difficulty: Mapped[str | None] = mapped_column(
        String(32),
        nullable=True,
        comment="beginner | intermediate | advanced",
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    phase: Mapped[PlanPhase] = relationship(
        "PlanPhase",
        back_populates="exercises",
    )
    sessions: Mapped[list[ExerciseSession]] = relationship(
        "ExerciseSession",
        back_populates="exercise",
    )
    feedback_events: Mapped[list[FeedbackEvent]] = relationship(
        "FeedbackEvent",
        back_populates="exercise",
        cascade="all, delete-orphan",
    )