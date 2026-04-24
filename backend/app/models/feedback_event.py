"""
A FeedbackEvent is one correction message sent to the patient during a live
session over the WebSocket channel.

These rows are the ground truth for session replay — the frontend can
reconstruct the full annotated timeline by querying all feedback events for
a session ordered by occurred_at.

Write path
----------
During a live session, feedback events are buffered in Redis (as a list keyed
by session_id) to avoid per-frame Postgres writes on the hot path.
The session_manager service flushes the buffer to Postgres in a single bulk
INSERT when the session ends.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.exercise import Exercise
    from app.models.session import ExerciseSession


class FeedbackSeverity(str, enum.Enum):
    INFO = "info"       # positive reinforcement / milestone
    WARNING = "warning" # form deviation, correctable
    ERROR = "error"     # significant deviation requiring immediate correction
    STOP = "stop"       # red-flag — stop the exercise


class FeedbackEvent(Base):
    """
    No UUID mixin here — uses Integer PK for efficient bulk inserts and
    TimescaleDB-friendly ordering.  session_id + occurred_at together
    act as the natural composite key for queries.
    """

    __tablename__ = "feedback_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # ── Ownership ──────────────────────────────────────────────────────────────
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exercise_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    exercise_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("exercises.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Timing ─────────────────────────────────────────────────────────────────
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Wall-clock timestamp when the feedback was generated on the server.",
    )
    frame_timestamp_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Client-side timestamp (milliseconds) of the frame that triggered this event.",
    )

    # ── Classification ─────────────────────────────────────────────────────────
    severity: Mapped[FeedbackSeverity] = mapped_column(
        Enum(FeedbackSeverity, name="feedback_severity"),
        nullable=False,
        index=True,
    )
    error_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment=(
            "Machine-readable error code from the pose_analyzer rules engine, "
            "e.g. 'knee_valgus', 'lumbar_hyperextension', 'shoulder_elevation'."
        ),
    )
    affected_joint: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="MediaPipe joint name that triggered the event.",
    )

    # ── Measurement ────────────────────────────────────────────────────────────
    actual_angle: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Measured joint angle in degrees at the time of the violation.",
    )
    expected_min_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_max_angle: Mapped[float | None] = mapped_column(Float, nullable=True)
    deviation_degrees: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Signed deviation from the nearest boundary of the acceptable range.",
    )
    form_score_at_event: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Rolling form quality score at the moment of the event (0–100).",
    )

    # ── Message ────────────────────────────────────────────────────────────────
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Patient-facing correction message sent over the WebSocket.",
    )
    from_cache: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
        comment="True if the message was served from the Redis feedback cache.",
    )

    # ── Replay data ────────────────────────────────────────────────────────────
    overlay_points: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment=(
            "Landmark coordinates to highlight in the frontend replay overlay. "
            "Schema: [{landmark_id: int, x: float, y: float, highlight: bool}]"
        ),
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    session: Mapped[ExerciseSession] = relationship(
        "ExerciseSession",
        back_populates="feedback_events",
    )
    exercise: Mapped[Exercise | None] = relationship(
        "Exercise",
        back_populates="feedback_events",
    )