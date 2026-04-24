"""
MediaFile tracks every video uploaded by a patient — both intake assessment
videos and session recordings.

Storage model
-------------
Files live in S3.  The DB stores only the object key and metadata; actual
bytes are never written to Postgres.  The video_tasks Celery worker reads
``processing_status`` to find pending files and updates it on completion.

Pre-signed URL flow
-------------------
  1. Client calls POST /media/upload-url → receives (presigned_url, s3_key, media_id).
  2. Client PUTs directly to S3 using the presigned URL.
  3. Client calls POST /media/{id}/process → enqueues Celery task.
  4. Celery worker sets processing_status to PROCESSING, runs MediaPipe,
     writes results, then sets status to DONE (or FAILED).
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

if TYPE_CHECKING:
    from app.models.patient import PatientProfile
    from app.models.session import ExerciseSession


class MediaType(str, enum.Enum):
    INTAKE = "intake"
    SESSION_RECORDING = "session_recording"


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class MediaFile(BaseModel):
    __tablename__ = "media_files"

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
        comment="NULL for intake videos that precede any session.",
    )

    # ── Storage ────────────────────────────────────────────────────────────────
    s3_key: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        unique=True,
        comment="Full S3 object key, e.g. 'patients/{patient_id}/intake/{uuid}.mp4'.",
    )
    s3_bucket: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Bucket name — stored explicitly so files can be moved between buckets.",
    )
    thumbnail_s3_key: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="S3 key of the auto-generated thumbnail image.",
    )

    # ── Metadata ───────────────────────────────────────────────────────────────
    media_type: Mapped[MediaType] = mapped_column(
        Enum(MediaType, name="media_type"),
        nullable=False,
        index=True,
    )
    duration_seconds: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        comment="Video duration extracted during processing.",
    )
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="e.g. 'video/mp4'",
    )

    # ── Processing ─────────────────────────────────────────────────────────────
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus, name="processing_status"),
        nullable=False,
        default=ProcessingStatus.PENDING,
        index=True,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    processing_error: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        comment="Error message if processing_status is FAILED.",
    )

    # ── Relationships ──────────────────────────────────────────────────────────
    patient: Mapped[PatientProfile] = relationship(
        "PatientProfile",
        back_populates="media_files",
    )
    session: Mapped[ExerciseSession | None] = relationship(
        "ExerciseSession",
        back_populates="media_files",
    )

    # ── Helpers ────────────────────────────────────────────────────────────────
    @property
    def is_processed(self) -> bool:
        return self.processing_status == ProcessingStatus.DONE