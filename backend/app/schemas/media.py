"""
Schemas for media upload endpoints:
  POST /api/v1/media/upload-url          — obtain a presigned S3 upload URL
  POST /api/v1/media/{id}/process        — confirm upload, enqueue Celery task
  POST /api/v1/media/processed-notify    — internal webhook called by Celery worker

Upload flow
-----------
  1. Client calls POST /media/upload-url with the media_type.
     Server creates a MediaFile DB row (status=PENDING) and returns a
     presigned S3 URL, the s3_key, and the media_id.

  2. Client PUTs the raw video bytes directly to S3 using the presigned URL.
     This never touches the API server, keeping upload bandwidth off the
     backend process.

  3. Client calls POST /media/{id}/process to confirm the upload is done.
     Server sets status=PROCESSING and enqueues the Celery video_tasks worker.

  4. Celery worker completes analysis and calls POST /media/processed-notify
     (internal route, bearer-token protected) to update the DB row.
     A push notification is sent to the patient on completion.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import Field, field_validator

from app.models.media import MediaType, ProcessingStatus
from app.schemas.base import AppBaseModel, AppResponseModel


# ── Requests ──────────────────────────────────────────────────────────────────

class UploadUrlRequest(AppBaseModel):
    """
    Client requests a presigned S3 upload URL.
    The session_id is required for session recordings; omit for intake videos.
    """

    media_type: MediaType = Field(
        ...,
        description="'intake' for initial assessment videos, 'session_recording' for live session captures.",
    )
    session_id: UUID | None = Field(
        None,
        description="Required when media_type='session_recording'.",
    )
    file_name: str = Field(
        ...,
        max_length=256,
        description="Original filename including extension, e.g. 'intake_video.mp4'.",
    )
    file_size_bytes: int = Field(
        ...,
        gt=0,
        description="Exact file size in bytes. Validated against S3_MAX_UPLOAD_BYTES server-side.",
    )
    mime_type: str = Field(
        "video/mp4",
        description="MIME type of the file being uploaded.",
    )

    @field_validator("mime_type")
    @classmethod
    def allowed_mime_types(cls, v: str) -> str:
        allowed = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}
        if v not in allowed:
            raise ValueError(
                f"Unsupported MIME type '{v}'. Allowed: {', '.join(sorted(allowed))}."
            )
        return v

    @field_validator("session_id", mode="after")
    @classmethod
    def session_required_for_recording(cls, v: UUID | None, info) -> UUID | None:
        data = info.data
        if data.get("media_type") == MediaType.SESSION_RECORDING and v is None:
            raise ValueError("session_id is required when media_type is 'session_recording'.")
        return v


class ProcessingNotifyRequest(AppBaseModel):
    """
    Sent by the Celery video worker to the internal webhook on completion.
    Protected by a shared internal bearer token, not the patient JWT.
    """

    media_id: UUID
    status: ProcessingStatus = Field(
        ...,
        description="The final processing status: 'done' or 'failed'.",
    )
    duration_seconds: int | None = Field(
        None,
        ge=1,
        description="Video duration extracted during processing.",
    )
    processing_error: str | None = Field(
        None,
        max_length=1024,
        description="Error message if status='failed'.",
    )
    # Baseline ROM extracted from intake videos (populated by video_intake_analyzer)
    baseline_rom: dict | None = Field(
        None,
        description=(
            "Per-joint baseline ROM measurements. Only present for intake videos "
            "after video_intake_analyzer has run. Written to PatientProfile.baseline_rom."
        ),
    )
    mobility_notes: str | None = Field(
        None,
        description="Plain-language mobility summary generated from the intake video.",
    )


# ── Responses ─────────────────────────────────────────────────────────────────

class UploadUrlResponse(AppResponseModel):
    """
    Returned immediately after POST /media/upload-url.
    The client must complete the S3 PUT upload within expires_in_seconds
    before the presigned URL becomes invalid.
    """

    media_id: UUID = Field(
        ...,
        description="DB record ID — pass to POST /media/{id}/process after the upload completes.",
    )
    presigned_url: str = Field(
        ...,
        description="Presigned S3 PUT URL. The client uploads directly to this URL.",
    )
    s3_key: str = Field(
        ...,
        description="S3 object key. Store this — it is required to reference the file later.",
    )
    expires_in_seconds: int = Field(
        ...,
        description="Seconds until the presigned URL expires. Default: 900 (15 minutes).",
    )
    max_file_size_bytes: int = Field(
        ...,
        description="Maximum accepted upload size enforced by the S3 bucket policy.",
    )


class MediaFileResponse(AppResponseModel):
    """Full media file record returned by GET /media/{id}."""

    id: UUID
    patient_id: UUID
    session_id: UUID | None
    s3_key: str
    media_type: MediaType
    processing_status: ProcessingStatus
    duration_seconds: int | None
    file_size_bytes: int | None
    mime_type: str | None
    processed_at: datetime | None
    processing_error: str | None = Field(
        None,
        description="Present only if processing_status='failed'.",
    )
    created_at: datetime


class ProcessConfirmResponse(AppResponseModel):
    """Returned by POST /media/{id}/process after the Celery task is enqueued."""

    media_id: UUID
    status: ProcessingStatus
    message: str = Field(
        ...,
        description="Human-readable confirmation, e.g. 'Video queued for processing.'",
    )
    estimated_processing_seconds: int = Field(
        ...,
        description=(
            "Rough estimate of how long processing will take. "
            "The client can poll GET /media/{id} or wait for a push notification."
        ),
    )