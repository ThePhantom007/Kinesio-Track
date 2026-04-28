"""
Media file upload and management endpoints:
  POST /api/v1/media/upload-url        — get presigned S3 PUT URL
  POST /api/v1/media/{id}/process      — confirm upload, enqueue Celery task
  POST /api/v1/media/processed-notify  — internal webhook called by Celery worker
  GET  /api/v1/media/{id}              — media file status and metadata
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, Header
from sqlalchemy import select

from app.api.deps import CurrentUser, DBSession, get_patient_profile
from app.core.config import settings
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.db.s3 import (
    generate_presigned_upload_url,
    intake_key,
    object_exists,
    session_recording_key,
)
from app.models.media import MediaFile, MediaType, ProcessingStatus
from app.models.user import UserRole
from app.schemas.media import (
    MediaFileResponse,
    ProcessConfirmResponse,
    ProcessingNotifyRequest,
    UploadUrlRequest,
    UploadUrlResponse,
)

router = APIRouter(prefix="/media", tags=["media"])


# ── Request presigned upload URL ───────────────────────────────────────────────

@router.post("/upload-url", response_model=UploadUrlResponse, status_code=201)
async def request_upload_url(
    body: UploadUrlRequest,
    db: DBSession,
    current_user: CurrentUser,
    patient=Depends(get_patient_profile),
):
    """
    Generate a presigned S3 PUT URL for direct client upload.

    Steps:
      1. Validate file size against S3_MAX_UPLOAD_BYTES.
      2. Build a structured S3 key from the patient and media type.
      3. Create a MediaFile DB row (status=PENDING) to track the upload.
      4. Return the presigned URL, s3_key, and media_id.

    The client must complete the PUT to S3 within ``expires_in_seconds``
    (default 15 minutes) then call POST /media/{id}/process to confirm.
    """
    if body.file_size_bytes > settings.S3_MAX_UPLOAD_BYTES:
        from app.core.exceptions import ValidationError
        raise ValidationError(
            f"File size {body.file_size_bytes} bytes exceeds the maximum "
            f"allowed size of {settings.S3_MAX_UPLOAD_BYTES} bytes.",
            detail={"max_bytes": settings.S3_MAX_UPLOAD_BYTES},
        )

    # Build S3 key
    if body.media_type == MediaType.INTAKE:
        s3_key = intake_key(patient.id)
    else:
        s3_key = session_recording_key(patient.id, body.session_id)

    # Generate presigned URL
    presigned_url = await generate_presigned_upload_url(
        s3_key=s3_key,
        content_type=body.mime_type,
        expires_in=settings.S3_PRESIGNED_EXPIRES,
    )

    # Create MediaFile tracking row
    media_file = MediaFile(
        patient_id=patient.id,
        session_id=body.session_id,
        s3_key=s3_key,
        s3_bucket=settings.S3_BUCKET_NAME,
        media_type=body.media_type,
        file_size_bytes=body.file_size_bytes,
        mime_type=body.mime_type,
        processing_status=ProcessingStatus.PENDING,
    )
    db.add(media_file)
    await db.flush()

    return UploadUrlResponse(
        media_id=media_file.id,
        presigned_url=presigned_url,
        s3_key=s3_key,
        expires_in_seconds=settings.S3_PRESIGNED_EXPIRES,
        max_file_size_bytes=settings.S3_MAX_UPLOAD_BYTES,
    )


# ── Confirm upload and enqueue processing ─────────────────────────────────────

@router.post("/{media_id}/process", response_model=ProcessConfirmResponse)
async def confirm_and_process(
    media_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """
    Confirm the S3 upload is complete and enqueue the Celery processing task.

    Verifies the S3 object actually exists before enqueueing (guards against
    clients calling this endpoint without completing the upload).
    """
    media_file = await _load_media(db, media_id, current_user)

    if media_file.processing_status != ProcessingStatus.PENDING:
        return ProcessConfirmResponse(
            media_id=media_file.id,
            status=media_file.processing_status,
            message="This file has already been submitted for processing.",
            estimated_processing_seconds=0,
        )

    # Verify the object actually landed in S3
    exists = await object_exists(media_file.s3_key, media_file.s3_bucket)
    if not exists:
        from app.core.exceptions import ValidationError
        raise ValidationError(
            "The file has not been uploaded to storage yet. "
            "Complete the S3 PUT upload before calling this endpoint.",
            detail={"s3_key": media_file.s3_key},
        )

    # Enqueue Celery task
    from app.workers.video_tasks import process_intake_video
    process_intake_video.delay(str(media_id))

    media_file.processing_status = ProcessingStatus.PROCESSING
    db.add(media_file)
    await db.flush()

    # Estimate based on typical processing time (~1s per second of video)
    estimated_seconds = 60  # conservative default before duration is known

    return ProcessConfirmResponse(
        media_id=media_file.id,
        status=ProcessingStatus.PROCESSING,
        message="Video queued for processing. You'll receive a notification when complete.",
        estimated_processing_seconds=estimated_seconds,
    )


# ── Internal webhook (called by Celery worker) ────────────────────────────────

@router.post("/processed-notify", include_in_schema=False)
async def processing_complete_notify(
    body: ProcessingNotifyRequest,
    db: DBSession,
    x_internal_token: str = Header(..., alias="X-Internal-Token"),
):
    """
    Internal webhook called by the Celery video_tasks worker on completion.

    Protected by a shared internal token (not the patient JWT).
    Writes the processing results to the DB and updates the media status.
    """
    # Validate internal token
    import secrets
    expected = settings.SECRET_KEY
    if not secrets.compare_digest(x_internal_token, expected):
        from app.core.exceptions import AuthenticationError
        raise AuthenticationError("Invalid internal token.")

    result = await db.execute(
        select(MediaFile).where(MediaFile.id == body.media_id)
    )
    media_file = result.scalar_one_or_none()
    if media_file is None:
        raise NotFoundError(f"MediaFile {body.media_id} not found.")

    media_file.processing_status = body.status
    media_file.processed_at      = datetime.now(timezone.utc)
    if body.duration_seconds:
        media_file.duration_seconds = body.duration_seconds
    if body.processing_error:
        media_file.processing_error = body.processing_error

    # Write baseline ROM and mobility notes to patient profile if provided
    if body.baseline_rom or body.mobility_notes:
        from app.models.patient import PatientProfile
        patient = await db.get(PatientProfile, media_file.patient_id)
        if patient:
            if body.baseline_rom:
                patient.baseline_rom   = body.baseline_rom
            if body.mobility_notes:
                patient.mobility_notes = body.mobility_notes
            db.add(patient)

    db.add(media_file)
    await db.flush()

    return {"status": "ok"}


# ── Get media file ─────────────────────────────────────────────────────────────

@router.get("/{media_id}", response_model=MediaFileResponse)
async def get_media_file(
    media_id: UUID,
    db: DBSession,
    current_user: CurrentUser,
):
    """Return media file metadata and processing status."""
    media_file = await _load_media(db, media_id, current_user)
    return MediaFileResponse(
        id=media_file.id,
        patient_id=media_file.patient_id,
        session_id=media_file.session_id,
        s3_key=media_file.s3_key,
        media_type=media_file.media_type,
        processing_status=media_file.processing_status,
        duration_seconds=media_file.duration_seconds,
        file_size_bytes=media_file.file_size_bytes,
        mime_type=media_file.mime_type,
        processed_at=media_file.processed_at,
        processing_error=media_file.processing_error,
        created_at=media_file.created_at,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _load_media(db, media_id: UUID, current_user: CurrentUser) -> MediaFile:
    result = await db.execute(select(MediaFile).where(MediaFile.id == media_id))
    media_file = result.scalar_one_or_none()
    if media_file is None:
        raise NotFoundError(f"Media file {media_id} not found.")

    if current_user.role == UserRole.PATIENT:
        from app.models.patient import PatientProfile
        from sqlalchemy import select as sa_select
        p = await db.execute(
            sa_select(PatientProfile).where(PatientProfile.user_id == current_user.id)
        )
        patient = p.scalar_one_or_none()
        if patient is None or media_file.patient_id != patient.id:
            raise PermissionDeniedError("You do not have access to this file.")
    return media_file