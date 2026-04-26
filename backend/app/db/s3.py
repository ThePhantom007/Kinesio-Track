"""
Boto3 S3 client wrapper for all video file operations.

All methods are async — they run the synchronous boto3 calls in a thread
pool executor so they never block the event loop.

Bucket structure
----------------
  patients/{patient_id}/intake/{uuid}.mp4
  patients/{patient_id}/sessions/{session_id}/{uuid}.mp4
  patients/{patient_id}/thumbnails/{media_id}.jpg

Access model
------------
  - Uploads:   client uploads directly to S3 via a presigned PUT URL.
               The API server never receives the raw bytes.
  - Downloads: the Celery worker uses download_video() to pull files to disk
               for MediaPipe processing.  Patients receive presigned GET URLs
               from the API — they also never stream through the backend.
  - Deletion:  soft-deleted in DB first, then hard-deleted from S3 by the
               Celery cleanup_tasks scheduled task.

Environment variables (from settings)
--------------------------------------
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME
  For local development with MinIO set endpoint_url in create_s3_client().
"""

from __future__ import annotations

import asyncio
import os
from functools import partial
from typing import Any
from uuid import UUID, uuid4

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.core.config import settings
from app.core.exceptions import ExternalServiceError, VideoDownloadError
from app.core.logging import get_logger

log = get_logger(__name__)

# ── Client factory ────────────────────────────────────────────────────────────

def _make_client():
    """Create a synchronous boto3 S3 client."""
    kwargs: dict[str, Any] = {
        "region_name":          settings.AWS_REGION,
        "aws_access_key_id":    settings.AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": settings.AWS_SECRET_ACCESS_KEY,
    }
    # MinIO / local override: set S3_ENDPOINT_URL env var
    endpoint = os.getenv("S3_ENDPOINT_URL")
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def _run_sync(fn, *args, **kwargs):
    """Run a synchronous callable in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, partial(fn, *args, **kwargs))


# ── S3 key builders ───────────────────────────────────────────────────────────

def intake_key(patient_id: UUID) -> str:
    """Return the S3 key for a new intake video."""
    return f"patients/{patient_id}/intake/{uuid4()}.mp4"


def session_recording_key(patient_id: UUID, session_id: UUID) -> str:
    """Return the S3 key for a session recording."""
    return f"patients/{patient_id}/sessions/{session_id}/{uuid4()}.mp4"


def thumbnail_key(patient_id: UUID, media_id: UUID) -> str:
    """Return the S3 key for a thumbnail image."""
    return f"patients/{patient_id}/thumbnails/{media_id}.jpg"


# ── Presigned URLs ────────────────────────────────────────────────────────────

async def generate_presigned_upload_url(
    s3_key: str,
    content_type: str = "video/mp4",
    expires_in: int | None = None,
) -> str:
    """
    Generate a presigned S3 PUT URL for direct client upload.

    The client must include the Content-Type header in the PUT request,
    otherwise S3 will reject it (the presigned URL is scoped to the
    content-type at signing time).

    Args:
        s3_key:       Full S3 object key (use key builders above).
        content_type: MIME type of the file being uploaded.
        expires_in:   URL lifetime in seconds. Defaults to S3_PRESIGNED_EXPIRES.

    Returns:
        Presigned URL string.

    Raises:
        ExternalServiceError: S3 client error or missing credentials.
    """
    expires = expires_in or settings.S3_PRESIGNED_EXPIRES
    client  = _make_client()

    try:
        url = await _run_sync(
            client.generate_presigned_url,
            "put_object",
            Params={
                "Bucket":      settings.S3_BUCKET_NAME,
                "Key":         s3_key,
                "ContentType": content_type,
            },
            ExpiresIn=expires,
        )
        log.debug(
            "presigned_upload_url_generated",
            s3_key=s3_key,
            expires_in=expires,
        )
        return url
    except (ClientError, NoCredentialsError) as exc:
        raise ExternalServiceError(
            f"Failed to generate presigned upload URL for {s3_key}: {exc}",
            detail={"s3_key": s3_key},
        ) from exc


async def generate_presigned_download_url(
    s3_key: str,
    expires_in: int = 900,
) -> str:
    """
    Generate a presigned S3 GET URL for temporary client access to a file.

    Args:
        s3_key:    Full S3 object key.
        expires_in: URL lifetime in seconds (default 15 minutes).

    Returns:
        Presigned URL string.

    Raises:
        ExternalServiceError: S3 client error.
    """
    client = _make_client()

    try:
        url = await _run_sync(
            client.generate_presigned_url,
            "get_object",
            Params={
                "Bucket": settings.S3_BUCKET_NAME,
                "Key":    s3_key,
            },
            ExpiresIn=expires_in,
        )
        return url
    except (ClientError, NoCredentialsError) as exc:
        raise ExternalServiceError(
            f"Failed to generate presigned download URL for {s3_key}: {exc}",
            detail={"s3_key": s3_key},
        ) from exc


# ── File operations ───────────────────────────────────────────────────────────

async def download_video(
    *,
    bucket: str,
    key: str,
    dest_path: str,
) -> None:
    """
    Download an S3 object to a local file path.

    Used exclusively by the Celery video worker — the API server never
    downloads video files directly.

    Args:
        bucket:    S3 bucket name.
        key:       S3 object key.
        dest_path: Absolute local path to write the file to.

    Raises:
        VideoDownloadError: Object not found or S3 error.
    """
    client = _make_client()

    try:
        await _run_sync(
            client.download_file,
            bucket,
            key,
            dest_path,
        )
        file_size = os.path.getsize(dest_path)
        log.info(
            "video_downloaded",
            bucket=bucket,
            key=key,
            dest=dest_path,
            size_mb=round(file_size / 1_048_576, 2),
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "unknown")
        if error_code == "404":
            raise VideoDownloadError(
                f"Video not found in S3: s3://{bucket}/{key}",
                detail={"bucket": bucket, "key": key},
            ) from exc
        raise VideoDownloadError(
            f"S3 download failed for s3://{bucket}/{key}: {exc}",
            detail={"bucket": bucket, "key": key, "error_code": error_code},
        ) from exc
    except Exception as exc:
        raise VideoDownloadError(
            f"Unexpected error downloading s3://{bucket}/{key}: {exc}",
            detail={"bucket": bucket, "key": key},
        ) from exc


async def delete_video(s3_key: str, bucket: str | None = None) -> None:
    """
    Delete a single object from S3.

    Args:
        s3_key: Full S3 object key.
        bucket: Bucket name. Defaults to S3_BUCKET_NAME from settings.

    Raises:
        ExternalServiceError: S3 client error (not raised for 404 — already gone).
    """
    bucket = bucket or settings.S3_BUCKET_NAME
    client = _make_client()

    try:
        await _run_sync(
            client.delete_object,
            Bucket=bucket,
            Key=s3_key,
        )
        log.info("video_deleted", bucket=bucket, s3_key=s3_key)
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "unknown")
        if error_code == "NoSuchKey":
            log.debug("delete_noop_already_gone", s3_key=s3_key)
            return
        raise ExternalServiceError(
            f"Failed to delete s3://{bucket}/{s3_key}: {exc}",
            detail={"s3_key": s3_key, "error_code": error_code},
        ) from exc


async def delete_videos_batch(s3_keys: list[str], bucket: str | None = None) -> int:
    """
    Delete multiple objects in a single S3 DeleteObjects request (max 1000).

    Args:
        s3_keys: List of S3 object keys to delete.
        bucket:  Bucket name. Defaults to S3_BUCKET_NAME from settings.

    Returns:
        Number of objects successfully deleted.

    Raises:
        ExternalServiceError: S3 client error.
    """
    if not s3_keys:
        return 0

    bucket = bucket or settings.S3_BUCKET_NAME
    client = _make_client()

    # S3 DeleteObjects is limited to 1000 keys per request
    deleted_count = 0
    for chunk_start in range(0, len(s3_keys), 1000):
        chunk = s3_keys[chunk_start : chunk_start + 1000]
        objects = [{"Key": k} for k in chunk]

        try:
            response = await _run_sync(
                client.delete_objects,
                Bucket=bucket,
                Delete={"Objects": objects, "Quiet": True},
            )
            errors = response.get("Errors", [])
            if errors:
                log.warning(
                    "s3_batch_delete_partial_failure",
                    error_count=len(errors),
                    first_error=errors[0],
                )
            deleted_count += len(chunk) - len(errors)
        except (ClientError, NoCredentialsError) as exc:
            raise ExternalServiceError(
                f"S3 batch delete failed: {exc}",
                detail={"key_count": len(chunk)},
            ) from exc

    log.info("videos_batch_deleted", count=deleted_count)
    return deleted_count


async def object_exists(s3_key: str, bucket: str | None = None) -> bool:
    """
    Check whether an S3 object exists without downloading it.

    Args:
        s3_key: Full S3 object key.
        bucket: Bucket name. Defaults to S3_BUCKET_NAME from settings.

    Returns:
        True if the object exists, False otherwise.
    """
    bucket = bucket or settings.S3_BUCKET_NAME
    client = _make_client()

    try:
        await _run_sync(client.head_object, Bucket=bucket, Key=s3_key)
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return False
        raise ExternalServiceError(
            f"S3 head_object failed for {s3_key}: {exc}",
        ) from exc


async def get_object_size(s3_key: str, bucket: str | None = None) -> int:
    """
    Return the size in bytes of an S3 object.

    Raises:
        ExternalServiceError: Object not found or S3 error.
    """
    bucket = bucket or settings.S3_BUCKET_NAME
    client = _make_client()

    try:
        response = await _run_sync(
            client.head_object, Bucket=bucket, Key=s3_key
        )
        return int(response["ContentLength"])
    except ClientError as exc:
        raise ExternalServiceError(
            f"Failed to get object size for {s3_key}: {exc}",
        ) from exc